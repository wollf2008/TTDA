import tqdm
import os
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .metrics import AverageMeter
from .losses import (
    compute_grad_gp,
    DomainInvariantPerceptualLoss,
    SemanticRelationConstrativeLoss)
from .utils import (
    requires_grad)

import torchvision.utils as vutils

def save_generated_images(X_gen, epoch, save_dir='./results', file_prefix='generated_image'):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 使用 epoch 编号生成文件路径
    file_path = os.path.join(save_dir, f"{file_prefix}_epoch_{epoch}.png")
    X_gen = (X_gen + 1) / 2

    # 保存图像
    vutils.save_image(X_gen, file_path, normalize=True)
class Trainer():

    def __init__(self):
        pass

    def create_metrics(
            self, hyperparameters):
        running_metrics = {
            "loss": AverageMeter(name="loss"),
            "disc_gan_loss": AverageMeter(name="disc_gan_loss"),
            "gen_gan_loss": AverageMeter(name="gen_gan_loss"),
            "style_rec_loss": AverageMeter(name="style_rec_loss"),
            "diversity_loss": AverageMeter(name="diversity_loss"),
            "cycle_consistency_loss": AverageMeter(
                name="cycle_consistency_loss"),
            "src_loss": AverageMeter(
                name="src_loss"),
            "perceptual_loss": AverageMeter(
                name="perceptual_loss"),
            "bcr_loss": AverageMeter(
                name="style_reg_loss"),
        }

        return running_metrics

    def reset_metrics(self, running_metrics):
        for metric_name, metric in running_metrics.items():
            metric.reset()

    def train(
            self,
            networks,
            dataloaders,
            optimizers,
            schedulers,
            augmentations,
            epochs,
            steps_per_epoch,
            hyperparameters,
            device,
            log_dir,
            checkpoint_path,
            save_every_epochs,
            rank=None):
        grad_scalers = {
            "SMG": torch.cuda.amp.GradScaler(
                enabled=hyperparameters["USE_MIXED_PRECISION"]),
            "D": torch.cuda.amp.GradScaler(
                enabled=hyperparameters["USE_MIXED_PRECISION"]),
                }

        criterions = {}
        if hyperparameters['LAMBDA_PERCEPTUAL_LOSS']:
            criterions["perceptual_loss"] = DomainInvariantPerceptualLoss(
                vgg_content_layers=hyperparameters["VGG_CONTENT_LAYERS"],
                vgg_style_layers=hyperparameters["VGG_STYLE_LAYERS"]).to(device)
        if hyperparameters['LAMBDA_SRC_LOSS']:
            criterions["src_loss"] = SemanticRelationConstrativeLoss(
                temperature=1/2,
                sampled_patches=1024,
                encoded_features_channels=networks['G'].module.encoder_channels,
                mlp_depth=2).to(device)

        use_logger = rank is None or rank == 0
        if use_logger:
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        running_metrics = self.create_metrics(hyperparameters)
        global_step = int(
            schedulers["SMG"].last_epoch * hyperparameters["STEPS_PER_EPOCH"])
        print(f"Last epoch was: {schedulers['SMG'].last_epoch}")
        print(f"Global step: {global_step}")
        for epoch in range(schedulers["SMG"].last_epoch, epochs):
            is_last_epoch = epoch == epochs-1
            dataloaders["train"]["src"].sampler.set_epoch(epoch)
            dataloaders["train"]["ref"].sampler.set_epoch(epoch+1)

            # Linear decay for diversity loss weight
            hyperparameters["LAMBDA_DS_DECAYED"] = hyperparameters[
                "LAMBDA_DIVERSITY_LOSS"] * (1 - epoch/epochs)

            # if use_logger:
            pbar = tqdm.tqdm(
                dataloaders["train"]["ref"],
                total=steps_per_epoch if steps_per_epoch is not None else len(
                    dataloaders["train"]),
                desc="Epoch: {}/{}".format(epoch+1, epochs))
            # else:
            #     pbar = dataloaders["train"]["ref"]

            self.reset_metrics(running_metrics)

            for batch_index, ((x_refs, d_ref), (x_src, d_src)) in enumerate(
                    zip(pbar, dataloaders["train"]["src"]), start=1):
                x_ref_1, x_ref_2 = x_refs
                outputs, losses = self.train_on_batch(
                    networks,
                    x_src,
                    x_ref_1,
                    x_ref_2,
                    d_src,
                    d_ref,
                    optimizers,
                    schedulers,
                    criterions,
                    grad_scalers,
                    augmentations,
                    hyperparameters,
                    device,
                    global_step,
                )
                global_step += 1
                for metric_name in running_metrics:
                    dist.all_reduce(
                        losses[metric_name],
                        op=dist.ReduceOp.SUM)
                    metric_val = losses[metric_name] / dist.get_world_size()
                    running_metrics[metric_name].update(
                        metric_val.item(),
                        int(x_src.size(0) * dist.get_world_size()))
                if batch_index == steps_per_epoch:
                    break

                # if use_logger:
                desc = "Epoch: {}/{} -- {}".format(
                    epoch,
                    epochs,
                    ", ".join(
                        ["{}: {:.3f}".format(metric_name, metric.avg)
                         for metric_name, metric in running_metrics.items()]))
                pbar.set_description(desc)

            if use_logger:

                if 'ada_aug' in augmentations:
                    writer.add_scalar(
                        "train/p_augment",
                        augmentations["diff_aug"].p,
                        global_step)

                for metric_name, metric in running_metrics.items():
                    writer.add_scalar(
                        "train/{}".format(metric_name),
                        metric.avg,
                        global_step)

                writer.add_scalar(
                    "train/lambda_ds",
                    hyperparameters["LAMBDA_DS_DECAYED"],
                    global_step)

                for output_name, output_value in outputs.items():
                    if output_value is not None:
                        img_tensor = ((output_value+1)/2).clip(0, 1)
                        writer.add_images(
                            tag=output_name,
                            img_tensor=img_tensor,
                            global_step=global_step)

            if (epoch+1) % save_every_epochs == 0 or is_last_epoch:
                print(f"Checkpoint -- Final checkpoint: {is_last_epoch}")
                checkpoint = {
                    "networks": {},
                    "optimizers": {},
                    "schedulers": {},
                    "augmentations": {},
                    "prototypes": {}
                }
                for model_name, model in networks.items():
                    checkpoint["networks"][model_name] = model.state_dict()
                for opt_name, opt in optimizers.items():
                    checkpoint["optimizers"][opt_name] = opt.state_dict()
                for sched_name, sched in schedulers.items():
                    checkpoint["schedulers"][sched_name] = sched.state_dict()
                for aug_name, aug in augmentations.items():
                    checkpoint["augmentations"][aug_name] = aug.state_dict()

                prototypes = networks["M"].module.get_prototypes(), networks["S"].module.get_prototypes()
                checkpoint["prototypes"]["mapping_prototypes"] = prototypes[0]
                checkpoint["prototypes"]["style_encoder_prototypes"] = prototypes[1]

                checkpoint["hyperparameters"] = hyperparameters
                
                checkpoint_dir = os.path.dirname(checkpoint_path)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                    
                torch.save(
                    checkpoint,
                    checkpoint_path)

            for sched_name, sche in schedulers.items():
                schedulers[sched_name].step()

    def train_on_batch(
            self,
            networks,
            x_src,
            x_ref_1,
            x_ref_2,
            d_src,
            d_ref,
            optimizers,
            schedulers,
            criterions,
            grad_scalers,
            augmentations,
            hyperparameters,
            device,
            global_step,):


        # Train the discriminator to distinguish real from generated images
        optimizers["SMG"].zero_grad()
        optimizers["D"].zero_grad()

        requires_grad(networks["S"], False)
        requires_grad(networks["M"], False)
        requires_grad(networks["G"], False)
        requires_grad(networks["D"], True)

        with torch.cuda.amp.autocast(
                enabled=hyperparameters["USE_MIXED_PRECISION"]):
            x_src = x_src.to(device)
            x_ref_1 = x_ref_1.to(device)
            x_ref_2 = x_ref_2.to(device)

            d_src = d_src.to(device)
            d_ref = d_ref.to(device)

            # z_ref_1 = torch.randn(
            #     x_ref_1.size(0), hyperparameters["LATENT_DIM"], device=device)
            # z_ref_2 = torch.randn(
            #     x_ref_2.size(0), hyperparameters["LATENT_DIM"], device=device)

            d_losses, _ = self.compute_d_loss(
                networks=networks,
                augmentations=augmentations,
                hyperparameters=hyperparameters,
                x_src=x_src,
                d_src=d_src,
                d_ref=d_ref,
                # z_ref=z_ref_1,
                x_ref=None)

        grad_scalers["D"].scale(d_losses["disc_loss"]).backward()
        grad_scalers["D"].step(optimizers["D"])
        grad_scalers["D"].update()

        optimizers["SMG"].zero_grad()
        optimizers["D"].zero_grad()
        with torch.cuda.amp.autocast(
                enabled=hyperparameters["USE_MIXED_PRECISION"]):
            d_losses, _ = self.compute_d_loss(
                networks=networks,
                augmentations=augmentations,
                hyperparameters=hyperparameters,
                x_src=x_src,
                d_src=d_src,
                d_ref=d_ref,
                # z_ref=None,
                x_ref=x_ref_1)

        grad_scalers["D"].scale(d_losses["disc_loss"]).backward()
        grad_scalers["D"].step(optimizers["D"])
        grad_scalers["D"].update()

        # # Train the generator to fool discriminator
        optimizers["SMG"].zero_grad()
        optimizers["D"].zero_grad()

        requires_grad(networks["S"], True)
        requires_grad(networks["M"], True)
        requires_grad(networks["G"], True)
        requires_grad(networks["D"], False)

        with torch.cuda.amp.autocast(
                enabled=hyperparameters["USE_MIXED_PRECISION"]):
            g_losses, _ = self.compute_g_loss(
                networks=networks,
                augmentations=augmentations,
                criterions=criterions,
                hyperparameters=hyperparameters,
                x_src=x_src,
                d_src=d_src,
                d_ref=d_ref,
                # z_refs=[z_ref_1, z_ref_2],
                x_refs=None,
            )

        grad_scalers["SMG"].scale(g_losses["loss"]).backward()
        grad_scalers["SMG"].step(optimizers["SMG"])
        grad_scalers["SMG"].update()

        # Train the generator to fool discriminator
        optimizers["SMG"].zero_grad()
        optimizers["D"].zero_grad()

        with torch.cuda.amp.autocast(
                enabled=hyperparameters["USE_MIXED_PRECISION"]):
            g_losses, outputs = self.compute_g_loss(
                networks=networks,
                augmentations=augmentations,
                criterions=criterions,
                hyperparameters=hyperparameters,
                x_src=x_src,
                d_src=d_src,
                d_ref=d_ref,
                # z_refs=None,
                x_refs=[x_ref_1, x_ref_2],
            )

        grad_scalers["SMG"].scale(g_losses["loss"]).backward()
        grad_scalers["SMG"].step(optimizers["SMG"])
        grad_scalers["SMG"].update()

        # Update Exponential Moving Average models
        networks["S_EMA"].update_ema_weights(networks["S"].module)
        networks["M_EMA"].update_ema_weights(networks["M"].module)
        networks["G_EMA"].update_ema_weights(networks["G"].module)

        losses = {**d_losses, **g_losses}
        return outputs, losses

    def compute_d_loss(
            self,
            networks,
            augmentations,
            hyperparameters,
            x_src,
            d_src,
            d_ref,
            # z_ref=None,
            x_ref=None):

        with torch.no_grad():
            if x_ref is not None:
                s_ref = networks["S"](x_ref, d_ref)
            else:
                s_ref = networks["M"](d_ref) ####################

            x_fake, _ = networks["G"](x_src, s_ref)

        if hyperparameters['LAMBDA_GP']:
            x_src.requires_grad_()

        x_src_aug = augmentations["diff_aug"](x_src)
        x_fake_aug = augmentations["diff_aug"](x_fake)

        d_real_aug = networks["D"](x_src_aug, d_src)
        d_fake_aug = networks["D"](x_fake_aug, d_ref)

        if hyperparameters['LAMBDA_GP']:
            gp_loss = hyperparameters["LAMBDA_GP"] * compute_grad_gp(
                d_real_aug, x_src_aug)
        else:
            gp_loss = torch.tensor(0., device=x_src.device)

        disc_gan_loss = hyperparameters["LAMBDA_ADV"] * (
            self.adv_loss(d_real_aug, 1) + self.adv_loss(d_fake_aug, 0))

        if hyperparameters["LAMBDA_BCR"]:
            bcr_fake = F.mse_loss(
                d_fake_aug,
                networks["D"](x_fake, d_ref))
            bcr_real = F.mse_loss(
                d_real_aug,
                networks["D"](x_src, d_src))
            bcr_loss = hyperparameters["LAMBDA_BCR"] * (
                bcr_real + bcr_fake)
        else:
            bcr_loss = torch.tensor(0, device=x_src.device)
        disc_loss = disc_gan_loss + gp_loss + bcr_loss
        losses = {
            "disc_loss": disc_loss,
            "disc_gan_loss": disc_gan_loss,
            "gp_loss": gp_loss,
            "bcr_loss": bcr_loss
        }
        outputs = {
            "x_src": x_src,
            "x_ref": x_ref,
            "x_fake": x_fake
        }
        return losses, outputs

    def compute_g_loss(
            self,
            networks,
            augmentations,
            criterions,
            hyperparameters,
            x_src,
            d_src,
            d_ref,
            # z_refs,
            x_refs):
        # Generate image from style vector based on ref images
        if x_refs is not None:
            x_ref_1, x_ref_2 = x_refs
            s_ref_1 = networks["S"](x_ref_1, d_ref)
            s_ref_2 = networks["S"](x_ref_2, d_ref)
        else:
            # Generate  image from random latent style vector
            # z_ref_1, z_ref_2 = z_refs
            s_ref_1 = networks["M"](d_ref) 
            s_ref_2 = networks["M"](d_ref) 


        x_fake_1, f_enc_src = networks["G"](x_src, s_ref_1)
        x_fake_2, _ = networks["G"](x_src, s_ref_2)
        x_fake_2 = x_fake_2.detach()
        
        if x_refs is not None:
            save_generated_images(x_src, epoch=33, save_dir='./results')
            save_generated_images(x_fake_1, epoch=11, save_dir='./results')
            save_generated_images(x_fake_2, epoch=22, save_dir='./results')
        else:
            save_generated_images(x_src, epoch=3, save_dir='./results')
            save_generated_images(x_fake_1, epoch=1, save_dir='./results')
            save_generated_images(x_fake_2, epoch=2, save_dir='./results')

        # Extract style vectors of generated images
        s_fake_1 = networks["S"](x_fake_1, d_ref)

        # Translate generated image back to original source image
        s_src = networks["S"](x_src, d_src)
        x_rec_1, f_enc_gen = networks["G"](x_fake_1, s_src)

        # Get discriminator outputs
        x_fake_aug = augmentations["diff_aug"](x_fake_1)
        d_fake = networks["D"](x_fake_aug, d_ref)

        # Adversarial loss
        disc_fake = self.adv_loss(d_fake, 1)
        gen_gan_loss = hyperparameters["LAMBDA_ADV"] * disc_fake

        # Style reconstruction loss
        style_rec_loss = hyperparameters["LAMBDA_STYLE_REC"] * F.l1_loss(
            s_fake_1, s_ref_1)

        # Diversity loss
        diversity_loss = - hyperparameters["LAMBDA_DS_DECAYED"] * F.l1_loss(
            x_fake_1, x_fake_2)

        # Cycle consistency loss
        cycle_consistency_loss = hyperparameters["LAMBDA_CYCLE_CONSISTENCY"] * F.l1_loss(
            x_rec_1, x_src)

        if hyperparameters["LAMBDA_PERCEPTUAL_LOSS"]:
            perceptual_loss = hyperparameters["LAMBDA_PERCEPTUAL_LOSS"] * \
                criterions["perceptual_loss"](x_gen=x_fake_1, x_src=x_src)
        else:
            perceptual_loss = torch.tensor(0., device=x_src.device)

        if hyperparameters["LAMBDA_SRC_LOSS"]:
            src_loss = hyperparameters["LAMBDA_SRC_LOSS"] * criterions[
                "src_loss"](
                    features_src=f_enc_src,
                    features_gen=f_enc_gen)
        else:
            src_loss = torch.tensor(0., device=x_src.device)

        loss = gen_gan_loss + style_rec_loss + diversity_loss +\
            cycle_consistency_loss + perceptual_loss + src_loss
        losses = {
            "loss": loss,
            "gen_gan_loss": gen_gan_loss,
            "style_rec_loss": style_rec_loss,
            "diversity_loss": diversity_loss,
            "cycle_consistency_loss": cycle_consistency_loss,
            "perceptual_loss": perceptual_loss,
            "src_loss": src_loss,
        }
        outputs = {
            "x_src": x_src,
            "x_ref": x_ref_1 if x_refs is not None else None,
            "x_fake": x_fake_1,
            "x_rec": x_rec_1
        }
        return losses, outputs

    def adv_loss(self, logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss
