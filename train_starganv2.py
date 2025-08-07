import os
import json
import pprint
import argparse
from PIL import Image

import torch
from torch import nn
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset
from dataset import *
from dataloaders import InfiniteDataLoader
from models.starganv2 import (
    StyleEncoder, MappingNetwork, Generator, Discriminator)
from models.ema import EMA
from trainers.starganv2 import Trainer
from trainers.diff_augment import DiffAugment
from utils.augmentations import RandomRotation90
from torchvision.transforms import AutoAugment, RandAugment
from utils.myTransforms import *
from torch.utils.data import Subset
import random

import warnings
warnings.filterwarnings("ignore")

# 打乱数据集索引并创建 Subset 数据集
def create_shuffled_subset(dataset):
    indices = list(range(len(dataset)))  # 创建索引列表
    random.shuffle(indices)  # 打乱索引
    return Subset(dataset, indices)  # 返回打乱后的 Subset 数据集

def main(
        json_config,
        log_dir,
        checkpoint_dir):

    # Get environment variables for distributed training
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    print('ENV VARIABLES'.center(50, '-'))
    print(f'rank: {rank}')
    print(f'local_rank: {local_rank}')
    print(f'world_size: {world_size}')
    print('-'*50+'\n')

    # Load json config file
    with open(json_config, "r") as f:
        hyperparameters = json.load(f)

    # Create log path and checkpoint from exp_name
    hyperparameters.update({
        "LOG_DIR": os.path.join(log_dir, hyperparameters["DATASET_NAME"],
            hyperparameters["EXP_NAME"]),
        "CHECKPOINT_PATH": os.path.join(
            checkpoint_dir, f'{hyperparameters["EXP_NAME"]}.pth')
    })
        
    print("Training on: {}".format(hyperparameters["DATASET_NAME"]))
    print(f"Exp_name: {hyperparameters['EXP_NAME']}")

    print('start init_process_group')
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',)
    print('end init_process_group')

    torch.cuda.set_device(local_rank)
    DEVICE = torch.device("cuda")
    hyperparameters["TRAIN_CENTERS"] 
    if rank == 0:
        pprint.pprint(hyperparameters)

    # Get train, val, test datasets and dataloaders
    torch.distributed.barrier()
    
    if hyperparameters["DATASET_NAME"] == "camelyon17":
        train_dataset_hf = load_dataset("wltjr1007/Camelyon17-WILDS", split="train")
        val_dataset_hf = load_dataset("wltjr1007/Camelyon17-WILDS", split="validation")
        test_dataset_hf = load_dataset("wltjr1007/Camelyon17-WILDS", split="test")

        train_centers = hyperparameters["TRAIN_CENTERS"]
        train_dataset_hf = train_dataset_hf.filter(lambda example: example['center'] in train_centers)
        val_id_dataset_hf = val_dataset_hf.filter(lambda example: example['center'] in train_centers)
        val_ood_dataset_hf = val_dataset_hf.filter(lambda example: example['center'] not in train_centers)
    elif hyperparameters["DATASET_NAME"] == "kather":
        train_dataset_hf, val_id_dataset_hf = load_train_19_data("./data/kather/train_19")
        val_ood_dataset_hf = load_test_16_data("./data/kather/test_16")

    
    augmentations = [
        transforms.Compose([HEDJitter(0.05,-0.05)]),
        transforms.Compose([HEDJitter(0.05,0.05)]),
        transforms.Compose([HEDJitter(-0.05,0.05)]),
        transforms.Compose([HEDJitter(-0.05,-0.05)])]

    
    
    datasets = {
        "train": {
            "ref": TwoStreamCamelyon17Dataset(
                hf_dataset=train_dataset_hf,
                split_name = "train",
                augmentations= augmentations,
                resize_size=hyperparameters["IMG_SIZE"],
                train_centers = hyperparameters["TRAIN_CENTERS"]),
            "src": OneStreamCamelyon17Dataset(
                hf_dataset=train_dataset_hf,
                split_name = "train",
                augmentations= augmentations,
                resize_size=hyperparameters["IMG_SIZE"],
                train_centers = hyperparameters["TRAIN_CENTERS"]
            )},
    }
    
    datasets['train']['ref'] = create_shuffled_subset(datasets['train']['ref'])
    datasets['train']['src'] = create_shuffled_subset(datasets['train']['src'])


    hyperparameters["NUM_DOMAINS"] = (len(augmentations)+1) * len(hyperparameters["TRAIN_CENTERS"])
    print(f"The number of domains is: {hyperparameters['NUM_DOMAINS']}")

    
    dataloaders = {}
    for split_name in datasets:
        if split_name not in dataloaders:
            dataloaders[split_name] = {}
        for dataset_name in datasets[split_name]:
            dataset = datasets[split_name][dataset_name]
            sampler = DistributedSampler(
                dataset=dataset,
                shuffle=True,
                drop_last=True)
            dataloaders[split_name][dataset_name] = InfiniteDataLoader(
                dataset=dataset,
                batch_size=hyperparameters["BATCH_SIZE"],
                num_workers=hyperparameters["NUM_WORKERS"],
                sampler=sampler)

    # Create the different networks of StarGanV2
    style_encoder = StyleEncoder(
        img_size=hyperparameters["IMG_SIZE"],
        style_dim=hyperparameters["STYLE_DIM"],
        num_domains=hyperparameters["NUM_DOMAINS"]).to(DEVICE)
    mapping_network = MappingNetwork(
        style_dim=hyperparameters["STYLE_DIM"],
        num_domains=hyperparameters["NUM_DOMAINS"]).to(DEVICE)
    generator = Generator(
        img_size=hyperparameters["IMG_SIZE"],
        style_dim=hyperparameters["STYLE_DIM"],
        use_sn=False).to(DEVICE)
    discriminator = Discriminator(
        img_size=hyperparameters["IMG_SIZE"],
        num_domains=hyperparameters["NUM_DOMAINS"],
        max_conv_dim=512,
        use_sn=False).to(DEVICE)

    # Create the exponential moving average models
    style_encoder_ema = EMA(
        style_encoder, decay=hyperparameters["EMA_DECAY"]).to(DEVICE).eval()
    mapping_network_ema = EMA(
        mapping_network, decay=hyperparameters["EMA_DECAY"]).to(DEVICE).eval()
    generator_ema = EMA(
        generator, decay=hyperparameters["EMA_DECAY"]).to(DEVICE).eval()

    # Create distributed data parallel model
    style_encoder = nn.parallel.DistributedDataParallel(
        style_encoder, device_ids=[local_rank], output_device=local_rank)
    mapping_network = nn.parallel.DistributedDataParallel(
        mapping_network, device_ids=[local_rank], output_device=local_rank)
    generator = nn.parallel.DistributedDataParallel(
        generator, device_ids=[local_rank], output_device=local_rank)
    discriminator = nn.parallel.DistributedDataParallel(
        discriminator, device_ids=[local_rank], output_device=local_rank)

    networks = {
        "S": style_encoder,
        "M": mapping_network,
        "G": generator,
        "D": discriminator,
        "S_EMA": style_encoder_ema,
        "G_EMA": generator_ema,
        "M_EMA": mapping_network_ema,
    }

    # Display number of weights per model:
    print('Number of trainable parameters:')
    for network_name, network in networks.items():
        num_params = sum((p.numel() for p in network.parameters() if p.requires_grad))
        print('{}: {:,}'.format(network_name, num_params))

    optimizers = {
        "SMG": Adam(
            lr=hyperparameters["GENERATOR_LR"],
            params=list(style_encoder.parameters()) + list(generator.parameters()) +\
            list(mapping_network.parameters())),
        "D": Adam(lr=hyperparameters["DISCRIMINATOR_LR"], params=discriminator.parameters()),
    }

    schedulers = {
        "SMG": LambdaLR(optimizers["SMG"], lr_lambda=lambda epoch: 1)
    }

    # Create DiffAugment module to performs data augmentation on GPU
    augmentations = {
        "diff_aug": DiffAugment(
            p=1).to(DEVICE)}

    if os.path.exists(hyperparameters["CHECKPOINT_PATH"]):
        print("Loading previous checkpoint")
        map_location = {
            'cuda:%d' % 0: 'cuda:%d' % local_rank}
        chkpt = torch.load(
            hyperparameters["CHECKPOINT_PATH"],
            map_location=map_location)
        for net_name, state_dict in chkpt["networks"].items():
            networks[net_name].load_state_dict(
                state_dict)
        for opt_name, state_dict in chkpt["optimizers"].items():
            optimizers[opt_name].load_state_dict(
                state_dict)
        for sched_name, state_dict in chkpt["schedulers"].items():
            schedulers[sched_name].load_state_dict(
                state_dict)
        if "augmentations" in chkpt:
            for aug_name, state_dict in chkpt["augmentations"].items():
                augmentations[aug_name].load_state_dict(
                    state_dict)

    print("Start training")
    trainer = Trainer()
    trainer.train(
        networks=networks,
        dataloaders=dataloaders,
        optimizers=optimizers,
        schedulers=schedulers,
        augmentations=augmentations,
        epochs=hyperparameters["EPOCHS"],
        steps_per_epoch=hyperparameters["STEPS_PER_EPOCH"],
        hyperparameters=hyperparameters,
        device=DEVICE,
        log_dir=hyperparameters["LOG_DIR"],
        checkpoint_path=hyperparameters["CHECKPOINT_PATH"],
        save_every_epochs=hyperparameters["SAVE_EVERY_EPOCHS"],
        rank=rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_config",
        help="Path to json config file for StarGANV2 training.", type=str)
    parser.add_argument(
        "--log_dir",
        help="Logs directory",
        type=str)
    parser.add_argument(
        "--checkpoint_dir",
        help="Checkpoints directory",
        type=str)
    args = parser.parse_args()
    main(
        json_config=args.json_config,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,)