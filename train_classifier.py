import os
import torch
import json
import cv2
import copy
import sys

from models.starganv2 import (
    StyleEncoder,
    MappingNetwork,
    Generator,
    StyleAugMix)
from models.ema import EMA

from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import AutoAugment, RandAugment
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import matplotlib.pyplot as plt

from datasets import load_dataset
from utils.myTransforms import *
from tqdm import tqdm
import numpy as np
from PIL import Image
import warnings
import random
from dataset import *
from trainers.eval import *

warnings.filterwarnings("ignore")
def save_attention_heatmap(attention_map, filename):
    # 将 attention_map 转换为 numpy 格式并进行归一化
    attention_map = attention_map.squeeze().cpu().detach().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())  # 归一化到[0, 1]
    
    # 将Attention Map转为热力图（使用颜色映射）
    heatmap = plt.cm.jet(attention_map)[:, :, :3]  # 使用'jet' colormap并获取 RGB 值

    # 保存热力图
    plt.imshow(heatmap)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def evaluate_models(models, dataloaders, style_encoder, generator, device):
    """
    评估模型在不同数据加载器和增强模式（TTA和非TTA）下的性能。
    """
    metrics = {
        'without_tta': {},
        'with_tta': {}
    }
    
    # 定义数据加载器和名称
    dataloader_names = {
        'val_id_loader': 'Validation ID',
        'val_ood_loader': 'Validation OOD',
        'test_loader': 'Test'
    }

    # 设置所有模型为评估模式
    for model in models:
        model.eval()

    # 对每个数据加载器进行评估
    for dataloader, split_name in zip(dataloaders, dataloader_names.values()):
        # Without TTA evaluation
        metrics['without_tta'][split_name] = evaluate_without_tta(
            classifiers=models,
            dataloader=dataloader,
            device=device
        )

        if generator is None:
            continue

        # With TTA evaluation
        metrics['with_tta'][split_name] = evaluate_with_tta(
            classifiers=models,
            mapping_network=style_encoder,
            generator=generator,
            dataloader=dataloader,
            device=device
        )

    return metrics

def save_results(metrics, model_name):
    """
    将评估结果保存到JSON文件。
    """
    # 打印评估结果
    for tta_type, tta_metrics in metrics.items():
        for split, result in tta_metrics.items():
            print(f"{model_name} {split} {tta_type} - Accuracy: {result[0]['acc']:.4f}, Weighted F1: {result[0]['weighted_f1']:.4f}")

    # 保存结果到 JSON 文件
    os.makedirs('./results', exist_ok=True)
    result_filename = f'TTDA_{model_name}.json'
    result_path = os.path.join('./results', result_filename)

    with open(result_path, "w") as f:
        json.dump({"metrics": metrics}, f, indent=4)
    
    print(f"Results saved to {result_path}")

def save_comparison_images(images1, x_adv_1, batch_idx, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    for i in range(4):  # 遍历 batch 中的每张图像
        # 获取单张图片
        img1 = images1[i]  # 原始图像
        adv_img1 = x_adv_1[i]  # 增强后的图像

        # 将图片并排拼接
        comparison_image = torch.cat((img1, adv_img1), dim=2)  # 在宽度上拼接

        # 保存图片到 ./results 目录
        save_image(comparison_image, os.path.join(output_dir, f"comparison_{batch_idx}_{i}.png"))


def attention_consistency_loss(attention_maps):
    num_models = len(attention_maps)
    
    # 初始化总损失
    total_loss = 0.0
    
    # 遍历所有模型对，计算平均 L2 损失
    for i in range(num_models):
        for j in range(i + 1, num_models):  # 只计算每对模型一次
            for layer_idx, layer in enumerate(attention_maps[i]):  # 遍历每个模型的层
                if layer_idx >= 0:
                    loss = nn.functional.mse_loss(attention_maps[i][layer], attention_maps[j][layer])  # L2 损失
                    total_loss += loss  # 累加总损失
    
    # 计算平均损失
    num_pairs = num_models * (num_models - 1) / 2  # 模型对的数量
    avg_loss = total_loss / (num_pairs * 4)
    return avg_loss

class AttentionExtractor(torch.nn.Module):
    def __init__(self, net):
        super(AttentionExtractor, self).__init__()
        # 提取DenseNet121的卷积部分
        self.features = net.features

        # 分别提取每个Dense Block
        self.block1 = nn.Sequential(*list(self.features.children())[:5])  # Dense Block 1
        self.block2 = nn.Sequential(*list(self.features.children())[5:7])  # Dense Block 2
        self.block3 = nn.Sequential(*list(self.features.children())[7:9])  # Dense Block 3
        self.block4 = nn.Sequential(*list(self.features.children())[9:])  # Dense Block 4

    def forward(self, x):
        attention_maps = {}

        # 每个Dense Block的输出
        out_block1 = self.block1(x)
        attention_maps['block1'] = torch.mean(out_block1, dim=1, keepdim=True)  # 计算block1的attention map

        out_block2 = self.block2(out_block1)
        attention_maps['block2'] = torch.mean(out_block2, dim=1, keepdim=True)  # 计算block2的attention map

        out_block3 = self.block3(out_block2)
        attention_maps['block3'] = torch.mean(out_block3, dim=1, keepdim=True)  # 计算block3的attention map

        out_block4 = self.block4(out_block3)
        attention_maps['block4'] = torch.mean(out_block4, dim=1, keepdim=True)  # 计算block4的attention map

        return attention_maps

def compute_classification_alignment_loss(network_outputs):
    num_networks = len(network_outputs)
    alignment_loss = 0.0
    # 计算每对网络的 KL 散度
    for i in range(num_networks):
        for j in range(i + 1, num_networks):
            alignment_loss += nn.functional.kl_div(nn.functional.log_softmax(network_outputs[i], dim=1), nn.functional.softmax(network_outputs[j], dim=1), reduction='batchmean')
    return alignment_loss

def train_model(models, mode, criterion, optimizers, schedulers, data_loader, device):
    num_models = len(models)
    running_loss = [0.0] * num_models
    total_model_loss = [0.0] * num_models
    total_consistency_loss = [0.0] * num_models
    correct = [0] * num_models
    total = [0] * num_models
    num_batches = [0] * num_models

    progress_bar = tqdm(data_loader, desc=f"Training Model", leave=False)
    for augmented_images, labels, _, domains in progress_bar:

        augmented_images = [[img.to(device, non_blocking=True) for img in sublist] for sublist in augmented_images]
        domains = [domain.to(device, non_blocking=True) for domain in domains]
        labels = labels.to(device, non_blocking=True)
        

        # 初始化模型梯度
        for optimizer in optimizers:
            optimizer.zero_grad()

        # 存储每个模型的注意力图和输出
        attention_maps_1 = [[] for _ in range(num_models)]
        attention_maps_2 = [[] for _ in range(num_models)]
        losses = []
        outputs_1 = []
        outputs_2 = []

        #####################
        batch_size = labels.size(0)
        domain = np.zeros((5, batch_size), dtype=int)
        for i in range(batch_size):
            domain[:, i] = np.random.permutation(5)
        domain = torch.from_numpy(domain).to(device, non_blocking=True)
        sample_idx = torch.arange(batch_size, device=device)
        views_1 = torch.stack([view[0] for view in augmented_images])
        views_2 = torch.stack([view[1] for view in augmented_images])
        #####################

        ##############################################
        # for batch_idx in range(len(augmented_images[0][0])):  # 遍历 batch size
        #     for domain_idx in range(5):  # 遍历 domain
        #         # 计算新的 domain 索引
        #         new_domain_idx = (batch_idx + domain_idx) % 5
        #         # 遍历 pair，并整体替换
        #         for pair_idx in range(2):  # 遍历 pairs，不拆散
        #             augmented_images[domain_idx][pair_idx][batch_idx] = augmented_images[new_domain_idx][pair_idx][batch_idx].clone()
        # ###############################

        for model_idx, model in enumerate(models):
            model.train()
            if mode == 'baseline':
                images1, images2 = augmented_images[1 + model_idx]
            else:
                d = domain[model_idx]
                images1, images2 = views_1[d, sample_idx], views_2[d, sample_idx]

            x_adv_1 = (images1 + 1)/2
            x_adv_2 = (images2 + 1)/2
            output1, output2 = model(x_adv_1), model(x_adv_2)
            outputs_1.append(output1)
            outputs_2.append(output2)
            loss = (criterion(output1, labels)+ criterion(output2, labels))/2
            losses.append(loss)

            _, predicted1 = torch.max(output1, 1)
            _, predicted2 = torch.max(output2, 1)

            total[model_idx] += labels.size(0)
            num_batches[model_idx] += 1
            correct[model_idx] += ((predicted1 == labels).sum().item()+(predicted2 == labels).sum().item())/2

            # 提取注意力图
            if mode == 'baseline':
                continue
            attention_extractor = AttentionExtractor(model)
            attention_map_1 = attention_extractor(x_adv_1)  # 计算激活
            attention_map_2 = attention_extractor(x_adv_2)  # 计算激活
            attention_maps_1[model_idx] = {
                'block1': attention_map_1['block1'],
                'block2': attention_map_1['block2'],
                'block3': attention_map_1['block3'],
                'block4': attention_map_1['block4']
            }
            attention_maps_2[model_idx] = {
                'block1': attention_map_2['block1'],
                'block2': attention_map_2['block2'],
                'block3': attention_map_2['block3'],
                'block4': attention_map_2['block4']
            }

        consistency_loss = classification_alignment_loss = torch.zeros((), device=device)
        if mode != 'baseline':
            consistency_loss_1 = attention_consistency_loss(attention_maps_1)
            consistency_loss_2 = attention_consistency_loss(attention_maps_2)
            consistency_loss = (consistency_loss_1 + consistency_loss_2)/2
            classification_alignment_loss_1 = compute_classification_alignment_loss(outputs_1)
            classification_alignment_loss_2 = compute_classification_alignment_loss(outputs_2)
            classification_alignment_loss= (classification_alignment_loss_1 + classification_alignment_loss_2)/2
        # 合并总的对齐损失
        alignment_loss_all = consistency_loss + classification_alignment_loss*10

        total_loss = sum(losses[idx] for idx in range(num_models)) + alignment_loss_all
        total_loss.backward()
 
        # 初始化模型损失列表
        model_losses = []
        for model_idx, optimizer in enumerate(optimizers):
            # 记录每个模型的损失
            model_loss = losses[model_idx].item()
            model_losses.append(model_loss)

            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清空梯度

            # 累加模型总损失
            total_model_loss[model_idx] += model_loss
            running_loss[model_idx] += model_loss

        # 更新进度条，显示各个损失
        progress_bar.set_postfix(
            L=[round(total_loss.item(), 4)],  # 用 batch 数计算平均损失
            A=[round(c / t * 100, 1) if t > 0 else 0 for c, t in zip(correct, total)],  # 准确率保留1位小数
            ML=[round(ml / b, 2) if b > 0 else 0 for ml, b in zip(total_model_loss, num_batches)],  # 模型损失保留2位小数
            CL=[round(consistency_loss.item(), 4)], # 一致性损失保留2位小数
            CA=[round(classification_alignment_loss.item(), 4)] 
        )

    for scheduler in schedulers:
        scheduler.step() 

    for model_idx in range(num_models):
        model_loss = losses[model_idx]  # 提取模型损失

        print(f"Model {model_idx + 1} Loss: {running_loss[model_idx] / num_batches[model_idx]:.4f}, "
            f"Accuracy: {100 * correct[model_idx] / total[model_idx]:.2f}%, "
            f"Model Loss: {model_loss:.4f}")



if __name__ == "__main__":
    args = {
        'dataset': 'camelyon17',  # 'kather' or 'camelyon17'
        'mode': 'ours',
        'train_centers': [0], 
        'batch_size': 16,
        'num_workers': 8,
        'num_epochs': 50,
        'learning_rate': 0.000005,
        'resize_size': 128,  # Resize input image to 128*128
        'num_class':2
    }
    if args['dataset'] == 'kather':
        args['num_class'] = 7


    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(111)

    if args['mode'] == 'baseline':
        augmentations = [transforms.Compose([HEDJitter(0.05, 0.05, symmetric=True)]) for _ in range(5)]
    else:
        augmentations = [
            transforms.Compose([HEDJitter(0.05,-0.05)]),
            transforms.Compose([HEDJitter(0.05,0.05)]),
            transforms.Compose([HEDJitter(-0.05,0.05)]),
            transforms.Compose([HEDJitter(-0.05,-0.05)])]

    # 从文件中读取 start_epoch
    exp_name = f"{args['dataset']}_{args['mode']}"
    checkpoint_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch_file = os.path.join(checkpoint_dir, 'start_epoch.txt')

    if os.path.exists(start_epoch_file):
        with open(start_epoch_file, 'r') as f:
            state = f.read().strip().split(',')
            start_epoch = int(state[0])
            best_val_ood_acc = float(state[1])
    else:
        start_epoch = 0
        best_val_ood_acc = 0.0
        with open(start_epoch_file, 'w') as f:
            f.write(f'{start_epoch},{best_val_ood_acc}')


    datasets = get_datasets(args['dataset'], args.get('train_centers', None), augmentations, args['resize_size'])
    train_loader, val_id_loader, val_ood_loader, test_loader = get_data_loaders(datasets, args['batch_size'], args['num_workers'])

    # Initialize models, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_models = []
    for _ in range(5):
        model = models.densenet121(pretrained=True).to(device)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, args['num_class']).to(device)
        training_models.append(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizers = [optim.Adam(model.parameters(), lr=args['learning_rate']) for model in training_models]
    lr_lambda = lambda epoch: 0.5 * (1 + np.cos(np.pi * (epoch / args['num_epochs'])))
    schedulers = [lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) for optimizer in optimizers]

    best_model_all = None
    best_val_ood_acc_all = best_val_ood_acc
    best_model_path_all = os.path.join(checkpoint_dir, 'best_model_all.pth')
    last_model_path_all = os.path.join(checkpoint_dir, 'last_model_all.pth')


    # 如果 start_epoch 不为 0，则加载保存的 training_models 参数
    if start_epoch > 0 and os.path.exists(last_model_path_all):
        last_state = torch.load(last_model_path_all, map_location=device)
        for i, model in enumerate(training_models):
            model.load_state_dict(last_state['models'][i])
            optimizers[i].load_state_dict(last_state['optimizers'][i])
            schedulers[i].load_state_dict(last_state['schedulers'][i])
    elif start_epoch > 0:
        print("未找到合适模型的检查点文件，无法继续训练")
        exit()


    # Load image to image translation model
    i2i_chkpt_path = f"./checkpoints/starganv2_{args['dataset']}.pth"
    style_encoder = generator = None
    if os.path.exists(i2i_chkpt_path):
        i2i_chkpt = torch.load(
            i2i_chkpt_path)
        i2i_hyperparams = i2i_chkpt["hyperparameters"]
        print("Loading pretrained StarGanV2 model...")

        style_encoder = StyleEncoder(
            img_size=i2i_hyperparams["IMG_SIZE"],
            style_dim=i2i_hyperparams["STYLE_DIM"],
            num_domains=i2i_hyperparams["NUM_DOMAINS"]).to(device)
        mapping_network = MappingNetwork(
            style_dim=i2i_hyperparams["STYLE_DIM"],
            num_domains=i2i_hyperparams["NUM_DOMAINS"]).to(device)
        generator = Generator(
            img_size=i2i_hyperparams["IMG_SIZE"],
            style_dim=i2i_hyperparams["STYLE_DIM"],
            use_sn=False).to(device)
        style_encoder = EMA(
            style_encoder, decay=i2i_hyperparams["EMA_DECAY"]).eval()
        mapping_network = EMA(
            mapping_network, decay=i2i_hyperparams["EMA_DECAY"]).eval()
        generator = EMA(generator, decay=i2i_hyperparams["EMA_DECAY"]).eval()
        style_encoder.load_state_dict(i2i_chkpt["networks"]["S_EMA"])
        mapping_network.load_state_dict(i2i_chkpt["networks"]["M_EMA"])
        generator.load_state_dict(i2i_chkpt["networks"]["G_EMA"])
        style_encoder = style_encoder.ema_model
        mapping_network = mapping_network.ema_model
        generator = generator.ema_model

        if "prototypes" in i2i_chkpt:
            mapping_network.prototypes = i2i_chkpt["prototypes"]["mapping_prototypes"]
            style_encoder.prototypes = i2i_chkpt["prototypes"]["style_encoder_prototypes"]

    for epoch in range(start_epoch,args['num_epochs']):
        print(f"Epoch {epoch+1}/{args['num_epochs']}")

        # Train each model with its corresponding data
        train_model(training_models, args['mode'], criterion, optimizers, schedulers, train_loader, device)

        # 评估 training_models
        metrics_training = evaluate_models(
            models=training_models,
            dataloaders = [val_id_loader, val_ood_loader],
            style_encoder=style_encoder,
            generator=generator,
            device=device
        )
        save_results(metrics_training, exp_name)

        # 从 metrics 中提取 val_id 和 val_ood 的无 TTA 准确率
        val_id_acc_all = metrics_training['without_tta']['Validation ID'][0]['acc']
        val_ood_acc_all = metrics_training['without_tta']['Validation OOD'][0]['acc']
        print(f"training_models Val ID Acc: {val_id_acc_all:.4f}, Val OOD Acc: {val_ood_acc_all:.4f}")

            
        if val_ood_acc_all > best_val_ood_acc_all:
            best_val_ood_acc_all = val_ood_acc_all
            best_model_all = {f'model_{i+1}': model.state_dict() for i, model in enumerate(training_models)}
            torch.save(best_model_all, best_model_path_all)  # Save the best model checkpoint

        torch.save({'models': [model.state_dict() for model in training_models],
                    'optimizers': [optimizer.state_dict() for optimizer in optimizers],
                    'schedulers': [scheduler.state_dict() for scheduler in schedulers]}, last_model_path_all)

        # 保存当前 epoch 和 best_val_ood_acc
        with open(start_epoch_file, 'w') as f:
            f.write(f'{epoch + 1},{best_val_ood_acc_all}')


##############test##############################

    # 加载所有单独的最佳训练模型
    best_training_models = torch.load(best_model_path_all, map_location=device)
    for i, training_model in enumerate(training_models):
        training_model.load_state_dict(best_training_models[f'model_{i+1}'])

    # 评估 training_models
    metrics_training = evaluate_models(
        models=training_models,
        dataloaders = [val_id_loader, val_ood_loader, test_loader],
        style_encoder=style_encoder,
        generator=generator,
        device=device
    )

    save_results(metrics_training, exp_name)
