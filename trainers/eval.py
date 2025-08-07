import os
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from torch.nn.functional import cosine_similarity
import torchvision.utils as vutils
from torchvision.utils import make_grid, save_image


def save_generated_images(X_gen, save_dir='./results', file_prefix='generated_image', only_once=True):
    if only_once:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 将图像保存为批量图片的单张网格
        file_path = os.path.join(save_dir, f"{file_prefix}.png")

        # 使用 torchvision.utils.save_image 保存图像
        vutils.save_image(X_gen, file_path, normalize=True)


def evaluate_without_tta(
    classifiers,  # classifiers 是多个模型的列表
    dataloader,
    device):
    y_true = []
    y_pred = []
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc="Evaluation without TTA")

    correct = [0] * len(classifiers)  # 累积每个模型的正确预测数
    total_correct = 0  # 累积多模型平均预测的正确数目
    total = 0  # 总样本数
    with torch.no_grad():
        for i, (X, target, _, _) in enumerate(pbar):
            X =  X[0].to(device) #1*(b*images)
            target = target.to(device) # (b)
            total += target.size(0)

            # 初始化预测输出为 0
            n, c, h, w = X.shape
            avg_prob = torch.zeros([n, classifiers[0].classifier.out_features], device=device)  # 用于累加概率
            prob_for_each_model = []

            # 对每个模型进行预测，并累加预测的概率分布
            for idx, classifier in enumerate(classifiers):
                prob = torch.softmax(classifier(X), dim=-1)  # 获取每个模型的概率分布 (b*classes)
                prob_for_each_model.append(prob)
                avg_prob += prob  # 累加模型的概率分布 (b*classes)

            # 计算每个模型的准确率
            for model_idx in range(len(classifiers)):
                _, predicted = torch.max(prob_for_each_model[model_idx], 1)
                correct[model_idx] += (predicted == target).sum().item()

            # 取所有模型的预测平均概率，并进行预测
            avg_prob /= len(classifiers)
            _, avg_pred = torch.max(avg_prob, 1)

            total_correct += (avg_pred == target).sum().item()
            
            #多模型平均预测结果
            y_true.append(target.cpu())
            y_pred.append(avg_pred.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    
    # 打印每个模型的单独准确率（仅当模型数量大于1时）
    if len(classifiers) > 1:
        for model_idx in range(len(classifiers)):
            print(f"Model {model_idx + 1} Accuracy: {correct[model_idx] / total:.4f}")  # 使用总样本数计算准确率

    # 计算多模型平均预测的准确率
    metrics = {
        "acc": total_correct / total,
        "weighted_f1": f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    }
    print(f"Accuracy: {metrics['acc']}, Weighted F1: {metrics['weighted_f1']}")

    epoch_acc = metrics['acc']

    return metrics, epoch_acc

def evaluate_with_tta(
        classifiers,  # classifiers 是多个模型的列表
        mapping_network,
        generator,
        dataloader,
        device):
    y_true = []
    y_pred = []
    image_saved = False
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc="Evaluation with TTA")

    correct = [0] * len(classifiers)  # 累积每个模型的正确预测数
    total_correct = 0  # 累积多模型平均预测的正确数目
    total = 0  # 总样本数
    with torch.no_grad():
        for i, (X, target, _, _) in enumerate(pbar):
            X = X[0].to(device)
            target = target.to(device)
            total += target.size(0)  # 累加样本数


            # Convert images from [0, 1] to [-1, 1] for StarGanV2 usage
            X = (X - 0.5)/0.5
            num_domains = mapping_network.num_domains

            # (N, C, H, W) --> (N * S, C, H, W)
            n, c, h, w = X.shape
            X = X.unsqueeze(1).repeat([1, num_domains, 1, 1, 1]).reshape([-1, c, h, w])

            prototypes = mapping_network.prototypes.to(device)  # [12, 64]
            s = prototypes.unsqueeze(0).repeat(n, 1, 1)
            s = s.view(-1, s.size(-1))

            X_gen, _ = generator(X, s)

            # Convert generated images from [-1, 1] to [0, 1] for classifier
            X_gen = (X_gen + 1) / 2
            # save_generated_images(X_gen)

            # 初始化预测输出为 0
            avg_prob = torch.zeros([n, classifiers[0].classifier.out_features], device=device)  # 用于累加概率
            prob_for_each_model = []

            # 对每个模型进行预测，并累加预测的概率分布
            for idx, classifier in enumerate(classifiers):
                predictions = classifier(X_gen)
                predictions = predictions.reshape([n, num_domains, -1]).mean(dim=1)
                prob = torch.softmax(predictions, dim=-1)  # 获取每个模型的概率分布
                prob_for_each_model.append(prob)
                avg_prob += prob  # 累加模型的概率分布

            # 计算每个模型的准确率
            for model_idx in range(len(classifiers)):
                _, predicted = torch.max(prob_for_each_model[model_idx], 1)
                correct[model_idx] += (predicted == target).sum().item()

            # 取所有模型的预测平均概率，并进行预测
            avg_prob /= len(classifiers)
            avg_pred = avg_prob.argmax(dim=-1)

            # 计算多模型平均预测的正确数目
            total_correct += (avg_pred == target).sum().item()

            #多模型平均预测结果
            y_true.append(target.cpu())
            y_pred.append(avg_pred.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    # 打印每个模型的单独准确率（仅当模型数量大于1时）
    if len(classifiers) > 1:
        for model_idx in range(len(classifiers)):
            print(f"Model {model_idx + 1} Accuracy: {correct[model_idx] / total:.4f}")  # 使用总样本数计算准确率

    # 计算多模型平均预测的准确率
    metrics = {
        "acc": total_correct / total,
        "weighted_f1": f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    }
    print(f"Accuracy: {metrics['acc']}, Weighted F1: {metrics['weighted_f1']}")

    epoch_acc = metrics['acc']

    return metrics, epoch_acc