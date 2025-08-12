import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models
from torchvision.transforms import AutoAugment, RandAugment
from torchvision.transforms import functional as F
from datasets import load_dataset
from datasets import Dataset as datasets_Dataset
from torch.utils.data import Subset

from utils import myTransforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import warnings
import random
from utils.augmentations import RandomRotation90
from tqdm import tqdm
from collections import defaultdict
import pickle
from skimage.feature import local_binary_pattern
from torchvision.utils import save_image


class TwoCropsTransform():

    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return self.transform1(x), self.transform2(x)

class TextureEnhancer:
    def __init__(self, method='laplacian', lbp_radius=1, lbp_points=8):
        """
        初始化增强类。
        method: 提取纹理信息的方法，默认为'Laplacian'，可以选择'LBP'。
        lbp_radius: LBP算法的半径。
        lbp_points: LBP算法的点数。
        """
        self.method = method
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points

    def enhance_luminance(self, luminance):
        """
        根据选定的方法对亮度通道进行纹理增强。
        """
        if self.method == 'laplacian':
            enhanced_luminance = cv2.Laplacian(luminance, cv2.CV_64F)
            enhanced_luminance = cv2.convertScaleAbs(enhanced_luminance)
        elif self.method == 'lbp':
            enhanced_luminance = local_binary_pattern(luminance, self.lbp_points, self.lbp_radius, method='uniform')
        else:
            raise ValueError("Unsupported method. Use 'laplacian' or 'lbp'.")
        
        return enhanced_luminance

    def enhance_texture(self, img):
        """
        增强输入图像的纹理信息，仅对亮度通道操作，保留颜色信息。
        输入: PIL 图像
        输出: PIL 图像 (3 通道)
        """
        # 将PIL图像转换为NumPy数组
        img = np.array(img)

        # 将RGB图像转换为HSV颜色空间
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # 提取亮度通道 (V)
        luminance = hsv_img[:, :, 2]

        # 对亮度通道进行纹理增强
        enhanced_luminance = self.enhance_luminance(luminance)

        # 将增强后的亮度通道替换回原始HSV图像
        hsv_img[:, :, 2] = enhanced_luminance

        # 将HSV图像转换回RGB颜色空间
        enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        # 将NumPy数组转换回PIL图像
        return Image.fromarray(np.uint8(enhanced_img))

    def __call__(self, img):
        """
        实现增强功能，使得对象可以直接通过 __call__ 被调用。
        """
        return self.enhance_texture(img)

class HEDJitter(object):
    def __init__(self, theta_alpha, theta_betti):
        self.hed_jitter = myTransforms.HEDJitter(theta_alpha = theta_alpha, theta_betti=theta_betti)
        self.random_elastic = myTransforms.RandomElastic(alpha=2, sigma=0.06)
        self.random_affine = myTransforms.RandomAffineCV2(alpha=0.1)

    def __call__(self, img):
        img = self.hed_jitter(img)
        return img

class Camelyon17Dataset(Dataset):
    def __init__(self, hf_dataset, augmentations=None, resize_size=128, train_centers = [0], classfication = False, val = False):
        self.classfication = classfication
        self.val = val
        self.dataset = hf_dataset
        self.resize_size = resize_size
        self.train_centers = train_centers
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                [self.resize_size, self.resize_size], scale=(0.75, 1), ratio=(0.8, 1.2),
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5))])
        self.class_tramsform = TwoCropsTransform(
            self.transform, self.transform)
        self.test_transform = transforms.Compose([
            transforms.Resize([resize_size,resize_size]),
            transforms.ToTensor()])
        self.augmentations = augmentations if augmentations else []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = int(idx)
        item = self.dataset[idx]
        image = item['image']
        if isinstance(image, Image.Image):  # 检查是否为 PIL.Image 对象
            image = image.convert("RGB")

        label = item['label']
        center = item['center']

        original_image = image

        augmented_images = []
        domain = []

        num_augmentations = len(self.augmentations) + 1

        if center in self.train_centers and self.val == False:
            center_base = self.train_centers.index(center) * num_augmentations
            for i in range(num_augmentations):
                if i == 0:
                    augmented_image = original_image
                else:
                    augmented_image = self.augmentations[i - 1](image)

                if self.classfication == False:
                    augmented_image_tensor = self.transform(augmented_image)
                else:
                    augmented_image_tensor = self.class_tramsform(augmented_image)
                augmented_images.append(augmented_image_tensor)
                domain.append(center_base + i)
        else:
            augmented_images.append(self.test_transform(original_image))
            domain.append(num_augmentations*len(self.train_centers)+int(center))

        return augmented_images, label, center, domain

def get_datasets(dataset_name, train_centers=None, augmentations=None, resize_size=128):
    if dataset_name == "camelyon17":
        train_dataset_hf = load_dataset("wltjr1007/Camelyon17-WILDS", split="train")
        val_dataset_hf = load_dataset("wltjr1007/Camelyon17-WILDS", split="validation")
        test_dataset_hf = load_dataset("wltjr1007/Camelyon17-WILDS", split="test")
        print(train_centers)
        train_dataset_hf = train_dataset_hf.filter(lambda example: example['center'] in train_centers, load_from_cache_file=True)
        val_id_dataset_hf = val_dataset_hf.filter(lambda example: example['center'] in train_centers, load_from_cache_file=True)
        val_ood_dataset_hf = val_dataset_hf.filter(lambda example: example['center'] not in train_centers, load_from_cache_file=True)
        print(f"Filtered Dataset Sizes:\n"
            f"  - Train dataset: {len(train_dataset_hf)} examples\n"
            f"  - Validation (ID) dataset: {len(val_id_dataset_hf)} examples\n"
            f"  - Validation (OOD) dataset: {len(val_ood_dataset_hf)} examples")

        train_dataset = Camelyon17Dataset(train_dataset_hf, augmentations, resize_size=resize_size, train_centers = train_centers, classfication = True)
        val_id_dataset = Camelyon17Dataset(val_id_dataset_hf, resize_size=resize_size, train_centers = train_centers, classfication = True, val = True)
        val_ood_dataset = Camelyon17Dataset(val_ood_dataset_hf, resize_size=resize_size, train_centers = train_centers, classfication = True)
        test_dataset = Camelyon17Dataset(test_dataset_hf, resize_size=resize_size, train_centers = train_centers, classfication = True)
    
    elif dataset_name == "kather":
        train_dataset_hf, val_id_dataset_hf = load_train_19_data("./data/kather/train_19")
        val_ood_dataset_hf = load_test_16_data("./data/kather/test_16")
        test_dataset_hf = load_test_16_data("./data/kather/test_16")
        print(f"Filtered Dataset Sizes:\n"
            f"  - Train dataset: {len(train_dataset_hf)} examples\n"
            f"  - Validation (ID) dataset: {len(val_id_dataset_hf)} examples\n"
            f"  - Validation (OOD) dataset: {len(val_ood_dataset_hf)} examples")
        train_dataset = Camelyon17Dataset(train_dataset_hf, augmentations, resize_size=resize_size, train_centers = train_centers, classfication = True)
        val_id_dataset = Camelyon17Dataset(val_id_dataset_hf, resize_size=resize_size, train_centers = train_centers, classfication = True, val = True)
        val_ood_dataset = Camelyon17Dataset(val_ood_dataset_hf, resize_size=resize_size, train_centers = train_centers, classfication = True)
        test_dataset = Camelyon17Dataset(test_dataset_hf, resize_size=resize_size, train_centers = train_centers, classfication = True)

    return train_dataset, val_id_dataset, val_ood_dataset, test_dataset

def get_data_loaders(datasets, batch_size=32, num_workers=0):
    train_dataset, val_id_dataset, val_ood_dataset, test_dataset = datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory = True)
    val_id_loader = DataLoader(val_id_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory = True)
    val_ood_loader = DataLoader(val_ood_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory = True)
    return train_loader, val_id_loader, val_ood_loader, test_loader


#########################################################

class OneStreamCamelyon17Dataset(Dataset):
    def __init__(self, hf_dataset, split_name, augmentations=None, resize_size=128, train_centers = [0]):
        self.split_name = split_name
        self.single_stream_dataset = Camelyon17Dataset(hf_dataset, augmentations, resize_size, train_centers)
        self.num_augmentations = len(augmentations) + 1 if augmentations else 1

    def __len__(self):
        return len(self.single_stream_dataset) * self.num_augmentations

    def __getitem__(self, idx):
        # Calculate the actual index in the original dataset and the augmentation index
        original_idx = idx // self.num_augmentations
        augmentation_idx = idx % self.num_augmentations

        augmented_images, label, center, domain = self.single_stream_dataset[original_idx]
        # if domain[augmentation_idx] == 2:
        #     save_image(augmented_images[augmentation_idx], "tensor_image_0.jpg")
        return augmented_images[augmentation_idx], domain[augmentation_idx]

class TwoStreamCamelyon17Dataset(Dataset):
    def __init__(self, hf_dataset, split_name, augmentations=None, resize_size=128, train_centers = [0]):
        self.split_name = split_name
        self.single_stream_dataset = Camelyon17Dataset(hf_dataset, augmentations, resize_size, train_centers)
        self.domain2indices = self._create_domain_to_indices_mapping()
        self.num_augmentations = len(augmentations) + 1 if augmentations else 1
        
    def _create_domain_to_indices_mapping(self, cache_file='domain2indices_cache.pkl'):
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                domain2indices = pickle.load(f)
            return domain2indices
    
        domain2indices = defaultdict(list)
        for idx in tqdm(range(len(self.single_stream_dataset)), desc="Creating domain to indices mapping"):
            _, _, center, domain_list = self.single_stream_dataset[idx]
            for domain in domain_list:
                domain2indices[domain].append(idx)
    
        # Convert defaultdict to dict before saving to pickle
        domain2indices = dict(domain2indices)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(domain2indices, f)
        
        return domain2indices
    
    def __len__(self):
        return len(self.single_stream_dataset) * self.num_augmentations

    def __getitem__(self, idx):
        # Calculate the actual index in the original dataset and the augmentation index
        original_idx = idx // self.num_augmentations
        augmentation_idx = idx % self.num_augmentations

        augmented_images_1, label_1, center_1, domain_1 = self.single_stream_dataset[original_idx]

        domain = domain_1[augmentation_idx]
        idx_2 = random.choice(self.domain2indices[domain])

        # 确保选择的两个样本是不同的
        while idx_2 == original_idx:
            idx_2 = random.choice(self.domain2indices[domain])

        augmented_images_2, label_2, center_2, domain_2 = self.single_stream_dataset[idx_2]
        # if domain_1[augmentation_idx] == 2:
        #     save_image(augmented_images_1[augmentation_idx], "tensor_image_1.jpg")
        #     save_image(augmented_images_2[augmentation_idx], "tensor_image_2.jpg")

        # 确保augmented_images_1和augmented_images_2具有相同的center和domain
        assert domain_1[augmentation_idx] == domain_2[augmentation_idx], f"Domain mismatch at index {augmentation_idx}: {domain_1[augmentation_idx]} vs {domain_2[augmentation_idx]}"
        assert center_1 == center_2, f"Center mismatch: {center_1} vs {center_2}"
        return [augmented_images_1[augmentation_idx], augmented_images_2[augmentation_idx]], domain_1[augmentation_idx]


###############################################################


# Utility function to load and transform images
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # 获取图像尺寸
    width, height = image.size
    
    # 计算中心裁剪的区域
    left = (width - 150) / 2
    top = (height - 150) / 2
    right = (width + 150) / 2
    bottom = (height + 150) / 2
    
    # 裁剪图像
    image = image.crop((left, top, right, bottom))
    
    return image

# 加载数据集并划分
def prepare_dataset(folder, center_value, train_split_ratio=0.8):
    train_data = []
    val_data = []
    train_labels = []
    val_labels = []
    train_centers = []
    val_centers = []
    
    # 遍历每个类别的文件夹
    for label, class_folder in enumerate(os.listdir(folder)):
        class_path = os.path.join(folder, class_folder)
        images = os.listdir(class_path)
        
        # 随机打乱图像列表
        random.shuffle(images)
        total_images = len(images)
        split_index = int(total_images * train_split_ratio)
        
        # 划分为训练集和验证集
        train_images = images[:split_index]
        val_images = images[split_index:]
        
        # 加载训练数据
        for img in train_images:
            image_path = os.path.join(class_path, img)
            train_data.append(load_image(image_path))
            train_labels.append(label)
            train_centers.append(center_value)
        
        # 加载验证数据
        for img in val_images:
            image_path = os.path.join(class_path, img)
            val_data.append(load_image(image_path))
            val_labels.append(label)
            val_centers.append(center_value)
    
    return train_data, train_labels, train_centers, val_data, val_labels, val_centers

def load_train_19_data(train_path):
    train_data, train_labels, train_centers, val_data, val_labels, val_centers = prepare_dataset(train_path, center_value=0, train_split_ratio=0.4)
    
    # 构建训练数据集
    train_dataset_hf = datasets_Dataset.from_dict({
        "image": train_data,
        "label": train_labels,
        "center": train_centers
    })

    # 构建验证集数据集
    val_id_dataset_hf = datasets_Dataset.from_dict({
        "image": val_data,
        "label": val_labels,
        "center": val_centers
    })
    
    return train_dataset_hf, val_id_dataset_hf

# 加载 test_16 数据集
def load_test_16_data(test_path):
    test_data = []
    test_labels = []
    test_centers = []
    
    for label, class_folder in enumerate(os.listdir(test_path)):
        class_path = os.path.join(test_path, class_folder)
        images = os.listdir(class_path)
        
        for img in images:
            image_path = os.path.join(class_path, img)
            test_data.append(load_image(image_path))
            test_labels.append(label)
            test_centers.append(1)
    
    val_ood_dataset_hf = datasets_Dataset.from_dict({
        "image": test_data,
        "label": test_labels,
        "center": test_centers
    })
    
    return val_ood_dataset_hf


