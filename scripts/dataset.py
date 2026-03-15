import torch
import pandas as pd
import numpy as np
import pydicom
import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

class DSADataset(Dataset):
    def __init__(self, csv_path, mode='train', target_size=(224, 224)):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.mode = mode
        self.target_size = target_size
        
        # 两个可能的根目录
        self.root_dirs = [
            "/mnt/pro/DSA/data/ori_data/DICOM1",
            "/autodl-fs/data/Pro/DSA/dicom_all"
        ]
        
        # 假设 CSV 中文件名的列名是 'filename'
        self.filenames = self.df['filename'].values
        self.labels = self.df['label'].values
        
        # 定义增强流程
        if self.mode == 'train':
            self.transform = v2.Compose([
                v2.RandomAffine(degrees=5, translate=(0.02, 0.02)), # 医疗影像动作要轻
                v2.ColorJitter(brightness=0.1, contrast=0.1),
                v2.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.transform = v2.Identity()

    def __len__(self):
        return len(self.labels)

    def load_dsa_data(self, filename):
        # 1. 在两个目录下寻找文件
        file_path = None
        for root in self.root_dirs:
            temp_path = os.path.join(root, filename)
            if os.path.exists(temp_path):
                file_path = temp_path
                break
        
        if file_path is None:
            raise FileNotFoundError(f"无法在指定目录找到文件: {filename}")

        # 2. 使用 pydicom 读取
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array.astype(np.float32) # [T, H, W] 或 [H, W]

        # 处理单帧情况
        if pixel_array.ndim == 2:
            pixel_array = pixel_array[np.newaxis, ...]

        # 3. 时间维度等间距采样 (解决帧数不一致)
        orig_t = pixel_array.shape[0]
        target_t = 16
        if orig_t >= target_t:
            indices = np.linspace(0, orig_t - 1, target_t).astype(int)
            sampled_data = pixel_array[indices]
        else:
            # 帧数不足则末帧填充
            pad_size = target_t - orig_t
            sampled_data = np.concatenate([pixel_array, np.stack([pixel_array[-1]] * pad_size)], axis=0)

        # 4. 空间维度 Resize
        processed_frames = []
        for frame in sampled_data:
            resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            processed_frames.append(resized)
        
        final_data = np.stack(processed_frames, axis=0) # [16, 224, 224]

        # 5. 归一化 (关键：保留序列内的对比度)
        d_min, d_max = final_data.min(), final_data.max()
        final_data = (final_data - d_min) / (d_max - d_min + 1e-8)
        
        return torch.from_numpy(final_data).unsqueeze(1) # [16, 1, 224, 224]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]
        
        images = self.load_dsa_data(filename)
        images = self.transform(images)
        
        # 全局归一化到 [-1, 1] 范围辅助收敛
        images = (images - 0.5) / 0.5
        
        return images, torch.tensor(label, dtype=torch.long)