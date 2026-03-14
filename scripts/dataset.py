import torch
import pydicom
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, 
    CenterSpatialCropd, 
    Resized, 
    ScaleIntensityRangePercentilesd,
    EnsureChannelFirstd
)

class DSADataset(Dataset):
    def __init__(self, csv_path, target_frames=32, spatial_size=(256, 256)):
        """
        Args:
            csv_path: 第一阶段生成的 cleansed_list.csv 路径
            target_frames: 统一重采样后的帧数
            spatial_size: 最终输出的图像分辨率
        """
        self.data_info = pd.read_csv(csv_path)
        self.target_frames = target_frames
        
        # 定义 MONAI 空间变换流水线
        self.spatial_transforms = Compose([
            # 删掉 EnsureChannelFirstd
            CenterSpatialCropd(keys=["image"], roi_size=(800, 800, -1)),
            # 统一缩放空间尺寸
            Resized(keys=["image"], spatial_size=spatial_size + (target_frames,), mode="trilinear"),
            # 强度归一化
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=1, upper=99, 
                b_min=0.0, b_max=1.0, clip=True
            )
        ])

    def __len__(self):
        return len(self.data_info)

    def _temporal_interpolation(self, video, frame_times_str):
        """
        核心物理逻辑：基于动态时间戳的线性插值 (已增强鲁棒性)
        """
        d, h, w = video.shape
        orig_indices = np.arange(d)
        
        # 1. 解析字符串 (加入 str() 和 strip() 防止空值或 nan 报错)
        if pd.isna(frame_times_str) or str(frame_times_str).strip() == "":
            frame_intervals = []
        else:
            frame_intervals = [float(x) for x in str(frame_times_str).split(',') if x.strip()]
        
        # 2. 【核心修复】强制时间间隔与实际帧数对齐
        # 要生成 d 个时间戳，我们需要 d - 1 个时间间隔
        required_intervals = d - 1 
        
        if len(frame_intervals) < required_intervals:
            # 如果间隔太少，用最后一个间隔值补齐；如果完全为空，默认用 66.67
            pad_val = frame_intervals[-1] if len(frame_intervals) > 0 else 66.67
            frame_intervals.extend([pad_val] * (required_intervals - len(frame_intervals)))
        elif len(frame_intervals) > required_intervals:
            # 如果间隔太多（比如元数据冗余），直接截断
            frame_intervals = frame_intervals[:required_intervals]
            
        # 3. 计算绝对时间轴 (现在的长度必定等于 d，和 orig_indices 完美匹配)
        abs_time = np.cumsum([0.0] + frame_intervals)
        total_duration = abs_time[-1]
        
        # 4. 目标均匀时间点
        target_time = np.linspace(0, total_duration, self.target_frames)
        
        # 5. 映射并执行插值
        interp_indices = np.interp(target_time, abs_time, orig_indices)
        
        resampled_video = []
        for t_idx in interp_indices:
            idx_low = int(np.floor(t_idx))
            idx_high = min(int(np.ceil(t_idx)), d - 1)
            weight = t_idx - idx_low
            
            frame = (1 - weight) * video[idx_low] + weight * video[idx_high]
            resampled_video.append(frame)
            
        return torch.stack([torch.from_numpy(f) for f in resampled_video])

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        
        # 1. 加载 DICOM 像素
        ds = pydicom.dcmread(row['file_path'])
        video = ds.pixel_array.astype(np.float32) # [D, H, W]
        
        # 2. 时域重采样 (变长 -> 等间隔定长)
        video_resampled = self._temporal_interpolation(video, row['frame_times'])
        
        # 3. 空间与强度变换 (使用 MONAI)
        # MONAI 期待字典格式: {"image": [D, H, W]} -> 注意：Resized 会处理 D 维
        # 为了符合 MONAI 习惯，我们将 D 放在最后作为深度维处理
        # 先变成 [H, W, T] (也就是 MONAI 喜欢的空间维度在前，深度在后)
        img_hw_t = video_resampled.permute(1, 2, 0).numpy()
        # 手动加上 Channel 维度，变成 [1, H, W, T]
        img_c_hw_t = np.expand_dims(img_hw_t, axis=0) 
        
        data_dict = {"image": img_c_hw_t}
        data_dict = self.spatial_transforms(data_dict)
        data_dict = self.spatial_transforms(data_dict)
        
        # 4. 最终输出形状调整为 [C, T, H, W]
        # MONAI 输出通常是 [C, H, W, T]
        final_tensor = data_dict["image"].permute(0, 3, 1, 2)
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return final_tensor, label