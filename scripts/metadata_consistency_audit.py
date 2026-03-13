import os
import pydicom
import pandas as pd
from pathlib import Path

def extract_dicom_metadata(data_dirs):
    """
    遍历多个路径提取 DICOM 核心元数据
    """
    records = []
    
    # 支持传入路径列表
    for data_dir in data_dirs:
        print(f"正在扫描路径: {data_dir}")
        path_obj = Path(data_dir)
        
        # 遍历目录下所有文件 (假设文件没有子目录，如有则用 rglob)
        for file_path in path_obj.glob('*'):
            if file_path.is_dir(): continue
            
            try:
                # 只读取元数据，不加载像素数据以提高速度 (stop_before_pixels=True)
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                
                # 提取关键字段，如果不存在则填充 None
                record = {
                    "filename": file_path.name,
                    "abs_path": str(file_path.absolute()),
                    "source_dir": data_dir,
                    # 1. 维度信息
                    "frames": getattr(ds, 'NumberOfFrames', 1),
                    "rows": getattr(ds, 'Rows', None),
                    "cols": getattr(ds, 'Columns', None),
                    # 2. 像素属性
                    "photometric": getattr(ds, 'PhotometricInterpretation', None), # 黑白极性
                    "bits_stored": getattr(ds, 'BitsStored', None),               # 位深
                    "pixel_spacing": str(getattr(ds, 'PixelSpacing', None)),      # 空间分辨率
                    # 3. 时间属性
                    "frame_time": getattr(ds, 'FrameTime', None),                # 帧间距(ms)
                    "frame_rate": getattr(ds, 'CineRate', None),                 # 帧率
                    # 4. 设备与厂商 (用于识别域差异)
                    "manufacturer": getattr(ds, 'Manufacturer', "Unknown"),
                    "model_name": getattr(ds, 'ManufacturerModelName', "Unknown"),
                    # 5. 像素值校准
                    "rescale_intercept": getattr(ds, 'RescaleIntercept', 0),
                    "rescale_slope": getattr(ds, 'RescaleSlope', 1),
                }
                records.append(record)
                
            except Exception as e:
                print(f"跳过文件 {file_path.name}: 无法读取 (可能非DICOM格式)")

    return pd.DataFrame(records)

# --- 配置路径 ---
paths_to_scan = [
    "/mnt/pro/DSA/data/ori_data/DICOM1",
    "/autodl-fs/data/Pro/DSA/dicom_all"
]

# --- 执行提取 ---
df = extract_dicom_metadata(paths_to_scan)

# --- 生成统计汇总 ---
print("\n" + "="*30)
print("数据一致性快速检查报告:")
print(f"总计成功读取文件数: {len(df)}")
print("-" * 30)

# 检查可能导致训练失败的关键因子
critical_cols = ['rows', 'cols', 'photometric', 'bits_stored', 'pixel_spacing', 'frame_time']
for col in critical_cols:
    unique_vals = df[col].unique()
    print(f"字段 [{col}] 的唯一值数量: {len(unique_vals)} -> {unique_vals}")

# --- 保存结果 ---
output_csv = "dicom_audit_report.csv"
df.to_csv(output_csv, index=False)
print(f"\n详细报告已保存至: {output_csv}")