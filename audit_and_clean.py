import os
import pydicom
import pandas as pd
from tqdm import tqdm

def clean_dsa_data(dicom, original_label_csv, output_csv):
    """
    Args:
        dicom (str or list): DICOM 存储目录路径 (str) 或 多个目录的列表 (list)
        original_label_csv (str): 原始标签 CSV 文件路径
        output_csv (str): 输出 CSV 文件路径
    """
    # 兼容单个目录（字符串）或 多个目录（列表）
    if isinstance(dicom, str):
        dicom_dirs = [dicom]
    else:
        dicom_dirs = dicom
        
    print(f"将在以下目录中搜索 DICOM 文件: {dicom_dirs}")

    # 1. 加载原始标签
    # 假设你的标签表里有 'filename' 和 'label' 两列
    filename_col = 'filename' # 请确认你 csv 表头是否为 filename
    label_col = 'label'       # 请确认你 csv 表头是否为 label
    
    try:
        df_labels = pd.read_csv(original_label_csv)
    except FileNotFoundError:
        print(f"[错误] 找不到标签文件: {original_label_csv}")
        return

    cleansed_data = []
    
    print(f"开始清洗数据并提取时序元数据，共 {len(df_labels)} 条记录...")
    
    for index, row in tqdm(df_labels.iterrows(), total=len(df_labels)):
        filename = str(row[filename_col])
        full_path = None
        
        # --- 核心修改：遍历多个目录寻找文件 ---
        for d_dir in dicom_dirs:
            temp_path = os.path.join(d_dir, filename)
            # 如果 filename 不带后缀，可能需要加 .dcm，根据实际情况调整
            # temp_path_dcm = temp_path + '.dcm' 
            
            if os.path.exists(temp_path):
                full_path = temp_path
                break
            # elif os.path.exists(temp_path_dcm):
            #     full_path = temp_path_dcm
            #     break
        
        if full_path is None:
            # print(f"[跳过] 未在任何目录找到文件: {filename}")
            continue
            
        try:
            ds = pydicom.dcmread(full_path, stop_before_pixels=True) # 只读头信息，速度快
            
            # --- 过滤逻辑 ---
            rows = getattr(ds, 'Rows', 0)
            cols = getattr(ds, 'Columns', 0)
            bits = getattr(ds, 'BitsStored', 0)
            num_frames = int(getattr(ds, 'NumberOfFrames', 1))
            
            # 你之前的过滤逻辑：剔除 750x750 或 12-bit 的离群值
            if rows == 750 or cols == 750 or bits == 12:
                # print(f"\n[跳过离群值] {filename}: {rows}x{cols}, {bits}-bit")
                continue
            
            # --- 提取动态帧间距 (关键点) ---
            frame_times_str = ""
            
            if 'FrameTimeVector' in ds:
                # FrameTimeVector 是一个列表
                ft_vector = ds.FrameTimeVector
                frame_times_str = ",".join([str(t) for t in ft_vector])
            elif 'FrameTime' in ds:
                # 固定帧率
                ft = float(ds.FrameTime)
                if num_frames > 1 and ft > 0:
                     # 有些 FrameTime 为 0 或者异常，需要注意
                    frame_times_str = ",".join([str(ft)] * (num_frames - 1))
                elif num_frames > 1:
                     # 如果 FrameTime 为 0 但有多帧，用默认值
                    frame_times_str = ",".join(["66.67"] * (num_frames - 1))
                elif num_frames == 1:
                    frame_times_str = str(ft)
                
            else:
                # 默认值，假设 15 FPS -> 66.67ms
                default_ft = "66.67"
                if num_frames > 1:
                    frame_times_str = ",".join([default_ft] * (num_frames - 1))
                else:
                    frame_times_str = default_ft

            cleansed_data.append({
                'filename': filename,        # 原始文件名
                'file_path': full_path,      # 找到的绝对路径（重要！后续只需读这个）
                'label': row[label_col],
                'rows': rows,
                'cols': cols,
                'total_frames': num_frames,
                'frame_times': frame_times_str # 存为逗号分隔的字符串
            })
            
        except Exception as e:
            print(f"[异常] 处理文件 {full_path} 出错: {e}")

    # 2. 保存结果
    if cleansed_data:
        new_df = pd.DataFrame(cleansed_data)
        new_df.to_csv(output_csv, index=False)
        print(f"\n清洗完成！共保留 {len(new_df)} 个有效样本。")
        print(f"清洗后的索引文件已保存至: {output_csv}")
    else:
        print("\n未找到任何有效数据，生成的 CSV 为空。")

# 执行
if __name__ == "__main__":
    # --- 配置区域 ---
    
    # 将多个 DICOM 目录放入列表
    DICOM_DIRS = [
        r"/mnt/pro/DSA/data/ori_data/DICOM1",      # 目录 1
        r"/autodl-fs/data/Pro/DSA/dicom_all"                          # 目录 2 (请修改为你真实的路径)
    ]
    
    ORIGINAL_CSV = r"/mnt/pro/DSA/dicom_audit_report_sorted.csv"
    OUTPUT_CSV = "cleansed_list.csv"
    
    clean_dsa_data(DICOM_DIRS, ORIGINAL_CSV, OUTPUT_CSV)