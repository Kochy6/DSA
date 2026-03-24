import os
import pandas as pd

# ================= 配置区 =================
DATA_DIR = r"/autodl-fs/data/Pro/DSA/dicom_all"
EXCEL_PATH = r"/autodl-fs/data/Pro/DSA/2602图像标记.xlsx"
ID_COLUMN = "图像编号"          # 存储文件名的列
LABEL_COLUMN = "6个月标签（0/1）"  # 存储 0 或 1 的列
# ==========================================

def prepare_index():
    # A. 读取 Excel 标签
    try:
        label_df = pd.read_excel(EXCEL_PATH, engine='openpyxl')
        print(f"成功读取 Excel，共 {len(label_df)} 条标签记录。")
    except Exception as e:
        print(f"读取 Excel 失败: {e}")
        return

    # B. 扫描文件夹中的 DICOM 文件
    files_in_folder = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    print(f"文件夹内共检测到 {len(files_in_folder)} 个文件。")

    # C. 数据对齐与清洗
    label_df[ID_COLUMN] = label_df[ID_COLUMN].astype(str)
    
    final_data = []
    missing_label_count = 0
    
    for file_name in files_in_folder:
        match = label_df[label_df[ID_COLUMN] == file_name]
        
        if not match.empty:
            label_value = match.iloc[0][LABEL_COLUMN]
            # 检查标签是否为 NaN
            if pd.isna(label_value):
                print(f"警告：文件 {file_name} 在 Excel 中的标签为空，已跳过。")
                missing_label_count += 1
                continue
            final_data.append({
                "filename": file_name,
                "label": int(label_value)
            })
        else:
            print(f"警告：文件 {file_name} 在 Excel 中找不到对应的标签。")
            missing_label_count += 1

    # D. 生成最终 CSV
    df_final = pd.DataFrame(final_data)
    df_final.to_csv("label.csv", index=False)
    print(f"\n--- 完成 ---")
    print(f"最终生成 label.csv，包含 {len(df_final)} 条样本（共跳过 {missing_label_count} 个缺失标签的文件）。")
    print(df_final.head())

if __name__ == "__main__":
    prepare_index()