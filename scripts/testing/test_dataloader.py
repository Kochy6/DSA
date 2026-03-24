from scripts.core.dataset import DSADataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test():
    dataset = DSADataset(csv_path=r"/mnt/pro/DSA/cleansed_list.csv")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 取一个 batch
    images, labels = next(iter(loader))
    
    print(f"Batch Image Shape: {images.shape}") # 预期应为 [2, 1, 32, 512, 512]
    print(f"Batch Label Shape: {labels.shape}")
    
    # 可视化第一例样本的中间帧（第 16 帧）
    sample_img = images[0, 0, 16].numpy()
    plt.imshow(sample_img, cmap='gray')
    plt.title(f"Label: {labels[0].item()}")
    plt.savefig("data_check.png")
    print("测试完成，预览图已保存为 data_check.png")

if __name__ == "__main__":
    test()