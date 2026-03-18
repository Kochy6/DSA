# DSA
DSA sequence prognostic prediction
# DSA-Prognosis-Prediction (DSA 序列预后预测系统)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![MONAI](https://img.shields.io/badge/MONAI-1.3+-darkgreen.svg)
![Status](https://img.shields.io/badge/Status-Active_Development-orange.svg)

## 📌 1. 项目背景
本项目旨在利用数字减影血管造影（DSA）的动态序列图像，通过深度学习与系统辨识技术，预测患者的预后情况（如血管再通状态、病情恶化风险等）。DSA 数据具有极高的时空分辨率，能够直观反映造影剂在血管内的血流动力学特征。本项目致力于在小样本（~500组）和高位深（10-bit）的约束下，构建一个具备极强泛化能力和时序建模能力的医疗影像分类系统。

## 🎯 2. 核心挑战与需求
如何解决伪影？
### 2.1 数据处理需求
* **多源数据整合：** 支持跨服务器路径（如 `/mnt/pro/` 与 `/autodl-fs/`）的无缝整合与标签对齐。
* **高位深与高分辨率：** 兼容 10-bit（0-1023 灰度级）原生 DICOM，处理 $1024 \times 1024$ 矩阵，防止关键微小血管特征在降采样中丢失。
* **时序与域对齐：** 解决不同成像设备间帧率（FPS）、光度学极性（黑白反转）不一致的问题。

### 2.2 算法性能需求
* **时空解耦建模：** 突破传统 3D 卷积的参数量限制，采用 `2D Backbone + Temporal Attention` 架构捕捉造影剂从充盈到消退的非线性动态特征。
* **小样本鲁棒性：** 在严格的 5 折交叉验证（5-Fold CV）下实现稳健的 AUC 与 Accuracy 表现，克服过拟合。

## 🛠️ 3. 技术栈
* **核心框架：** PyTorch, MONAI (Medical Open Network for AI)
* **数据处理与审计：** Pydicom, Pandas, Scikit-learn
* **硬件环境：** NVIDIA RTX 4090 (24GB VRAM) 

## 🚀 4. 快速开始 (Quick Start)

### 4.1 环境依赖
```bash
pip install torch torchvision torchaudio
pip install monai pydicom pandas scikit-learn
