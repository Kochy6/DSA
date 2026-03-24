# Project Structure

## Current layout

```text
.
├── audit_and_clean.py
├── cleansed_list.csv
├── csv/
├── scripts/
│   ├── core/
│   │   ├── dataset.py
│   │   └── model.py
│   ├── data_prep/
│   │   ├── generate_label_csv_from_excel.py
│   │   └── metadata_consistency_audit.py
│   ├── experiments/
│   │   └── temporal_cnn/
│   │       ├── dataset.py
│   │       ├── model.py
│   │       └── train.py
│   ├── testing/
│   │   ├── test_dataloader.py
│   │   └── test_dicom_read.py
│   └── training/
│       ├── train_kfold.py
│       └── baselines/
│           ├── train_pretrained_baseline.py
│           └── train_scratch_baseline_debug.py
└── ...
```

## Naming conventions used

- `core/`: production-ready core modules shared by training/inference.
- `training/`: primary training entrypoints.
- `training/baselines/`: baseline and ablation training scripts.
- `experiments/`: experiment-specific pipelines kept separate from core.
- `data_prep/`: data indexing/audit/preprocessing scripts.
- `testing/`: quick verification scripts.
