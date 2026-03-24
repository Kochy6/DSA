import pydicom
# 验证这个绝对路径是否真的能打开
test_ds = pydicom.dcmread("/mnt/pro/DSA/data/ori_data/DICOM1/1")
print(test_ds.PatientName)