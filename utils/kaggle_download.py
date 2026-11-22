# DOWNLOAD DATASET 
import kagglehub

target_dir = "./datasets/"

path = kagglehub.dataset_download(
    "youssef019/mnist-multi-digit-regression-dataset",
    output_dir=target_dir
)

print("Path to dataset files:", path)