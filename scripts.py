import os
import kagglehub

path = "./datasets"
mnist_multi_digit_dataset = "daitoo/mnist-multi-digit-cache-gray-datasets"

def download_datasets(dataset_name, path):
    """
    Downloads the specified dataset from Kaggle and saves it to the given path.

    Args:
        dataset_name (str): The name of the dataset to download.
        path (str): The directory where the dataset will be saved.
    """
    if os.path.exists(path) and os.listdir(path):
        print(f"✅ Dataset {dataset_name} already exists at {path}. Skipping download.")
        return

    kagglehub.dataset_download(dataset_name, output_dir=path)
    print(f"✅ Downloaded {dataset_name} to {path}")

if __name__ == "__main__":
    # Download the MNIST multi-digit dataset
    download_datasets(mnist_multi_digit_dataset, path)