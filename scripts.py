import os
import kagglehub

path = "./datasets"

traffic_volume_dataset = "pooriamst/metro-interstate-traffic-volume"
traffic_path = os.path.join(path, "TrafficVolume")

mnist_multi_digit_dataset = "youssef019/mnist-multi-digit-regression-dataset"
mnist_path = os.path.join(path, "MnistMultiDigit")

def download_datasets(dataset_name, path):
    """
    Downloads the specified dataset from Kaggle and saves it to the given path.

    Args:
        dataset_name (str): The name of the dataset to download.
        path (str): The directory where the dataset will be saved.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(path):
        print(f"✅ Dataset {dataset_name} already exists at {path}. Skipping download.")
        return

    kagglehub.dataset_download(dataset_name, output_dir=path)
    print(f"✅ Downloaded {dataset_name} to {path}")

if __name__ == "__main__":
    # Download the traffic volume dataset
    download_datasets(traffic_volume_dataset, traffic_path)

    # Download the MNIST multi-digit dataset
    download_datasets(mnist_multi_digit_dataset, mnist_path)