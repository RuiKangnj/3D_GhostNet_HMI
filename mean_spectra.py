import csv
import numpy as np
from data.dataloader import load_config, collect_file_paths_and_labels, HyperspectralDataset,load_json_paths
import json
from tqdm import tqdm
def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['train_paths'], data['train_labels']

def get_num_bands(image_path):
    """获取单个图像的波段数"""
    image = np.load(image_path)
    return image.shape[-1]  # 假设波段在最后一个维度

def append_to_csv(batch_average_spectra, batch_labels, csv_file):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for spectrum, label in zip(batch_average_spectra, batch_labels):
            writer.writerow([label] + list(spectrum))

def compute_spectrum(image):
    non_zero_mask = image > 0
    sum_spectrum = np.sum(image, axis=(0, 1))
    count_non_zero = np.sum(non_zero_mask, axis=(0, 1))
    average_spectrum = sum_spectrum / np.maximum(count_non_zero, 1)
    return average_spectrum

def compute_average_spectrum_batch(image_paths, labels, csv_file, batch_size=30):
    header_written = False
    num_bands = get_num_bands(image_paths[0])
    print(num_bands,len(image_paths))
    # 初始化 tqdm 进度条
    pbar = tqdm(total=len(image_paths), desc="Processing Images", leave=True)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_average_spectra = []

        for path in batch_paths:
            image = np.load(path)
            spectrum = compute_spectrum(image)
            batch_average_spectra.append(spectrum)

        batch_labels = labels[i:i + batch_size]

        # Append the batch results to CSV
        if not header_written:
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['label'] + [f'band_{j + 1}' for j in range(num_bands)])
            header_written = True

        append_to_csv(batch_average_spectra, batch_labels, csv_file)
        # 更新 tqdm 进度条
        pbar.update(len(batch_paths))

    pbar.close()




image_paths, labels = load_data('3D_bacteria_cells.json')

# 使用函数
compute_average_spectrum_batch(image_paths, labels, 'train_spectra_labels.csv')

