import torch
import os
from torch.utils.data import Dataset, DataLoader
import yaml
import numpy as np
import json

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def save_json_paths(data, file_name):
    with open(file_name, 'w') as file:
        json.dump(data, file)

def load_json_paths(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def collect_file_paths_and_labels(base_folder):
    file_paths = []
    labels = []
    label_dict = {'Ecoli': 0, 'Li': 1, 'Sa': 2, 'St': 3}

    for label_folder in os.listdir(base_folder):
        label_path = os.path.join(base_folder, label_folder)
        if os.path.isdir(label_path):
            label = label_dict[label_folder]
            for file in os.listdir(label_path):
                if file.endswith('.npy'):  # 假设文件是 .nyp 格式
                    file_paths.append(os.path.join(label_path, file))
                    labels.append(label)
    return file_paths, labels

def collect_file_paths(base_folder):
    file_paths = []
    for file in os.listdir(base_folder):
        if file.endswith('.npy'):  # 假设文件是 .nyp 格式
            file_paths.append(os.path.join(base_folder, file))
    return file_paths

class HyperspectralDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = np.load(self.file_paths[idx])
        img = torch.tensor(img).permute(2, 0, 1).float()
        label = self.labels[idx]
        label = torch.tensor(label)
        return img, label

class HyperspectralDataset_nolabel(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        img = np.load(self.file_paths[idx])
        img = torch.tensor(img).permute(2, 0, 1).float()
        return img

