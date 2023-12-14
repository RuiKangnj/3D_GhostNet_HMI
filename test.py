from dataset_prepare import convert_hyp_to_npy
import spectral.io.envi as envi
import numpy as np
from scipy.ndimage import rotate
from dataset_prepare import envi_open
from nn.net import GhostNet
import torch.nn as nn

from torch.utils.data import DataLoader
from data.dataloader import load_config, collect_file_paths_and_labels, HyperspectralDataset,load_json_paths
import torch.optim as optim
import torch
from data.dataloader import save_json_paths
from sklearn.model_selection import train_test_split


###load model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GhostNet(num_classes=4)
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()  # 将模型设置为评估模式

####load data
loaded_data = load_json_paths('3D_bacteria_cells.json')
test_paths, test_labels = loaded_data["train_paths"], loaded_data["train_labels"]
test_dataset = HyperspectralDataset(test_paths, test_labels)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


#####inference
total = 0
correct = 0
test_loss = 0.0
criterion = nn.CrossEntropyLoss()  # 假设使用交叉熵损失

with torch.no_grad():  # 在评估过程中不计算梯度
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(2)  ###[32,256,1,64,64]
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader.dataset)
test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')




#config = load_config('config/default.yaml')
#dataset_folder = config['dataset']['dataset_dir']

#file_paths, labels = collect_file_paths_and_labels(dataset_folder)

# 创建数据集
#dataset = HyperspectralDataset(file_paths, labels)


# # 划分训练集和剩余部分
# train_paths, temp_paths, train_labels, temp_labels = train_test_split(
#     file_paths, labels, test_size=0.3, random_state=42, stratify=labels)
#
# # 划分验证集和测试集
# val_paths, test_paths, val_labels, test_labels = train_test_split(
#     temp_paths, temp_labels, test_size=1/3, random_state=42, stratify=temp_labels)
#
# data_to_save = {
#     "train_paths": train_paths,
#     "val_paths": val_paths,
#     "test_paths": test_paths,
#     "train_labels": train_labels,
#     "val_labels": val_labels,
#     "test_labels": test_labels
# }
#
# save_json_paths(data_to_save, '3D_bateria_cells.json')
#
#





# target_path='dataset'
# folder_path='../Single-cell Hypercube dataset'
# convert_hyp_to_npy(folder_path,target_path)