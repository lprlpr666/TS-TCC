import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform
from sklearn.preprocessing import StandardScaler
import random

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        self.x_data = dataset["samples"]
        self.y_data = dataset["labels"]

        self.len = self.x_data.shape[0]
        self.config = config
        
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            # tmp = self.x_data[index].unsqueeze(0)
            # aug1, aug2 = DataTransform(tmp, self.config)
            # aug1 = aug1.squeeze(0)
            # aug2 = aug2.squeeze(0)
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len
    
def load_data(index, data_path, subject):
    train_dataset = None
    data_path = os.path.join(data_path, str(subject))
    
    for i in index:
        # print("index:",i)
        path_temp = os.path.join(data_path, f"{i}.pt")
        tmp = torch.load(path_temp)
        if train_dataset == None:
            train_dataset = tmp
        else:
            train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp["samples"]), 0)
            train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp["labels"]), 0)
    train_dataset["labels"] = train_dataset["labels"].squeeze(1)
   
    return train_dataset

def load_data_seed(index, data_path, subject):
    train_dataset = None
    data_path = os.path.join(data_path, str(subject))
    
    for i in index:
        # print("index:",i)
        path_temp = os.path.join(data_path, f"train_{i}.pt")
        tmp = torch.load(path_temp)
        if train_dataset == None:
            train_dataset = tmp
        else:
            train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp["samples"]), 0)
            train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp["labels"]), 0)
   
    return train_dataset

def get_dataset(seed, subject):
    data_path =  "../autodl-tmp/SEED-Dataset/SEED-IV-blocks/"
    random.seed(seed)
    selected_numbers = random.sample(range(18), 15)
    train_index = random.sample(selected_numbers, 12)
    valid_index = [elem for elem in selected_numbers if elem not in train_index]
    test_index = [elem for elem in range(18) if elem not in selected_numbers]
    
    train_dataset = load_data(train_index, data_path, str(subject))
    valid_dataset = load_data(valid_index, data_path, str(subject))
    test_dataset = load_data(test_index, data_path, str(subject))

    return train_dataset, valid_dataset, test_dataset


def get_dataset_seed(seed, subject):
    data_path =  "../autodl-tmp/SEED-Dataset/SEED-block/"
    random.seed(seed)
    selected_numbers = random.sample(range(15), 12)
    train_index = random.sample(selected_numbers, 9)
    valid_index = [elem for elem in selected_numbers if elem not in train_index]
    test_index = [elem for elem in range(15) if elem not in selected_numbers]
    print(train_index, valid_index, test_index)
    train_dataset = load_data_seed(train_index, data_path, str(subject))
    valid_dataset = load_data_seed(valid_index, data_path, str(subject))
    test_dataset = load_data_seed(test_index, data_path, str(subject))
    
    return train_dataset, valid_dataset, test_dataset
    
def data_generator_sample(training_mode, configs, id, ratio, file_seed, file_name):    
    if file_name == 'SEED4':
        train_dataset, valid_dataset, test_dataset = get_dataset(file_seed, id)
        # print(valid_dataset["labels"].shape, test_dataset["labels"].shape)
        print("get!")
    elif file_name == 'SEED':
        train_dataset, valid_dataset, test_dataset = get_dataset_seed(file_seed, id)
        print("get seed!")

    batch_size = 256
    
    if ratio != 1:
        samples = train_dataset["samples"]
        labels = train_dataset["labels"]
        if ratio <= 0.01:
            batch_size = 16
        elif ratio <= 0.05:
            batch_size = 32
        elif ratio <= 0.5:
            batch_size = 128

        # 获取每个标签的索引
        label_indices = {}
        for i, label in enumerate(labels):
            if label.item() not in label_indices:
                label_indices[label.item()] = []
            label_indices[label.item()].append(i)
        # 从每个标签中随机采样 10% 的数据
        sampled_indices = []
        for label in label_indices:
            label_samples = label_indices[label]
            sample_size = int(ratio * len(label_samples))  # 计算每个标签的采样数量
            print(sample_size)
            sampled_indices.extend(random.sample(label_samples, sample_size))  # 从每个标签的样本中随机采样

        # 从原始数据中采样得到的样本和标签
        sampled_samples = [samples[i].tolist() for i in sampled_indices]
        sampled_labels = [labels[i].item() for i in sampled_indices]
        

        # 将采样后的数据转换为 PyTorch 张量
        sampled_samples_tensor = torch.tensor(sampled_samples)
        sampled_labels_tensor = torch.tensor(sampled_labels)
        print(sampled_samples_tensor.shape)
        print(sampled_labels_tensor.shape)
        
        train_dataset["samples"] = sampled_samples_tensor
        train_dataset["labels"] = sampled_labels_tensor

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)
    

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    
    return train_loader, valid_loader, test_loader


def data_generator_seed4_kfolder(seed, config, training_mode):
    data_path =  "../autodl-tmp/SEED-Dataset/SEED-IV-blocks/"
    random.seed(seed)
    selected_numbers = random.sample(range(18), 15)
    train_index = random.sample(selected_numbers, 12)
    valid_index = [elem for elem in selected_numbers if elem not in train_index]
    test_index = [elem for elem in range(18) if elem not in selected_numbers]
    print(train_index, valid_index, test_index)
    
    train_dataset = load_data(train_index, data_path, "1")
    valid_dataset = load_data(valid_index, data_path, "1")
    test_dataset = load_data(test_index, data_path, "1")
    for subject in range(2,16):
        # print("subject ",subject)
        tmp_train_dataset = load_data(train_index, data_path, subject)
        tmp_valid_dataset = load_data(valid_index, data_path, subject)
        tmp_test_dataset = load_data(test_index, data_path, subject)
        
        train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
        train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
        
        valid_dataset["samples"] = torch.cat((valid_dataset["samples"], tmp_valid_dataset["samples"]), 0)
        valid_dataset["labels"] = torch.cat((valid_dataset["labels"], tmp_valid_dataset["labels"]), 0)
        
        test_dataset["samples"] = torch.cat((test_dataset["samples"], tmp_test_dataset["samples"]), 0)
        test_dataset["labels"] = torch.cat((test_dataset["labels"], tmp_test_dataset["labels"]), 0)
    
    train_dataset = Load_Dataset(train_dataset, config, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, config, training_mode)
    test_dataset = Load_Dataset(test_dataset, config, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                               shuffle=True, drop_last=config.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                                               shuffle=False, drop_last=config.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    
    return train_loader, valid_loader, test_loader

def data_generator_seed_kfolder(seed, config, training_mode):
    data_path =  "../autodl-tmp/SEED-Dataset/SEED-block/"
    random.seed(seed)
    selected_numbers = random.sample(range(15), 12)
    train_index = random.sample(selected_numbers, 9)
    valid_index = [elem for elem in selected_numbers if elem not in train_index]
    test_index = [elem for elem in range(15) if elem not in selected_numbers]
    print(train_index, valid_index, test_index)
    
    train_dataset = load_data_seed(train_index, data_path, "1")
    valid_dataset = load_data_seed(valid_index, data_path, "1")
    test_dataset = load_data_seed(test_index, data_path, "1")
    for subject in range(2,16):
        # print("subject ",subject)
        tmp_train_dataset = load_data_seed(train_index, data_path, subject)
        tmp_valid_dataset = load_data_seed(valid_index, data_path, subject)
        tmp_test_dataset = load_data_seed(test_index, data_path, subject)
        
        train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
        train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
        
        valid_dataset["samples"] = torch.cat((valid_dataset["samples"], tmp_valid_dataset["samples"]), 0)
        valid_dataset["labels"] = torch.cat((valid_dataset["labels"], tmp_valid_dataset["labels"]), 0)
        
        test_dataset["samples"] = torch.cat((test_dataset["samples"], tmp_test_dataset["samples"]), 0)
        test_dataset["labels"] = torch.cat((test_dataset["labels"], tmp_test_dataset["labels"]), 0)
    
    train_dataset = Load_Dataset(train_dataset, config, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, config, training_mode)
    test_dataset = Load_Dataset(test_dataset, config, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                                               shuffle=True, drop_last=config.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=config.batch_size,
                                               shuffle=False, drop_last=config.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    
    return train_loader, valid_loader, test_loader
    
    

