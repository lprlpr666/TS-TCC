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

def get_z_score(train_dataset, valid_dataset, test_dataset):
    combined_dataset = torch.cat([train_dataset["samples"], valid_dataset["samples"], test_dataset["samples"]], dim=0)
    
    mean = combined_dataset.mean()
    std = combined_dataset.std()
    
    train_dataset = (train_dataset["samples"] - mean) / std
    test_dataset = (test_dataset["samples"] - mean) / std
    valid_dataset = (valid_dataset["samples"] - mean) / std
    
    return train_dataset, valid_dataset, test_dataset

def data_generator(data_path, configs, training_mode):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))
    
    # train_dataset["samples"], valid_dataset["samples"], test_dataset["samples"] = get_z_score(train_dataset, valid_dataset, test_dataset)

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader

def data_generator_partial(root_path, configs, training_mode, id, num):
    train_dataset = torch.load(os.path.join(root_path, "SEED-per-session", str(id), "session_1","train.pt"))
    valid_dataset = torch.load(os.path.join(root_path,"SEED", str(id), "val.pt"))
    test_dataset = torch.load(os.path.join(root_path,"SEED", str(id), "test.pt"))
    
    if num == 2 or num == 3:
        tmp_train_dataset = torch.load(os.path.join(root_path, "SEED-per-session", str(id), "session_2","train.pt"))
        train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
        train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
    if num == 3:
        tmp_train_dataset = torch.load(os.path.join(root_path, "SEED-per-session", str(id), "session_3","train.pt"))
        train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
        train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
        
        
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)
    
    return train_loader, valid_loader, test_loader


def data_generator_free(root_path, configs, training_mode, id, num):
    if num > 9:
        print("num should be less than 10")
    train_dataset = torch.load(os.path.join(root_path, "SEED-few-shot", str(id), "train_1.pt"))
    
    
    valid_dataset = torch.load(os.path.join(root_path,"SEED", str(id), "val.pt"))
    test_dataset = torch.load(os.path.join(root_path,"SEED", str(id), "test.pt"))
    if num == 0.1:
        # 获取数据集大小
        dataset_size = len(train_dataset["samples"])

        # 计算要抽取的数据量
        sample_size = int(dataset_size * 0.1)  # 10% 的数据量

        # 生成随机索引
        random_indices = torch.randperm(dataset_size)[:sample_size]

        # 使用随机索引获取抽样数据
        train_dataset["samples"] = train_dataset["samples"][random_indices]
        train_dataset["labels"] = train_dataset["labels"][random_indices]
        
        print(len(train_dataset["samples"]))
    elif num == 0.5:
        # 获取数据集大小
        dataset_size = len(train_dataset["samples"])

        # 计算要抽取的数据量
        sample_size = int(dataset_size * 0.5)  # 10% 的数据量

        # 生成随机索引
        random_indices = torch.randperm(dataset_size)[:sample_size]

        # 使用随机索引获取抽样数据
        train_dataset["samples"] = train_dataset["samples"][random_indices]
        train_dataset["labels"] = train_dataset["labels"][random_indices]
        
        print(len(train_dataset["samples"]))
    
    else:
        for i in range(2, int(num) + 1):
            tmp_train_dataset = torch.load(os.path.join(root_path, "SEED-few-shot", str(id), f"train_{i}.pt"))
            train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
            train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
        
    train_dataset = Load_Dataset(train_dataset, configs,  training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs,  training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256,
                                               shuffle=True, drop_last=False, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=256,
                                               shuffle=True, drop_last=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256,
                                              shuffle=True, drop_last=False, num_workers=0)
    
    return train_loader, valid_loader, test_loader

def data_generator_all_seed4(configs, training_mode):
    data_path = "../autodl-tmp/SEED-Dataset/SEED-IV-raw/"
    
    path1 = data_path + "1/"
    train_dataset = torch.load(os.path.join(path1, "train.pt"))
    valid_dataset = torch.load(os.path.join(path1, "val.pt"))
    test_dataset = torch.load(os.path.join(path1, "test.pt"))
    
    for i in range(2,16):
        path_temp = data_path + str(i) + "/"
        tmp_train_dataset = torch.load(os.path.join(path_temp, "train.pt"))
        tmp_valid_dataset = torch.load(os.path.join(path_temp, "val.pt"))
        tmp_test_dataset = torch.load(os.path.join(path_temp, "test.pt"))
        
        # tmp_train_dataset["samples"], tmp_valid_dataset["samples"] ,tmp_test_dataset["samples"] = get_z_score(tmp_train_dataset, tmp_valid_dataset, tmp_test_dataset)
        
        train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
        train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
        
        valid_dataset["samples"] = torch.cat((valid_dataset["samples"], tmp_valid_dataset["samples"]), 0)
        valid_dataset["labels"] = torch.cat((valid_dataset["labels"], tmp_valid_dataset["labels"]), 0)
        
        test_dataset["samples"] = torch.cat((test_dataset["samples"], tmp_test_dataset["samples"]), 0)
        test_dataset["labels"] = torch.cat((test_dataset["labels"], tmp_test_dataset["labels"]), 0)
        
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader
def data_generator_all(configs, training_mode):
    data_path = "../autodl-tmp/SEED-Dataset/SEED/"
    
    path1 = data_path + "1/"
    train_dataset = torch.load(os.path.join(path1, "train.pt"))
    valid_dataset = torch.load(os.path.join(path1, "val.pt"))
    test_dataset = torch.load(os.path.join(path1, "test.pt"))
    
    # train_dataset["samples"], valid_dataset["samples"], test_dataset["samples"] = get_z_score(train_dataset, valid_dataset, test_dataset)
    
    for i in range(2,16):
        path_temp = data_path + str(i) + "/"
        tmp_train_dataset = torch.load(os.path.join(path_temp, "train.pt"))
        tmp_valid_dataset = torch.load(os.path.join(path_temp, "val.pt"))
        tmp_test_dataset = torch.load(os.path.join(path_temp, "test.pt"))
        
        # tmp_train_dataset["samples"], tmp_valid_dataset["samples"] ,tmp_test_dataset["samples"] = get_z_score(tmp_train_dataset, tmp_valid_dataset, tmp_test_dataset)
        
        train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
        train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
        
        valid_dataset["samples"] = torch.cat((valid_dataset["samples"], tmp_valid_dataset["samples"]), 0)
        valid_dataset["labels"] = torch.cat((valid_dataset["labels"], tmp_valid_dataset["labels"]), 0)
        
        test_dataset["samples"] = torch.cat((test_dataset["samples"], tmp_test_dataset["samples"]), 0)
        test_dataset["labels"] = torch.cat((test_dataset["labels"], tmp_test_dataset["labels"]), 0)
        
    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader


def data_generator_all_seed_34(configs, training_mode):
    data_path = "../autodl-tmp/SEED-Dataset/SEED/"
    path1 = data_path + "1/"
    train_dataset = torch.load(os.path.join(path1, "train.pt"))
    valid_dataset = torch.load(os.path.join(path1, "val.pt"))
    test_dataset = torch.load(os.path.join(path1, "test.pt"))
    
    for i in range(2,16):
        path_temp = data_path + str(i) + "/"
        tmp_train_dataset = torch.load(os.path.join(path_temp, "train.pt"))
        tmp_valid_dataset = torch.load(os.path.join(path_temp, "val.pt"))
        tmp_test_dataset = torch.load(os.path.join(path_temp, "test.pt"))
        
        train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
        train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
        
        valid_dataset["samples"] = torch.cat((valid_dataset["samples"], tmp_valid_dataset["samples"]), 0)
        valid_dataset["labels"] = torch.cat((valid_dataset["labels"], tmp_valid_dataset["labels"]), 0)
        
        test_dataset["samples"] = torch.cat((test_dataset["samples"], tmp_test_dataset["samples"]), 0)
        test_dataset["labels"] = torch.cat((test_dataset["labels"], tmp_test_dataset["labels"]), 0)
    
    # 加入seed 4
    data_path_4 = "../autodl-tmp/SEED-Dataset/SEED-IV/"
    for i in range(1,2):
        path_temp = data_path_4 + str(i) + "/"
        tmp_train_dataset = torch.load(os.path.join(path_temp, "train.pt"))
        tmp_valid_dataset = torch.load(os.path.join(path_temp, "val.pt"))
        tmp_test_dataset = torch.load(os.path.join(path_temp, "test.pt"))
        
        train_dataset["samples"] = torch.cat((train_dataset["samples"], tmp_train_dataset["samples"]), 0)
        train_dataset["labels"] = torch.cat((train_dataset["labels"], tmp_train_dataset["labels"]), 0)
        
        valid_dataset["samples"] = torch.cat((valid_dataset["samples"], tmp_valid_dataset["samples"]), 0)
        valid_dataset["labels"] = torch.cat((valid_dataset["labels"], tmp_valid_dataset["labels"]), 0)
        
        test_dataset["samples"] = torch.cat((test_dataset["samples"], tmp_test_dataset["samples"]), 0)
        test_dataset["labels"] = torch.cat((test_dataset["labels"], tmp_test_dataset["labels"]), 0)

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader

def data_generator_sample(configs,training_mode, id, ratio, seed, file_name):
    data_path = "../autodl-tmp/SEED-Dataset/"
    random.seed(seed) 
    # 加载 train.pt 文件
    train_dataset = torch.load(os.path.join(data_path,file_name, str(id), "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path,file_name, str(id), "val.pt"))
    test_dataset = torch.load(os.path.join(data_path,file_name, str(id), "test.pt"))
    
    batch_size = 256
    if ratio != 1:
        if ratio <= 0.01:
            batch_size = 16
        elif ratio <= 0.05:
            batch_size = 32
        elif ratio < 0.3:
            batch_size = 64
        # 获取样本和标签
        samples = train_dataset["samples"]
        labels = train_dataset["labels"]
        print(labels.shape)
        print(samples.shape)

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

    train_dataset = Load_Dataset(train_dataset,configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs,training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)
    

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=False, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=256,
                                               shuffle=True, drop_last=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256,
                                              shuffle=True, drop_last=False, num_workers=0)
    
    return train_loader, valid_loader, test_loader