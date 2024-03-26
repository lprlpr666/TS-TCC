import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from .augmentations import DataTransform


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


def data_generator(data_path, configs, training_mode):

    # 在模型初始化之后
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode, device)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode, device)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode, device)

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
    for i in range(1,4):
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