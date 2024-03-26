import os
import numpy as np
from scipy.io import loadmat
import torch

def make_window(signal, fs, overlap, window_size_sec):
    window_size = fs * window_size_sec
    overlap = int(window_size * (overlap / 100))
    start = 0
    segmented = np.zeros((1, signal.shape[0], window_size), dtype=int)
    while (start + window_size <= signal.shape[1]):
        segment = signal[:, start:start + window_size]
        segment = segment.reshape(1, segment.shape[0], segment.shape[1])
        segmented = np.append(segmented, segment, axis=0)
        start = start + window_size - overlap
    return segmented[1:]



# 给seed 3用的
def process_data_seed(file_path, labels, save_path):
    file_list = os.listdir(file_path)
    data_train = np.zeros((1, 62, 200), dtype=int)
    label_train = np.zeros((1, 1), dtype=int)
    data_valid = np.zeros((1, 62, 200), dtype=int)
    label_valid = np.zeros((1, 1), dtype=int)
    data_test = np.zeros((1, 62, 200), dtype=int)
    label_test = np.zeros((1, 1), dtype=int)
    
    for file_name in file_list:
        data = loadmat(os.path.join(file_path, file_name))
        keys = data.keys()
        keys = [x for x in keys if '_eeg' in x]
        
        for key in keys:
            index = int(key.split('g')[-1])
            processed_data_sub = make_window(data[key], 200, 0 , 5)
            if index <= 9:
                data_train = np.append(data_train, processed_data_sub, axis=0)
                label_train = np.append(label_train, np.full((processed_data_sub.shape[0], 1), labels[index-1]), axis=0)
            elif index <=12:
                data_valid = np.append(data_valid, processed_data_sub, axis=0)
                label_valid = np.append(label_valid, np.full((processed_data_sub.shape[0], 1), labels[index-1]), axis=0)
            else :
                data_test = np.append(data_test, processed_data_sub, axis=0)
                label_test = np.append(label_test, np.full((processed_data_sub.shape[0], 1), labels[index-1]), axis=0)
    data_train = torch.tensor(data_train[1:])
    label_train = torch.tensor(label_train[1:])
    label_train = torch.squeeze(label_train)
    data_valid = torch.tensor(data_valid[1:])
    label_valid = torch.tensor(label_valid[1:])
    label_valid = torch.squeeze(label_valid)
    data_test = torch.tensor(data_test[1:])
    label_test = torch.tensor(label_test[1:])
    label_test = torch.squeeze(label_test)
    torch.save({"samples":data_train, "labels":label_train}, os.path.join(save_path, "train.pt"))
    torch.save({"samples":data_valid, "labels":label_valid}, os.path.join(save_path, "val.pt"))
    torch.save({"samples":data_test, "labels":label_test}, os.path.join(save_path, "test.pt"))


if __name__ == "__main__":
    for i in range(15):
        path = os.path.join('/userhome2/dairenjie/TimeSeries/data/SEED5s', str(i + 1))
        if not os.path.exists(path):
            os.mkdir(path)

    # seed 3数据集用的
    file_path0 = "/userhome2/dairenjie/BCMI/dataset/SEED/Preprocessed_EEG"
    labels = loadmat(os.path.join(file_path0, 'label'))['label'][0] + 1
    # labels = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    print(labels)
    root_path = '/userhome2/dairenjie/BCMI/dataset/SEED/Preprocessed_EEG_Person'
    save_root_path = '/userhome2/dairenjie/TimeSeries/data/SEED5s'
    for i in range(15):
        file_path = os.path.join(root_path, str(i + 1))
        save_path = os.path.join(save_root_path, str(i + 1))
        process_data_seed(file_path, labels, save_path)