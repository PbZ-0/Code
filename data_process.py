import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
def task_generator(config,  data, label):
    srcdat = data
    srclbl = label
    def create_tasks( srcdat,  srclbl):
        tasks_data = []
        tasks_labels = []
        for i in range(len(srcdat)):
            # 当前任务的查询集
            query_data = srcdat[i]
            query_labels = srclbl[i]
            support_data = np.concatenate([srcdat[j] for j in range(len(srcdat)) if j != i], axis=0)
            support_labels = np.concatenate([srclbl[j] for j in range(len(srclbl)) if j != i], axis=0)
            tasks_data.append((support_data,  query_data))
            tasks_labels.append((support_labels, query_labels))

        return tasks_data, tasks_labels

    tasks_data, tasks_labels = create_tasks(srcdat=srcdat, srclbl=srclbl)

    return [tasks_data, tasks_labels]

def divied_data(data):
    group_sizes = [240, 240, 240, 240, 210, 240, 240, 240, 240, 210]##driver subject number
    ##pilot subject number:
    #group_sizes = [188, 206, 190, 207, 151, 204, 211, 200, 209, 199, 222, 181, 216, 209, 218, 211, 202, 213, 206, 278, 217, 203, 206, 212, 186, 229, 207, 210, 205]
    indexed_data = {}
    current_index = 0
    for i, size in enumerate(group_sizes):
        end_index = current_index + size

        indexed_data[i] = data[current_index:end_index, :, :]
        current_index = end_index
    return indexed_data

def divied_label(labels):
    group_sizes = [240, 240, 240, 240, 210, 240, 240, 240, 240, 210]##driver subject number
    ##pilot subject number:
    #group_sizes = [188, 206, 190, 207, 151, 204, 211, 200, 209, 199, 222, 181, 216, 209, 218, 211, 202, 213, 206, 278, 217, 203, 206, 212, 186, 229, 207, 210, 205]

    indexed_labels = {}
    current_index = 0

    for i, size in enumerate(group_sizes):
        end_index = current_index + size
        indexed_labels[i] = labels[current_index:end_index]
        current_index = end_index
    return indexed_labels

def load_dataset(path):
    Y = np.load(path+'Label_pilot.npy')
    X1 = np.load(path+'EMG_pilot_downsample.npy')
    X2 = np.load(path+'ECG_pilot_downsample.npy')
    X3 = np.load(path+'EDA_128_pilot_downsample.npy')
    X4 = np.load(path+'RESP_pilot_downsample.npy')
    X = np.concatenate((X1, X2, X3, X4), axis=1)
    X = divied_data(X)
    Y = divied_label(Y)
    return X, Y
def data_process(data, labels, train_index, test_index):
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for index in train_index:
        patient_data = data[index]
        patient_data = z_score_normalize(patient_data)
        patient_label = labels[index]
        X_train_list.append(patient_data)
        y_train_list.append(patient_label)


    for index in test_index:
        patient_test_data = data[index]
        patient_test_data = z_score_normalize(patient_test_data)
        patient_test_label = labels[index]
        X_test_list.append(patient_test_data)
        y_test_list.append(patient_test_label)

    return X_train_list, y_train_list, X_test_list, y_test_list

def preprocess_signals(X_train, X_test):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    return apply_standardizer(X_train, ss),  apply_standardizer(X_test, ss)
def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp
def z_score_normalize(arr):
    mean = np.mean(arr)
    std_dev = np.std(arr)
    return (arr - mean) / std_dev

class Dataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """
    def __init__(self, signals: np.ndarray,  labels: np.ndarray):
        super(Dataset, self).__init__()
        self.data = signals
        self.label = labels

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.tensor(x.copy(), dtype=torch.float)
        y = self.label[index]
        y = torch.tensor(y, dtype=torch.float)


        return x, y

    def __len__(self):
        return len(self.data)


def load_datasets(train_index, val_index):
    data, raw_labels = load_dataset('./data/')
    X_train, y_train, X_test, y_test = data_process(data, raw_labels, train_index, val_index)
    ds_train = Dataset(X_train, y_train)
    ds_test = Dataset(X_test, y_test)

    return ds_train, ds_test