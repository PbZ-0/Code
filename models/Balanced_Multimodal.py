import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cluster import KMeans

import torch
from Config import config
def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)

    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def reinit_score(config, train_ECG, train_EMG, train_EDA, train_RESP, train_label, val_ECG, val_EMG, val_EDA, val_RESP, val_label):
    all_feature = [train_ECG, val_ECG[0], train_EMG, val_EMG[0], train_EDA, val_EDA[0], train_RESP, val_RESP[0]]
    stages = ['train_ECG', 'val_ECG', 'train_EMG', 'val_EMG', 'train_EDA', 'val_EDA', 'train_RESP', 'val_RESP']
    all_purity = []
    ##uncertain
    uncertain_ECG = uncertain(val_ECG[0])
    uncertain_EMG = uncertain(val_EMG[0])
    uncertain_EDA = uncertain(val_EDA[0])
    uncertain_RESP = uncertain(val_RESP[0])
    uncertain_All = np.exp(1.0 / uncertain_ECG) + np.exp(1.0 / uncertain_EMG) + np.exp(1.0 / uncertain_EDA) + np.exp(1.0 / uncertain_RESP)
    uncertain_lambda_ECG = (np.exp(1.0 / uncertain_ECG)) / uncertain_All
    uncertain_lambda_EMG = (np.exp(1.0 / uncertain_EMG)) / uncertain_All
    uncertain_lambda_EDA = (np.exp(1.0 / uncertain_EDA)) / uncertain_All
    uncertain_lambda_RESP = (np.exp(1.0 / uncertain_RESP)) / uncertain_All
    ##
    for idx, fea in enumerate(all_feature):
        # print('Computing t-SNE embedding')
        result = fea
        # 归一化处理
        result = result.reshape(result.size(0), -1)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        result = scaler.fit_transform(result)
        y_pred = KMeans(n_clusters=config.n_classes, random_state=0).fit_predict(result)

        if (stages[idx][:5] == 'train'):
            purity = purity_score(np.array(train_label), y_pred)
        else:
            purity = purity_score(np.array(val_label[0]), y_pred)
        all_purity.append(purity)

        print('%s purity= %.4f' % (stages[idx], purity))

    purity_gap_ECG = np.abs(all_purity[0] - all_purity[1])
    purity_gap_EMG = np.abs(all_purity[2] - all_purity[3])
    purity_gap_EDA = np.abs(all_purity[4] - all_purity[5])
    purity_gap_RESP = np.abs(all_purity[6] - all_purity[7])

    weight_ECG = torch.tanh(uncertain_lambda_ECG * purity_gap_ECG)
    weight_EMG = torch.tanh(uncertain_lambda_EMG * purity_gap_EMG)
    weight_EDA = torch.tanh(uncertain_lambda_EDA * purity_gap_EDA)
    weight_RESP = torch.tanh(uncertain_lambda_RESP * purity_gap_RESP)


    return weight_ECG, weight_EMG, weight_EDA, weight_RESP


def reinit(config, model, checkpoint, weight_ECG, weight_EMG, weight_EDA, weight_RESP):
    print("Start reinit ... ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, param in model.named_parameters():

        if 'ECG_net' in name:
            init_weight = checkpoint[name]
            current_weight = param.data

            if len(init_weight.shape) == 1 and len(current_weight.shape) == 1:
                continue
            else:
                if weight_ECG.shape[0] == init_weight.shape[0]:
                    new_weight = weight_ECG[:64].to(device) * init_weight.to(device) + (1 - weight_ECG[0:64]).to(
                        device) * current_weight.to(device)
                    param.data = new_weight

        elif 'EMG_net' in name:
            init_weight = checkpoint[name]
            current_weight = param.data
            if len(init_weight.shape) == 1 and len(current_weight.shape) == 1:
                continue
            else:
                if weight_EMG.shape[0] == init_weight.shape[0]:
                    new_weight = weight_EMG[:64].to(device) * init_weight.to(device) + (1 - weight_EMG[0:64]).to(
                        device) * current_weight.to(device)
                    param.data = new_weight
        elif 'EDA_net' in name:
            init_weight = checkpoint[name]
            current_weight = param.data
            if len(init_weight.shape) == 1 and len(current_weight.shape) == 1:
                continue
            else:
                if weight_EDA.shape[0] == init_weight.shape[0]:
                    new_weight = weight_EDA[:64].to(device) * init_weight.to(device) + (1 - weight_EDA[0:64]).to(
                        device) * current_weight.to(device)
                    param.data = new_weight
        elif 'RESP_net' in name:
            init_weight = checkpoint[name]
            current_weight = param.data
            if len(init_weight.shape) == 1 and len(current_weight.shape) == 1:
                continue
            else:
                if weight_RESP.shape[0] == init_weight.shape[0]:
                    new_weight = weight_RESP[:64].to(device) * init_weight.to(device) + (1 - weight_RESP[0:64]).to(
                        device) * current_weight.to(device)
                    param.data = new_weight
    return model

def uncertain(x):
    ##uncertain
    std = (x.var(dim=2, keepdim=False) + 1e-6).sqrt()
    t = (std.var(dim=0, keepdim=True) + 1e-6).sqrt()
    t = t.repeat(std.shape[0], 1)
    return t

