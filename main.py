
from resnet1d_wang import  Multimodal
import os
import time
import random
import argparse
from datetime import datetime

from ptflops import get_model_complexity_info
import torch.nn.functional as F
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import higher
import copy
import utils as ut
from Balanced_Multimodal import reinit_score, reinit
from data_generator import task_generator,  load_datasets
from Config import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


def main(config, tasks_data, tasks_labels, model, iteration, device,  mini_task_size, criterion, checkpoint, Train=False):


# To control update parameter
    head_params = [p for name, p in model.named_parameters() if 'classifier' in name]
    body_params = [p for name, p in model.named_parameters() if 'classifier' not in name]

    # outer optimizer
    meta_optimizer = torch.optim.Adam([{'params': body_params, 'lr': config.meta_lr},
                                       {'params': head_params, 'lr': config.meta_lr
                                       if iteration != 0 and (iteration + 1) % config.freeze_epoch == 0 else 0}])

    inner_optimizer = torch.optim.Adam([{'params': body_params, 'lr': config.task_lr},
                                        {'params': head_params, 'lr': config.task_lr
                                        if iteration != 0 and (iteration + 1) % config.freeze_epoch == 0 else 0}])

    meta_optimizer.zero_grad()
    inner_optimizer.zero_grad()

    total_loss = torch.tensor(0., device=device)
    accuracy = torch.tensor(0., device=device)
    precision = torch.tensor(0., device=device)
    recall = torch.tensor(0., device=device)
    F1 = torch.tensor(0., device=device)

    for task_idx, (task_data, task_label) in enumerate(zip(tasks_data, tasks_labels)):
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fnet, diffopt)

            outer_loss = torch.tensor(0., device=device)

            spt_data, qry_data = task_data[0], task_data[1]

            spt_label, qry_label = task_label[0], task_label[1]


            spt_data = torch.tensor(spt_data)
            spt_label = torch.tensor(spt_label)

            src_tensor = TensorDataset(spt_data, spt_label)
            src_loader = DataLoader(src_tensor,
                                    batch_size=config.inner_batch_size if Train else config.test_batch_size, shuffle=True, drop_last=False)

            if Train:
                ##  reinit
                all_ECG_fea = []
                all_EMG_fea = []
                all_EDA_fea = []
                all_RESP_fea = []
                all_label = []
                query_all_ECG_fea = []
                query_all_EMG_fea = []
                query_all_EDA_fea = []
                query_all_RESP_fea = []
                query_all_label = []
                ##
                for batch_idx, (inputs, labels) in enumerate(src_loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    spt_logits, ECG_fea, EMG_fea, EDA_fea, RESP_fea = fnet(inputs)
                    ###reinit
                    all_ECG_fea.append(ECG_fea.data.cpu())
                    all_EMG_fea.append(EMG_fea.data.cpu())
                    all_EDA_fea.append(EDA_fea.data.cpu())
                    all_RESP_fea.append(RESP_fea.data.cpu())
                    all_label.append(labels.data.cpu())
                    ###
                    spt_loss = F.cross_entropy(spt_logits, labels.long())#fusion
                    diffopt.step(spt_loss)
                ##reinit
                all_ECG_fea = torch.cat(all_ECG_fea)
                all_EMG_fea = torch.cat(all_EMG_fea)
                all_EDA_fea = torch.cat(all_EDA_fea)
                all_RESP_fea = torch.cat(all_RESP_fea)
                all_label = torch.cat(all_label)
                ##
                inputs, labels = qry_data.to(device), qry_label.to(device)
                query_logit, ECG_fea, EMG_fea, EDA_fea, RESP_fea = fnet(inputs)
                ## reinit
                query_all_ECG_fea.append(ECG_fea.data.cpu())
                query_all_EMG_fea.append(EMG_fea.data.cpu())
                query_all_EDA_fea.append(EDA_fea.data.cpu())
                query_all_RESP_fea.append(RESP_fea.data.cpu())
                query_all_label.append(labels.data.cpu())
                ###
                outer_loss += F.cross_entropy(query_logit, labels.long())

                with torch.no_grad():
                    query_logit = torch.argmax(query_logit, dim=1).cpu().numpy()
                    accuracy += accuracy_score(query_logit, labels.cpu())

                    precision += precision_score(query_logit, labels.cpu(), average='macro')

                    recall += recall_score(query_logit, labels.cpu(), average='macro', zero_division=0)

                    F1 += f1_score(query_logit, labels.cpu(), average='macro')

                total_loss += outer_loss

                if task_idx != 0 and (task_idx+1) % config.meta_batch_size == 0 or (task_idx + 1) == len(
                        tasks_data):
                    total_loss.div_(config.meta_batch_size)
                    total_loss.backward()
                    meta_optimizer.step()
                    total_loss = torch.tensor(0.).to(device)

            else:
                for batch_idx, (inputs, labels) in enumerate(src_loader):
                    inner_optimizer.zero_grad()
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    logit,_,_,_,_ = model(inputs)
                    loss = criterion(logit, labels.long())
                    loss.backward()
                    inner_optimizer.step()

                inputs, labels = torch.from_numpy(qry_data).to(device), torch.from_numpy(qry_label).to(device)
                query_logit,_,_,_,_ = model(inputs)
                outer_loss += F.cross_entropy(query_logit, labels.long())

                with torch.no_grad():
                    query_logit = torch.argmax(query_logit, dim=1).cpu().numpy()
                    accuracy += accuracy_score(query_logit, labels.cpu())
                    precision += precision_score(query_logit, labels.cpu(), average='macro')
                    recall += recall_score(query_logit, labels.cpu(), average='macro')
                    F1 += f1_score(query_logit, labels.cpu(), average='macro')
                    total_loss += outer_loss
                return total_loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(), precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), F1.detach().cpu().numpy()
    if Train:
        accuracy.div_(task_idx + 1)
        precision.div_(task_idx + 1)
        recall.div_(task_idx + 1)
        F1.div_(task_idx + 1)
        total_loss.div_(task_idx + 1)


    return total_loss.detach().cpu().numpy(), accuracy.detach().cpu().numpy(), precision.detach().cpu().numpy(), recall.detach().cpu().numpy(), F1.detach().cpu().numpy(),\
all_ECG_fea, all_EMG_fea, all_EDA_fea, all_RESP_fea, all_label, query_all_ECG_fea, query_all_EMG_fea, query_all_EDA_fea, query_all_RESP_fea, query_all_label


def train(config, train_index, val_index):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(config.seed)
    model = Multimodal(config).to(torch.float32).to(device)


    torch.save(model.state_dict(), 'init_para.pkl')
    PATH = 'init_para.pkl'
    checkpoint = torch.load(PATH)
    print('get init weight')

    criterion = nn.CrossEntropyLoss()

    dataset_train, dataset_test = load_datasets(train_index, val_index)

    srcdat = []
    srclbl = []
    zrdat = []
    zrlbl =[]
    for source_data, source_label in dataset_train:
        srcdat.append(source_data)
        srclbl.append(source_label)

    for taget_data, taget_label in dataset_test:
        zrdat.append(taget_data)
        zrlbl.append(taget_label)

    tasks_data_spt, tasks_labels_spt = task_generator(config, srcdat, srclbl)

    tasks_data_val = []
    tasks_labels_val = []
    sup_data = np.concatenate([srcdat[j] for j in range(len(srcdat))], axis=0)
    qur_data = np.concatenate([zrdat[j] for j in range(len(zrdat))], axis=0)
    sup_lab = np.concatenate([srclbl[j] for j in range(len(srclbl))], axis=0)
    qur_lab = np.concatenate([zrlbl[j] for j in range(len(zrlbl))], axis=0)
    tasks_data_val.append((sup_data, qur_data))
    tasks_labels_val.append((sup_lab, qur_lab))


    train_iter = []
    train_loss = []
    train_acc = []
    train_pre = []
    train_rec = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_pre = []
    val_rec = []
    val_f1 = []

    best_acc = -1
    flag_reinit = 0
    for meta_iteration in range(config.batch_iter):
        iter_time = time.time()
        print('Meta iteration =', meta_iteration + 1)
        if meta_iteration != 0 and (meta_iteration + 1) % config.freeze_epoch == 0:
            print("Update head parameters")

        mini_task_size = config.mini_task_size
        meta_train_loss, meta_train_acc, meta_train_pre, meta_train_rec, meta_train_f1, \
        all_ECG_fea, all_EMG_fea, all_EDA_fea, all_RESP_fea, all_label, \
        query_all_ECG_fea, query_all_EMG_fea, query_all_EDA_fea, query_all_RESP_fea, query_all_label = main(config, tasks_data_spt, tasks_labels_spt, model, meta_iteration, device,  mini_task_size, criterion,
                                               Train=True, checkpoint= checkpoint)

        ##  reinit
        if ((meta_iteration % config.reinit_epoch == 0) & (meta_iteration > 0)):
            flag_reinit += 1
            if (flag_reinit <= config.reinit_num):

                weight_ECG, weight_EMG, weight_EDA, weight_RESP = reinit_score(config, all_ECG_fea, all_EMG_fea, all_EDA_fea, all_RESP_fea, all_label,
                                                           query_all_ECG_fea, query_all_EMG_fea, query_all_EDA_fea, query_all_RESP_fea, query_all_label)
                model = reinit(config, model, checkpoint, weight_ECG, weight_EMG, weight_EDA, weight_RESP)

        train_iter.append(meta_iteration + 1)
        train_loss.append(meta_train_loss)
        train_acc.append(meta_train_acc)
        train_pre.append(meta_train_pre)
        train_rec.append(meta_train_rec)
        train_f1.append(meta_train_f1)

        copy_model = copy.deepcopy(model)

        meta_val_loss, meta_val_acc, meta_val_pre, meta_val_rec, meta_val_f1 = main(config, tasks_data_val, tasks_labels_val, copy_model, meta_iteration, device,
                                           mini_task_size, criterion, Train=False, checkpoint= checkpoint)

        if meta_val_acc > best_acc:
            best_acc = meta_val_acc

            if not os.path.exists(config.ckpt_path):
                os.mkdir(config.ckpt_path)
            model_name ='Dataset{}_subject_{}'.format(config.dataset, str(i))
            saved_dict = {'model': model.state_dict()}
            save_dir = os.path.join(config.ckpt_path, model_name)

            torch.save(saved_dict, save_dir)
        else:
            pass

        val_loss.append(meta_val_loss)
        val_acc.append(meta_val_acc)
        val_pre.append(meta_val_pre)
        val_rec.append(meta_val_rec)
        val_f1.append(meta_val_f1)


        print("Train Iter Loss = {:.4f}".format(meta_train_loss))
        print("Train Iter Acc = {:.4f}".format(meta_train_acc))
        print("Train Iter Pre = {:.4f}".format(meta_train_pre))
        print("Train Iter Rec = {:.4f}".format(meta_train_rec))
        print("Train Iter F1 = {:.4f}".format(meta_train_f1))
        print("Validation Iter Loss = {:.4f}".format(meta_val_loss))
        print("Validation Iter Acc = {:.4f}".format(meta_val_acc))
        print("Validation Iter Pre = {:.4f}".format(meta_val_pre))
        print("Validation Iter Rec = {:.4f}".format(meta_val_rec))
        print("Validation Iter F1 = {:.4f}".format(meta_val_f1))
        max_acc = max(val_acc)
        print("Best Validation Iter Acc = {:.4f}".format(max_acc))
        m, s = divmod(time.time() - iter_time, 60)
        h, m = divmod(m, 60)

        print("Iteration total_time = {} mins {:.6} secs".format(m, s))
        print("=" * 30)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


if __name__ == '__main__':