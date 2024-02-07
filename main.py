import argparse
import pandas as pd
import numpy as np
import logging
import random
import time
import json
from collections import Counter
import matplotlib.pyplot as plt

import click
from utils.write2txt import writer2txt
import os
import re
from BiasedAD import BiasedAD
from datasets.nb15_contamination import NB15_contamination_Dataset
from datasets.nb15_contamination_for_BADM import NB15_contamination_for_BADM
from datasets.nb15 import NB15Dataset
from datasets.nb15_for_BADM import nb15_for_BADM
from datasets.fashionmnist import FashionMNIST_Dataset
from datasets.fashionmnist_for_BADM import fashionmnist_for_BADM
from datasets.sqb import SQBDataset
from datasets.sqb_for_BADM import sqb_for_BADM

args = argparse.ArgumentParser()
args.add_argument('--model_type', type=click.Choice(["BiasedAD", "BiasedADM", "Gcon"]), default="BiasedAD")
args.add_argument('--dir_path', type=str, default="./result/DEBUG")
args.add_argument('--dataset_name', type=str, default="nb15")
args.add_argument('--device', type=str, default="cuda")
args.add_argument('--gpu', type=str, default="3")
args.add_argument('--intermediate_flag', type=bool, default=False)
args.add_argument("--random_seed" , type=int, default = 0)
args.add_argument('--debug', type=bool, default=False)

args.add_argument("--lr" , type=float, default = None)
args.add_argument("--epoch" , type=int, default = None)
args.add_argument("--batch_size" , type=int, default = None)
args.add_argument("--weight_decay" , type=float, default = 5e-7)

args.add_argument("--ae_lr" , type=float, default = None)
args.add_argument("--ae_epoch" , type=int, default = None)
args.add_argument("--ae_batch_size" , type=int, default = None)
args.add_argument("--ae_weight_decay" , type=float, default = 1e-6)
args.add_argument("--sample_count" , type=int, default = 100)

args.add_argument('--times', type=int, default=10)
args.add_argument('--eta_0', type=int, default=None)
args.add_argument('--eta_1', type=int, default=None)
args.add_argument('--eta_2', type=int, default=None)

# The follow three options are useful when the dataset is the fashionMNIST dataset
args.add_argument("--contamination_for_FMNIST", type=float, default=0.02)
args.add_argument("--labeled_target_outlier_number", type=int, default=100)
args.add_argument("--normal_class" , type=int, default = 4)
args.add_argument("--non_target_outlier_class" , type=int, default = 0)
args.add_argument("--target_outlier_class" , type=int, default = 6)

# The follow three options are useful when the dataset is the nb15 dataset with a fixed contamination ratio of 2%.
args.add_argument("--s_normal" , type=float, default = 1.0)
args.add_argument("--s_non_target" , type=int, default = 100)
args.add_argument("--s_target" , type=int, default = 100)
# Controls the number of non-target categories
args.add_argument("--nb15_non_target_class_num" , type=int, default = 4)
# Controls target categories
args.add_argument("--nb15_target_class", nargs="+", type=str, default=["DoS", "Generic", "Backdoor"], choices=["DoS", "Generic", "Backdoor"])

args.add_argument("--sqb_test_frac" , type=int, default = None)
args.add_argument("--update_anchor" , type=str, default = "default")
args.add_argument("--update_epoch" , type=int, default = 10)

args = args.parse_args()

# args = args.parse_args(["--model_type", "BiasedAD", "--dir_path", "./kdd_results/BAD_20240201/nb15", "--dataset_name", "nb15", "--gpu", "1", "--random_seed", "0", "--s_normal", "0.01", "--s_non_target", "1", "--s_target", "1"])
# args = args.parse_args(["--model_type", "BiasedADM", "--dir_path", "./kdd_results/BADM_20240201/nb15", "--dataset_name", "nb15", "--gpu", "0", "--sample_count", "1000", "--random_seed", "0", "--s_normal", "0.01", "--s_target", "1",])
# args = args.parse_args(["--model_type", "BiasedADM", "--dir_path", "./kdd_results/RF_20240201/nb15", "--dataset_name", "nb15", "--gpu", "3", "--sample_count", "1000", "--random_seed", "0", "--s_normal", "0.01", "--s_target", "1", "--times", "1",])
# args = args.parse_args(["--model_type", "BiasedAD", "--dir_path", "./kdd_results/RF_20240201/nb15", "--dataset_name", "nb15", "--gpu", "2", "--random_seed", "0", "--s_normal", "0.01", "--s_non_target", "1", "--s_target", "1", "--times", "1",])
# args = args.parse_args(["--model_type", "BiasedADM", "--dir_path", "./kdd_results/RF_20240201/nb15", "--dataset_name", "nb15", "--gpu", "3", "--sample_count", "1000", "--random_seed", "0", "--s_normal", "0.01", "--s_target", "1", "--times", "1",])


# args = args.parse_args(["--model_type", "BiasedADM", "--dir_path", "./result/BADM_20240129/fmnist", "--dataset_name", "fashionmnist", "--normal_class", "4", "--non_target_outlier_class", "4", "--target_outlier_class", "6", "--gpu", "2", "--random_seed", "0"])
# args = args.parse_args(["--model_type", "BiasedAD", "--dir_path", "./result/BAD_20240129/nb15", "--dataset_name", "nb15", "--gpu", "0", "--random_seed", "0",])
# args = args.parse_args(["--model_type", "BiasedADM", "--dir_path", "./result/BADM_20240129/nb15", "--dataset_name", "nb15", "--gpu", "0", "--sample_count", "1000", "--random_seed", "0",])
# args = args.parse_args(["--model_type", "BiasedADM", "--dir_path", "./result/BADM_20240129/SQB", "--dataset_name", "SQB", "--gpu", "0", "--sample_count", "200", "--random_seed", "0",])

# args = args.parse_args(["--model_type", "BiasedADM", "--dir_path", "./result/BADM_20240129/fmnist", "--dataset_name", "fashionmnist", "--normal_class", "4", "--non_target_outlier_class", "2", "--target_outlier_class", "0", "--gpu", "2", "--random_seed", "0", "--intermediate_flag", "True", "--times", "1", "--contamination_for_FMNIST", "0.02", "--labeled_target_outlier_number", "100"])
# args = args.parse_args(["--model_type", "BiasedAD", "--dir_path", "./result/BAD_20240129/fmnist", "--dataset_name", "fashionmnist", "--normal_class", "4", "--non_target_outlier_class", "4", "--target_outlier_class", "6", "--gpu", "3", "--random_seed", "0", "--intermediate_flag", "True", "--times", "1"])
# args = args.parse_args(["--model_type", "BiasedAD", "--dir_path", "./result/BAD_20240129/nb15", "--dataset_name", "nb15", "--gpu", "0", "--random_seed", "0", "--intermediate_flag", "True", "--times", "1"])
# args = args.parse_args(["--model_type", "BiasedAD", "--dir_path", "./result/BAD_20240129/SQB", "--dataset_name", "SQB", "--gpu", "0", "--random_seed", "0", "--intermediate_flag", "True", "--times", "1", "--sqb_test_frac", "100"])

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

file_save_path = args.dir_path

if args.intermediate_flag:
    args.times = 1

for i in range(args.times):
    contaminationRate = 2
    
    #The contaminated dataset name of nb-15 is "7_Unlabelled_data," where the number 7 indicates a contamination ratio of 7*2=14%.
    if re.search("Unlabelled_data", args.dataset_name):
        file_name = "dataset=NB15_contamination"

        if args.model_type == "BiasedAD":
            dataset = NB15_contamination_Dataset(args.dataset_name, args.random_seed)
            default_eta_0 = 20
            default_eta_1 = 1
            default_eta_2 = 1
        elif args.model_type == "BiasedADM":
            args.s_non_target = 0
            dataset = NB15_contamination_for_BADM(args.dataset_name, s_non_target = 0, s_target = args.s_target, nb15_non_target_class_num = 4, seed = args.random_seed)
            default_eta_0 = 10
            default_eta_1 = 1
            default_eta_2 = 1
        
        file_name += ","
        file_name += "s_non_target=" + str(args.s_non_target) + ",s_target=" + str(args.s_target) + ",nb15_non_target_class_num=" +str(args.nb15_non_target_class_num) + ","
        net_name = 'mlp_for_nb15'
        contaminationRate = int(args.dataset_name.split("_")[0]) * 2
        args.ae_lr = 0.0001 if args.ae_lr is None else args.ae_lr
        args.ae_batch_size = 128 if args.ae_batch_size is None else args.ae_batch_size
        args.ae_epoch = 30 if args.ae_epoch is None else args.ae_epoch
        BAD_lr = 0.00001
        BAD_batch_size = 128
        BAD_epoch = 50

    elif args.dataset_name == "nb15":
        if args.model_type == "BiasedAD":
            dataset = NB15Dataset(args)
            default_eta_0 = 20
            default_eta_1 = 1
            default_eta_2 = 1
        elif args.model_type == "BiasedADM":
            args.s_non_target = 0
            dataset = nb15_for_BADM(args)
            default_eta_0 = 10
            default_eta_1 = 1
            default_eta_2 = 1
        net_name = 'mlp_for_nb15'
        file_name = f"s_normal={args.s_normal}," + "s_non_target=" + str(args.s_non_target) + ",s_target=" + str(args.s_target) + ",nb15_non_target_class_num=" +str(args.nb15_non_target_class_num) + "," + f'nb15_target_class={"_".join(args.nb15_target_class)},'
        args.ae_lr = 0.0001 if args.ae_lr is None else args.ae_lr
        args.ae_batch_size = 128 if args.ae_batch_size is None else args.ae_batch_size
        args.ae_epoch = 30 if args.ae_epoch is None else args.ae_epoch
        BAD_lr = 0.00001
        BAD_batch_size = 128
        BAD_epoch = 50

    elif args.dataset_name == "fashionmnist":
        if args.normal_class != args.non_target_outlier_class:
            unlabeled_normal_number = int(5000 * args.s_normal)
            labeled_normal_number = int(0 * args.s_normal)
            test_normal = 1000

            unlabeled_non_target_outlier_number = int(50 * args.s_normal)
            labeled_non_target_outlier_number = int(100 * args.s_normal)
            test_non_target_outlier = 100

            unlabeled_target_outlier_number = int(50 * args.s_normal)
            labeled_target_outlier_number = int(100 * args.s_normal)
            test_target_outlier = 100
        else:
            unlabeled_normal_number = int(5000 * args.s_normal)
            labeled_normal_number = int(0 * args.s_normal)
            test_normal = 1000

            unlabeled_non_target_outlier_number = int(0 * args.s_normal)
            labeled_non_target_outlier_number = int(0 * args.s_normal)
            test_non_target_outlier = 0

            unlabeled_target_outlier_number = int(100 * args.s_normal)
            labeled_target_outlier_number = args.labeled_target_outlier_number
            test_target_outlier = 100
        if args.model_type == "BiasedADM":
            labeled_non_target_outlier_number = 0

        if args.model_type == "BiasedAD":
            dataset = FashionMNIST_Dataset("./data",
                                        args.normal_class,
                                        unlabeled_normal_number,
                                        labeled_normal_number,
                                        test_normal,

                                        args.non_target_outlier_class,
                                        unlabeled_non_target_outlier_number,
                                        labeled_non_target_outlier_number,
                                        test_non_target_outlier,

                                        args.target_outlier_class,
                                        unlabeled_target_outlier_number,
                                        labeled_target_outlier_number,
                                        test_target_outlier,
                                        
                                        args.random_seed)
            default_eta_0 = 1
            default_eta_1 = 1
            default_eta_2 = 1
        elif args.model_type == "BiasedADM":
            contaminationRate = args.contamination_for_FMNIST

            if contaminationRate != 0.02:
                unlabeled_target_outlier_number = (unlabeled_normal_number + labeled_normal_number + unlabeled_non_target_outlier_number + labeled_non_target_outlier_number + labeled_target_outlier_number) * args.contamination_for_FMNIST / (1 - args.contamination_for_FMNIST)
                unlabeled_target_outlier_number = int(round(unlabeled_target_outlier_number))

            dataset = fashionmnist_for_BADM("./data",
                                            args.normal_class,
                                            unlabeled_normal_number,
                                            labeled_normal_number,
                                            test_normal,

                                            args.non_target_outlier_class,
                                            unlabeled_non_target_outlier_number,
                                            labeled_non_target_outlier_number,
                                            test_non_target_outlier,
                                            
                                            args.target_outlier_class,
                                            unlabeled_target_outlier_number,
                                            labeled_target_outlier_number,
                                            test_target_outlier,

                                            args.random_seed)
            default_eta_0 = 1
            default_eta_1 = 1
            default_eta_2 = 1

        net_name = "fmnist_LeNet"
        if args.normal_class == args.non_target_outlier_class:
            file_save_path = args.dir_path + "/" + str(args.normal_class) + str(args.non_target_outlier_class) + "x"
        else:
            file_save_path = args.dir_path + "/" + str(args.normal_class) + "x" + str(args.target_outlier_class)

        # file_name = "normal=" + str(args.normal_class) + ",non_target=" + str(args.non_target_outlier_class) + ",target=" + str(args.target_outlier_class) + ","
        file_name = f"normal={args.normal_class},non_target={args.non_target_outlier_class},target={args.target_outlier_class},s_normal={args.s_normal},labeled_target_outlier_number={args.labeled_target_outlier_number},"
        args.ae_lr = 0.0001 if args.ae_lr is None else args.ae_lr
        args.ae_batch_size = 128 if args.ae_batch_size is None else args.ae_batch_size
        args.ae_epoch = 10 if args.ae_epoch is None else args.ae_epoch
        BAD_lr = 0.0001
        BAD_batch_size = 128
        BAD_epoch = 30

    elif args.dataset_name == "SQB":
        if args.model_type == "BiasedAD":
            dataset = SQBDataset(args.sqb_test_frac)
            default_eta_0 = 1
            default_eta_1 = 1
            default_eta_2 = 1
        elif args.model_type == "BiasedADM":
            dataset = sqb_for_BADM(args.sqb_test_frac)
            default_eta_0 = 10
            default_eta_1 = 1
            default_eta_2 = 1

        file_name = f'dataset=sqb,sqb_test_frac={args.sqb_test_frac},'
        net_name = "mlp_for_sqb"
        args.lr, args.epoch, args.batch_size
        args.ae_lr = 0.0001 if args.ae_lr is None else args.ae_lr
        args.ae_batch_size = 128 if args.ae_batch_size is None else args.ae_batch_size
        args.ae_epoch = 30 if args.ae_epoch is None else args.ae_epoch
        BAD_lr = 0.0001
        BAD_batch_size = 128
        BAD_epoch = 50
        
    if not (args.lr is None):
        BAD_lr = args.lr
    if not (args.epoch is None):
        BAD_epoch = args.epoch
    if not (args.batch_size is None):
        BAD_batch_size = args.batch_size
    if args.eta_0 is None:
        args.eta_0 = default_eta_0
    if args.eta_1 is None:
        args.eta_1 = default_eta_1
    if args.eta_2 is None:
        args.eta_2 = default_eta_2

    time.sleep(round(random.uniform(0.001, 0.01), 3))
    if not os.path.exists(file_save_path):
        os.makedirs(file_save_path)
    
    if not os.path.exists("./log/{}_log".format(args.dir_path.split("/")[-1])):
        os.makedirs("./log/{}_log".format(args.dir_path.split("/")[-1]))

    writer = writer2txt()
    # output_name = file_name + 'contaminationRate={},eta0={},eta1={},eta2={},BAD_lr={},BAD_batchsize={},BAD_epoch={},sample_count={},model_type={},update_anchor={}'.format(str(contaminationRate), str(args.eta_0), str(args.eta_1), str(args.eta_2), str(BAD_lr), str(BAD_batch_size), str(BAD_epoch), str(args.sample_count), args.model_type, args.update_anchor)
    # output_name = file_name + f'contaminationRate={contaminationRate},eta0={args.eta_0},model_type={args.model_type},update_anchor={args.update_anchor},update_epoch={args.update_epoch},sample_count={args.sample_count}'
    output_name = file_name + f'contaminationRate={contaminationRate},eta0={args.eta_0},model_type={args.model_type},sample_count={args.sample_count}'
    
    from sklearn.ensemble import RandomForestClassifier #集成学习中的随机森林
    from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score,confusion_matrix, classification_report, f1_score
    import seaborn as sns

    x_train = dataset.train_set.data
    y_train = dataset.train_set.labels
    target_y_train = dataset.train_set.semi_targets

    x_test = dataset.test_set.data
    y_test = dataset.test_set.labels
    target_y_test = dataset.test_set.semi_targets

    
    # if args.RF == "RF3":
    #     # 三类
    #     random.seed(args.random_seed)
    #     non_target_samples = x_train[np.where(target_y_train==-2)[0]]
    #     target_samples = x_train[np.where(target_y_train==-1)[0]]
    #     normal_samples = x_train[random.sample(np.where(target_y_train==0)[0].tolist(), len(non_target_samples))]
    #     rf_x_train = np.concatenate([normal_samples, non_target_samples, target_samples], axis=0)
    #     rf_y_train = np.concatenate([np.zeros(normal_samples.shape[0]),
    #                                 np.zeros(non_target_samples.shape[0]) - 2,
    #                                 np.zeros(target_samples.shape[0]) - 1,], axis=0)#建立模型
    #     for temp in range(10):
    #         rfc = RandomForestClassifier()
    #         rfc = rfc.fit(rf_x_train, rf_y_train)
    #         y_pred = rfc.predict(x_test)
    #         print(Counter(y_pred))
    #         print(Counter(target_y_test))
    #         y_prob = rfc.predict_proba(x_test)

    #         cm = confusion_matrix(target_y_test, y_pred)
    #         print(cm)
    #         print(classification_report(target_y_test, y_pred))

    #         precision, recall, threshold = precision_recall_curve(y_test, y_prob[:,1])
    #         rf_test_AUPRC = auc(recall, precision)
    #         rf_test_AUROC = roc_auc_score(y_test, y_prob[:,1])
    #         f1 = np.nanmax(2 * recall * precision / (recall + precision))
    #         print(rf_test_AUPRC)
    #         print(rf_test_AUROC)
    #         output_name = file_name + f'contaminationRate={contaminationRate}'
    #         with open("{}/{}.txt".format(file_save_path, output_name), 'a+') as f:
    #             f.write('Test AUC: {:.2f}% | Test PRC: {:.2f}% | Test F1: {:.2f}'.format(100. * rf_test_AUROC, 100. * rf_test_AUPRC, f1 * 100))
    #             f.write('\n')
    #         print()
    #     continue
        
    # if args.RF == "RF2":
    #     # 两类
    #     random.seed(args.random_seed)
    #     target_samples = x_train[np.where(target_y_train==-1)[0]]
    #     normal_samples = x_train[random.sample(np.where(target_y_train==0)[0].tolist(), len(target_samples))]
    #     rf_x_train = np.concatenate([normal_samples, target_samples], axis=0)
    #     rf_y_train = np.concatenate([np.zeros(normal_samples.shape[0]),
    #                                     np.zeros(target_samples.shape[0]) + 1,], axis=0)#建立模型
    #     for temp in range(10):
    #         rfc = RandomForestClassifier()
    #         rfc = rfc.fit(rf_x_train, rf_y_train)
    #         y_pred = rfc.predict(x_test)
    #         print(Counter(y_pred))
    #         print(Counter(y_test))
    #         y_prob = rfc.predict_proba(x_test)

    #         cm = confusion_matrix(y_test, y_pred)
    #         print(cm)
    #         print(classification_report(y_test, y_pred))

    #         precision, recall, threshold = precision_recall_curve(y_test, y_prob[:,1])
    #         rf_test_AUPRC = auc(recall, precision)
    #         rf_test_AUROC = roc_auc_score(y_test, y_prob[:,1])
    #         f1 = np.nanmax(2 * recall * precision / (recall + precision))
    #         print(rf_test_AUPRC)
    #         print(rf_test_AUROC)
    #         output_name = file_name + f'contaminationRate={contaminationRate}'
    #         with open("{}/{}.txt".format(file_save_path, output_name), 'a+') as f:
    #             f.write('Test AUC: {:.2f}% | Test PRC: {:.2f}% | Test F1: {:.2f}'.format(100. * rf_test_AUROC, 100. * rf_test_AUPRC, f1 * 100))
    #             f.write('\n')
    #         print()
    #     continue

    if args.intermediate_flag:
        x_train = np.array(dataset.train_set.data)
        y_train = np.array(dataset.train_set.labels)
        target_y_train = np.array(dataset.train_set.semi_targets)
        x_test = np.array(dataset.test_set.data)
        y_test = np.array(dataset.test_set.labels)
        target_y_test = np.array(dataset.test_set.semi_targets)
        
        np.savez(f"./processed_data/{args.model_type}/{output_name}",
                 x_train=x_train,
                 y_train=y_train,
                 target_y_train=target_y_train,
                 x_test=x_test,
                 y_test=y_test,
                 target_y_test=target_y_test)
    
    
    
    model = BiasedAD(args.eta_0, args.eta_1, args.eta_2, args.model_type, args.update_anchor, args.debug, args.update_epoch)
    model.set_network(net_name)
    model.pretrain(dataset, optimizer_name='adam',
                    lr=args.ae_lr,
                    n_epochs=args.ae_epoch,
                    batch_size=args.ae_batch_size,
                    weight_decay = args.ae_weight_decay,
                    device=args.device,
                    n_jobs_dataloader=0)
    if args.intermediate_flag:
        # save the characterization obtained after AE pretrain
        train_loader, test_loader = dataset.loaders(batch_size=128, drop_last_train = False)
        train_data_input, train_data_label, train_data_semi_target = model.intermediate_result(train_loader)
        test_data_input, test_data_label, test_data_semi_target = model.intermediate_result(test_loader)
        npz_name = f'./intermediate_results/{args.model_type}/{output_name}.npz'
        np.savez(npz_name , x_train = train_data_input, y_train = train_data_label, target_y_train = train_data_semi_target, x_test = test_data_input, y_test = test_data_label, target_y_test = test_data_semi_target)
        # np.savez(file_save_path + "/" + file_name + "train_data.npz", train_data_input = train_data_input, train_data_label = train_data_label, train_data_semi_target = train_data_semi_target)
        # np.savez(file_save_path + "/" + file_name + "test_data.npz", test_data_input = test_data_input, test_data_label = test_data_label, test_data_semi_target = test_data_semi_target)
    else:
        writer.set_output_name(output_name)
        writer.set_file_save_path(file_save_path)
        writer.set_path("{}/{}.txt".format(file_save_path, output_name), "./log/{}_log/{}".format(args.dir_path.split("/")[-1], output_name))
        model.train(dataset,
                    optimizer_name='adam',
                    lr=BAD_lr,
                    n_epochs=BAD_epoch,
                    batch_size=BAD_batch_size,
                    weight_decay=args.weight_decay,
                    device=args.device,
                    n_jobs_dataloader=0,
                    sample_count=args.sample_count)
        model.test(dataset, device=args.device, n_jobs_dataloader=0)
    # with open(f'result/running_time_20231118/{args.model_type}_{args.dataset_name}.txt' , "a", encoding="utf-8") as f:
    #     f.write(f'ae_train_time={model.ae_results["train_time"]},SAD_train_time={model.results["train_time"]},total_train_time={model.results["train_time"] + model.ae_results["train_time"]}')
    #     f.write("\n")