import torch.optim as optim
import torch
import torch.nn as nn
import json
import os
from dataset import ConsistentImageDataset
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from config import args
from models import Resnet18, Projector, Linear_Classifier
from small_models import SmallResnet
from criterions import NTXEntLoss
from helper import visualize, adjust_lr, adjust_linear_lr, LARC

from sklearn.linear_model import LogisticRegression


def Train_CNN():
    model1.train()
    model2.train()
    train_loss1 = 0.0
    train_loss2 = 0.0
    opt_cnn.zero_grad()
    for ix, (sample1, sample2) in enumerate(trainloader):
        data_i, data_j = sample1["image"], sample2["image"]
        data_i, data_j = data_i.to(device), data_j.to(device)
        f_i, h_i = model1(data_i)
        f_j, h_j = model2(data_i)
        zh_i = gh(h_i)
        zh_j = gh(h_j)
        zf_i = gf(f_i)
        zf_j = gf(f_j)
        l1 = criterion1(zh_i, zh_j)
        l2 = criterion1(zf_i, zf_j)
        loss = l1 - torch.nn.functional.relu(5-l2)  # NT-Xent Loss in the paper
        loss = loss / accum
        loss.backward()
        train_loss1 += l1.item()
        train_loss2 += l2.item()
        if (ix + 1) % accum == 0:
            opt_cnn.step()
            opt_cnn.zero_grad()
        if ((ix + 1) % (5 * accum)) == 0:
            print("L-train loss:{} {}".format(train_loss1 * accum / (ix + 1), train_loss2 * accum / (ix + 1)))

    record_cnn["train_loss"].append((train_loss1 * accum / (ix + 1), train_loss2 * accum / (ix + 1)))


def Eval_CLF():
    model1.eval()
    model2.eval()
    feat_ = []
    label_ = []
    with torch.no_grad():
        for ix, (sample1, sample2) in enumerate(Linear_trainloader):
            data, label = sample1["image"], sample1["label"]
            data, label = data.to(device), label.to(device)
            _, feature = model1(data)
            feature = feature.cpu()
            label = label.cpu()

            feat_.append(feature)
            label_.append(label)

    feat_ = torch.cat(feat_).numpy()
    label_ = torch.cat(label_).numpy()

    clf = LogisticRegression(random_state=0).fit(feat_, label_)

    feat_ = []
    label_ = []
    with torch.no_grad():
        for ix, (sample1, sample2) in enumerate(Linear_testloader):
            data, label = sample1["image"], sample1["label"]
            data, label = data.to(device), label.to(device)
            _, feature = model1(data)
            feature = feature.cpu()
            label = label.cpu()

            feat_.append(feature)
            label_.append(label)

    feat_ = torch.cat(feat_).numpy()
    label_ = torch.cat(label_).numpy()

    test_acc = clf.score(feat_, label_)

    print("Test Accuracy:", test_acc)
        
    record_clf["test_acc"].append(test_acc)
    return clf, test_acc


def record_saver(record, path):
    with open(path, 'w') as f:
        json.dump(record, f)


if __name__ == "__main__":
    # ========== [param] ==========
    for arg in vars(args):
        print(arg, '===>', getattr(args, arg))
    lr = args.lr
    clf_lr = args.clf_lr
    batch_size = args.batch
    epoch = args.epoch
    clf_epoch = args.clf_epoch
    classNum = args.classNum
    temp = args.temperature
    data_root = args.data_root
    num_worker = args.workers
    dir_ckpt = args.dir_ckpt
    dir_log = args.dir_log
    os.makedirs(dir_log, exist_ok=True)
    os.makedirs(dir_ckpt, exist_ok=True)
    accum = args.accumulate
    aug_s = args.strength
    useLARS = args.useLARS
    decay = args.weight_decay
    momentnum = args.momentnum
    warm = args.warmup
    project_in = args.pro_in
    project_hidden = args.pro_hidden
    project_out = args.pro_out
    linear_in = args.linear_in
    eval_routine = args.eval_routine
    subset_interval = args.subset_interval
    record_cnn = {"train_loss": []}
    record_clf = {"train_loss": [],
                  "train_acc": [],
                  "test_loss": [],
                  "test_acc": []}

    # ========== [data] ==========
    train_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=32),
        transforms.ToTensor()]
    )

    traindata = torchvision.datasets.CIFAR10(
        root=data_root, 
        train=True,
        download=True
    )

    trainset = ConsistentImageDataset(traindata, transform=train_aug, subset_interval=subset_interval)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_worker
    )

    Linear_train_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=32),
        transforms.ToTensor()]
    )

    Linear_traindata = torchvision.datasets.CIFAR10(
        root=data_root, 
        train=True,
        download=True
    )

    Linear_trainset = ConsistentImageDataset(Linear_traindata, transform=Linear_train_aug, subset_interval=subset_interval)

    Linear_trainloader = DataLoader(
        Linear_trainset,
        batch_size=256,
        shuffle=True,
        num_workers=num_worker
    )

    Linear_test_aug = transforms.Compose([
        transforms.ToTensor()]
    )

    Linear_testdata = torchvision.datasets.CIFAR10(
        root=data_root, 
        train=False,
        download=True, 
    )

    Linear_testset = ConsistentImageDataset(Linear_testdata, transform=Linear_test_aug, subset_interval=subset_interval)

    Linear_testloader = DataLoader(
        Linear_testset,
        batch_size=256,
        shuffle=False,
        num_workers=num_worker
    )

    # ========== [visualize] ==========
    if batch_size >= 64:
        visualize(trainloader, dir_log + '/' + 'visual.png')

    # ========== [device] =============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========== [cnn model] ==========
    model1 = SmallResnet(split=True)
    model2 = SmallResnet(split=True)
    model1.to(device)
    model2.to(device)
    gh = Projector(input_size=project_in, hidden_size=project_hidden, output_size=project_out)
    gf = Projector(input_size=project_in//2, hidden_size=project_hidden, output_size=project_out)
    gh.to(device)
    gf.to(device)

    # ========== [optim for cnn] ==========
    opt_cnn = optim.SGD(
        list(model1.parameters()) + list(model2.parameters()) + list(gh.parameters()) + list(gf.parameters()),
        lr=lr,
        momentum=momentnum,
        weight_decay=decay
    )

    criterion1 = NTXEntLoss(temp=temp)
    criterion2 = nn.CrossEntropyLoss()

    if useLARS:
        opt_cnn = LARC(opt_cnn)  # LARS on SGD optimizer

    best_acc = 0.0
    for i in range(1, epoch + 1):
        print("========== [Unsupervised Training] ==========")
        print("[epoch {}/{}]".format(i, epoch))
        print("[lr {}]".format(adjust_lr(opt=opt_cnn, epoch=i, lr_init=lr, T=epoch, warmup=warm)))
        Train_CNN()
        record_saver(record_cnn, dir_log + '/' + "cnn.txt")
        if (i % eval_routine) == 0:
            linear_clf, test_acc = Eval_CLF()

            print("save the last model: {} || best model: {}".format(test_acc, best_acc))
            torch.save({"cnn": model1.state_dict(), "clf": linear_clf, "epoch": i}, dir_ckpt + '/' + "last.pt")
            record_saver(record_clf, dir_log + '/' + "clf.txt")
            if test_acc > best_acc:
                best_acc = test_acc
                print("save the best model: {}".format(best_acc))
                torch.save({"cnn": model1.state_dict(), "clf": linear_clf, "epoch": i}, dir_ckpt + '/' + "best.pt")
