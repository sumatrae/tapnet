from __future__ import division
from __future__ import print_function

import math
import sys
import time
import argparse

import torch.optim as optim
from models import TapNet
from utils import *
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()

# dataset settings
parser.add_argument('--data_path', type=str, default="./data/",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="NATOPS_IMBALANCE",  # NATOPS
                    help='time series dataset. Options: See the datasets list')

# cuda settings
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

# Training parameter settings
parser.add_argument('--epochs', type=int, default=3000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='Initial learning rate. default:[0.00001]')
parser.add_argument('--wd', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters). default: 5e-3')
parser.add_argument('--stop_thres', type=float, default=1e-9,
                    help='The stop threshold for the training error. If the difference between training losses '
                         'between epoches are less than the threshold, the training will be stopped. Default:1e-9')

# Model parameters


parser.add_argument('--use_cnn', type=boolean_string, default=True,
                    help='whether to use CNN for feature extraction. Default:False')
parser.add_argument('--use_lstm', type=boolean_string, default=True,
                    help='whether to use LSTM for feature extraction. Default:False')
parser.add_argument('--use_rp', type=boolean_string, default=True,
                    help='Whether to use random projection')
parser.add_argument('--rp_params', type=str, default='-1,3',
                    help='Parameters for random projection: number of random projection, '
                         'sub-dimension for each random projection')
parser.add_argument('--use_metric', action='store_true', default=False,
                    help='whether to use the metric learning for class representation. Default:False')
parser.add_argument('--metric_param', type=float, default=0.01,
                    help='Metric parameter for prototype distances between classes. Default:0.000001')
parser.add_argument('--filters', type=str, default="256,256,128",
                    help='filters used for convolutional network. Default:256,256,128')
parser.add_argument('--kernels', type=str, default="8,5,3",
                    help='kernels used for convolutional network. Default:8,5,3')
parser.add_argument('--dilation', type=int, default=1,
                    help='the dilation used for the first convolutional layer. '
                         'If set to -1, use the automatic number. Default:-1')
parser.add_argument('--layers', type=str, default="500,300",
                    help='layer settings of mapping function. [Default]: 500,300')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability). Default:0.5')
parser.add_argument('--lstm_dim', type=int, default=128,
                    help='Dimension of LSTM Embedding.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
args.sparse = True
args.layers = [int(l) for l in args.layers.split(",")]
args.kernels = [int(l) for l in args.kernels.split(",")]
args.filters = [int(l) for l in args.filters.split(",")]
args.rp_params = [float(l) for l in args.rp_params.split(",")]

if not args.use_lstm and not args.use_cnn:
    print("Must specify one encoding method: --use_lstm or --use_cnn")
    print("Program Exiting.")
    exit(-1)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

print("Loading dataset", args.dataset, "...")
# Model and optimizer
model_type = "TapNet"

if model_type == "TapNet":

    features, labels, idx_train, idx_val, idx_test, nclass = load_raw_ts(args.data_path, dataset=args.dataset)

    # update random permutation parameter
    if args.rp_params[0] < 0:
        dim = features.shape[1]
        args.rp_params = [3, math.floor(dim / (3 / 2))]
    else:
        dim = features.shape[1]
        args.rp_params[1] = math.floor(dim / args.rp_params[1])

    args.rp_params = [int(l) for l in args.rp_params]
    print("rp_params:", args.rp_params)

    # update dilation parameter
    if args.dilation == -1:
        args.dilation = math.floor(features.shape[2] / 64)

    print("Data shape:", features.size())
    model = TapNet(nfeat=features.shape[1],
                   len_ts=features.shape[2],
                   layers=args.layers,
                   nclass=nclass,
                   dropout=args.dropout,
                   use_lstm=args.use_lstm,
                   use_cnn=args.use_cnn,
                   filters=args.filters,
                   dilation=args.dilation,
                   kernels=args.kernels,
                   use_metric=args.use_metric,
                   use_rp=args.use_rp,
                   rp_params=args.rp_params,
                   lstm_dim=args.lstm_dim
                   )

    # cuda
    if args.cuda:
        # model = nn.DataParallel(model) Used when you have more than one GPU. Sometimes work but not stable
        model.cuda()
        features, labels, idx_train = features.cuda(), labels.cuda(), idx_train.cuda()
    input = (features, labels, idx_train, idx_val, idx_test)

# init the optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.wd)

weight = torch.tensor([0.2, 0.8]).cuda()


# training function
def train():
    loss_list = [sys.maxsize]
    test_best_possible, best_so_far = 0.0, sys.maxsize

    for epoch in range(args.epochs):

        t = time.time()
        model.train()
        optimizer.zero_grad()

        output, proto_dist = model(input)

        loss_train = F.cross_entropy(output[idx_train], torch.squeeze(labels[idx_train]), weight=weight)
        # loss_train = F.binary_cross_entropy(output[idx_train], torch.squeeze(labels[idx_train]))
        # weight=torch.tensor([30 / 33, 3 / 33]))
        if args.use_metric:
            loss_train = loss_train + args.metric_param * proto_dist

        if abs(loss_train.item() - loss_list[-1]) < args.stop_thres \
                or loss_train.item() > loss_list[-1]:
            break
        else:
            loss_list.append(loss_train.item())

        acc_train, recall_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]), weight=weight)
        # loss_val = F.binary_cross_entropy(output[idx_val], torch.squeeze(labels[idx_val]),)
        # weight=torch.tensor([30 / 33, 3 / 33]))
        acc_val, recall_val = accuracy(output[idx_val], labels[idx_val])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.8f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'recall_train: {:.4f}'.format(recall_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'recall_val: {:.4f}'.format(recall_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

        if acc_val.item() > test_best_possible:
            test_best_possible = acc_val.item()
        if best_so_far > loss_train.item():
            best_so_far = loss_train.item()
            test_acc = acc_val.item()
            test_recall = recall_val.item()
    print("test_acc: " + str(test_acc))
    print("test_recall: " + str(test_recall))
    print("best possible: " + str(test_best_possible))


# test function
def test():
    output, proto_dist = model(input)
    loss_test = F.cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]), weight=weight)
    # loss_test = F.binary_cross_entropy(output[idx_test], torch.squeeze(labels[idx_test]),)
    # weight=torch.tensor([30 / 33, 3 / 33]))
    if args.use_metric:
        loss_test = loss_test - args.metric_param * proto_dist

    acc_test, recall_test = accuracy(output[idx_test], labels[idx_test])
    print(args.dataset, "Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "recall= {:.4f}".format(recall_test.item()))

    plot_confusion(output[idx_test], labels[idx_test])


# Train model
t_total = time.time()
train()
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
