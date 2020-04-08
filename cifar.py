#!/bin/env python 
import os
import re
import sys
import argparse
import torch 
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

name2net={
    'vgg19'       : models.vgg19,
    'resnet18'    : models.resnet18,
    'resnet50'    : models.resnet50,
    'mobilenet'   : models.mobilenet,
    'mobilenetv2' : models.mobilenet_v2
}

def parse_args():
    # Training configurations 
    parser = argparse.ArgumentParser(description='Configuration for cifar training')
    parser.add_argument('--lr', default=0.01, type=float, help='Learing Rate')
    parser.add_argument('--batchsize', type=int, default=128, help='Batchsize for training')
    parser.add_argument('--epoch', type=int, default=200, help='The number of epochs')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum value for optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the optimizer')
    
    parser.add_argument('--cpu', default=False, action='store_true', help='Only use CPU to train')
    parser.add_argument('--gpuid', default='0', type=str, help='Gpus used for training')
    parser.add_argument('--arch', choices=list(name2net.keys()), default='vgg19', help='Network type used for training')
    parser.add_argument('--training', action='store_true', default=True, help='Train the model')
    parser.add_argument('--inference', action='store_true', default=False, help='Use the pretrained model to inference')
    return parser.parse_args()


def build_net(args):
    if args.training:
        net = name2net[args.arch](pretrained=False)
    else: 
        net = name2net[args.arch](pretrained=True)
    if args.cpu:
        # Use CPU to run the model
        args.gpu = False
        return net
    if not cuda.is_available() or cuda.device_count() == 0:
        # If the cuda is not available or GPU count equals to 0
        args.gpu =False
        return net
    args.gpu = True
    gpus = list(filter(lambda x:len(x)>0, re.split(',', args.gpuid)))
    if len(gpus) > 1:
        net = nn.DataParallel(net)
    return net.cuda()


def train_epoch(net, train_loader, optimizer, args):
    net.train()
    loss_sum = 0
    count = 0
    correct = 0
    for bid, (data, target) in enumerate(train_loader):
        if args.gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        count += data.size(0)
        _, predict = output.max(1)
        correct += predict.eq(target).sum().item()
    print('Loss: %.2f , Accuracy: %.2f' %(loss_sum/len(train_loader), correct/count) )


def validate(net, data_loader, args):
    net.eval()
    loss_sum = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for bid, (data, target) in enumerate(data_loader):
            if args.gpu:
                data, target = data.cuda(), target.cuda()
            output = net(data)
            loss = F.cross_entropy(output, target)
            _, predict = output.max(1)
            loss_sum += loss.item()
            count += target.size(0)
            correct += predict.eq(target).sum().item()
        print('Loss: %.2f  Accuracy: %2f' % (loss_sum/len(data_loader), correct/count))

def prepare_data(args):
    print('==> Preparing Data...')
    cifar_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', transform=cifar_transform_train, train=True, download=True)
    val_set = torchvision.datasets.CIFAR10(root='./data', transform=cifar_transform_val, train=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchsize, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batchsize, shuffle=False)
    return train_loader, val_loader



def main(args):
    train_loader, val_loader = prepare_data(args)
    net = build_net(args)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.training:
        for epochid in range(args.epoch):
            print("==> Training Epoch %d" % epochid)
            train_epoch(net, train_loader, optimizer, args)
            print('==> Validating ')
            validate(net, val_loader, args)
    else:
        print('==>Validating')
        validate(net, val_loader, args)

    

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    main(args)