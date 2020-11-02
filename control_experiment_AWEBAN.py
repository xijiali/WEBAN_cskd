from __future__ import print_function

import argparse
import csv
import os, logging

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import models
from utils import progress_bar, set_logging_defaults
from datasets_aug import load_dataset

#added
from tensorboardX import SummaryWriter
from updater import AWEBANUpdater
from models.hypernetwork import HyperNetwork_FC
#global variable
best_val = 0  # best validation accuracy

def main():
    parser = argparse.ArgumentParser(description='CS-KD Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model', default="CIFAR_ResNet18", type=str,
                        help='model type (32x32: CIFAR_ResNet18, CIFAR_DenseNet121, 224x224: resnet18, densenet121)')
    parser.add_argument('--name', default='2_teachers_AWEBAN', type=str, help='name of run')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')#30
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--ngpu', default=2, type=int, help='number of gpu')
    parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
    parser.add_argument('--dataset', default='cifar100', type=str,
                        help='the name for dataset cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67')
    parser.add_argument('--dataroot', default='/gruntdata4/xiaoxi.xjl/classification_datasets/', type=str,
                        help='data directory')
    parser.add_argument('--saveroot', default='./control_experiment', type=str, help='save directory')
    parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
    parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')
    # added
    parser.add_argument("--n_gen", type=int, default=5)
    parser.add_argument("--resume_gen", type=int, default=2)
    parser.add_argument('--alpha', default=0.8, type=float, help='ce loss weight ratio')
    parser.add_argument('--evaluate', default=False, help='evaluate ensembling checkpoints')
    parser.add_argument('--testdir', default='./AWEBAN_results', type=str, help='save directory')
    parser.add_argument("--hypernetwork_lr", type=float, default=0.001)
    parser.add_argument('--cosine_annealing', default=True, help='cosine annealing')



    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    global best_val
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    cudnn.benchmark = True

    # Data
    print('==> Preparing dataset: {}'.format(args.dataset))

    trainloader, valloader = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)

    num_class = trainloader.dataset.num_classes
    print('Number of train dataset: ', len(trainloader.dataset))
    print('Number of validation dataset: ', len(valloader.dataset))

    # Model
    print('==> Building model: {}'.format(args.model))

    net = models.load_model(args.model, num_class)
    # print(net)
    #added
    hypernetwork = HyperNetwork_FC(args.resume_gen, num_class)

    if use_cuda:
        torch.cuda.set_device(args.sgpu)
        net.cuda()
        print(torch.cuda.device_count())
        print('Using CUDA..')
        #added
        hypernetwork.cuda()

    if args.ngpu > 1:
        device_ids=list(range(args.sgpu, args.sgpu + args.ngpu))
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        #added
        hypernetwork=torch.nn.DataParallel(hypernetwork, device_ids=device_ids)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    # cosine annealing
    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    #added
    hypernetwork_optimizer=optim.SGD(hypernetwork.parameters(), lr=args.hypernetwork_lr, momentum=0.9, weight_decay=args.decay)
    # cosine annealing
    if args.cosine_annealing:
        hypernet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hypernetwork_optimizer, T_max=args.epoch)

    logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
    set_logging_defaults(logdir, args)
    logger = logging.getLogger('main')
    logname = os.path.join(logdir, 'log.csv')

    # Evaluate
    if args.evaluate:
        testdir = os.path.join(args.testdir, args.dataset, args.model, args.name)
        # Load checkpoint.
        print('==> Evaluating ensembling checkpoints..')
        model_lst = []
        for i in range(args.n_gen):
            temp_model = models.load_model(args.model, num_class).cuda()
            temp_model.eval()
            temp_model= torch.nn.DataParallel(temp_model, device_ids=device_ids)
            model_name = 'model' + str(i) + '.pth.tar'
            checkpoint = torch.load(os.path.join(testdir, model_name))
            temp_model.load_state_dict(checkpoint)
            model_lst.append(temp_model)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch'] + 1
        # rng_state = checkpoint['rng_state']
        # torch.set_rng_state(rng_state)
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = torch.zeros(inputs.size(0), num_class).cuda()
                for j in range(args.n_gen):
                    outputs += model_lst[j](inputs)
                outputs = outputs / args.n_gen
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum().float()
        acc = 100. * correct / total
        print('acc is {}'.format(acc))
        return

    criterion = nn.CrossEntropyLoss()
    kld_criterion=KDLoss(args.temp)

    # Logs
    # added
    kwargs = {
        "model": net,
        "hypernetwork": hypernetwork,
        "optimizer": optimizer,
        "hypernetwork_optimizer": hypernetwork_optimizer,
        "n_gen": args.n_gen,
        "model_name": args.model,
        "alpha": args.alpha,
        "num_class": num_class,
    }
    updater = AWEBANUpdater(**kwargs)

    writer = SummaryWriter()
    best_loss_list = []
    last_model_weight_lst=[]

    for gen in range(0,args.resume_gen):
        pretrained_weight=torch.load(os.path.join(logdir, "model"+str(gen)+".pth.tar"))
        last_model_weight_lst.append(pretrained_weight)

    updater.gen = args.resume_gen
    updater.register_last_model(last_model_weight_lst, device_ids)

    for gen in range(args.resume_gen, args.n_gen):
        print('\nGEN: %d' % gen)
        for epoch in range(start_epoch, args.epoch):
            # train_loss, train_acc, train_cls_loss = train(epoch)
            # train_loss, train_acc, train_cls_loss = train(epoch, net, trainloader, use_cuda, criterion, optimizer)
            train_loss, train_kld_loss,hypernetwok_ce_loss = updater.update(epoch, trainloader, criterion, kld_criterion)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("kld_loss", train_kld_loss, epoch)
            writer.add_scalar("hypernetwok_ce_loss", hypernetwok_ce_loss, epoch)

            # val_loss, val_acc = val(epoch)
            val_loss, val_acc = val(epoch, updater.model, valloader, use_cuda, criterion, optimizer, logdir, gen)
            writer.add_scalar("val_loss", val_loss, epoch)

            if args.cosine_annealing:
                # cosine annealing
                scheduler.step()
                #hypernet_scheduler.step()
            else:
                adjust_learning_rate(optimizer, epoch, args.lr, args.epoch)
                #adjust_learning_rate(hypernetwork_optimizer, epoch, args.lr, args.epoch)

        last_model_weight=torch.load(os.path.join(logdir, "model"+str(gen)+".pth.tar"))
        last_model_weight_lst.append(last_model_weight)
        updater.register_last_model(last_model_weight_lst,device_ids)
        updater.gen += 1
        best_loss_list.append(best_val)
        best_val=0
        # initialize self (mode and optimizer)
        net = models.load_model(args.model, num_class).cuda()
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
        hypernetwork = HyperNetwork_FC(updater.gen, num_class).cuda()
        hypernetwork=torch.nn.DataParallel(hypernetwork, device_ids=device_ids)
        hypernetwork_optimizer = optim.SGD(hypernetwork.parameters(), lr=args.hypernetwork_lr, momentum=0.9, weight_decay=1e-4)
        # cosine annealing
        if args.cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
            hypernet_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hypernetwork_optimizer, T_max=args.epoch)
        updater.model = net
        updater.optimizer = optimizer
        updater.hypernetwork = hypernetwork
        updater.hypernetwork_optimizer = hypernetwork_optimizer
        # reload the dataloader
        #trainloader, valloader = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)

    logger = logging.getLogger('best')
    for gen in range(len(best_loss_list)):
        print("Gen: ", gen+args.resume_gen,
              ", best Accuracy: ", best_loss_list[gen])
        logger.info('[GEN {}] [Acc {:.3f}]'.format(gen,best_loss_list[gen]))
    # print("Best Accuracy : {}".format(best_val))
    # logger = logging.getLogger('best')
    # logger.info('[Acc {:.3f}]'.format(best_val))

    return


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

def train(epoch,net,trainloader,use_cuda,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_cls_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = torch.mean(criterion(outputs, targets))
        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum().float().cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | Cls: %.3f '
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, train_cls_loss/(batch_idx+1)))

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [cls {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        train_loss/len(trainloader),
        train_cls_loss/len(trainloader),
        100.*correct/total))

    return train_loss/len(trainloader), 100.*correct/total, train_cls_loss/len(trainloader)

def val(epoch,net,valloader,use_cuda,criterion,optimizer,logdir,gen):
    global best_val
    net.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    # Define a data loader for evaluating
    loader = valloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets))

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        val_loss/(batch_idx+1),
        acc))

    if acc > best_val:
        best_val = acc
        checkpoint(net,logdir,gen)

    return (val_loss/(batch_idx+1), acc)


def checkpoint(net,logdir,gen):
    # Save checkpoint.
    print('Saving..')
    # state = {
    #     'net': net.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'acc': acc,
    #     'epoch': epoch,
    #     'rng_state': torch.get_rng_state()
    # }
    torch.save(net.state_dict(), os.path.join(logdir, "model"+str(gen)+".pth.tar"))


def adjust_learning_rate(optimizer, epoch,initial_lr,max_epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = initial_lr
    if epoch >= 0.5 * max_epoch:
        lr /= 10
    if epoch >= 0.75 * max_epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()