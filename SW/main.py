import torch
import torch.optim as optim
import argparse
import torchvision.datasets as ds
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler as lrs
import torch.nn as nn
import models.ResNet20 as ResNet20
import models.resnet20_8bit as resnet20
import models.resnet20_mixed as resne20tmix
import models.resnet18_mixed as resnet18mix
import models.mobilenet_mix as mobilenet_mix
import models.resnet18_Tiny as rs18Tiny
import models.ResNet18Im as rsImaNet
import math
import torch.nn.functional as F
import func as f
import data as d
import models.Qconf as q
import json
from torch import amp
from configs.exp_configs import EXPERIMENTS
import configs.module_configs as module_configs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser('Options for training SqueezeNet in pytorch')
parser.add_argument('--batch-size', '-bz', type=int, default=40, metavar='N', help='batch size of train')
# 单次输入模型的样本数量 与显存容量强相关，值越大训练越快但占用显存越多
parser.add_argument('--epochs', '-es', type=int, default=100, metavar='N', help='number of epochs to train for')
parser.add_argument('--epochf', '-ef', type=int, default=100, metavar='N', help='number of epochs to train for')
parser.add_argument('--begin_epoch', '-be', type=int, default=0, metavar='N', help='number of epochs to train for')
# 训练总轮数
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
# 动量系数（加速梯度下降，缓解局部最优）
parser.add_argument('--no-cuda', '-cpu', action='store_true', default=False, help='use cuda for training')
parser.add_argument('--log-schedule', '-log', type=int, default=200, metavar='N',
                    help='number of epochs to save snapshot after')
# 每隔多少batch输出一次训练日志
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
# 固定随机种子保证实验可复现性
parser.add_argument('--model_name', '-n', type=str, default=None, help='Use a pretrained model')
# 预训练模型路径（用于继续训练或测试）权重路径
parser.add_argument('--want_to_test', '-t', action='store_true', default=False,
                    help='make true if you just want to test.py')
# 仅运行测试模式
parser.add_argument('--Cosine_MaxRate', '-lr', type=float, default=0.1, help="the init rate of Cosine")
# parser.add_argument('--Cosine_T0', type=int, default=55, help="the restart cyc")
parser.add_argument('--outdir', '-d', type=str, default='saveDir', help='Use a pretrained model')

parser.add_argument('--train_beta', '-beta', type=float, default=None, help="train loss about complexity")
parser.add_argument('--tao', '-tao', type=float, default=-0.0045, help="tui huo wen du ")
parser.add_argument('--netname', '-netname', type=str, default='res20mix', help="tui huo wen du ")
parser.add_argument('--sensitivity', '-sen', type=json.loads, default=None, help="sensitivity value")
parser.add_argument('--func', '-f', type=float, default=0, help="train func")
parser.add_argument('--dataset', '-data', type=str, default='cifar10', help="used dataset")
parser.add_argument('--exp_config',type=str,required=True,help="experiment name")

args = parser.parse_args()


exp_name = args.exp_config
assert exp_name in EXPERIMENTS, f"Unknown experiment: {exp_name}"
cfg = EXPERIMENTS[exp_name]
args.epochs       = cfg["epochs"]
args.epochf       = cfg["epochf"]
args.begin_epoch  = cfg["begin_epoch"]
args.train_beta         = cfg["beta"]
args.tao          = cfg["tao"]
args.netname      = cfg["netname"]
args.dataset         = cfg["data"]
args.sensitivity  = cfg["sensitivity"]
args.model_name   = cfg["pretrained"]
args.batch_size   = cfg["batch_size"]
args.Cosine_MaxRate   = cfg["Cosine_MaxRate"]
quant_used = {
    "FIRST_CONV": module_configs.FIRST_CONV,
    "CONV3x3": module_configs.CONV3x3,
    "CONV1x1": module_configs.CONV1x1,
    "FC": module_configs.FC,
}

save_obj = {
    "experiment": exp_name,
    "config": cfg,
    "module_configs": quant_used
}

config_path = os.path.join(args.outdir, f"{exp_name}_config.json")

with open(config_path, "w") as config_file:
    json.dump(save_obj, config_file, indent=4)




args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# -------- dataset switch --------
if args.dataset == 'cifar10':
    train_loader, test_loader = d.getdataloader(args)
    print("use cifar10")

elif args.dataset == 'cifar100':
    train_loader, test_loader = d.getdataloader_cifar100(args)
    print("use cifar100")

elif args.dataset == 'tiny':
    train_loader, test_loader = d.get_tiny_imagenet_dataloaders(
        data_path='/home/user/disk_nvme/tiny-imagenet',
        batch_size=args.batch_size)
    print("use tiny imagenet")

elif args.dataset == 'ImageNet':
    train_loader, test_loader = d.get_ImageNet_getdataloader(args)
    print("use ImageNet")

else:
    raise ValueError("dataset error")

if args.netname == 'res20mix':
    if args.dataset == 'cifar10':
        net = resne20tmix.ResNet20(q.qconfig_dict_res20, s_layer=args.sensitivity)
        print("ues res20 mix cifar10")
    if args.dataset == 'cifar100':
        net = resne20tmix.ResNet20_cifar100(q.qconfig_dict_res20, num_classes=100)
        print("ues res20 mix cifar100")
else:
    if args.netname == 'res208bit':
        net = resnet20.ResNet20(q.qconfig_dict_res20)
        print("ues res20 8bit")
    else:
        if args.netname == 'res18mix':
            if args.dataset == 'cifar10':
                net = resnet18mix.ResNet18(q.qconfig_dict_res18, num_classes=10, s_layer=args.sensitivity)
                print("ues res18 mix")

            if args.dataset == 'cifar100':
                net = resnet18mix.ResNet18(q.qconfig_dict_res18, num_classes=100, s_layer=args.sensitivity)
                print("ues res18 mix")

            if args.dataset == 'tiny':
                net = resnet18mix.ResNet18(q.qconfig_dict_res18, num_classes=200, s_layer=args.sensitivity)
                print("ues res18 mix")

            if args.dataset == 'ImageNet':
                net = rsImaNet.ResNet18(q.qconfig_dict_res18, num_classes=1000, s_layer=args.sensitivity)
                print("ues res18 mix")

        else:
            if args.netname == 'mobv2mix':
                if args.dataset == 'cifar10':
                    net = mobilenet_mix.MobileNetV2(q.qconfigmb_dict_mob, output_size=10, s_layer=args.sensitivity)
                    print("ues mobv2 mix")

                if args.dataset == 'cifar100':
                    net = mobilenet_mix.MobileNetV2(q.qconfigmb_dict_mob, output_size=100, s_layer=args.sensitivity)
                    print("ues mobv2 mix")

                if args.dataset == 'tiny':
                    net = mobilenet_mix.MobileNetV2(q.qconfigmb_dict_mob, output_size=200, s_layer=args.sensitivity)
                    print("ues mobv2 mix")

            else:
                print("point net failure")
                raise ValueError

device = torch.device('cuda')
net = net.to(device)
net = nn.DataParallel(net)
scaler = amp.GradScaler()


def main():
    if args.model_name is not None:
        print("loading pre trained weights")
        pretrained_weights = torch.load(args.model_name)
        net.module.load_state_dict(pretrained_weights, strict=False)

        f.val(net, test_loader, args, 0)

    if not args.want_to_test:

        fig2, ax2 = plt.subplots()
        train_acc, val_acc = list(), list()

        optimizerSW = optim.SGD(
            net.parameters(),
            momentum=0.9,
            weight_decay=1e-4,
            lr=args.Cosine_MaxRate)
        schedulerSW = lrs.CosineAnnealingWarmRestarts(
            optimizerSW,
            T_0=args.epochs,
            T_mult=2)

        t = 5
        if args.func == 0:
            for i in range(1, args.epochs + 1):
                if i <= args.begin_epoch:
                    schedulerSW.step()
                    continue

                t = 5 * (math.exp(args.tao) ** i)
                net.module.change_t(t)

                train_acc.append(
                    f.trainCosine(i, net, optimizerSW, schedulerSW, train_loader, args, scaler))

                val_acc.append(
                    f.val(net, test_loader, args, i))
                ax2.plot(train_acc, 'g', label='Train Accuracy')
                ax2.plot(val_acc, 'b', label='Val Accuracy')

                fig2.savefig(f'{args.outdir}/train_val_accuracy.jpg')
        net.module.show_params()
        net.module.last_params()
        # weight train
        net.module.is_need_grad(0)

        optimizerF = optim.SGD(
            net.parameters(),
            momentum=0.9,
            weight_decay=1e-4,
            lr=args.Cosine_MaxRate)
        schedulerF = lrs.CosineAnnealingWarmRestarts(
            optimizerF,
            T_0=args.epochf,
            T_mult=2)

        for i in range(args.epochs + 1, args.epochs + args.epochf + 1):
            if i <= args.begin_epoch:
                schedulerF.step()
                continue


            train_acc.append(
                f.trainCosine((i - args.epochs), net, optimizerF, schedulerF, train_loader, args, scaler))

            val_acc.append(
                f.val(net, test_loader, args, i))


    # 当需要测试时（跳过训练直接评估）
    else:  # TODO 图
        # fig2, ax2 = plt.subplots()
        t = 0
        for i in range(1, args.epoch + 1):
            t = t + 0.06
            net.change_t(t)
            test_acc = f.test(net, test_loader, args)

            print("Testing accuracy on CIFAR-10 data is {:.2f}% {:.6f}".format(test_acc, t))



if __name__ == '__main__':
    main()


