import torch
import torch.optim as optim
import argparse
import torchvision.datasets as ds
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler as lrs
import json

def opendict(name):
    with open(name, 'r') as f:
        config = json.load(f)
    return config

def savedict(name,dict):
    with open(name, 'w') as f:
        json.dump(dict, f, indent=4)

def save_torchscript_model(model, model_filename):
    torch.jit.save(torch.jit.script(model), model_filename)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model


fig1, ax1 = plt.subplots()
avg_loss = list()
best_accuracy = 0.0  # 跟踪测试集最佳准确率（用于模型保存判断）

#train
def trainCosine(epoch, model, optimizer, scheduler, train_loader, args, scaler):
    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    global avg_loss
    correct = 0
    model.train()
    beta = 0
    if args.dataset == 'cifar10':
        classify = 10
    elif args.dataset == 'cifar100':
        classify = 100
    elif args.dataset == 'tiny':
        classify = 200
    if args.train_beta != None:
        beta = args.train_beta#TODO 模型大小重要程度
    print(beta)
    for b_idx, (data, targets) in enumerate(train_loader):


        if args.cuda:
            data, targets = data.cuda(), targets.cuda()

        optimizer.zero_grad()


        with torch.amp.autocast('cuda'):
            loss_n,loss_com, pred = model.forward(data,targets=targets,beta=beta)

        loss_n = loss_n.mean()
        loss_com = loss_com.mean()
        loss = loss_n + loss_com
        correct += pred.eq(targets.data).cpu().sum()

        avg_loss.append(loss.item())


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 

        if b_idx % args.log_schedule == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_n: {:.6f}\tLoss_com: {:.6f}'.format(
                epoch,
                (b_idx + 1) * len(data),  # 已处理样本数
                len(train_loader.dataset),  # 总样本数
                100. * (b_idx + 1) * len(data) / len(train_loader.dataset),  # 进度百分比
                loss_n.item(),
                loss_com.item()
            ))


            ax1.plot(avg_loss)

            fig1.savefig(f"{args.outdir}/Squeezenet_loss.jpg")

    scheduler.step()  # 更新参数
    train_accuracy = correct / float(len(train_loader.dataset))
    print("training accuracy ({:.2f}%)".format(100 * train_accuracy))
    return (train_accuracy * 100.0)


def val(model, test_loader, args, epoch):
    global best_accuracy

    correct = 0
    model.eval()
    i = 0
    for idx, (data, target) in enumerate(test_loader):
        if (len(test_loader.dataset) / args.batch_size * 0.75) <= idx:
            break

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        score = model.forward(data)

        # 计算预测结果
        pred = score.data.max(1)[1].detach().squeeze()
        correct += pred.eq(target.data.detach()).cpu().sum()

        i = i + 1

    val_accuracy = correct / (i * args.batch_size) * 100
    print("predicted {} out of {} and accuracy = {:.2f}%".format(correct, i * args.batch_size, val_accuracy))
    if val_accuracy > best_accuracy:
        print("save the best pth")
        best_accuracy = val_accuracy
        torch.save(model.module.state_dict(), f'{args.outdir}/{epoch}the_best({best_accuracy:.2f}).pth')
        best = 1
    else:
        torch.save(model.module.state_dict(), f'{args.outdir}/{epoch}this_round({val_accuracy:.2f}).pth')
        best = 0
    return val_accuracy, best


def test(model, test_loader, args):
    model.eval()

    test_correct = 0
    total_examples = 0
    for idx, (data, target) in enumerate(test_loader):
        if (len(test_loader.dataset) / args.batch_size * 0.75) >= idx:
            continue

        total_examples += len(target)

        if args.cuda:
            data, target = data.cuda(), target.cuda()

        scores = model(data)

        pred = scores.data.max(1)[1].detach().squeeze()
        test_correct += pred.eq(target.data.detach()).cpu().sum()

    print("Predicted {} out of {} correctly {}".format(test_correct, total_examples, test_correct/total_examples))

    return 100.0 * test_correct / float(total_examples)



def print_model_structure(module, prefix="", depth=0, max_depth=None, name=""):

    if max_depth is not None and depth > max_depth:
        return

    indent = "│   " * (depth - 1) + "├── " if depth > 0 else ""

    module_info = f"{prefix}{indent}{name} ({module.__class__.__name__})"

    if hasattr(module, "out_channels"):
        module_info += f" [out: {module.out_channels}]"
    elif hasattr(module, "num_features"):
        module_info += f" [features: {module.num_features}]"
    elif isinstance(module, nn.Linear):
        module_info += f" [in: {module.in_features}, out: {module.out_features}]"

    print(module_info)

    if hasattr(module, "named_children"):
        children = list(module.named_children())
        for i, (child_name, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            new_prefix = prefix + ("    " if depth == 0 else "│   " * (depth) if not is_last else "    " * (depth))
            print_model_structure(
                child,
                prefix=new_prefix,
                depth=depth + 1,
                max_depth=max_depth,
                name=child_name  # 传递子模块名称
            )

def print_layer_params(module, name="model", indent=0):
    params = sum(p.numel() for p in module.parameters(recurse=False))
    line = f"{' ' * indent}{name}: {params}"
    for child_name, child_module in module.named_children():
        print_layer_params(child_module, child_name, indent + 2)

if __name__ == '__main__':
    import model_S.ResNet20 as rs
    import model_S.Qconf as q
    net = rs.ResNet20(q.qconfig_dict)

    print_layer_params(net)