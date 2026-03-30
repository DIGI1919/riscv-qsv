import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import torch.ao.quantization as ao
import torch.nn.utils.prune as prune

import models.quant_layer as quant_layer

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, planes):
        super(LambdaLayer, self).__init__()
        self.planes = planes

    def forward(self, x):
        # 实现原始lambda函数的功能
        return F.pad(x[:, :, ::2, ::2],
                     (0, 0, 0, 0, self.planes // 4, self.planes // 4),
                     "constant", 0.0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, qconfig_dict, stride=1, option='A', sensitivity_conv = None):
        super(BasicBlock, self).__init__()
        #  s 是灵敏度
        if sensitivity_conv is None:
            sensitivity_conv = [1, 1]
        self.conv1 = quant_layer.QuantConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                             bits= None, t = 5, mask_level=[1,2,4], block_size=4)#多出的参数
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = quant_layer.QuantConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                             bits=None, t = 5, mask_level=[1,2,4], block_size=4)  # 多出的参数
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

        self.sconv = quant_layer.QuantConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False,
                                             bits=None, t = 5, mask_level=[1,2,4], block_size=4)  # 多出的参数
        self.sbn   = nn.BatchNorm2d(self.expansion * planes)

        self.conv1.qconfig = qconfig_dict['conv1']
        self.conv2.qconfig = qconfig_dict['conv2']
        self.sconv.qconfig = qconfig_dict['sconv']
        self.sensitivity_conv = sensitivity_conv

        if qconfig_dict:
            for name, module in self.named_modules():
                if name in qconfig_dict:
                    # print('BB'+name)
                    module.qconfig = qconfig_dict[name]

    def forward(self, x):
        # 统一量化输入
        # 主路径
        out = self.bn1(self.conv1(x))
        out = self.relu1(out)
        out = self.bn2(self.conv2(out))

        # shortcut路径（在量化域处理）
        shortcut_out = self.sconv(x)
        shortcut_out = self.sbn(shortcut_out)
        # shortcut_out = self.shortcut(x_quant)  # 使用量化后的输入
        # 在浮点域相加
        out += shortcut_out
        out = self.relu2(out)
        return out

    def change_t(self,t):
        self.conv1.change_t(t)
        self.conv2.change_t(t)
        self.sconv.change_t(t)
        return 0

    def complexity(self):
        out = 0.0
        out += self.conv1.complexity() * self.sensitivity_conv[0]
        out += self.conv2.complexity() * self.sensitivity_conv[1]
        out += self.sconv.complexity()
        return out

    def show_params(self):
        print("    conv1:")
        self.conv1.show_params()
        print("    conv2:")
        self.conv2.show_params()
        self.sconv.show_params()
        return 0
    def last_params(self):
        self.conv1.last_params()
        self.conv2.last_params()
        self.sconv.last_params()
        return 0
    #type = 1 alpha 开启权重 w冻结 权重  type=0 alpha冻结 w开启权重
    def is_need_grad(self,type):
        self.conv1.is_need_grad(type)
        self.conv2.is_need_grad(type)
        self.sconv.is_need_grad(type)
        return 0

class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, qconfig_dict, stride=1, option='A', sensitivity_block = None):
        super(BasicBlock2, self).__init__()

        if sensitivity_block is None:
            sensitivity_block = [[1,1], [1,1], [1,1]]
        self.sensitivity_block = sensitivity_block


        self.b0 = BasicBlock(in_planes, planes, qconfig_dict['b0'], stride, option=option, sensitivity_conv=self.sensitivity_block[0])
        self.b1 = BasicBlock(planes, planes, qconfig_dict['b1'], stride=1, option=option, sensitivity_conv=self.sensitivity_block[1])

    def forward(self, x):
        out = self.b0(x)
        out = self.b1(out)
        return out

    def change_t(self,t):
        self.b0.change_t(t)
        self.b1.change_t(t)
        return 0

    def complexity(self):
        out = 0.0
        out += self.b0.complexity()
        out += self.b1.complexity()
        return out

    def show_params(self):
        print("         b0:")
        self.b0.show_params()
        print("         b1:")
        self.b1.show_params()
        return 0
    def last_params(self):
        self.b0.last_params()
        self.b1.last_params()
        return 0
    #type = 1 alpha 开启权重 w冻结 权重  type=0 alpha冻结 w开启权重
    def is_need_grad(self,type):
        self.b0.is_need_grad(type)
        self.b1.is_need_grad(type)
        return 0

class ResNet18(nn.Module):
    def __init__(self, qconfig_dict, num_classes=1000, s_layer = None):
        super(ResNet18, self).__init__()

        if s_layer is None:
            s_layer = [[[1,1],[1,1]],
                       [[1,1],[1,1]],
                       [[1,1],[1,1]],
                       [[1,1],[1,1]],
                       [[1,1],[1,1]],
                       [[1,1],[1,1]]]
        self.s_layer = s_layer
        self.num_classes = num_classes
        self.in_planes = 16

        self.conv1 = quant_layer.QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False,
                                             bits= [8], t = 5, mask_level=[1], block_size=1)#多出的参数
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock2(64, 64, qconfig_dict['layer1'], stride=1,option='B', sensitivity_block= self.s_layer[1])

        self.layer2 = BasicBlock2(64, 128, qconfig_dict['layer2'], stride=2,option='B', sensitivity_block= self.s_layer[2])
        self.layer3 = BasicBlock2(128, 256, qconfig_dict['layer3'], stride=2,option='B', sensitivity_block= self.s_layer[3])
        self.layer4 = BasicBlock2(256, 512, qconfig_dict['layer4'], stride=2, option='B', sensitivity_block= self.s_layer[4])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = quant_layer.QuantLastFc(512, num_classes, bits = [8], t = 5, mask_level=[1,2,4], block_size=4)
        self.softmax = nn.LogSoftmax(dim=1)
        self.apply(_weights_init)

        self.relu2 = nn.ReLU()

        self.conv1.qconfig  = qconfig_dict['conv1']
        self.linear.qconfig = qconfig_dict['linear']

        self.pruning_layers = []
        if qconfig_dict:
            self.conv1.qconfig  = qconfig_dict['conv1']
            self.linear.qconfig = qconfig_dict['linear']
            # for name, module in self.named_modules():
            #     if name in qconfig_dict:
            #         # print('glb'+name)
            #         module.qconfig = qconfig_dict[name]

    def forward(self, x,targets=None,beta=0):

        out = self.bn1(self.conv1(x))
        out = self.relu1(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)


        out = self.linear(out)
        # out = self.relu2(out)


        out = self.softmax(out)
        if targets is not None:
            out = out.view(out.size(0), self.num_classes)
            loss = F.nll_loss(out, targets) 
            pred = out.data.max(1)[1]
            loss_n = loss.unsqueeze(0)

            loss_com = beta * self.complexity()
            loss_com = loss_com.unsqueeze(0)
            # print(loss_com.size)
            # print(loss_n.size)

            return loss_n,loss_com, pred
        else:
            return out 

    def change_t(self,t):
        self.conv1.change_t(t)
        self.layer1.change_t(t)
        self.layer2.change_t(t)
        self.layer3.change_t(t)
        self.layer4.change_t(t)
        self.linear.change_t(t)
        return 0

    def complexity(self):
        out = 0.0
        out += self.conv1.complexity() * self.s_layer[0][0][0]
        out += self.layer1.complexity()
        out += self.layer2.complexity()
        out += self.layer3.complexity()
        out += self.layer4.complexity()
        out += self.linear.complexity() * self.s_layer[4][0][0]
        return out

    def show_params(self):
        print("conv1:")
        self.conv1.show_params()
        print("             layer1:")
        self.layer1.show_params()
        print("             layer2:")
        self.layer2.show_params()
        print("             layer3:")
        self.layer3.show_params()
        print("             layer4:")
        self.layer4.show_params()
        print("fc:")
        self.linear.show_params()
        return 0

    def last_params(self):
        self.conv1.last_params()
        self.layer1.last_params()
        self.layer2.last_params()
        self.layer3.last_params()
        self.layer4.last_params()
        self.linear.last_params()
        return 0
    #type = 1 alpha 开启权重 w冻结 权重  type=0 alpha冻结 w开启权重
    def is_need_grad(self,type):
        self.conv1.is_need_grad(type)
        self.layer1.is_need_grad(type)
        self.layer2.is_need_grad(type)
        self.layer3.is_need_grad(type)
        self.layer4.is_need_grad(type)
        self.linear.is_need_grad(type)
        return 0



if __name__ == "__main__":
    # for net_name in __all__:
    #     if net_name.startswith('resnet'):
    #         print(net_name)
    #         test(globals()[net_name]())
    #         print()
    import Qconf as q

    a = ResNet18(q.qconfig_dict)
    print_model_structure(a)
    # check_quantization_status(a)
    # for name, module in a.named_modules():
    #     print(name)
    # a.maskinit()
    # print(a.pruning_layers)

