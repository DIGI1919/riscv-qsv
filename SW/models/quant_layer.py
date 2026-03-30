
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.core.evalf import scaled_zero
from torch.nn.modules.module import T

import models.prune_layer as pr
from torch.nn.parameter import Parameter


class acti_asym_min_max_quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bit):
        xmax = x.max()
        xmin = x.min()
        return acti_min_max_quantize_common(x, xmin, xmax, bit)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

def acti_min_max_quantize_common(x, xmin, xmax, bit):

    ch_range= xmax - xmin
    if ch_range == 0:
        ch_range = 1
    n_steps = 2 ** bit - 1
    scale = ch_range / n_steps
    zero_point = 0 - torch.round(xmin / scale)
    zero_point = zero_point.clamp(0, xmax)

    y = x.div(scale).round().mul(scale)

    return y



class weight_asym_min_max_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        xmax= x.abs().max()
        xmin= x.min()
        return weight_min_max_quantize_common(x, xmin, xmax, bit)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

# DJP (TODO: are clones necessary?)
def weight_min_max_quantize_common(x, xmin, xmax, bit):

    ch_range= xmax
    # ch_range.masked_fill_(ch_range.eq(0), 1)    #如果range 有0 替换为1  ch_range.eq(0)创建一个mask 等于0的位置 true
    if ch_range == 0:
        ch_range = 1
    n_steps = 2 ** (bit - 1) - 1
    scale = xmax / n_steps
    q_x = torch.round(x / scale)
    q_x = q_x.clamp(-n_steps, n_steps)
    y = q_x * scale
    return y

#dilation 空洞卷积 决定卷积核元素之间的间隔 当 dilation>1 卷积核元素之间插入 空洞 扩大感受野
#groups 分组卷积 决定输入通道与输出通道的连接方式 groups=1 标准卷积 每个输出通道连接所有输入通道 groups = in_channels Depthwise卷积  grpups=N 分组卷积
class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 bits=None, t=1, mask_level = None, block_size = None):#相比于 官方 多了bits 和 t  mask_level block_size
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if bits is None:
            bits = [4, 8]
        if mask_level is None:
            mask_level = [1]
        if block_size is None:
            block_size = 1

        self.layer_type = "QConv2d"
        self.mask_level = mask_level
        self.block_size = block_size
        self.bits = bits

        self.kernel_size = kernel_size

        self.alpha_weight = nn.Parameter(
            torch.tensor([0.01] * len(bits), dtype=torch.float32),  # 明确指定dtype
            requires_grad=True
        )

        #TODO 配套输入 剪枝空间长度 根据 备选剪枝方案 生成 相同数量的 alpha
        self.alpha_mask_level = nn.Parameter(
            torch.tensor([0.01] * len(mask_level), dtype=torch.float32),
            requires_grad=True
        )

        self.t = t
        self.sw_weight = None               #weight alpha ratio
        self.sw_mask_level = None           #mask alpha ratio

    #输入激活值
    def forward(self, input):
        mix_prune_mask = []         #剪枝掩码
        mix_quant_weight = []       #存储量化后权重
        mix_quant_activate = []     #存储量化后激活值

        sw_mask_level = F.softmax(self.alpha_mask_level / self.t, dim=0)
        sw_weight = F.softmax(self.alpha_weight / self.t, dim=0)

        self.sw_mask_level = sw_mask_level
        self.sw_weight = sw_weight      #save sw

        #剪枝
        for i, ratio in enumerate(self.mask_level):
            prune_mask = pr.gen_mask(self.weight, ratio, self.block_size)
            scaled_prune_mask = prune_mask * sw_mask_level[i]
            mix_prune_mask.append(scaled_prune_mask)

        mix_mask = torch.stack(mix_prune_mask).sum(0)
        mix_prune_weight = self.weight * mix_mask

        #量化
        for i, bit in enumerate(self.bits):
            #权重量化
            quant_weight = weight_asym_min_max_quantize.apply(mix_prune_weight, bit)
            scaled_quant_weight = quant_weight * sw_weight[i]
            mix_quant_weight.append(scaled_quant_weight)

            #激活值量化
            quant_activate = acti_asym_min_max_quantize.apply(input, bit)
            scaled_quant_activate = quant_activate * sw_weight[i]
            mix_quant_activate.append(scaled_quant_activate)

        #求和放入conv
        mix_weight = torch.stack(mix_quant_weight).sum(0)
        mix_activate = torch.stack(mix_quant_activate).sum(0)
        out = F.conv2d(mix_activate, mix_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def change_t(self, t):
        self.t = t
        return 0
    def complexity(self):
        out1 = 0
        out2 = 0
        for i , sw_w in enumerate(self.sw_weight):
            out1 += (self.in_channels *
                    self.out_channels *
                    self.kernel_size *
                    sw_w *
                    self.bits[i])
        for i,sw_m in enumerate(self.sw_mask_level):
            out2 += out1 * sw_m * self.mask_level[i]
        out2 = out2 / self.block_size
        return out2
    
    def complexity_cycle(self):
        out1 = 0
        out2 = 0
        for i , sw_w in enumerate(self.sw_weight):
            out1 += (sw_w * self.bits[i])
        for i,sw_m in enumerate(self.sw_mask_level):
            out2 += out1 * sw_m * self.mask_level[i]
        out2 = out2 / self.block_size
        return out2
    def show_params(self):
        print("conv weight alpha:", self.alpha_weight)
        # print("      conv weightp:",self.sw_weight)
        print("conv mask alpha:", self.alpha_mask_level)
        # print("      conv maskp:", self.sw_mask_level)
        # print("      conv tao:", self.t)
        # print("      conv weight:", self.weight)
        return 0
    # def confirm_params(self):

    def last_params(self):
        # 找出最大值的索引
        with torch.no_grad():
            # 找到最大值的位置
            idx_w = torch.argmax(self.alpha_weight).item()
            idx_m = torch.argmax(self.alpha_mask_level).item()

            # 创建 one-hot
            self.alpha_weight.zero_()
            self.alpha_mask_level.zero_()

            self.alpha_weight[idx_w] = 1
            self.alpha_mask_level[idx_m] = 1

        # 如果不希望再训练
        self.alpha_weight.requires_grad = False
        self.alpha_mask_level.requires_grad = False
        return 0
    #type = 1 alpha 开启权重 w冻结 权重  type=0 alpha冻结 w开启权重
    def is_need_grad(self,type):
        if type == 1:
            self.alpha_weight.requires_grad = True
            self.alpha_mask_level.requires_grad = True
            self.weight.requires_grad = False
            return 0
        else:
            self.alpha_weight.requires_grad = False
            self.alpha_mask_level.requires_grad = False
            self.weight.requires_grad = True
            return 0
        
    def save_alpha(self):
        out = []
        out.append(self.alpha_weight)
        out.append(self.alpha_mask_level)
        return out




class QuantLastFc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 bits=None, t=1, mask_level = None, block_size = None):#相比于 官方 多了bits 和 t mask_level block_size
        if mask_level is None:
            mask_level = [1]
        if block_size is None:
            block_size = 1
        if bits == None:
            bits = [4,8]
        super(QuantLastFc,self).__init__(in_features, out_features, bias)
        self.layer_type = "LFc"

        self.mask_level = mask_level
        self.block_size = block_size
        self.bits = bits

        self.alpha_weight = nn.Parameter(
            torch.tensor([0.01] * len(self.bits), dtype=torch.float32),
            requires_grad=True
        )
        #TODO 配套输入 剪枝空间长度 根据 备选剪枝方案 生成 相同数量的 alpha
        self.alpha_mask_level = nn.Parameter(
            torch.tensor([0.01] * len(mask_level), dtype=torch.float32),
            requires_grad=True
        )

        self.t = t
        self.sw_weight = None
        self.sw_mask_level = None           #mask alpha ratio

    def forward(self, input):
        mix_prune_mask = []         #剪枝掩码
        mix_quant_weight = []
        mix_quant_activate = []

        sw_mask_level = F.softmax(self.alpha_mask_level / self.t, dim=0)
        sw_weight = F.softmax(self.alpha_weight / self.t, dim=0)

        self.sw_mask_level = sw_mask_level
        self.sw_weight = sw_weight  # save sw

        #剪枝
        for i, ratio in enumerate(self.mask_level):
            prune_mask = pr.gen_mask(self.weight, ratio, self.block_size)
            scaled_prune_mask = prune_mask * sw_mask_level[i]
            mix_prune_mask.append(scaled_prune_mask)

        mix_mask = torch.stack(mix_prune_mask).sum(0)
        mix_prune_weight = self.weight * mix_mask

        # 对位宽参数 alpha soft
        for i, bit in enumerate(self.bits):
            # 权重量化
            quant_weight = weight_asym_min_max_quantize.apply(mix_prune_weight, bit)
            scaled_quant_weight = quant_weight * sw_weight[i]
            mix_quant_weight.append(scaled_quant_weight)

            quant_activate = acti_asym_min_max_quantize.apply(input, bit)
            scaled_quant_activate = quant_activate * sw_weight[i]
            mix_quant_activate.append(scaled_quant_activate)

        # 求和放入linear
        mix_weight = torch.stack(mix_quant_weight).sum(0)
        mix_activate = torch.stack(mix_quant_activate).sum(0)
        return F.linear(mix_activate, mix_weight, self.bias)

    def change_t(self,t):
        self.t = t
        return 0

    def complexity(self):
        out1 = 0
        out2 = 0
        for i,sw_w in enumerate(self.sw_weight):
                out1 += (self.in_features * self.out_features * sw_w * self.bits[i])

        for i,sw_m in enumerate(self.sw_mask_level):
            out2 += out1 * sw_m * self.mask_level[i]
        out2 = out2 / self.block_size

        return out2
    def complexity_cycle(self):
        out1 = 0
        out2 = 0
        for i , sw_w in enumerate(self.sw_weight):
            out1 += (sw_w * self.bits[i])
        for i,sw_m in enumerate(self.sw_mask_level):
            out2 += out1 * sw_m * self.mask_level[i]
        out2 = out2 / self.block_size
        return out2
    
    def show_params(self):
        print("      conv weight alpha:", self.alpha_weight)
        # print("      conv weightp:",self.sw_weight)
        print("      conv mask alpha:", self.alpha_mask_level)
        # print("      conv maskp:", self.sw_mask_level)
        # print("      conv tao:", self.t)
        # print("      conv weight:", self.weight)
        return 0

    def last_params(self):
        # 找出最大值的索引
        with torch.no_grad():
            # 找到最大值的位置
            idx_w = torch.argmax(self.alpha_weight).item()
            idx_m = torch.argmax(self.alpha_mask_level).item()

            # 创建 one-hot
            self.alpha_weight.zero_()
            self.alpha_mask_level.zero_()

            self.alpha_weight[idx_w] = 1
            self.alpha_mask_level[idx_m] = 1

        # 如果不希望再训练
        self.alpha_weight.requires_grad = False
        self.alpha_mask_level.requires_grad = False
        return 0

    #type = 1 alpha 开启权重 w冻结 权重  type=0 alpha冻结 w开启权重
    def is_need_grad(self,type):
        if type == 1:
            self.alpha_weight.requires_grad = True
            self.alpha_mask_level.requires_grad = True
            self.weight.requires_grad = False
            return 0
        else:
            self.alpha_weight.requires_grad = False
            self.alpha_mask_level.requires_grad = False
            self.weight.requires_grad = True
            return 0
        
    def save_alpha(self):
        out = []
        out.append(self.alpha_weight)
        out.append(self.alpha_mask_level)
        return out