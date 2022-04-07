import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.profiler import profile, record_function, ProfilerActivity

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x+3, inplace = True) /6

        return out

class hsigmoid(nn.Module):
    def forward(self,x ):
        out = F.relu6(x+3, inplace = True) / 6
       
        return out

class SeModule(nn.Module):

    def __init__(self, in_size, reduction = 4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
   
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        print(x.shape)
        out = self.nolinear1(self.bn1(self.conv1(x)))
        print(out.shape)
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

Blocks = [
    Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
    Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
    Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
    Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
    Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
    Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
    Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
    Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
    Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
    Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
    Block(5, 96, 576, 96, hswish(), SeModule(96), 1)
]

Blocks_shape = [
    (2, 16, 112, 112),
    (2, 16, 56, 56),
    (2, 24, 28, 28),
    (2, 24, 28, 28),
    (2, 40, 14, 14),
    (2, 40, 14, 14),
    (2, 40, 14, 14),
    (2, 48, 14, 14),
    (2, 48, 14, 14),
    (2, 96, 7, 7),
    (2, 96, 7, 7)
]


class MobileNetV3_block(nn.Module):
    def __init__(self, blk_idx=0, num_classes=1000):
        super(MobileNetV3_block, self).__init__()
        self.bneck = Blocks[blk_idx]
        # self.bneck = Block(kernel_size = 3, in_size = 16, expand_size = 72, out_size = 24, nolinear = nn.ReLU(inplace=True), semodule = None, stride = 2)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.bneck(x)
        return out

class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

# # block 0
# net = MobileNetV3_block(0)
# x = torch.randn(Blocks_shape[0])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(0))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 1
# net = MobileNetV3_block(1)
# x = torch.randn(Blocks_shape[1])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(1))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 2
# net = MobileNetV3_block(2)
# x = torch.randn(Blocks_shape[2])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(2))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 3
# net = MobileNetV3_block(3)
# x = torch.randn(Blocks_shape[3])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(3))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 4
# net = MobileNetV3_block(4)
# x = torch.randn(Blocks_shape[4])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(4))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 5
# net = MobileNetV3_block(5)
# x = torch.randn(Blocks_shape[5])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(5))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 6
# net = MobileNetV3_block(6)
# x = torch.randn(Blocks_shape[6])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(6))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 7
# net = MobileNetV3_block(7)
# x = torch.randn(Blocks_shape[7])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(7))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 8
# net = MobileNetV3_block(8)
# x = torch.randn(Blocks_shape[8])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(8))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# block 9
net = MobileNetV3_block(9)
x = torch.randn(Blocks_shape[9])

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(9))) as prof:
    with record_function("model_inference"):
        net(x)        

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))

# # block 10
# net = MobileNetV3_block(10)
# x = torch.randn(Blocks_shape[10])

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,  on_trace_ready=torch.profiler.tensorboard_trace_handler('./mobilenet/log', 'block_'+str(10))) as prof:
#     with record_function("model_inference"):
#         net(x)        

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))
