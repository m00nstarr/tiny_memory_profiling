# tiny_memory_profiling


##1. 계획
- [x]  tinyTL/count_activation_size, model_size import하는 방법 구하기
- [x]  layer 별로 activation size 분석하기.
- [x]  memory constraint 기준 설정
- [ ]  액티베이션 사이즈가 큰 순서대로 quantization 시도
- [ ]  pruning 시도


##2. MobileNetV2 Inverted Residual Block Memory Profiling

InvertedResidual(
  (conv): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
torch.Size([1, 32, 112, 112])
**blk_idx : 0 / activation_size =  5.41Mb

InvertedResidual(
  (conv): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): ConvBNReLU(
      (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
      (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
torch.Size([1, 16, 112, 112])
**blk_idx : 1 / activation_size =  12.72Mb 

InvertedResidual(
  (conv): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): ConvBNReLU(
      (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
      (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
torch.Size([1, 24, 56, 56])
**blk_idx : 2 / activation_size =  4.76Mb 

InvertedResidual(
  (conv): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): ConvBNReLU(
      (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
      (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
torch.Size([1, 32, 28, 28])
**blk_idx : 3 / activation_size =  1.60Mb 

InvertedResidual(
  (conv): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): ConvBNReLU(
      (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
      (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
torch.Size([1, 64, 14, 14])
**blk_idx : 4 / activation_size =  1.29Mb 

InvertedResidual(
  (conv): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): ConvBNReLU(
      (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
      (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
torch.Size([1, 96, 14, 14])
**blk_idx : 5 / activation_size =  1.20Mb 

InvertedResidual(
  (conv): Sequential(
    (0): ConvBNReLU(
      (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): ConvBNReLU(
      (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
      (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
)
torch.Size([1, 160, 7, 7])
**blk_idx : 6 / activation_size =  0.82Mb 
