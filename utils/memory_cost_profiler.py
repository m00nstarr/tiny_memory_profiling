import copy
import torch
import torch.nn as nn

__all__ = ['count_model_size', 'count_activation_size', 'profile_memory_cost']
class Hswish(nn.Module):

	def __init__(self, inplace=True):
		super(Hswish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		return x * F.relu6(x + 3., inplace=self.inplace) / 6.

	def __repr__(self):
		return 'Hswish()'


class Hsigmoid(nn.Module):

	def __init__(self, inplace=True):
		super(Hsigmoid, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		return F.relu6(x + 3., inplace=self.inplace) / 6.

	def __repr__(self):
		return 'Hsigmoid()'


class MyConv2d(nn.Conv2d):
	"""
	Conv2d with Weight Standardization
	https://github.com/joe-siyuan-qiao/WeightStandardization
	"""

	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
	             padding=0, dilation=1, groups=1, bias=True):
		super(MyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
		self.WS_EPS = None

	def weight_standardization(self, weight):
		if self.WS_EPS is not None:
			weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
			weight = weight - weight_mean
			std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + self.WS_EPS
			weight = weight / std.expand_as(weight)
		return weight

	def forward(self, x):
		if self.WS_EPS is None:
			return super(MyConv2d, self).forward(x)
		else:
			return F.conv2d(x, self.weight_standardization(self.weight), self.bias,
			                self.stride, self.padding, self.dilation, self.groups)

	def __repr__(self):
		return super(MyConv2d, self).__repr__()[:-1] + ', ws_eps=%s)' % self.WS_EPS


def count_model_size(net, trainable_param_bits=32, frozen_param_bits=8, print_log=True):
	frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits

	trainable_param_size = 0
	frozen_param_size = 0
	for p in net.parameters():
		if p.requires_grad:
			trainable_param_size += trainable_param_bits / 8 * p.numel()
		else:
			frozen_param_size += frozen_param_bits / 8 * p.numel()
	model_size = trainable_param_size + frozen_param_size
	if print_log:
		print('Total: %d' % model_size,
		      '\tTrainable: %d (data bits %d)' % (trainable_param_size, trainable_param_bits),
		      '\tFrozen: %d (data bits %d)' % (frozen_param_size, frozen_param_bits))
	# Byte
	return model_size


def count_activation_size(net, input_size=(1, 3, 224, 224), require_backward=True, activation_bits=32):
	act_byte = activation_bits / 8
	model = copy.deepcopy(net)

	# noinspection PyArgumentList
	def count_convNd(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte // m.groups])  # bytes

	# noinspection PyArgumentList
	def count_linear(m, x, y):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_bn(m, x, _):
		# count activation size required by backward
		if m.weight is not None and m.weight.requires_grad:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_relu(m, x, _):
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() / 8])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	# noinspection PyArgumentList
	def count_smooth_act(m, x, _):
		# count activation size required by backward
		if require_backward:
			m.grad_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes
		else:
			m.grad_activations = torch.Tensor([0])
		# temporary memory footprint required by inference
		m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])  # bytes

	def add_hooks(m_):
		if len(list(m_.children())) > 0:
			return

		m_.register_buffer('grad_activations', torch.zeros(1))
		m_.register_buffer('tmp_activations', torch.zeros(1))

		if type(m_) in [nn.Conv1d, nn.Conv2d, nn.Conv3d, MyConv2d]:
			fn = count_convNd
		elif type(m_) in [nn.Linear]:
			fn = count_linear
		elif type(m_) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm]:
			fn = count_bn
		elif type(m_) in [nn.ReLU, nn.ReLU6, nn.LeakyReLU]:
			fn = count_relu
		elif type(m_) in [nn.Sigmoid, nn.Tanh, Hswish, Hsigmoid]:
			fn = count_smooth_act
		else:
			fn = None

		if fn is not None:
			_handler = m_.register_forward_hook(fn)

	model.eval()
	model.apply(add_hooks)

	x = torch.zeros(input_size).to(model.parameters().__next__().device)
	with torch.no_grad():
		model(x)

	memory_info_dict = {
		'peak_activation_size': torch.zeros(1),
		'grad_activation_size': torch.zeros(1),
		'residual_size': torch.zeros(1),
	}

	for m in model.modules():
		if len(list(m.children())) == 0:
			def new_forward(_module):
				def lambda_forward(_x):
					current_act_size = _module.tmp_activations + memory_info_dict['grad_activation_size'] + \
					                   memory_info_dict['residual_size']
					memory_info_dict['peak_activation_size'] = max(
						current_act_size, memory_info_dict['peak_activation_size']
					)
					memory_info_dict['grad_activation_size'] += _module.grad_activations
					return _module.old_forward(_x)

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

		if (isinstance(m, ResidualBlock) and m.shortcut is not None) or \
				(isinstance(m, InvertedResidual) and m.use_res_connect) or \
				type(m) in [BasicBlock, Bottleneck]:
			def new_forward(_module):
				def lambda_forward(_x):
					memory_info_dict['residual_size'] = _x.numel() * act_byte
					result = _module.old_forward(_x)
					memory_info_dict['residual_size'] = 0
					return result

				return lambda_forward

			m.old_forward = m.forward
			m.forward = new_forward(m)

	with torch.no_grad():
		model(x)

	return memory_info_dict['peak_activation_size'].item(), memory_info_dict['grad_activation_size'].item()


def profile_memory_cost(net, input_size=(1, 3, 224, 224), require_backward=True,
                        activation_bits=32, trainable_param_bits=32, frozen_param_bits=8, batch_size=8):
	param_size = count_model_size(net, trainable_param_bits, frozen_param_bits, print_log=True)
	activation_size, _ = count_activation_size(net, input_size, require_backward, activation_bits)

	memory_cost = activation_size * batch_size + param_size
	return memory_cost, {'param_size': param_size, 'act_size': activation_size}

