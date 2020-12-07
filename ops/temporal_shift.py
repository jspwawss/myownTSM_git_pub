# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_, ones_, zeros_
import time
from torchsummary  import summary

class TemporalShift(nn.Module):
	def __init__(self, net, n_segment=3, n_div=8, inplace=False, concat = False, prune_list = [], prune = False):
		super(TemporalShift, self).__init__()
		self.net = net
		self.n_segment = n_segment
		self.fold_div = n_div
		self.inplace = inplace
		self.Concat = concat
		self.prune_list = prune_list
		self.Prune = prune
		
		if inplace:
			print('=> Using in-place shift...')
		if prune:
			print('=> Pruning conv1 input channels...')
			new_input_channel = self.net.in_channels - len(prune_list) + 1
			self.net = self.fill_param_to_extra_channels(self.net, new_input_channel)
		if concat:
			print('=> Concatenate after shifting...')
			new_input_channel = self.net.in_channels + self.net.in_channels // n_div // 2 * 2
			print("new_input_channel=",new_input_channel)
			self.net = self.fill_param_to_extra_channels(self.net, new_input_channel)
		
		print('=> Using fold div: {}'.format(self.fold_div))

	def forward(self, x):
		
		#x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
		#print("in temporal_shift forward")
		#print("x size=",x.size())
		self.n_segment = x.size()[0]
		#self.fold_div = x.size()[0]
		#print(self.inplace)		#False
		#exit()
		x = self.shift(x, self.n_segment,fold_div= self.fold_div, inplace =self.inplace)
		if self.Prune:
			#print("prune")
			x = self.prune(x, prune_list = self.prune_list)
		
		if self.Concat:
			#print("concat")
			x = self.concat(x, self.n_segment, fold_div=self.fold_div)
		#print("after concate x size",x.size())
		#print("going to resnet50-")
		#print(self.net)
		return self.net(x)

	@staticmethod
	def shift(x, n_segment, fold_div=8, inplace=False, concat = False):
		
		nt, c, h, w = x.size()
		n_batch = nt // n_segment
		#n_batch = 1
		#print("n_batch=",n_batch)
		#print("x.size()=",x.size())
		#print("n_segment",n_segment)
		#print("fold_div",fold_div)
		x = x.view(n_batch, n_segment, c, h, w)
		#x = x.view(1,-1,c,h,w)
		

		fold = c // fold_div
		
		if inplace:
			out = InplaceShift.apply(x, fold)
			return out.view(nt, c, h, w)
		else:
			out = torch.zeros_like(x)
			#print("fold,",fold)
			#print(out[0,0,0,0,0])
			#print("out size",out.size())
			out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left(up)
			#print(out[0,0,0,0,0])
			#print(out.size())
			out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right(down)
			#print(out[0,0,0,0,0])
			#print(out.size())
			out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
			#print(out[0,0,0,0,0])
			#print(out.size())
			return out.view(nt, c, h, w)

	def prune(self, x, prune_list = []):
		if prune_list == []:
			return x
		else:
			remain_list = [int(i) for i in range(x.size(1)) if i not in prune_list[1:]]
			remain_list = torch.autograd.Variable(torch.LongTensor(remain_list).cuda())

			x_prune = x.clone().detach()
			x_prune = x_prune[:, prune_list, :, :].mean(dim=1, keepdim=True)
			x[:, prune_list[0]:prune_list[0]+1, :, :] = x_prune
			x_prune = x[:, remain_list, :, :]

			return x_prune

	def concat(self, x, n_segment, fold_div=8):
		#print("x,size(),",x.size())
		nt, c, h, w = x.size()
		n_batch = nt // n_segment
		x = x.view(n_batch, n_segment, c, h, w)
		#x = x.view(1,-1,c,h,w)

		fold = c // fold_div
		fold2 = fold // 2
		#print("in concat")
		#print("fold={},fold2={}".format(fold,fold2))
		out = torch.zeros(n_batch, n_segment, c+2*fold2, h, w, dtype=x.dtype, layout=x.layout, device=x.device)
		#print(out[0,0,0,0,0])
		out[:, :-2, c:c+fold2] = x[:, 2:, c - 2 * fold2:c - fold2]		# shift left(up)
		#print(out[0,0,0,0,0])
		out[:, 2:, c+fold2:c+2*fold2] = x[:, :-2, c - fold2:]			# shift right(down) 
		#print(out[0,0,0,0,0])
		out[:, :, :c] = x[:, :, :]  									# not shift
		#print(out[0,0,0,0,0])

		return out.view(nt, c+2*fold2, h, w)

	def fill_param_to_extra_channels(self, net, new_input_channel):
		#print("in tmp_shift.py ::fill_param_to_extra_channels")
		params = [x.clone() for x in net.parameters()]
		kernel_size = params[0].size()
		#print("kernel_size",kernel_size)
		new_kernel_size = kernel_size[:1] + (int(new_input_channel), ) + kernel_size[2:]
		#print("new_kernel_size",new_kernel_size)
		new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

		new_conv = nn.Conv2d(new_kernel_size[1], net.out_channels,
							 net.kernel_size, net.stride, net.padding,
							 bias=True if len(params) == 2 else False)
		
		new_conv.weight.data = new_kernels
		if len(params) == 2:
			new_conv.bias.data = params[1].data # add bias if neccessary

		return new_conv


class InplaceShift(torch.autograd.Function):
	# Special thanks to @raoyongming for the help to this function
	@staticmethod
	def forward(ctx, input, fold):
		# not support higher order gradient
		# input = input.detach_()
		ctx.fold_ = fold
		n, t, c, h, w = input.size()
		buffer = input.data.new(n, t, fold, h, w).zero_()
		buffer[:, :-1] = input.data[:, 1:, :fold]
		input.data[:, :, :fold] = buffer
		buffer.zero_()
		buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
		input.data[:, :, fold: 2 * fold] = buffer
		return input

	@staticmethod
	def backward(ctx, grad_output):
		# grad_output = grad_output.detach_()
		fold = ctx.fold_
		n, t, c, h, w = grad_output.size()
		buffer = grad_output.data.new(n, t, fold, h, w).zero_()
		buffer[:, 1:] = grad_output.data[:, :-1, :fold]
		grad_output.data[:, :, :fold] = buffer
		buffer.zero_()
		buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
		grad_output.data[:, :, fold: 2 * fold] = buffer
		return grad_output, None


class TemporalPool(nn.Module):
	def __init__(self, net, n_segment):
		super(TemporalPool, self).__init__()
		self.net = net
		self.n_segment = n_segment

	def forward(self, x):
		x = self.temporal_pool(x, n_segment=self.n_segment)
		return self.net(x)

	@staticmethod
	def temporal_pool(x, n_segment):
		nt, c, h, w = x.size()
		n_batch = nt // n_segment
		x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
		x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
		x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
		return x

def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False, concat = "", prune_list = {}, prune = False):
	#summary(net,(3,256,256))
	#exit()

	if temporal_pool:
		n_segment_list = [n_segment, n_segment, n_segment // 2, n_segment // 2]
	else:
		n_segment_list = [n_segment] * 4
	assert n_segment_list[-1] > 0
	print('=> n_segment per stage: {}'.format(n_segment_list))

	if concat == "All":
		concat_list = [True, True, True, True]
	elif concat == "First":
		concat_list = [True, False, False, False]
	else:
		concat_list = [False, False, False, False]


	if place == 'block':
		def make_block_temporal(stage, this_segment):
			blocks = list(stage.children())

			print('=> Processing stage with {} blocks'.format(len(blocks)))
			for i, b in enumerate(blocks):
				blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div, concat = concat)
			return nn.Sequential(*(blocks))

		net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
		net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
		net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
		net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

	elif 'blockres' in place:
		n_round = 1
		if len(list(net.layer3.children())) >= 23:
			n_round = 2
			print('=> Using n_round {} to insert temporal shift'.format(n_round))

		def make_block_temporal(stage, this_segment, n_div = 8, concat = False, layer_count = 0):
			blocks = list(stage.children())
			print('=> Processing stage with {} blocks residual'.format(len(blocks)))
			for i, b in enumerate(blocks):
				if not prune:
					if i % n_round == 0:
						blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div, concat = concat)
				else:
					if layer_count != 0:
						blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div, concat = concat, prune_list = prune_list[layer_count-1], prune = prune)
					else:
						blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div, concat = concat)
					layer_count += 1
				
			return nn.Sequential(*blocks)
		#print("n segment list",n_segment_list)
		#print(concat_list)
		#exit()
		net.layer1 = make_block_temporal(net.layer1, n_segment_list[0], n_div, concat_list[0], layer_count = 0)
		#net.layer1 = make_block_temporal(net.layer1, 8, 1, concat_list[0], layer_count = 0)
		
		net.layer2 = make_block_temporal(net.layer2, n_segment_list[1], n_div, concat_list[1], layer_count = 3)
		net.layer3 = make_block_temporal(net.layer3, n_segment_list[2], n_div, concat_list[2], layer_count = 7)
		net.layer4 = make_block_temporal(net.layer4, n_segment_list[3], n_div, concat_list[3], layer_count = 13)

def make_efficientnet_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False, concat = ""):
	if concat == "All":
		concat_list = '1' * 16		#shifting starts from second MBConv block, so totally only shifts 15 blocks
	elif concat == "First":
		concat_list = '1' + '0' * 15
	else:
		concat_list = '0' * 16

	shift_list = ['0','0','1','0','1','0','1','1','0','1','1','0','1','1','1','0']
	if place == 'blockres':
		def make_block_temporal(stage, this_segment):
			blocks = list(stage.children())
			print('=> Processing stage with {} blocks residual'.format(len(blocks)))
			MBConv_count = 0
			for i, b in enumerate(blocks):
				if hasattr(b, '_expand_conv') and shift_list[MBConv_count] == '1':
					blocks[i]._expand_conv = TemporalShift(b._expand_conv, n_segment=this_segment, n_div=n_div, concat = True if concat_list[MBConv_count] == '1' else False)
				MBConv_count += 1
			return nn.Sequential(*(blocks))

		net._blocks = make_block_temporal(net._blocks, 8)
	else:
		raise NotImplementedError(place)

def make_temporal_pool(net, n_segment):
	
	print('=> Injecting nonlocal pooling')
	net.layer3 = TemporalPool(net.layer3, n_segment)


if __name__ == '__main__':
	# test inplace shift v.s. vanilla shift
	tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False)
	tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

	print('=> Testing CPU...')
	# test forward
	with torch.no_grad():
		for i in range(10):
			x = torch.rand(2 * 8, 3, 224, 224)
			y1 = tsm1(x)
			y2 = tsm2(x)
			assert torch.norm(y1 - y2).item() < 1e-5

	# test backward
	with torch.enable_grad():
		for i in range(10):
			x1 = torch.rand(2 * 8, 3, 224, 224)
			x1.requires_grad_()
			x2 = x1.clone()
			y1 = tsm1(x1)
			y2 = tsm2(x2)
			grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
			grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
			assert torch.norm(grad1 - grad2).item() < 1e-5

	print('=> Testing GPU...')
	tsm1.cuda()
	tsm2.cuda()
	# test forward
	with torch.no_grad():
		for i in range(10):
			x = torch.rand(2 * 8, 3, 224, 224).cuda()
			y1 = tsm1(x)
			y2 = tsm2(x)
			assert torch.norm(y1 - y2).item() < 1e-5

	# test backward
	with torch.enable_grad():
		for i in range(10):
			x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
			x1.requires_grad_()
			x2 = x1.clone()
			y1 = tsm1(x1)
			y2 = tsm2(x2)
			grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
			grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
			assert torch.norm(grad1 - grad2).item() < 1e-5
	print('Test passed.')


