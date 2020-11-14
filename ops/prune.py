import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_

class Prune(nn.Module):
	def __init__(self, net, layer = 'conv1', prune_ratio = 0.1):
		super(Prune, self).__init__()
		self.prune = net
		self.prune_ratio = prune_ratio
		
		if layer == 'conv1':
			new_output_channel = net.out_channels - int(net.out_channels * prune_ratio)
			self.prune = self.adjust_output_channels(self.prune, new_output_channel)
		elif layer == 'conv2':
			new_input_channel = net.in_channels - int(net.in_channels * prune_ratio)
			self.prune = self.adjust_input_channels(self.prune, new_input_channel)
		elif layer == 'bn1':
			new_num_features = net.num_features - int(net.num_features * prune_ratio)
			self.prune = nn.BatchNorm2d(new_num_features)
		

	def forward(self, x):

		return self.prune(x)

	def adjust_input_channels(self, net, new_input_channel):
		params = [x.clone() for x in net.parameters()]
		kernel_size = params[0].size()
		new_kernel_size = kernel_size[:1] + (int(new_input_channel), ) + kernel_size[2:]
		new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

		new_conv = nn.Conv2d(new_kernel_size[1], net.out_channels,
							 net.kernel_size, net.stride, net.padding,
							 bias=True if len(params) == 2 else False)
		
		new_conv.weight.data = new_kernels
		if len(params) == 2:
			new_conv.bias.data = params[1].data # add bias if neccessary

		return new_conv

	def adjust_output_channels(self, net, new_output_channel):
		params = [x.clone() for x in net.parameters()]
		kernel_size = params[0].size()
		#print("kernel size=",kernel_size)
		
		new_kernel_size = (int(new_output_channel), ) + kernel_size[1:]
		#print("new_kernel_size=",new_kernel_size)
		new_kernels = params[0].data.mean(dim=0, keepdim=True).expand(new_kernel_size).contiguous()
		#print("new_kernels=",new_kernels)

		new_conv = nn.Conv2d(net.in_channels, new_kernel_size[0],
							 net.kernel_size, net.stride, net.padding,
							 bias=True if len(params) == 2 else False)
		
		new_conv.weight.data = new_kernels
		if len(params) == 2:
			new_conv.bias.data = params[1].data # add bias if neccessary
		#exit()
		return new_conv

def make_prune_conv(net, place='blockres'):
	if 'blockres' in place:
		n_round = 1
		if len(list(net.layer3.children())) >= 23:
			n_round = 2
			print('=> Using n_round {} to insert temporal shift'.format(n_round))

		def make_block_prune(stage):
			blocks = list(stage.children())
			print('=> Processing stage with {} blocks residual'.format(len(blocks)))

			for i, b in enumerate(blocks):
				blocks[i].conv1.net = Prune(b.conv1.net, layer = 'conv1')
				blocks[i].bn1 = Prune(b.bn1, layer = 'bn1')
				blocks[i].conv2 = Prune(b.conv2, layer = 'conv2')
			return nn.Sequential(*blocks)

		net.layer1 = make_block_prune(net.layer1)
		net.layer2 = make_block_prune(net.layer2)
		net.layer3 = make_block_prune(net.layer3)
		net.layer4 = make_block_prune(net.layer4)