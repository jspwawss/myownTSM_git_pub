import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalModeling(nn.Module):
	def __init__(self, num_segment = 8, num_class = 101, backbone = 'ResNet'):
		super(TemporalModeling, self).__init__()

		self.num_segment = num_segment
		self.num_class = num_class

		if backbone == 'ResNet':
			self.extra_layer_conv1 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 2, dilation = 1, bias = False)
		else:
			self.extra_layer_conv1 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 2, dilation = 1, bias = False)

		self.extra_layer_conv2 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 2, dilation = 1, bias = False)
		self.extra_layer_batchnorm1 = nn.BatchNorm2d(64)
		self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')
		
		self.tanh = nn.Tanh()
		self.extra_layer_conv3 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = 1, bias = False)
		self.extra_layer_conv4 = nn.Conv2d(128, 128, kernel_size = (2, 2), stride = 1, bias = False)
		self.extra_layer_batchnorm2 = nn.BatchNorm2d(128)
		self.relu = nn.ReLU()
		self.globalavgpool = nn.AdaptiveAvgPool2d(2)
		self.extra_layer_fc1 = nn.Linear(512, self.num_class)

	def TFDEM(self, x):
																				# x -> (Batch_size * Depth, Channel, Height, Width)
		
		x = self.extra_layer_conv1(x)
		x = self.extra_layer_conv2(x)
		x = self.extra_layer_batchnorm1(x)
		c = x.size(1)
		x = self.upsample(x)

		sub = x.view(-1, self.num_segment, x.size(1), x.size(2), x.size(3))
		temp = sub.clone()
		sub[:, 1:, :, :, :] = temp[:, 1:, :, :, :] - sub[:, :-1, :, :, :]
		sub = sub.view(-1, sub.size(2), sub.size(3), sub.size(4))

		x = self.tanh(sub)
		x = self.extra_layer_conv3(x)
		x = self.extra_layer_conv4(x)
		x = self.extra_layer_batchnorm2(x)
		x = self.relu(x)
		x = self.globalavgpool(x)
		x = torch.flatten(x, 1)						

		output = self.extra_layer_fc1(x)

		return output

	def forward(self, x):

		return self.TFDEM(x)
