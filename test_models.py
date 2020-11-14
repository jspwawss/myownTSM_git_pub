# Notice that this file has been modified to support ensemble testing

import argparse
import time

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config
from torch.nn import functional as F
import torchvision

import numpy as np
# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--wholeFrame', default=False, action="store_true", help="use wholeFrame")
parser.add_argument('--full_res', default=False, action="store_true",
					help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
					help='number of data loading workers (default: 8)')

# for true test
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default=None)
parser.add_argument('--save-scores', type=str, default=None)
parser.add_argument('--softmax', default=False, action="store_true", help='use softmax')

parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim',type=int, default=256)
parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
parser.add_argument('--pretrain', type=str, default='imagenet')

args = parser.parse_args()


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)
	#print("output=",output)
	#print("output.topk(maxk, 1, True, True)=",output.topk(maxk, 1, True, True))
	_, pred = output.topk(maxk, 1, True, True)
	#print("pred",pred)
	pred = pred.t()
	#print("pred T =",pred)
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	#print("target=",target)
	#print("target.view(1, -1)",target.view(1, -1))
	#print("target.view(1, -1).expand_as(pred)=",target.view(1, -1).expand_as(pred))
	#print("correct=",correct)
	res = []
	for k in topk:
		#print("correct[:k].view(-1)=",correct[:k].view(-1))
		#print("correct[:k].view(-1).float()=",correct[:k].view(-1).float())
		#print('correct[:k].view(-1).float().sum(0)=',correct[:k].view(-1).float().sum(0))
		correct_k = correct[:k].view(-1).float().sum(0)
		#print("correct_k",correct_k)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def parse_shift_option_from_log_name(log_name):
	if 'shift' in log_name:
		strings = log_name.split('_')
		for i, s in enumerate(strings):
			if 'shift' in s:
				break
		return True, int(strings[i].replace('shift', '')), strings[i + 1]
	else:
		return False, None, None

prune_conv1in_list = {}
prune_conv1out_list = {}
def input_dim_L2distance(sd, n_div, prune_ratio_out = 0.1, prune_ratio_in = 0.1):
	count_out = 0
	count_in = 0
	for k, v in sd.items():
		if 'extra' in k:
			continue
		if 'conv1.net' in k:
			kernel = v.squeeze()
			output_c, input_c = kernel.size()
			kernel1 = kernel.unsqueeze(dim = 1).expand(output_c, output_c, input_c)
			kernel2 = kernel.unsqueeze(dim = 0).expand(output_c, output_c, input_c)
			GM_distance = torch.norm(kernel1 - kernel2, dim = 2, keepdim = True).sum(1).reshape(1, -1)
			sort, index = torch.sort(GM_distance, 1)

			GM_kernel = kernel[index[:,0]].expand(output_c, input_c)
			distance = torch.norm(kernel - GM_kernel, dim = 1, keepdim = True).reshape(1, -1)
			sort, index = torch.sort(distance, 1)
			
			prune_conv1out_list[count_out] = [int(index[:, i]) for i in range(int(distance.size(1) * prune_ratio_out))]

			count_out += 1
			
		if 'layer' in k and 'conv3' in k:
			kernel = v.squeeze()
			output_c, input_c = kernel.size()
			kernel1 = kernel.unsqueeze(dim = 1).expand(output_c, output_c, input_c)
			kernel2 = kernel.unsqueeze(dim = 0).expand(output_c, output_c, input_c)
			GM_distance = torch.norm(kernel1 - kernel2, dim = 2, keepdim = True).sum(1).reshape(1, -1)
			
			sort, index = torch.sort(GM_distance, 1)

			GM_kernel = kernel[index[:,0]].expand(output_c, input_c)
			distance = torch.norm(kernel - GM_kernel, dim = 1, keepdim = True).reshape(1, -1)
			sort, index = torch.sort(distance, 1)

			prune_conv1in_list[count_in] = [int(index[:, i]) for i in range(int(distance.size(1) * prune_ratio_in))]
			
			count_in += 1


weights_list = args.weights.split(',')

test_segments_list = [int(s) for s in args.test_segments.split(',')]
assert len(weights_list) == len(test_segments_list)
if args.coeff is None:
	coeff_list = [1] * len(weights_list)
else:
	coeff_list = [float(c) for c in args.coeff.split(',')]

if args.test_list is not None:
	test_file_list = args.test_list.split(',')
else:
	test_file_list = [None] * len(weights_list)


data_iter_list = []
net_list = []
modality_list = []

total_num = None
print("weights_list=",weights_list)
print("test_segments_list=",test_segments_list)
print("test_file_list=",test_file_list)
#exit()

for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
	is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
	if 'RGB' in this_weights:
		modality = 'RGB'
	elif 'Depth' in this_weights:
		modality = 'Depth'
	else:
		modality = 'Flow'

	if 'concatAll' in this_weights:
		concat = "All" 
	elif "concatFirst" in this_weights:
		concat = "First"
	else:
		concat = ""

	if 'extra' in this_weights:
		extra_temporal_modeling = True

	if 'conv1d' in this_weights:
		args.crop_fusion_type = "conv1d"
	else:
		args.crop_fusion_type = "avg"

	if 'prune' in this_weights:
		args.prune = 'inout'
		args.tune_from = r'/home/ubuntu/backup_kevin/myownTSM/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e35_extra_finetune_MSTSM_TFDEM_split3/ckpt.best.pth.tar'
		#args.tune_from = '/home/share/UCF101/Saved_Models/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e35_extraLSTM_finetune_MSTSM_TFDEM_split2/ckpt.best.pth.tar'
	else:
		args.prune = ''

	if args.prune in ['input', 'inout'] and args.tune_from:
		sd = torch.load(args.tune_from)
		sd = sd['state_dict']
		sd = input_dim_L2distance(sd, shift_div)

	if modality in ["RGB", "Depth"]:
		new_length = 2 if "fuse" in this_weights else 1
	else:
		new_length = 5
	
	this_arch = this_weights.split('TSM_')[1].split('_')[2]
	modality_list.append(modality)
	num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,
																							modality)
	#print("num_class, args.train_list, val_list, root_path, prefix ={}\n{}\n{}\n{}\n{}".format(num_class, args.train_list, val_list, root_path, prefix ))																					
	print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
	#exit()
	net = TSN(num_class, this_test_segments if is_shift else 1, modality,
			  base_model=this_arch, new_length = new_length,
			  consensus_type=args.crop_fusion_type,
			  img_feature_dim=args.img_feature_dim,
			  pretrain=args.pretrain,
			  is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
			  non_local='_nl' in this_weights,
			  concat = concat,
			  extra_temporal_modeling = extra_temporal_modeling,
			  prune_list = [prune_conv1in_list, prune_conv1out_list],
			  is_prune = args.prune,
			  )
	#print(net)
	#exit()
	if 'tpool' in this_weights:
		from ops.temporal_shift import make_temporal_pool
		make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel

	checkpoint = torch.load(this_weights)
	checkpoint = checkpoint['state_dict']

	# base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
	base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
	replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
					'base_model.classifier.bias': 'new_fc.bias',
					}
	for k, v in replace_dict.items():
		if k in base_dict:
			base_dict[v] = base_dict.pop(k)

	net.load_state_dict(base_dict)

	input_size = net.scale_size if args.full_res else net.input_size
	if args.test_crops == 1:
		cropping = torchvision.transforms.Compose([
			GroupScale(net.scale_size),
			GroupCenterCrop(input_size),
		])
	elif args.test_crops == 3:  # do not flip, so only 5 crops
		cropping = torchvision.transforms.Compose([
			GroupFullResSample(input_size, net.scale_size, flip=False)
		])
	elif args.test_crops == 5:  # do not flip, so only 5 crops
		cropping = torchvision.transforms.Compose([
			GroupOverSample(input_size, net.scale_size, flip=False)
		])
	elif args.test_crops == 10:
		cropping = torchvision.transforms.Compose([
			GroupOverSample(input_size, net.scale_size)
		])
	else:
		raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))
	print("test file=",test_file)
	#print("args.dense_sample=",args.dense_sample)
	#print("twice_sample=",args.twice_sample)
	#print("wholeFrame=",args.wholeFrame)
	#print("*"*50)
	#exit()
	data_loader = torch.utils.data.DataLoader(
			TSNDataSet(root_path, test_file if test_file is not None else val_list, num_segments=this_test_segments,
					   new_length=1 if modality in ["RGB", "Depth"] else 5,
					   modality=modality,
					   image_tmpl=prefix,
					   test_mode=True,
					   remove_missing=len(weights_list) == 1,
					   transform=torchvision.transforms.Compose([
						   cropping,
						   Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
						   ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
						   GroupNormalize(net.input_mean, net.input_std),
					   ]), dense_sample=args.dense_sample, twice_sample=args.twice_sample, wholeFrame=args.wholeFrame,\
					   data_fuse = True if "fuse" in this_weights else False),
			batch_size=args.batch_size, shuffle=False,
			num_workers=args.workers, pin_memory=True,
	)

	if args.gpus is not None:
		devices = [args.gpus[i] for i in range(args.workers)]
	else:
		devices = list(range(args.workers))

	net = torch.nn.DataParallel(net.cuda())
	net.eval()

	data_gen = enumerate(data_loader)

	if total_num is None:
		total_num = len(data_loader.dataset)
	else:
		assert total_num == len(data_loader.dataset)

	data_iter_list.append(data_gen)
	net_list.append(net)
#print(net_list)
#print("*"*50)
#print(len(net_list))
#exit()
output = []
#from tensorboardX import SummaryWriter
#writer = SummaryWriter(logdir = "/home/u9210700/myownTSM/featuremap/")
def visualize_featuremap(x, model, step, label, this_test_segments, writer):

	x = x.view(-1, x.size(2), x.size(3), x.size(4)).cuda()
	model.eval()
	layer_name = ['module.base_model.conv1', 'module.base_model.bn1', 'module.base_model.relu', 'module.base_model.maxpool',
				  'module.base_model.extra_layer0.extra_layer_conv1', 'module.base_model.extra_layer0.extra_layer_conv2', 'module.base_model.extra_layer0.extra_layer_batchnorm1',
				  'module.base_model.extra_layer0.tanh', 'module.base_model.extra_layer0.extra_layer_conv3',
				  'module.base_model.extra_layer0.extra_layer_conv4', 'module.base_model.extra_layer0.extra_layer_batchnorm2']

	target_layer = ['module.base_model.extra_layer0.extra_layer_conv1', 'module.base_model.extra_layer0.extra_layer_conv2', 'module.base_model.extra_layer0.extra_layer_batchnorm1',
					'module.base_model.extra_layer0.tanh', 'module.base_model.extra_layer0.extra_layer_conv3',
					'module.base_model.extra_layer0.extra_layer_conv4', 'module.base_model.extra_layer0.extra_layer_batchnorm2']
	
	with torch.no_grad():
		for name, module in model.named_modules():
			if name in layer_name:
				if 'unpool' in name:
					batch, c, h, w = x.size()
					indices = torch.tensor([range(0,h*w*4,2)]).cuda().view(h, -1)
					indices = indices[:,0::2].expand(batch, c, -1, -1).contiguous()
					x = module(x, indices)
				else:
					x = module(x)
				if name == target_layer[2]:
					for i in range(this_test_segments):
						img_grid = torchvision.utils.make_grid(x[i].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, nrow=4)
						writer.add_image('figure{}/{}_feature_maps'.format(step, name), img_grid, global_step=i)
					
					x = x.view(-1, this_test_segments, x.size(1), x.size(2), x.size(3))
					x[:, 1:, :, :, :] -= x[:, :-1, :, :, :]
					x = x.view(-1, x.size(2), x.size(3), x.size(4))

					for i in range(this_test_segments):
						img_grid = torchvision.utils.make_grid(x[i].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, nrow=4)
						writer.add_image('figure{}/{}_feature_maps_after_sub'.format(step, name), img_grid, global_step=i)

				elif name in target_layer:
					for i in range(this_test_segments):
						img_grid = torchvision.utils.make_grid(x[i].detach().cpu().unsqueeze(dim=1), normalize=True, scale_each=True, nrow=4)
						writer.add_image('figure{}/{}_feature_maps'.format(step, name), img_grid, global_step=i)

def check_mode(net):
	for m in net.modules():
		if m.training == True:
			print(m)

def eval_video(video_data, net, this_test_segments, modality):
	net.eval()
	check_mode(net)
	
	with torch.no_grad():
		i, data, label = video_data
		batch_size = label.numel()
		num_crop = args.test_crops
		if args.dense_sample:
			num_crop *= 10  # 10 clips for testing when using dense sample

		if args.twice_sample:
			num_crop *= 2

		if modality == 'RGB':
			length = 3
		elif modality == 'Depth':
			length = 1
		elif modality == 'Flow':
			length = 10
		elif modality == 'RGBDiff':
			length = 18
		else:
			raise ValueError("Unknown modality "+ modality)

		data_in = data.view(-1, length, data.size(2), data.size(3))
		print("data size=",data.size())
		print("data_in size=",data_in.size())
		#print(data_in[0] == data_in[1])
		#print("*"*50)
		#print((data_in[0] == data_in[1]).size())
		#print(net)
		if is_shift:
			#data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
			data_in = data_in.view(-1, this_test_segments, length, data_in.size(2), data_in.size(3))
			pass
		rst = net(data_in)
		print("in eval video, get model ouput")
		print("rst=",rst[0].size())
		print("batch_size=",batch_size)
		print("num_crop=",num_crop)
		print("label type=",type(label))
		print("label = ",label)
		rst = rst[0].reshape(batch_size, num_crop, -1).mean(1)

		if args.softmax:
			# take the softmax to normalize the output to probability
			rst = F.softmax(rst, dim=1)

		rst = rst.data.cpu().numpy().copy()

		if net.module.is_shift:
			#rst = rst.reshape(batch_size, num_class)
			rst = rst.reshape(-1, num_class)
		else:
			rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

		
		#visualize_featuremap(data_in, net, i, label, this_test_segments, writer)

		for x in range(label.numel()):
			with open("logit2_p.txt", "a") as f:
				import numpy as np
				np.set_printoptions(precision=4, threshold=np.inf)
				f.write("{}\t{}\n".format(np.argmax(rst[x]), label))

		return i, rst, label


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else total_num

top1 = AverageMeter()
top5 = AverageMeter()

for i, data_label_pairs in enumerate(zip(*data_iter_list)):
	with torch.no_grad():
		if i >= max_num:
			break
		this_rst_list = []
		this_label = None
		#print(type(data_label_pairs))
		#print("data label pairs=",data_label_pairs)
		
		
		for n_seg, (_, (data, label)), net, modality in zip(test_segments_list, data_label_pairs, net_list, modality_list):
			#print("data=",data)
			print("data size=",data.size())
			#exit()
			rst = eval_video((i, data, label), net, n_seg, modality)
			this_rst_list.append(rst[1])
			this_label = label
		#print("coeff_list==",coeff_list)
		#print("this_rst_list=",this_rst_list)
		assert len(this_rst_list) == len(coeff_list)
		for i_coeff in range(len(this_rst_list)):
			#print("i_coeff=",i_coeff)
			#print("this_rst_list[i_coeff]=",this_rst_list[i_coeff])
			#print("coeff_list[i_coeff]=",coeff_list[i_coeff])
			this_rst_list[i_coeff] *= coeff_list[i_coeff]
		ensembled_predict = sum(this_rst_list) / len(this_rst_list)
		#print("ensembled_predict=",ensembled_predict)
		print("ensembled_predict=",np.shape(ensembled_predict))
		for p, g in zip(ensembled_predict, this_label.cpu().numpy()):
			#print("ensembled_predict=",ensembled_predict)
			#print("p=",p)
			#print("p[None, ...]=",p[None, ...])
			output.append([p[None, ...], g])
		cnt_time = time.time() - proc_start_time
		prec1, prec5 = accuracy(torch.from_numpy(ensembled_predict), this_label, topk=(1, 5))
		top1.update(prec1.item(), this_label.numel())
		top5.update(prec5.item(), this_label.numel())
		if i % 20 == 0:
			print('video {} done, total {}/{}, average {:.3f} sec/video, '
				  'moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i * args.batch_size, i * args.batch_size, total_num,
															  float(cnt_time) / (i+1) / args.batch_size, top1.avg, top5.avg))
		
		
video_pred = [np.argmax(x[0]) for x in output]
video_pred_top5 = [np.argsort(np.mean(x[0], axis=0).reshape(-1))[::-1][:5] for x in output]

video_labels = [x[1] for x in output]

if args.save_scores is not None:
	save_scores_list = args.save_scores.split(',')

	for i, score_name in enumerate(save_scores_list):
		name_list = [x.strip().split()[0] for x in open(test_file_list[i])]
		order_dict = {e:i for i, e in enumerate(sorted(name_list))}

		reorder_output = [None] * len(output)
		reorder_label = [None] * len(output)
		reorder_name = [None] * len(output)

		for i in range(len(output)):
			idx = order_dict[name_list[i]]
			reorder_output[idx] = output[i]
			reorder_label[idx] = video_labels[i]
			reorder_name[idx] = name_list[i]

		np.savez(score_name, scores=reorder_output, labels=reorder_label, names=reorder_name)

if args.csv_file is not None:
	print('=> Writing result to csv file: {}'.format(args.csv_file))
	with open(test_file_list[0].replace('test_videofolder.txt', 'category.txt')) as f:
		categories = f.readlines()
	categories = [f.strip() for f in categories]
	with open(test_file_list[0]) as f:
		vid_names = f.readlines()
	vid_names = [n.split(' ')[0] for n in vid_names]
	assert len(vid_names) == len(video_pred)
	if args.dataset != 'somethingv2':  # only output top1
		with open(args.csv_file, 'w') as f:
			for n, pred in zip(vid_names, video_pred):
				f.write('{};{}\n'.format(n, categories[pred]))
	else:
		with open(args.csv_file, 'w') as f:
			for n, pred5 in zip(vid_names, video_pred_top5):
				fill = [n]
				for p in list(pred5):
					fill.append(p)
				f.write('{};{};{};{};{};{}\n'.format(*fill))


cf = confusion_matrix(video_labels, video_pred).astype(float)

np.save(args.save_scores.replace('scores', 'cm.npy'), cf)
cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)
#print(video_labels)
#print(video_pred)
cls_acc = cls_hit / cls_cnt
print(cls_acc)
upper = np.mean(np.max(cf, axis=1) / cls_cnt)
print('upper bound: {}'.format(upper))

print('-----Evaluation is finished------')
print('Class Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))
