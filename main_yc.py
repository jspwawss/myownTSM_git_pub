import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


from ops.dataset_v2 import YouCookDataSetRcg
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter
from torchsummaryX import summary

best_prec1 = 0
prune_conv1in_list = {}
prune_conv1out_list = {}

def main():
	global args, best_prec1
	global crop_size
	args = parser.parse_args()

	num_class, train_list, val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)
	if args.train_list == "":
		args.train_list = train_list
	if args.val_list == "":
		args.val_list = val_list

	full_arch_name = args.arch
	if args.shift:
		full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
	if args.concat != "":
		full_arch_name += '_concat{}'.format(args.concat)
	if args.temporal_pool:
		full_arch_name += '_tpool'
	args.store_name = '_'.join(
		['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'lr%.5f' % args.lr, 'dropout%.2f' % args.dropout, 'wd%.5f' % args.weight_decay,
		 'batch%d' % args.batch_size, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
	if args.data_fuse:
		args.store_name += '_fuse'
	if args.extra_temporal_modeling:
		args.store_name += '_extra'
	if args.tune_from is not None:
		args.store_name += '_finetune'
	if args.pretrain != 'imagenet':
		args.store_name += '_{}'.format(args.pretrain)
	if args.lr_type != 'step':
		args.store_name += '_{}'.format(args.lr_type)
	if args.dense_sample:
		args.store_name += '_dense'
	if args.non_local > 0:
		args.store_name += '_nl'
	if args.suffix is not None:
		args.store_name += '_{}'.format(args.suffix)
	print('storing name: ' + args.store_name)

	check_rootfolders()

	if args.prune in ['input', 'inout'] and args.tune_from:
		sd = torch.load(args.tune_from)
		sd = sd['state_dict']
		sd = input_dim_L2distance(sd, args.shift_div)

	model = TSN(num_class, args.num_segments, args.modality,
				base_model=args.arch,
				new_length = 2 if args.data_fuse else None,
				consensus_type=args.consensus_type,
				dropout=args.dropout,
				img_feature_dim=args.img_feature_dim,
				partial_bn=not args.no_partialbn,
				pretrain=args.pretrain,
				is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
				fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
				temporal_pool=args.temporal_pool,
				non_local=args.non_local,
				concat = args.concat,
				extra_temporal_modeling = args.extra_temporal_modeling,
				prune_list = [prune_conv1in_list, prune_conv1out_list],
				is_prune = args.prune,
				)

	print(model)
	#summary(model, torch.zeros((16, 24, 224, 224)))
	#exit(1)
	if args.dataset == 'ucf101':		#twice sample & full resolution
		twice_sample = True
		crop_size = model.scale_size	#256 x 256
	else:
		twice_sample = False
		crop_size = model.crop_size		#224 x 224
	crop_size = 256
	scale_size = model.scale_size
	input_mean = model.input_mean
	input_std = model.input_std
	policies = model.get_optim_policies(args.concat)
	train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset or 'nvgesture' in args.dataset else True)

	model = torch.nn.DataParallel(model).cuda()

	optimizer = torch.optim.SGD(policies,
								args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	if args.resume:
		if args.temporal_pool:  # early temporal pool so that we can load the state_dict
			make_temporal_pool(model.module.base_model, args.num_segments)
		if os.path.isfile(args.resume):
			print(("=> loading checkpoint '{}'".format(args.resume)))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print(("=> loaded checkpoint '{}' (epoch {})"
				   .format(args.evaluate, checkpoint['epoch'])))
		else:
			print(("=> no checkpoint found at '{}'".format(args.resume)))

	if args.tune_from:
		print(("=> fine-tuning from '{}'".format(args.tune_from)))
		tune_from_list = args.tune_from.split(',')
		sd = torch.load(tune_from_list[0])
		sd = sd['state_dict']

		model_dict = model.state_dict()
		replace_dict = []
		for k, v in sd.items():
			if k not in model_dict and k.replace('.net', '') in model_dict:
				print('=> Load after remove .net: ', k)
				replace_dict.append((k, k.replace('.net', '')))
		for k, v in model_dict.items():
			if k not in sd and k.replace('.net', '') in sd:
				print('=> Load after adding .net: ', k)
				replace_dict.append((k.replace('.net', ''), k))
		for k, v in model_dict.items():
			if k not in sd and k.replace('.prune', '') in sd:
				print('=> Load after adding .prune: ', k)
				replace_dict.append((k.replace('.prune', ''), k))

		if args.prune in ['input', 'inout']:
			sd = adjust_para_shape_prunein(sd, model_dict)
		if args.prune in ['output', 'inout']:
			sd = adjust_para_shape_pruneout(sd, model_dict)

		if args.concat != "" and "concat" not in tune_from_list[0]:
			sd = adjust_para_shape_concat(sd, model_dict)

		for k, k_new in replace_dict:
			sd[k_new] = sd.pop(k)
		keys1 = set(list(sd.keys()))
		keys2 = set(list(model_dict.keys()))
		set_diff = (keys1 - keys2) | (keys2 - keys1)
		print('#### Notice: keys that failed to load: {}'.format(set_diff))
		if args.dataset not in tune_from_list[0]:  # new dataset
			print('=> New dataset, do not load fc weights')
			sd = {k: v for k, v in sd.items() if 'fc' not in k}
		if args.modality != 'Flow' and 'Flow' in tune_from_list[0]:
			sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}

		model_dict.update(sd)
		model.load_state_dict(model_dict)

	if args.temporal_pool and not args.resume:
		make_temporal_pool(model.module.base_model, args.num_segments)

	cudnn.benchmark = True

	# Data loading code
	if args.modality != 'RGBDiff':
		normalize = GroupNormalize(input_mean, input_std)
	else:
		normalize = IdentityTransform()

	if args.modality in ['RGB']:
		data_length = 1
	elif args.modality in ['Depth']:
		data_length = 1
	elif args.modality in ['Flow', 'RGBDiff']:
		data_length = 5

	'''
	dataRoot = r"/home/share/YouCook/downloadVideo"
	for dirPath, dirnames, filenames in os.walk(dataRoot):
		for filename in filenames:
			print(os.path.join(dirPath,filename) +"is {}".format("exist" if os.path.isfile(os.path.join(dirPath,filename))else "NON"))
			train_data = torchvision.io.read_video(os.path.join(dirPath,filename),start_pts=0,end_pts=1001, )
			tmp = torchvision.io.read_video_timestamps(os.path.join(dirPath,filename),)
			print(tmp)
			print(len(tmp[0]))
			print(train_data[0].size())
			exit()
	exit()
	'''
	'''
	train_loader = torch.utils.data.DataLoader(
		TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
				   new_length=data_length,
				   modality=args.modality,
				   image_tmpl=prefix,
				   transform=torchvision.transforms.Compose([
					   train_augmentation,
					   Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
					   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
					   normalize,
				   ]), dense_sample=args.dense_sample, data_fuse = args.data_fuse),
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True,
		drop_last=True)  # prevent something not % n_GPU
	

	
	val_loader = torch.utils.data.DataLoader(
		TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
				   new_length=data_length,
				   modality=args.modality,
				   image_tmpl=prefix,
				   random_shift=False,
				   transform=torchvision.transforms.Compose([
					   GroupScale(int(scale_size)),
					   GroupCenterCrop(crop_size),
					   Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
					   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
					   normalize,
				   ]), dense_sample=args.dense_sample, twice_sample=twice_sample, data_fuse = args.data_fuse),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)
	'''
	
	train_loader = torch.utils.data.DataLoader(
		YouCookDataSetRcg(args.root_path, args.train_list,train=True,inputsize=crop_size),
		#shuffle=True,
		

	)
	val_loader = torch.utils.data.DataLoader(
		YouCookDataSetRcg(args.root_path, args.val_list,val=True,inputsize=crop_size)

	)
	# define loss function (criterion) and optimizer
	if args.loss_type == 'nll':
		criterion = torch.nn.CrossEntropyLoss().cuda()
	else:
		raise ValueError("Unknown loss type")

	for group in policies:
		print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
			group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

	if args.evaluate:
		validate(val_loader, model, criterion, 0)
		return

	log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
	with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
		f.write(str(args))
	#print(os.path.join(args.root_log, args.store_name, 'args.txt'))
	#exit()
	tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

	for epoch in range(args.start_epoch, args.epochs):
		adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
		print("265")
		# train for one epoch
		######
		train(train_loader, model, criterion, optimizer, epoch, log_training, tf_writer)
		######
		print("268")
		# evaluate on validation set
		if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
			prec1 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

			# remember best prec@1 and save checkpoint
			is_best = prec1 > best_prec1
			best_prec1 = max(prec1, best_prec1)
			tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

			output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
			#print(output_best)
			log_training.write(output_best + '\n')
			log_training.flush()

			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_prec1': best_prec1,
			}, is_best)

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	losses_extra = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	top1_extra = AverageMeter()
	top5_extra = AverageMeter()

	if args.no_partialbn:
		model.module.partialBN(False)
	else:
		model.module.partialBN(True)

	# switch to train mode
	model.train()
	#print("308")
	end = time.time()
	for i, data in enumerate(train_loader):
		#print("main 311")
		#print(data)
		#exit()
		if len(data)==1:
			continue
		# measure data loading time
		data_time.update(time.time() - end)
		#print("315")
		#print("/"*50)
		#print(data)
		'''
		for d in data:
			print(d)
			print("317")
			#[label, video, caption] = d #video = [[batch,frames,h,w,c], [audio], video_fps:tensor, audio_fps:tensor]
			[segment,sentence,id,videos] = d
			print("label=",label)
			#print("video=",video)
			#print(type(video))
			#print(np.shape(video[0]))
			print(caption)
			label = label.cuda()
			video = video[0].cuda()
			print(video.size())
			video = video.transpose(2,4)
			print(video.size())
			#caption = caption.cuda()
			output, extra = model(video)
			######################################################################################################################
		'''
		
		[label, fileURL, filecontents] = data		##[label, URL, [[segment,sentence,id,videos], [], [], ...]
		#print("label={},fileURL={}".format(label, fileURL))
		label=label[0].cuda()
		fileURL=fileURL[0] 				#for recording filename in result
		
		#print("label={},fileURL={}".format(label, fileURL))
		input=torch.tensor([],dtype=torch.float).cuda()
		for filecontent in filecontents:
			#print(filecontent)
			for f in filecontent[3]:

				input = torch.cat((input, f.float().cuda()),1)
		#videoclip = videoclip.view(8,3,224,224)
		input = input.view(8,3,crop_size,crop_size)
		#print(input.size())

		#target = target.cuda()
		target = label.cuda()
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(label)

		# compute output
		wExtraLoss = 1 if args.prune == '' else 0.1
		output, extra = model(input_var)
		loss_main = criterion(output, target_var)
		extra_loss = criterion(extra, target_var)*wExtraLoss
		loss = loss_main + extra_loss

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		losses.update(loss_main.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))
		top5.update(prec5.item(), input.size(0))

		prec1_extra, prec5_extra = accuracy(extra.data, target, topk=(1, 5))
		losses_extra.update(extra_loss.item(), input.size(0))
		top1_extra.update(prec1_extra.item(), input.size(0))
		top5_extra.update(prec5_extra.item(), input.size(0))

		# compute gradient and do SGD step
		loss.backward()

		if args.clip_gradient is not None:
			total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

		optimizer.step()
		optimizer.zero_grad()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Loss_h {losses_extra.val:.4f} ({losses_extra.avg:.4f})\t'
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					  .format(
				epoch, i, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, losses_extra=losses_extra, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
			print(output)
			log.write(output + '\n')
			log.flush()
		#print("**"*50)
		#break

	tf_writer.add_scalar('loss/train', losses.avg, epoch)
	tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
	tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
	tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, data in enumerate(val_loader):
			if len(data) == 1:
				continue
			[label, fileURL, filecontents] = data		##[label, URL, [[segment,sentence,id,videos], [], [], ...]
			#print("label={},fileURL={}".format(label, fileURL))
			label=label[0].cuda()
			fileURL=fileURL[0] 				#for recording filename in result
			
			#print("label={},fileURL={}".format(label, fileURL))
			input=torch.tensor([],dtype=torch.float).cuda()
			for filecontent in filecontents:
				print("filecontent")
				print(len(filecontent[3]))
				for f in filecontent[3]:
					print("{0:*^50}".format("f"))
					print(f.size())
					input = torch.cat((input, f.float().cuda()),1)
					print(input.size())
			#videoclip = videoclip.view(8,3,224,224)
			input = input.view(8,3,crop_size,crop_size)
			#print(input.size())																				
			# compute output
			output, extra = model(input)
			#print("output size",output.size())
			target = label.cuda()
			if args.dataset == 'ucf101':		#twice sample & full resolution
				output = output.reshape(target.numel(), 2, -1).mean(1)

			loss = criterion(output, target)

			# measure accuracy and record loss
			prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

			losses.update(loss.item(), input.size(0))
			top1.update(prec1.item(), input.size(0))
			top5.update(prec5.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				output = ('Test: [{0}/{1}]\t'
						  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
						  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
						  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					i, len(val_loader), batch_time=batch_time, loss=losses,
					top1=top1, top5=top5))
				print(output)
				if log is not None:
					log.write(output + '\n')
					log.flush()

	output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
			  .format(top1=top1, top5=top5, loss=losses))
	print(output)
	if log is not None:
		log.write(output + '\n')
		log.flush()

	if tf_writer is not None:
		tf_writer.add_scalar('loss/test', losses.avg, epoch)
		tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
		tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

	return top1.avg


def save_checkpoint(state, is_best):
	filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	if lr_type == 'step':
		decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
		lr = args.lr * decay
		decay = args.weight_decay
	elif lr_type == 'cos':
		import math
		lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
		decay = args.weight_decay
	else:
		raise NotImplementedError
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr * param_group['lr_mult']
		param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
	"""Create log and model folder"""
	folders_util = [args.root_log, args.root_model,
					os.path.join(args.root_log, args.store_name),
					os.path.join(args.root_model, args.store_name)]
	for folder in folders_util:
		if not os.path.exists(folder):
			print('creating folder ' + folder)
			os.mkdir(folder)

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

def adjust_para_shape_pruneout(sd, model_dict):
	remaining_kernel_index = []
	count = 0
	for k, v in sd.items():
		if 'extra' in k:
			continue
		if 'conv1.net' in k:# and 'layer1.0' not in k:
			remaining_kernel_index = list(set(range(v.size(0))) - set(prune_conv1out_list[count]))
			sd[k] = v.data[remaining_kernel_index, :, :, :]
			
			count += 1
			
		elif 'conv2' in k:
			sd[k] = v.data[:, remaining_kernel_index, :, :]
		elif 'layer' in k and 'bn1.weight' in k:
			sd[k] = v.data[remaining_kernel_index]
		elif 'layer' in k and 'bn1.bias' in k:
			sd[k] = v.data[remaining_kernel_index]
		elif 'layer' in k and 'bn1.running_mean' in k:
			sd[k] = v.data[remaining_kernel_index]
		elif 'layer' in k and 'bn1.running_var' in k:
			sd[k] = v.data[remaining_kernel_index]

	return sd

def adjust_para_shape_prunein(sd, model_dict):
	remaining_kernel_index = []
	count = 0
	for k, v in sd.items():
		if 'conv1.net' in k and 'layer1.0' not in k:
			new_kernel_size = model_dict[k.replace('.net', '.net.prune')].size(1)
			remaining_kernel_index = list(set(range(v.size(1) // 9 * 8)) - set(prune_conv1in_list[count][1:]))
			remaining_kernel = v.data[:, remaining_kernel_index, :, :]
			sd[k] = remaining_kernel.mean(dim=1, keepdim = True).expand(-1, new_kernel_size, -1, -1).contiguous()
			sd[k].data[:, :remaining_kernel.size(1), :, :] = remaining_kernel.data

			extra_kernel_size = new_kernel_size - remaining_kernel.size(1)
			sd[k].data[:, -extra_kernel_size:, :, :] = remaining_kernel.data[:, -extra_kernel_size:, :, :]

			count += 1

	return sd

def adjust_para_shape_concat(sd, model_dict, n_div = 8):
	
	for k, v in sd.items():
		if "net" in k:
			kernel_size = v.size()
			new_kernel_size = model_dict[k].size()
			sd[k] = v.data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
			sd[k].data[:,:kernel_size[1],:,:] = v.data
	return sd

if __name__ == '__main__':
	main()

