# Notice that this file has been modified to support ensemble testing

import argparse
import time
import os

import torch.backends.cudnn as cudnn

import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from ops.dataset_slice_v2 import YouCookDataSetRcg
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
parser.add_argument('--clipnums', type = str, default = "", help='numbers of clips')
parser.add_argument('--epoch', type = str, default = "", help='numbers of clips')
parser.add_argument('--data_fuse', default = False, action = "store_true", help = 'concatenate skeleton to depth')

parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')
parser.add_argument('--concat', type = str, default = "", choices = ['All', 'First'], help = 'use concatenation after shifting')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
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

def AUC(output, target):
    #print("output is ",output)
    #print("target is ",target)
    #>0.8 => True   || 0.8 is my threshold
    #output_cpu=output.cpu()
    #target_cpu = target.cpu()
    #output = target
    
    pred = torch.where(output>0.8,torch.ones(output.size()).cuda(),torch.zeros(output.size()).cuda()).cuda()
    #pred[0] = 1
    #print(pred)

    tp = torch.eq(pred,target).cuda()
    #print("tp=",tp)
    acc = torch.sum(tp)
    accdata = acc.float()
    #print("intersection=",accdata)
    #print(output.size())
    #print(type(target.size()))
    if not output.size():
        a = 1+1-accdata
    else:
        a = output.size()[0]+target.size()[0]-accdata
    #print("area =",a)
    auc = float(acc.data/a.data)
    #print("auc in auc is ",auc)
    return auc


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
#print("weights_list=",weights_list)
#print("test_segments_list=",test_segments_list)
#print("test_file_list=",test_file_list)
#exit()

for this_weights, this_test_segments, test_file in zip(weights_list, test_segments_list, test_file_list):
	#print("ayaya")
	is_shift, shift_div, shift_place = parse_shift_option_from_log_name(this_weights)
	print("is shift={is_shift}, shift_div={shift_div}, shift_place={shift_place}".format(is_shift=is_shift, shift_div = shift_div, shift_place = shift_place))
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
	else:
		extra_temporal_modeling = False

	if 'conv1d' in this_weights:
		args.crop_fusion_type = "conv1d"
	else:
		args.crop_fusion_type = "avg"

	if 'prune' in this_weights:
		print("prune in this weights")
		args.prune = 'inout'
		args.tune_from = r'/home/ubuntu/backup_kevin/myownTSM/checkpoint/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e35_extra_finetune_MSTSM_TFDEM_split3/ckpt.best.pth.tar'
		#args.tune_from = '/home/share/UCF101/Saved_Models/TSM_ucf101_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e35_extraLSTM_finetune_MSTSM_TFDEM_split2/ckpt.best.pth.tar'
	else:
		args.prune = ''
	#print("args.prune={}".format(args.prune))
	if args.prune in ['input', 'inout'] and args.tune_from:
		sd = torch.load(args.tune_from)
		sd = sd['state_dict']
		sd = input_dim_L2distance(sd, shift_div)

	if modality in ["RGB", "Depth"]:
		new_length = 2 if "fuse" in this_weights else 1
	else:
		new_length = 5
	print("this weight\n",this_weights)
	this_arch = this_weights.split('TSM_')[2].split('_')[2]
	modality_list.append(modality)
	num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset,modality)
	
	num_class = 1																					
	print("num_class, args.train_list, val_list, root_path, prefix ={}\n{}\n{}\n{}\n{}".format(num_class, args.train_list, val_list, root_path, prefix ))																					
	print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
	#exit()
	'''
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
	'''
	net = TSN(num_class, this_test_segments if is_shift else 1, modality,
			base_model=this_arch,
			new_length = 2 if args.data_fuse else None,
			consensus_type=args.crop_fusion_type,
			#dropout=args.dropout,
			img_feature_dim=args.img_feature_dim,
			#partial_bn=not args.no_partialbn,
			pretrain=args.pretrain,
			is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
			#fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
			#temporal_pool=args.temporal_pool,
			non_local='_nl' in this_weights,
			concat = concat,
			extra_temporal_modeling = extra_temporal_modeling,
			prune_list = [prune_conv1in_list, prune_conv1out_list],
			is_prune = args.prune,
			)
	print(net)
	#print(args.shift)
	#exit()
	if 'tpool' in this_weights:
		from ops.temporal_shift import make_temporal_pool
		make_temporal_pool(net.base_model, this_test_segments)  # since DataParallel


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
			YouCookDataSetRcg(root_path, val_list,val=True,inputsize=input_size,hasPreprocess = False,\
			#clipnums=args.clipnums,
			hasWordIndex = True,)
	)

	if args.gpus is not None:
		devices = [args.gpus[i] for i in range(args.workers)]
	else:
		devices = list(range(args.workers))

	policies = net.get_optim_policies(args.concat)
	optimizer = torch.optim.SGD(policies,
							lr=0.0025,
							#momentum=args.momentum,
							#weight_decay=args.weight_decay,
							)
	net = torch.nn.DataParallel(net.cuda())
	cudnn.benchmark = True

	#net.eval()

	data_gen = enumerate(data_loader)

	if total_num is None:
		total_num = len(data_loader.dataset)
	else:
		assert total_num == len(data_loader.dataset)

	data_iter_list.append(data_gen)
	net_list.append(net)

#print(net_list)
print("*"*50)
#print(len(net_list))
#exit()
crop_size = 256

def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
	model.eval()
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	AUCs = AverageMeter()

	# switch to evaluate mode
	model.eval()
	
	end = time.time()
	with torch.no_grad():
		with open(os.path.join(args.save_scores, "val.txt"), "a+") as txt:
			txt.write("epoch\t"+str(epoch)+"\n")
		for idx, data in enumerate(val_loader):
			totalOutput = torch.tensor([],dtype=torch.float).cuda()
			totalTarget = torch.tensor([],dtype=torch.float).cuda()
			data_time.update(time.time() - end)
	#		if isinstance(data, bool):
		#		continue
			#if not torch.any(data):
	#			continue
			if not data:
				continue
			try:
				[URL, id, sift, label, clips] = data	
			except Exception as e:
				print(e)
				print(data)
				with open("errr.txt","a+") as txt:
					txt.write(str(data))

			#cap_nums = []
			for i, s in enumerate(sift):
				input_video=torch.tensor([],dtype=torch.float).cuda()
				input_caption=torch.tensor([],dtype=torch.long).cuda()
				#print("i={},s={}".format(i,s))
				if i+1 < len(sift):
					if int(sift[i+1])-int(s) < 50:
						#print( int(sift[i+1])-int(s))
						for clip in range(s,sift[i+1]):
							#print(clip)
							video = clips[clip][0]
							caption = clips[clip][1]
							input_video = torch.cat((input_video.float().cuda(),video.float().cuda()),1)
							input_caption = torch.cat((input_caption, caption.long().cuda()),0)
							target = label[0,s: sift[i+1]].cuda()
					else:
						for clip in range(s,sift[i+1],(int(sift[i+1])-int(s))//50):
							#print(clip)
							if (clip - s)/((int(sift[i+1])-int(s))//50) >= 50:
								break
							video = clips[clip][0]
							caption = clips[clip][1]
							input_video = torch.cat((input_video.float().cuda(), video.float().cuda()),1)
							input_caption = torch.cat((input_caption, caption.long().cuda()),0)
							target = label[0,s: sift[i+1]: (sift[i+1]-s) //50].cuda()
							target = target[:50]
				else:
					length = len(clips)
					if int(length)- int(s) < 50:
						for clip in range(s,length):
							video = clips[clip][0]
							caption = clips[clip][1]
							input_video = torch.cat((input_video.float().cuda(),video.float().cuda()),1)
							input_caption = torch.cat((input_caption, caption.long().cuda()),0)
							target = label[0,s: length].cuda()
					else:
						for clip in range(s,length,(int(length)-int(s))//50):
							if (clip - s)/((int(length)-int(s))//50) >= 50:
								break
							video = clips[clip][0]
							caption = clips[clip][1]
							input_video = torch.cat((input_video.float().cuda(), video.float().cuda()),1)
							input_caption = torch.cat((input_caption, caption.long().cuda()),0)
							target = label[0,s:length: (length-s) //50].cuda()
							target = target[:50]
				#print("input video.size()",input_video.size())
				#print("input_caption.size()",input_caption.size())

				input_video = input_video.view(1,-1,3,crop_size,crop_size)
				#input_caption = input_caption.view(1,-1,input_caption.size()[-1],1)
				input_caption = input_caption.view(1,-1,64,1)



				input_video_var = torch.autograd.Variable(input_video)
				input_caption_var = torch.autograd.Variable(input_caption)
				target_var = torch.autograd.Variable(target)
				#print("input_video_var.size()",input_video_var.size())
				#print("target size",target.size())
				# compute output
				wExtraLoss = 1 if args.prune == '' else 0.1
				output = model(input_video_var, input_caption_var)
				#print("output size=",output.size())
				#print("target)var=",target_var.size())
				#print(output)
				#print(target_var)
				#print("totalOutput size=",totalOutput.size())
				#print("output size=",output.size())
				#print("squeeze size=",output.squeeze().size())
				#print(totalOutput)
				#print("/*"*50)
				#print(output)
				totalOutput = torch.cat((totalOutput.float().cuda(),output.float().cuda()),dim=1)
				#print("totalTarget size()",totalTarget.size())
				#print("target_Var.size()",target_var.size())
				#print("target size",target.size())
				#print(target_var)
				if len(target_var.size()) != 0:
					totalTarget = torch.cat((totalTarget.float().cuda(),target_var.float().cuda()),dim=0)
				loss = criterion(output.squeeze(), target.squeeze())
				#print("loss = ",loss)
				#auc = AUC(output.squeeze(), target_var.squeeze())
				#print("auc=",auc)

				#AUCs.update(auc)

				losses.update(loss.item(), )


			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			auc = AUC(totalOutput.squeeze(), totalTarget.squeeze())
			AUCs.update(auc)
			txtoutput = ('Test: [{0}/{1}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
						"output\target\t{output}\t{target}"		
						'AUC {top5.val:.3f} ({top5.avg:.3f})\t'
						"URL id = {URL} {id}".format(
				idx, len(val_loader), batch_time=batch_time, loss=losses,output=totalOutput, target=totalTarget,
					top5=AUCs, URL=URL, id= id))	
			if idx % args.print_freq == 0:
				#output = ('Test: [{0}/{1}]\t'
				#		  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				#		  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				#		  "output/target {output}/{target}"
				#		  'AUC {aucs.val:.3f} ({aucs.avg:.3f})'.format(
				#	i, len(val_loader), batch_time=batch_time, loss=losses,output=output,target=target,
				#	 aucs=AUCs))
				print(txtoutput)
				if log is not None:
					log.write(txtoutput + '\n')
					log.flush()


			with open(os.path.join(args.save_scores, "val.txt"), "a+") as txt:
				txt.write(txtoutput+"\n")

			#break

		txtoutput = ('Testing Results: auc {auc.avg:.3f}  Loss {loss.avg:.5f}'
			  	.format(auc=AUCs, loss=losses))
		with open(os.path.join(args.save_scores, "val.txt"), "a+") as txt:
			txt.write(txtoutput+"\n")
	print(txtoutput)
	if log is not None:
		log.write(txtoutput + '\n')
		log.flush()

	if tf_writer is not None:
		tf_writer.add_scalar('loss/test', losses.avg, epoch)
		#tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
		#tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
		tf_writer.add_scalar('auc/test', AUCs.avg, epoch)
	return AUCs.avg


def train(train_loader, model, criterion, optimizer, epoch, log=None, tf_writer=None):
	model.train()
	#global trainDataloader, valDataloader

	#print(len(train_loader))
	#print(trainDataloader._getMode())
	#print(valDataloader._getMode())
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	AUCs = AverageMeter()
	#losses_extra = AverageMeter()
	#top1 = AverageMeter()
	#top5 = AverageMeter()
	#top1_extra = AverageMeter()
	#top5_extra = AverageMeter()

	#if args.no_partialbn:
	#	model.module.partialBN(False)
	#else:
	#	model.module.partialBN(True)

	# switch to train mode
	model.train()
	#print("308")
	end = time.time()
	#print("in train")
	for idx, data in enumerate(train_loader):
		
		#print("data fetch finish ",i)
		

		#print(data)
		#if isinstance(data, bool): #img DNE
	#		continue
		if not data:
			continue
		data_time.update(time.time() - end)


		#print(data[0])
		#print(data[1])
		[URL, id, sift, label, clips] = data	


		print("label.size=",label.size())
		#cap_nums = []
		print("sift=",sift)
		for i, s in enumerate(sift):
			input_video=torch.tensor([],dtype=torch.float).cuda()
			input_caption=torch.tensor([],dtype=torch.long).cuda()
			print("i={},s={},lensift={},idx={},epoch={}".format(i, s, len(sift), idx, epoch))
			if i+1 < len(sift):
				
				print(sift[i+1])
				print(s)
				#print(int(sift[i+1])-int(s))
				if int(sift[i+1])-int(s) < 50:
					print("566")
					for clip in range(s,sift[i+1]):
						video = clips[clip][0]
						caption = clips[clip][1]
						input_video = torch.cat((input_video.float().cuda(), video.float().cuda()),1)
						#print(input_video.size())
						input_caption = torch.cat((input_caption, caption.long().cuda()),0)
						#print(input_caption.size())
						#print(sift[i+1])
						#print(s)
						#print(label[0,s: sift[i+1]])
					target = label[0,s: sift[i+1]].cuda()
					print("taget assgin size=",target.size())
				else:
					print("579")
					for clip in range(s,sift[i+1],(int(sift[i+1])-int(s))//50):
						#print("clip={},s={},sift[i+1]={}".format(clip, s, sift[i+1]))
						if (clip - s)/((int(sift[i+1])-int(s))//50) >= 50:
							break
						video = clips[clip][0]
						caption = clips[clip][1]
						input_video = torch.cat((input_video.float().cuda(), video.float().cuda()),1)
						input_caption = torch.cat((input_caption, caption.long().cuda()),0)
					target = label[0,s: sift[i+1]: ((int(sift[i+1])-int(s)) //50)].cuda()
					target = target[:50].cuda()
			else:
				length = len(clips)
				if int(length)- int(s) < 50:
					print("593")
					for clip in range(s,length):
						video = clips[clip][0]
						caption = clips[clip][1]
						input_video = torch.cat((input_video.float().cuda(),video.float().cuda()),1)
						input_caption = torch.cat((input_caption, caption.long().cuda()),0)
					target = label[0,s: length].cuda()
				else:
					print("601")
					for clip in range(s,length,(int(length)-int(s))//50):
						if (clip - s)/((int(length)-int(s))//50) >= 50:
							break
						video = clips[clip][0]
						caption = clips[clip][1]
						input_video = torch.cat((input_video.float().cuda(), video.float().cuda()),1)
						input_caption = torch.cat((input_caption, caption.long().cuda()),0)
					target = label[0,s: length: (length-s) //50].cuda()
					target = target[:50]
				'''
				video = clips[-1][0]
				caption = clips[-1][1]
				input_video = torch.cat((input_video.float().cuda(), video.float().cuda()),1)
				input_caption = torch.cat((input_caption, caption.long().cuda()),0)
				target = label[0,-1].cuda()
				'''
			input_video = input_video.view(1,-1,3,crop_size,crop_size)
			#input_caption = input_caption.view(1,-1,input_caption.size()[-1],1)
			input_caption = input_caption.view(1,-1,64,1)
			print("target=",target.size())
			input_video_var = torch.autograd.Variable(input_video)
			input_caption_var = torch.autograd.Variable(input_caption)
			target_var = torch.autograd.Variable(target)

			# compute output
			wExtraLoss = 1 if args.prune == '' else 0.1
			output = model(input_video_var, input_caption_var)
			#print("outputsize=",output.size())
			#print("target_var=",target_var.size())
			#print("target=",target.size())
			loss_main = criterion(output.squeeze(), target_var.squeeze())

			loss = loss_main

			losses.update(loss_main.item(), )

			auc = AUC(output.squeeze(), target_var.squeeze())
			#print("auc=",auc)
			AUCs.update(auc)
			#print("after AUCs update")

			# compute gradient and do SGD step
			st = time.time()
			loss.backward()
			#print("{0:*^50}".format("after backward\t"+str(time.time()-st)))
			#if args.clip_gradient is not None:
			#	total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
			st = time.time()
			optimizer.step()
			#print("{0:*^50}".format("after step\t"+str(time.time()-st)))
			st = time.time()
			optimizer.zero_grad()
			#print("{0:*^50}".format("after zero_grad\t"+str(time.time()-st)))
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
		'''
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
		'''
		if idx % args.print_freq == 0:
			txtoutput = ('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  "output/target\t{output}/\t{target}\t"
					  'auc {auc.val:.4f} ({auc.avg:.3f})\t'
					  .format(
				epoch, idx, len(train_loader), batch_time=batch_time,output=output, target=target,
				data_time=data_time, loss=losses,  auc=AUCs, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
			print(txtoutput)
			#log.write(txtoutput + '\n')
			#log.flush()
		print("**"*50)
		break
		#exit()

	#tf_writer.add_scalar('loss/train', losses.avg, epoch)
	#tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
	#tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
	#tf_writer.add_scalar('auc/train', AUCs.avg, epoch)
	#tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)


criterion = torch.nn.BCELoss()
best_prec1 = 0
for epoch in range(12,int(args.epoch)+1):
	print(this_weights.format(str(epoch+1)))
	checkpoint = torch.load(this_weights.format(str(epoch+1)))
	checkpoint = checkpoint['state_dict']
	#for key, value in checkpoint.items():
	#	print(key)
	
	
	#print(checkpoint)
	
	netdict = [ "module.base_model.layer1.0.conv1.weight", "module.base_model.layer1.1.conv1.weight", "module.base_model.layer1.2.conv1.weight", \
		"module.base_model.layer2.0.conv1.weight", "module.base_model.layer2.1.conv1.weight", "module.base_model.layer2.2.conv1.weight",\
			 "module.base_model.layer2.3.conv1.weight", "module.base_model.layer3.0.conv1.weight", "module.base_model.layer3.1.conv1.weight", \
				 "module.base_model.layer3.2.conv1.weight", "module.base_model.layer3.3.conv1.weight", "module.base_model.layer3.4.conv1.weight", \
					 "module.base_model.layer3.5.conv1.weight", "module.base_model.layer4.0.conv1.weight", "module.base_model.layer4.1.conv1.weight", \
						 "module.base_model.layer4.2.conv1.weight"]
	storedict = [ "module.base_model.layer1.0.conv1.net.weight", "module.base_model.layer1.1.conv1.net.weight", "module.base_model.layer1.2.conv1.net.weight", \
		"module.base_model.layer2.0.conv1.net.weight", "module.base_model.layer2.1.conv1.net.weight", "module.base_model.layer2.2.conv1.net.weight", \
			"module.base_model.layer2.3.conv1.net.weight", "module.base_model.layer3.0.conv1.net.weight", "module.base_model.layer3.1.conv1.net.weight",\
				 "module.base_model.layer3.2.conv1.net.weight", "module.base_model.layer3.3.conv1.net.weight", "module.base_model.layer3.4.conv1.net.weight", \
					 "module.base_model.layer3.5.conv1.net.weight", "module.base_model.layer4.0.conv1.net.weight", "module.base_model.layer4.1.conv1.net.weight",\
						  "module.base_model.layer4.2.conv1.net.weight"]
	#replace_dict = dict(n,s) for n,s in zip(netdict,storedict)
	# base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
	##base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
	#replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
	#				'base_model.classifier.bias': 'new_fc.bias',
	#				}
	#for k, v in replace_dict.items():
	#	if k in base_dict:
	#		base_dict[v] = base_dict.pop(k)
	#for n,s in zip(netdict, storedict):
		
		#print(checkpoint)
	#	checkpoint[n] = checkpoint.pop(s)
	
	
	net.load_state_dict(checkpoint)
	net = net.cuda()
	print("{0:*^50}".format("checkpoint"))
	##net.load_state_dict(base_dict)
	#print("265")
	# train for one epoch
	######
	#print(trainDataloader._getMode())
	#print(valDataloader._getMode())

	######
	#print("268")
	# evaluate on validation set
	#model = model.load_state_dict(torch.load("/home/ubuntu/backup_kevin/myownTSM_git/checkpoint/TSM_youcook_RGB_resnet50_shift8_blockres_concatAll_conv1d_lr0.00025_dropout0.70_wd0.00050_batch16_segment8_e20_finetune_slice_v1_clipnum500/ckpt_"+str(epoch+1)+".pth.tar"))
	
	#train(data_loader, net, criterion, optimizer, epoch)

	prec1 = validate(data_loader, net, criterion, epoch)

	# remember best prec@1 and save checkpoint
	######is_best = prec1 > best_prec1
	######best_prec1 = max(prec1, best_prec1)
	#tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)

	######output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
	#print(output_best)
	#log_training.write(output_best + '\n')
	#log_training.flush()



	print("test pass")



output = []
#from tensorboardX import SummaryWriter
#writer = SummaryWriter(logdir = "/home/u9210700/myownTSM/featuremap/")
'''
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
'''