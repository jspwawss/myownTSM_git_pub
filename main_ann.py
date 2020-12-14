import os
import time
import shutil
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F


from ops.dataset_ann import YouCookDataSetRcg
from ops.models_ann import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, AUC
from ops.temporal_shift import make_temporal_pool
from archs.Resnet_ann_v2 import AttnDecoderRNN

from nltk.translate.bleu_score import sentence_bleu

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
	num_class = 1
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
	args.store_name += '_'.join(
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
	if args.clipnums:
		args.store_name +=  "_clip{}".format(args.clipnums)
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
	#print(type(policies))
	#print(policies)
	#exit()
	train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset or 'nvgesture' in args.dataset else True)

	model = torch.nn.DataParallel(model).cuda()

	

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
		#print(sd.keys())
		#print("*"*50)
		#print(model_dict.keys())
		model_dict.update(sd)
		model.load_state_dict(model_dict)

	if args.temporal_pool and not args.resume:
		make_temporal_pool(model.module.base_model, args.num_segments)

	decoder = AttnDecoderRNN().cuda()
	if args.decoder_resume:
		decoder_chkpoint = torch.load(args.decoder_resume)
		
		decoder.load_state_dict(decoder_chkpoint["state_dict"])
	print(decoder.parameters())
	policies.append({"params":decoder.parameters(), "lr_mult":5, "decay_mult":1, "name": "Attndecoder_weight"})
	cudnn.benchmark = True
	optimizer = torch.optim.SGD(policies,
								args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)
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
	#global trainDataloader, valDataloader, train_loader, val_loader
	trainDataloader = YouCookDataSetRcg(args.root_path, args.train_list,train=True,inputsize=crop_size,hasPreprocess = False,\
			#clipnums=args.clipnums,
			hasWordIndex = True,)
	valDataloader = YouCookDataSetRcg(args.root_path, args.val_list,val=True,inputsize=crop_size,hasPreprocess = False,\
			#clipnums=args.clipnums,
			hasWordIndex = True,)

	#print(trainDataloader._getMode())
	#print(valDataloader._getMode())
	#exit()
	train_loader = torch.utils.data.DataLoader(
		trainDataloader,
		#shuffle=True,



	)
	val_loader = torch.utils.data.DataLoader(
		valDataloader
	)
	index2wordDict = trainDataloader.getIndex2wordDict()
	#print(train_loader is val_loader)
	#print(trainDataloader._getMode())
	#print(valDataloader._getMode())

	#print(trainDataloader._getMode())
	#print(valDataloader._getMode())
	#print(len(train_loader))


	
	#exit()
	# define loss function (criterion) and optimizer
	if args.loss_type == 'nll':
		criterion = torch.nn.NLLLoss().cuda()
	elif args.loss_type == "MSELoss":
		criterion = torch.nn.MSELoss().cuda()
	elif args.loss_type == "BCELoss":
		#print("BCELoss")
		criterion = torch.nn.BCELoss().cuda()
	elif args.loss_type == "CrossEntropyLoss":
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
		#print("265")
		# train for one epoch
		######
		#print(trainDataloader._getMode())
		#print(valDataloader._getMode())
		train(train_loader, model,decoder, criterion, optimizer, epoch, log_training, tf_writer, index2wordDict)
		######
		#print("268")
		# evaluate on validation set
		if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
			prec1 = validate(val_loader, model,decoder, criterion, epoch, log_training, tf_writer, index2wordDict=index2wordDict)

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
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': decoder.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_prec1': best_prec1,
			}, is_best, filename="decoder")
		else:
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_prec1': best_prec1,
			}, False)
			save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': decoder.state_dict(),
				'optimizer': optimizer.state_dict(),
				'best_prec1': best_prec1,
			}, is_best, filename="decoder")
		#break
		print("test pass")

def train(train_loader, model, decoder, criterion, optimizer, epoch, log, tf_writer, index2wordDict):
	#global trainDataloader, valDataloader

	#print(len(train_loader))
	#print(trainDataloader._getMode())
	#print(valDataloader._getMode())
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	BLEUs = AverageMeter()
	
	teacher_forcing_ratio = 0.5
	#losses_extra = AverageMeter()
	#top1 = AverageMeter()
	#top5 = AverageMeter()
	#top1_extra = AverageMeter()
	#top5_extra = AverageMeter()

	if args.no_partialbn:
		model.module.partialBN(False)
	else:
		model.module.partialBN(True)

	# switch to train mode
	model.train()
	#print("308")
	end = time.time()
	#print("in train")
	with open(os.path.join(args.root_model, args.store_name, "train.txt"), "a+") as txt:
		txt.write("epoch\t"+str(epoch)+"\n")
	for i, data in enumerate(train_loader):
		loss = 0
		#print("data fetch finish ",i)
		

		#print(data)
		#if isinstance(data, bool): #img DNE
	#		continue
		if not data:
			continue
		data_time.update(time.time() - end)


		#print(data[0])
		#print(data[1])
		[URL, id, label, clips] = data	

		input_video=torch.tensor([],dtype=torch.float).cuda()
		input_caption=torch.tensor([],dtype=torch.long).cuda()
		#cap_nums = []

		for clip in range(0,len(clips),max(len(clips)//50,1)):
			#print(type(input_video.size()))
			#print(input_video.size())
			if list(input_video.size()) != [0]:
				if input_video.size()[1] > 50:
					break
			video = clips[clip][0]
			caption = clips[clip][1]
			input_video = torch.cat((input_video, video.float().cuda()),1)
			input_caption = torch.cat((input_caption, caption.long().cuda()),0)
			#cap_nums.append(clip[2])

	
		#print(input_video.size())
		#print(input_caption.size())

		input_video = input_video.view(1,-1,3,crop_size,crop_size)
		input_caption = input_caption.view(1,-1,input_caption.size()[-1],1)
		#print(input_video.size())
		#print(input_caption.size())

		target = label.cuda()
		input_video_var = torch.autograd.Variable(input_video)
		input_caption_var = torch.autograd.Variable(input_caption)
		target_var = torch.autograd.Variable(target)

		# compute output
		wExtraLoss = 1 if args.prune == '' else 0.1
		#print("input video size=",input_video.size())
		encoder_output = model(input_video_var, input_caption_var) #size=(1, frames, 2048)
		
		#print("encoder_output size=",encoder_output.size())
		decoder_outputs = []
		decoder_input = torch.tensor([[0]]).cuda() #SOS
		decoder_hidden = decoder.initHidden()
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
		#if False:
		if use_teacher_forcing:
			print("use_teacher_forceing")
			#print("target size=,",target.size())
		
			#print("target= ", target)
			for tar in target[0]:
				decoder_output, decoder_hidden , _ = decoder(decoder_input, decoder_hidden, encoder_output)
				#print("tar = ",tar)
				#print("tar size = ",tar.size())
				decoder_output_word = decoder_output.argmax().cpu()
				decoder_outputs.append(decoder_output_word)
				#print("decoder_output_word=",decoder_output_word)
				#ttar = torch.zeros(17469, dtype=torch.long).unsqueeze(0).cuda()
				#ttar[0,tar.data] = torch.tensor(1, dtype=torch.long).cuda()
				#print("ttar size={}, decoder_output size={}".format(ttar.size(), decoder_output.size()))
				#print(decoder_output)
				#print(args.loss_type)
				#print(criterion)
				loss += criterion(decoder_output,tar.unsqueeze(0))
				decoder_input = tar  # Teacher forcing
		else:
			print("without teacher forcing")
			# Without teacher forcing: use its own predictions as the next input
			for tar in target[0]:
				decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
				decoder_output_word = decoder_output.argmax().cpu()
				decoder_outputs.append(decoder_output_word)
				topv, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze().detach()  # detach from history as input

				loss += criterion(decoder_output, tar.unsqueeze(0))
				#print("-"*50)
				#print("decoder_input item=",decoder_input.item())
				#print(loss)
				if decoder_input.item() == 1:
					break
		print("{0:*^50}".format("loss "+ str(loss)))
		
		#exit()
		#print("target)var=",target_var.size())
		#print(output)
		#print(target_var)
		#print("label size=",label.size())
		#print(output.size())
		#print(target_var.size())
		#loss_main = criterion(output.squeeze(), target_var.squeeze())
		#loss_main = 0
		#lloss = torch.sub(output, target_var).squeeze()
		#for l in lloss:
		#	loss_main += l**2
		#extra_loss = criterion(extra, target_var)*wExtraLoss
		#loss = loss_main + extra_loss
		#loss = loss_main
		#print("loss=",loss)
		losses.update(loss.item(), )
		#print(decoder_outputs)
		#print(decoder_outputs[0].item())
		decoder_sentence = [index2wordDict[index.item()] for index in decoder_outputs]
		print(decoder_sentence)
		#print(target.size())
		
		target_sentence = [index2wordDict[index.item()] for index in target[0]]
		
		print(target_sentence)
		bleu = sentence_bleu(target_sentence, decoder_sentence ) #default weight is [0.25,0.25,0.25,0.25] = bleu-4
		print("bleu =",bleu)

		BLEUs.update(bleu)
		

		# compute gradient and do SGD step
		st = time.time()
		loss.backward()
		#print("{0:*^50}".format("after backward\t"+str(time.time()-st)))
		if args.clip_gradient is not None:
			total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
		st = time.time()
		optimizer.step()
		#print("{0:*^50}".format("after step\t"+str(time.time()-st)))
		st = time.time()
		optimizer.zero_grad()
		#print("{0:*^50}".format("after zero_grad\t"+str(time.time()-st)))
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		txtoutput = ('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  "output/target{output}/{target}"
					  'bleu {bleu.val:.4f} ({bleu.avg:.3f})\t'
					  .format(
				epoch, i, len(train_loader), batch_time=batch_time,output=decoder_sentence, target=target_sentence,
				data_time=data_time, loss=losses,  bleu=BLEUs, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
		if i % args.print_freq == 0:
			
			print(txtoutput)
			log.write(txtoutput + '\n')
			log.flush()
		with open(os.path.join(args.root_model, args.store_name, "train.txt"), "a+") as txt:
			txt.write(txtoutput+"\n")
		print("**"*50)
		#break
		#exit()

	tf_writer.add_scalar('loss/train', losses.avg, epoch)
	#tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
	#tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
	tf_writer.add_scalar('auc/train', BLEUs.avg, epoch)
	tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, decoder, criterion, epoch, log=None, tf_writer=None, index2wordDict=None):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	#losses = AverageMeter()
	BLEUs = AverageMeter()
	
	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		with open(os.path.join(args.root_model, args.store_name, "val.txt"), "a+") as txt:
			txt.write("epoch\t"+str(epoch)+"\n")
		for i, data in enumerate(val_loader):
			loss = 0
			data_time.update(time.time() - end)
		#if isinstance(data, bool):
		#		continue
			#if not torch.any(data):
		#	continue
			if not data:
				continue
			try:
				[URL, id, label, clips] = data	
			except Exception as e:
				print(e)
				print(data)
				with open("errr.txt","a+") as txt:
					txt.write(str(data))
			input_video=torch.tensor([],dtype=torch.float).cuda()
			input_caption=torch.tensor([],dtype=torch.long).cuda()
			#cap_nums = []
			for clip in range(0,len(clips),max(len(clips)//50,1)):
				if list(input_video.size()) != [0]:
					
					if input_video.size()[1] > 50:
						break
				video = clips[clip][0]
				caption = clips[clip][1]
				input_video = torch.cat((input_video, video.float().cuda()),1)
				input_caption = torch.cat((input_caption, caption.long().cuda()),0)
				#cap_nums.append(clip[2])			#for recording filename in result

			input_video = input_video.view(1,-1,3,crop_size,crop_size)
			input_caption = input_caption.view(1,-1,input_caption.size()[-1],1)
			#print(input_video.size())
			#print(input_caption.size())

			target = label.cuda()
			input_video_var = torch.autograd.Variable(input_video)
			input_caption_var = torch.autograd.Variable(input_caption)
			target_var = torch.autograd.Variable(target)

			# compute output
			wExtraLoss = 1 if args.prune == '' else 0.1
			encoder_output = model(input_video_var, input_caption_var) #size=(1, frames, 2048)
			decoder_outputs = []
			decoder_input = torch.tensor([[0]]).cuda() #SOS
			decoder_hidden = decoder.initHidden()

			print("without teacher forcing")
			# Without teacher forcing: use its own predictions as the next input
			while decoder_input.squeeze().item != 1 and len(decoder_outputs)<64:	
				decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
				decoder_output_word = decoder_output.argmax().cpu()
				decoder_outputs.append(decoder_output_word)
				topv, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze().detach()  # detach from history as input

				#loss += criterion(decoder_output, tar.unsqueeze(0))
				print("-"*50)
				print("decoder_input item=",decoder_input.item())
				#print(loss)
				if decoder_input.item() == 1:
					break
			
			#print("output size=",output.size())
			#print("target)var=",target_var.size())
			#print(output)
			#print(target_var)

			decoder_sentence = [index2wordDict[index.item()] for index in decoder_outputs]
			print(decoder_sentence)
			#print(target.size())
			
			target_sentence = [index2wordDict[index.item()] for index in target[0]]
			
			print(target_sentence)
			bleu = sentence_bleu(target_sentence, decoder_sentence ) #default weight is [0.25,0.25,0.25,0.25] = bleu-4
			#print("bleu =",bleu)

			BLEUs.update(bleu)
			

			#auc = AUC(output.squeeze(), target_var.squeeze())
			#print("auc=",auc)

			#AUCs.update(auc)

			#losses.update(loss.item(), )


			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			txtoutput = ('Test: [{0}/{1}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'BLEU {top5.val:.3f} ({top5.avg:.3f}\t'
						"output/target {output}/{target}"
						).format(
				i, len(val_loader), batch_time=batch_time,output=decoder_sentence,target=target_sentence,
					top5=BLEUs)	
			if i % args.print_freq == 0:
				print(txtoutput)
				if log is not None:
					log.write(txtoutput + '\n')
					log.flush()


			with open(os.path.join(args.root_model, args.store_name, "val.txt"), "a+") as txt:
				txt.write(txtoutput+"\n")

			#break

		txtoutput = ('Testing Results: BLEUs {auc.avg:.3f} '.format(auc=BLEUs))
		with open(os.path.join(args.root_model, args.store_name, "val.txt"), "a+") as txt:
			txt.write(txtoutput+"\n")	

	print(txtoutput)
	if log is not None:
		log.write(txtoutput + '\n')
		log.flush()

	if tf_writer is not None:
		#tf_writer.add_scalar('loss/test', losses.avg, epoch)
		#tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
		#tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
		tf_writer.add_scalar('bleu/test', BLEUs.avg, epoch)
	return BLEUs.avg


def save_checkpoint(state, is_best, filename=None,):
	if filename == None:
		filename = '%s/%s/ckpt_%s.pth.tar' % (args.root_model, args.store_name, state['epoch'])
	else:
		filename = '%s/%s/%s_ckpt_%s.pth.tar' % (args.root_model, args.store_name, filename, state['epoch'])
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

