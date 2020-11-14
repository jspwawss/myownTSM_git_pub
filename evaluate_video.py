# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

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
import cv2

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)

# may contain splits
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=str, default=25)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample as I3D')
parser.add_argument('--twice_sample', default=False, action="store_true", help='use twice sample for ensemble')
parser.add_argument('--full_res', default=False, action="store_true",
					help='use full resolution 256x256 for test as in Non-local I3D')

parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--coeff', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
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

action_list = ['ApplyEyeMakeup','ApplyLipstick','Archery','BabyCrawling','BalanceBeam','BandMarching','BaseballPitch','Basketball','BasketballDunk','BenchPress','Biking','Billiards','BlowDryHair',
			   'BlowingCandles','BodyWeightSquats','Bowling','BoxingPunchingBag','BoxingSpeedBag','BreastStroke','BrushingTeeth','CleanAndJerk','CliffDiving','CricketBowling','CricketShot',
			   'CuttingInKitchen','Diving','Drumming','Fencing','FieldHockeyPenalty','FloorGymnastics','FrisbeeCatch','FrontCrawl','GolfSwing','Haircut','Hammering','HammerThrow','HandstandPushups',
			   'HandstandWalking','HeadMassage','HighJump','HorseRace','HorseRiding','HulaHoop','IceDancing','JavelinThrow','JugglingBalls','JumpingJack','JumpRope','Kayaking','Knitting','LongJump',
			   'Lunges','MilitaryParade','Mixing','MoppingFloor','Nunchucks','ParallelBars','PizzaTossing','PlayingCello','PlayingDaf','PlayingDhol','PlayingFlute','PlayingGuitar','PlayingPiano',
			   'PlayingSitar','PlayingTabla','PlayingViolin','PoleVault','PommelHorse','PullUps','Punch','PushUps','Rafting','RockClimbingIndoor','RopeClimbing','Rowing','SalsaSpin','ShavingBeard',
			   'Shotput','SkateBoarding','Skiing','Skijet','SkyDiving','SoccerJuggling','SoccerPenalty','StillRings','SumoWrestling','Surfing','Swing','TableTennisShot','TaiChi','TennisSwing',
			   'ThrowDiscus','TrampolineJumping','Typing','UnevenBars','VolleyballSpiking','WalkingWithDog','WallPushups',' WritingOnBoard','YoYo']

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
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		 correct_k = correct[:k].view(-1).float().sum(0)
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


weights_list = args.weights.split(',')
'''
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
'''

data_iter_list = []
net_list = []
modality_list = []
num_class = 0
prune_conv1in_list = {}
prune_conv1out_list = {}
def load_model(weights):
	global num_class
	is_shift, shift_div, shift_place = parse_shift_option_from_log_name(weights)
	if 'RGB' in weights:
		modality = 'RGB'
	elif 'Depth' in weights:
		modality = 'Depth'
	else:
		modality = 'Flow'

	if 'concatAll' in weights:
		concat = "All" 
	elif "concatFirst" in weights:
		concat = "First"
	else:
		concat = ""

	if 'extra' in this_weights:
		extra_temporal_modeling = True

	args.prune = ""

	if 'conv1d' in weights:
		args.crop_fusion_type = "conv1d"
	else:
		args.crop_fusion_type = "avg"

	this_arch = weights.split('TSM_')[1].split('_')[2]
	modality_list.append(modality)
	num_class, args.train_list, val_list, root_path, prefix = dataset_config.return_dataset(args.dataset, modality)

	print('=> shift: {}, shift_div: {}, shift_place: {}'.format(is_shift, shift_div, shift_place))
	net = TSN(num_class, int(args.test_segments) if is_shift else 1, modality,
			  base_model=this_arch,
			  consensus_type=args.crop_fusion_type,
			  img_feature_dim=args.img_feature_dim,
			  pretrain=args.pretrain,
			  is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
			  non_local='_nl' in weights,
			  concat = concat,
			  extra_temporal_modeling = extra_temporal_modeling,
			  prune_list = [prune_conv1in_list, prune_conv1out_list],
			  is_prune = args.prune)

	if 'tpool' in weights:
		from ops.temporal_shift import make_temporal_pool
		make_temporal_pool(net.base_model, args.test_segments)  # since DataParallel

	checkpoint = torch.load(weights)
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

	transform=torchvision.transforms.Compose([
						   cropping,
						   Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
						   ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
						   GroupNormalize(net.input_mean, net.input_std),])

	if args.gpus is not None:
		devices = [args.gpus[i] for i in range(args.workers)]
	else:
		devices = list(range(args.workers))

	net = torch.nn.DataParallel(net.cuda())
	return is_shift, net, transform



def eval_video(video_data, net, this_test_segments, modality, is_shift):
	net.eval()
	with torch.no_grad():
		data = video_data
		#print(data.size())
		batch_size = 1
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

		data_in = data.view(-1, length, data.size(1), data.size(2))
		if is_shift:
			data_in = data_in.view(batch_size * num_crop, this_test_segments, length, data_in.size(2), data_in.size(3))
		rst = net(data_in)
		rst = rst[0].reshape(batch_size, num_crop, -1).mean(1)

		if args.softmax:
			# take the softmax to normalize the output to probability
			rst = F.softmax(rst, dim=1)

		rst = rst.data.cpu().numpy().copy()

		if net.module.is_shift:
			rst = rst.reshape(batch_size, num_class)
		else:
			rst = rst.reshape((batch_size, -1, num_class)).mean(axis=1).reshape((batch_size, num_class))

		return rst

def main():
	print("preparing RGB model...")
	is_shift_RGB, net_RGB, transform = load_model(weights_list[0])

	cap_RGB = cv2.VideoCapture('/home/ubuntu/myownTSM/merge.avi')
	fps = cap_RGB.get(cv2.cv.CV_CAP_PROP_FPS)
	size = (int(cap_RGB.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap_RGB.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
	RGB_out = cv2.VideoWriter('Merge_result.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
	

	frame_count = 0
	frame_list = []
	action_predicted = None
	while True:
		frame_count += 1
		result_RGB, frame_ori_RGB = cap_RGB.read()
		
		if not result_RGB:
			break

		frame_RGB = [Image.fromarray(frame_ori_RGB).convert('RGB')]
		
		if frame_count % 13 == 7:
			frame_list.extend(frame_RGB)
			
		if frame_count % 104 == 0:
			
			frame = transform(frame_list)
			frame_var_RGB = torch.autograd.Variable(frame.cuda())
			start = time.time()
			predict_RGB = eval_video(frame_var_RGB, net_RGB, int(args.test_segments), "RGB", is_shift_RGB)
			
			end = time.time()
			action_predicted = action_list[np.argmax(predict_RGB)]
			model_fps = 8/(end - start)
			print("RGB predict:\t{}, final predict:\t{}, fps:\t{}".format(np.argmax(predict_RGB), action_predicted, model_fps))
			frame_list = []
		
		cv2.rectangle(frame_ori_RGB, (10, 40), (680, 100), (0, 0, 0), -1)
		if action_predicted != None:	
			cv2.putText(frame_ori_RGB, action_predicted + ', fps=' + str(round(model_fps,2)), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.CV_AA)
		
		RGB_out.write(frame_ori_RGB)
		cv2.imshow('RGB', frame_ori_RGB)
		cv2.waitKey(33)
		
	print(frame_count)
	cap_RGB.release()
	RGB_out.release()
			

if __name__ == '__main__':
	main()
