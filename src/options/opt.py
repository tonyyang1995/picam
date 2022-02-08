import argparse
import os
import torch

class Options():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
		self.parser.add_argument('--name', type=str, default='PICAM')
		self.parser.add_argument('--model_name', type=str, default='resnet3d', help='resnet3d, autoendoer')
		self.parser.add_argument('--model_depth',type=int, default=10, help='resnet10, 18, 34, 50, 101, 152')
		self.parser.add_argument('--load_model', type=str, default='')

		self.parser.add_argument('--dataroot', type=str, default='dataset/data/labels/supervised/CrossVal0/adnc_train.txt')
		self.parser.add_argument('--train_model', type=int, default=1)
		self.parser.add_argument('--num_classes', type=int, default=2)
		self.parser.add_argument('--batch_size', type=int, default=8)
		self.parser.add_argument('--lr', type=float, default=1e-4)

		self.parser.add_argument('--display_freq', type=int, default=20)
		self.parser.add_argument('--val_freq', type=int, default=1)
		self.parser.add_argument('--save_freq', type=int, default=1)

		self.parser.add_argument('--gpu_ids', type=str, default='0,1')

		self.parser.add_argument('--start_epochs', type=int, default=0)
		self.parser.add_argument('--epochs', type=int, default=20)

		self.parser.add_argument('--dataset_mode', type=str, default='bet', help='singleModality, ROI, Rawdata')
		self.parser.add_argument('--input_channels', type=int, default=1)

		self.parser.add_argument('--gamma', type=float, default=2.0) # parameters for focal loss
		self.parser.add_argument('--alpha', type=float, default=0.25)
		self.parser.add_argument('--ae_input', type=int, default=3)
		self.initialized = True
		return self.parser


	def print_options(self, opt):
		message = ''
		message += '------------------------ OPTIONS -----------------------------\n'
		for k,v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}\n'.format(str(k), str(v), comment)
		message += '------------------------  END   ------------------------------\n'
		print(message)

		# save to the disk
		expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
		if not os.path.exists(expr_dir):
			os.makedirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write(message)
			opt_file.write('\n')

	def parse(self):
		# parse the options, create checkpoint, and set up device
		if not self.initialized:
			self.initialize()

		self.opt = self.parser.parse_args()

		os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
		# set gpu ids
		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)
		if len(self.opt.gpu_ids) > 0:
			self.opt.device = torch.device('cuda')
		else:
			self.opt.device = torch.device('cpu')

		self.print_options(self.opt)
		return self.opt