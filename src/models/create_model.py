import torch
from torch import nn

def CreateModel(opt):
	if opt.model_name == 'resnet3d':
		from .ResNet3D import resnet3d
		model = resnet3d()
	elif opt.model_name == 'picam':
		from .PICAM import PIcam
		model = PIcam()
	else:
		raise NotImplementedError("model depth should be in [10, 18, 34, 50, 101, 152, 200], but got %d" % (opt.model_depth))

	model.initialize(opt)
	print(opt.gpu_ids)
	if len(opt.gpu_ids) > 0:
		model = model.cuda()
	if len(opt.gpu_ids) > 1:
		model = nn.DataParallel(model)
	return model