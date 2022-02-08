def get_dataset(opt, train=1):
	dataset = None
	name = opt.dataset_mode
	if name == 'bet':
		from .BetLoader import SingleModelDataset
		dataset = SingleModelDataset(opt, train)
	elif name == 'picam':
		from .PicamLoader import PicamDataset
		dataset = PicamDataset(opt, train)
	return dataset