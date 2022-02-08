import os, sys
import numpy as np
import torch

# set up an abstract class for all network strucutres
class BaseModel(torch.nn.Module):
	def name(self):
		return 'BaseModel'

	def initialize(self, opt):
		self.opt = opt

	def set_input(self, input):
		pass

	def forward(self):
		pass

	def inference(self):
		pass

	def save(self, path, epoch):
		name = self.opt.name
		dirs = os.path.join(self.opt.checkpoint_dir, name)
		if not os.path.exists(dirs):
			os.makedirs(dirs)
		path = os.path.join(self.opt.checkpoint_dir, name, name+'_'+str(epoch)+'.pth')
		torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, path)

	def load(self, model_path):
		from collections import OrderedDict
		new_state_dict = OrderedDict()
		state_dict = torch.load(model_path)

		if 'optimizer_state_dict' in state_dict:
			# print(state_dict['optimizer_state_dict'])
			# self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
			if len(self.opt.gpu_ids) > 1:
				for k, v in state_dict['model_state_dict'].items():
					if not 'module.' in k:
						name = 'module.' + k
						new_state_dict[name] = v
					else:
						new_state_dict[k] = v
			elif len(self.opt.gpu_ids) == 1:
				for k, v in state_dict['model_state_dict'].items():
					if 'module.' in k:
						name = k[7:]
						new_state_dict[name] = v
					else:
						new_state_dict[k] = v
			self.model.load_state_dict(new_state_dict)
			return
		else:
			# means not store optimizer info
			if len(self.opt.gpu_ids) > 1:
				for k,v in state_dict.items():
					if not 'module.' in k:
						name = 'module.' + k
						new_state_dict[name] = v
					else:
						new_state_dict[k] = v
			elif len(self.opt.gpu_ids) == 1:
				for k, v in state_dict.items():
					if 'module.' in k:
						name = k[7:]
						new_state_dict[name] = v
					else:
						new_state_dict[k] = v
			self.model.load_state_dict(new_state_dict)

	def resolve_version(self):
		import torch._utils
		try:
			torch._utils._rebuild_tensor_v2
		except AttributeError:
			def _rebuild_tensor_v2(storage, storage_offset, size, stride, requireds_grad, backward_hooks):
				tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
				tensor.requireds_grad = requireds_grad
				tensor._backward_hooks = backward_hooks
				return tensor

			torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

	def set_requireds_grad(self, nets, requireds_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requireds_grad = requireds_grad