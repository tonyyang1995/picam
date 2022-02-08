import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Backbones import resnet_3d, BasicBlock, BottoleneckBlock
from .BaseModel import BaseModel

from torch.autograd import Variable

from captum.attr._core.guided_grad_cam import GuidedGradCam
from captum.attr import IntegratedGradients
from captum.attr import DeepLift
from captum.attr import GradientShap
from captum.attr import DeepLiftShap

class resnet3d(BaseModel):
	def name(self):
		return 'resnet3d'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.criterion = nn.CrossEntropyLoss().cuda()
		self.opt = opt
		if opt.model_depth == 10:
			self.model = resnet_3d(BasicBlock, [1,1,1,1], num_classes=opt.num_classes)
		elif opt.model_depth == 18:
			self.model = resnet_3d(BasicBlock, [2,2,2,2], num_classes=opt.num_classes)
		elif opt.model_depth == 34:
			self.model = resnet_3d(BasicBlock, [3,4,6,3], num_classes=opt.num_classes)
		elif opt.model_depth == 50:
			self.model = resnet_3d(BottoleneckBlock, [3,4,6,3], num_classes=opt.num_classes)
		elif opt.model_depth == 101:
			self.model = resnet_3d(BottoleneckBlock, [3,4,23,3], num_classes=opt.num_classes)
		elif opt.model_depth == 152:
			self.model = resnet_3d(BottoleneckBlock, [3,8,36,3], num_classes=opt.num_classes)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt.lr)

		if len(self.opt.gpu_ids) > 1:
			self.model = torch.nn.DataParallel(self.model).to(opt.device)
		elif len(self.opt_gpu_ids) > 0:
			self.model = self.model.to(opt.device)

		self.gbp = GuidedGradCam(self.model.module, self.model.module.layer4, device_ids=opt.gpu_ids)
		# self.ig = IntegratedGradients(self.model.module)
		self.dl = DeepLift(self.model.module)
		self.gshap = GradientShap(self.model.module)
		# self.dlshap = DeepLiftShap(self.model.module)

	def visualize(self, img, target):
		self.model.eval()
		# attr_gbp = self.gbp.attribute(img, target=target)
		# print(self.imgs.shape, self.labels.shape)
		self.imgs.required_grad = True
		attr_gbp = self.gbp.attribute(self.imgs, target=self.labels, interpolate_mode='trilinear')
		attr_dl = self.dl.attribute(self.imgs, target=self.labels, baselines=self.imgs*0)
		# attr_ig = self.ig.attribute(img, target=target, baselines=img*0)
		attr_gshap = self.gshap.attribute(img, target=target, baselines=img*0)
		# attr_dlshap = self.dlshap.attribute(img, target=target, baselines=img*0)

		return [attr_gbp, attr_dl, attr_gshap]
	
	def set_input(self, input, mode='train'):
		self.imgs = Variable(input['img']).to(self.opt.device)
		if 'masks' in input.keys():
			self.masks = input['masks']
		if mode == 'train':
			self.labels = Variable(input['label']).to(self.opt.device)
			# self.masks = Variable(input['mask'].to(self.opt.device))
			# print(type(self.masks))

	def cal_current_loss(self):
		# print(self.outputs.shape, self.labels.shape)
		self.loss = self.criterion(self.outputs, self.labels)
		return self.loss

	def get_current_loss(self):
		return self.loss

	def forward(self):
		# self.outputs, self.feat = self.model(self.imgs)
		self.outputs = self.model(self.imgs)

	def backward(self):
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()

	def inference_batch(self):
		self.model.eval()
		self.outputs = self.model(self.imgs)
		output = []
		for i in range(self.outputs.size(0)):
			output.append(self.outputs[i].argmax())
		return output

	@torch.no_grad()
	def inference_mask(self):
		self.model.eval()
		# out_probs = []
		# output = self.model(self.imgs)
		# out_probs.append(output.max())

		visualize = torch.zeros(self.imgs.shape)
		for mask in self.masks:
			mask = Variable(mask).to(self.opt.device)
			output = self.model(mask)
			#out_probs.append(output.max())
			idx = (mask > 0)
			empty = torch.zeros(self.imgs.shape)
			# print(output.max()) # batchsize = 1
			empty[idx] = 1 * (output.max().detach().cpu().item())
			visualize = visualize + empty

		return visualize

	def inference(self):
		self.model.eval()
		output = self.model(self.imgs)
		return output.argmax()
		# return output.argmax()

	def optimize_parameters(self):
		self.forward()
		self.loss = self.criterion(self.outputs, self.labels)
		# self.loss = self.criterion(self.outputs, self.labels, self.masks)
		# self.loss = F.cross_entropy(self.outputs, self.labels, self.masks)
		self.backward()


	def roc(self):
		self.model.eval()
		output = self.model(self.imgs)
		return output.sigmoid()
	
	def get_weight(self):
		# self.model.eval()
		# self.set_requireds_grad(self.model, requireds_grad=True)
		# self.imgs = Variable(self.imgs).requires_grad_()
		# output,feat = self.model(self.imgs)

		# pred_0 = output[:, 0]
		# pred_1 = output[:, 1]
		# grad_0 = feat.grad
		# print(grad_0.size())
		# self.model.zero_grad()
		# self.img.grad.zero_()
		# pred_1.backward()
		# grad_1 = self.imgs.grad

		# #print(grad_0.size(), grad_1.size())

		# return grad_0, grad_1, feat
		######################################################################

		self.model.eval()
		self.fmap_pool = {}
		self.grad_pool = {}
		self.handlers = []
		#self.candidate_layers = candidate_layers
	
		def save_fmaps(key):
			def forward_hook(module, input, output):
				if len(output) == 1:
					self.fmap_pool[key] = output.detach()
				else:
					self.fmap_pool[key] = output[0].detach()
			return forward_hook
		
		def save_grads(key):
			def backward_hook(module, grad_in, grad_out):
				self.grad_pool[key] = grad_out[0].detach()
			return backward_hook

		for name, module in self.model.named_modules():
			self.handlers.append(module.register_forward_hook(save_fmaps(name)))
			self.handlers.append(module.register_backward_hook(save_grads(name)))
	
		self.set_requireds_grad(self.model, requireds_grad=True)
		self.imgs = Variable(self.imgs).requires_grad_()
		output = self.model(self.imgs)
		prob = F.softmax(output, dim=1)

		pred_0 = prob[:, 0]
		pred_1 = prob[:, 1]

		pred_0.backward(retain_graph=True)

		# feat = self.fmap_pool['module.layer4.0.relu']
		# grad_0 = self.grad_pool['module.layer4.0.relu']
		feat = self.fmap_pool['module.layer4']
		grad_0 = self.grad_pool['module.layer4']

		self.model.zero_grad()
		pred_1.backward(retain_graph=True)
		# grad_1 = self.grad_pool['module.layer4.0.relu']
		grad_1 = self.grad_pool['module.layer4']

		return -grad_0, -grad_1, feat



class _BaseWrapper(object):
	def __init__(self, model):
		super(_BaseWrapper, self).__init__()
		self.model = model
		self.handlers = []
	
	def _encode_one_hot(self, ids):
		ont_hot = torch.zero_like(self.logits).cuda()
		ont_hot.scatter_(1, ids, 1.0)
	
	def forward(self, img):
		self.logits, feat = self.model(img)
		self.probs = F.softmax(self.logits, dim=1)
		return self.probs.sort(dim=1, descending=True)
	
	def backward(self, ids):
		ont_hot = self._encode_one_hot(ids)
		self.model.zero_grad()
		self.logits.backward(gradient=one_hot, retain_graph=True)
	
	def generate(self):
		raise NotImplementedError
	
	def remove_hook(self):
		for handle in self.handlers:
			handle.remove()

class BackPropagation(_BaseWrapper):
	def forward(self, img):
		self.img = img.requires_grad_()
		return super(BackPropagation, self).forward(self.img)
	
	def generate(self):
		gradient = self.img.grad.clone()
		self.img.grad.zero_()
		return gradient

class GuidedBackPropagation(BackPropagation):
	def __init__(self, model):
		super(GuidedBackPropagation, self).__init__(model)
	
		def backward_hook(module, grad_in, grad_out):
			if isinstance(module, nn.ReLU):
				return (F.relu(grad_in[0]), )
		for module in self.model.named_modules():
			self.handlers.append(module[1].register_backward_hook(backward_hook))
	
class GradCAM(_BaseWrapper):
	def __init__(self, model, candidate_layers=None):
		super(GradCAM, self).__init__(model)
		self.fmap_pool = {}
		self.grad_pool = {}
		self.candidate_layers = candidate_layers
	
		def save_fmaps(key):
			def forward_hook(module, input, output):
				self.fmap_pool[key] = output.detach()
			return forward_hook
		
		def save_grads(key):
			def backward_hook(module, grad_in, grad_out):
				self.grad_pool[key] = grad_out[0].detach()
			return backward_hook

		for name, module in self.model.named_modules():
			if self.candidate_layers is None or name in self.candidate_layers:
				self.handlers.append(module.register_forward_hook(save_fmaps(name)))
				self.handlers.append(module.register_backward_hook(save_grads(name)))
	
	def _find(self, pool, target_layer):
		if target_layer in pool.keys():
			return pool[target_layer]
		
		else:
			raise ValueError('not find layer')
	
	def generate(self, target_layer):
		fmaps = self._find(self.fmap_pool, target_layer)
		grads = self._find(self.grad_pool, target_layer)
		weights = F.adaptive_avg_pool3d(grads,1)
		gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
		gcam = F.relu(gcam)
		return gcam
	