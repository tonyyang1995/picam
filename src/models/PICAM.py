import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Backbones import resnet_trunk, BasicBlock, BottoleneckBlock
from .BaseModel import BaseModel
import copy
from torch.autograd import Variable

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    raise RuntimeError('activation should be relu/gelu')

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory, reference, query_pos, memory_pos):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                        key=self.with_pos_embed(memory, memory_pos),
                                        value=reference)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
    
    def forward(self, tgt, memory, reference, querypos, memorypos):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, reference, querypos, memorypos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output # (N * b * c)


class cvt(nn.Module):
    def __init__(self, latent_dim, nhead, dim_feedforward, dropout=0.1, activation='relu', num_decoder_layers=6):
        super(cvt, self).__init__()
        decoder_layer = TransformerDecoderLayer(latent_dim, nhead, dim_feedforward, dropout, activation)
        decoder_norm = nn.LayerNorm(latent_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self._reset_parameters()

        self.d_model = latent_dim
        self.nhead = nhead
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, key, value, querypos, memorypos):
        hs = self.decoder(query, key, value, querypos, memorypos)
        hs = hs.permute(1,2,0) # b*c*N
        return hs

class picam(nn.Module):
    def __init__(self, opt):
        super(picam, self).__init__()
        self.latent_dim = 512
        self.query_dim = 170
        self.nhead = 8
        self.feed_forward = 2048
        self.trunk = resnet_trunk(BasicBlock, [1,1,1,1])
        self.key2query = nn.Sequential(
            nn.Linear(6*7*6, self.query_dim),
            nn.ReLU(True),
            nn.Linear(170, self.query_dim),
            nn.ReLU(True)
        )
        self.qpos = nn.Embedding(self.query_dim, self.latent_dim)
        self.kpos = nn.Embedding(6*7*6, self.latent_dim)
        self.neck = cvt(self.latent_dim, self.nhead, self.feed_forward)
        # mlp head
        self.head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, opt.num_classes)
        ) 
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.qpos.weight)
        nn.init.uniform_(self.kpos.weight)

    def forward(self, img, template):
        key_out = self.trunk(img)
        value_out = self.trunk(template)
        # print(key_out.shape)

        bs, c, d, h, w = key_out.shape
        key = key_out.view(bs, c, (d*h*w))
        value = value_out.view(bs, c, (d*h*w))

        query = self.key2query(key)
        i = torch.arange((d*h*w), device=key.device)
        key_pos = self.kpos(i).unsqueeze(0).repeat(bs, 1, 1) # b * N * c
        key = key.permute(2,0,1) # N * b * c
        value = value.permute(2,0,1)
        key_pos = key_pos.permute(1,0,2)


        i = torch.arange(self.query_dim, device=query.device)
        query_pos = self.qpos(i).unsqueeze(0).repeat(bs, 1, 1)

        query = query.permute(2, 0, 1) # N * b * c
        query_pos = query_pos.permute(1,0,2)
        hs = self.neck(query, key, value, query_pos, key_pos)
        hs = hs.mean(dim=2) # b * c * n -> b * c
        out = self.head(hs)
        return out, key_out, value_out

class PIcam(BaseModel):
    def name(self):
        return "PIcam"
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.model = picam(opt)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opt.lr)

        if len(self.opt.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model).to(opt.device)
        elif len(self.opt.gpu_ids) > 0:
            self.model = self.model.to(opt.device)
        
        self.celoss = nn.CrossEntropyLoss().cuda()
        self.l1loss = nn.SmoothL1Loss().cuda()
        self.loss_stat = {}

    def set_input(self, input, mode='train'):
        self.imgs = Variable(input['img']).to(self.opt.device)
        self.template = Variable(input['template']).to(self.opt.device)
        if mode == 'train':
            self.labels = Variable(input['label']).to(self.opt.device)
    
    def forward(self):
        self.model.train()
        # print(self.imgs.shape)
        self.outputs, self.key, self.value = self.model(self.imgs, self.template)
        # print(self.outputs.shape)

    def inference(self):
        self.model.eval()
        self.outputs, self.key, self.value = self.model(self.imgs, self.template)
        return self.outputs.argmax(dim=1)

    def backward(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

    def compute_loss(self):
        self.clsloss = self.celoss(self.outputs, self.labels)
        self.conloss = self.l1loss(self.key, self.value)
        alpha = 1
        beta = 50
        self.loss = alpha * self.clsloss + beta * self.conloss
        self.loss_stat['loss'] = self.loss
        self.loss_stat['clsloss'] = self.clsloss * alpha
        self.loss_stat['conloss'] = self.conloss * beta
    
    def optimize_parameters(self):
        self.forward()
        self.compute_loss()
        self.backward()
    
    def get_current_loss(self):
        return self.loss_stat