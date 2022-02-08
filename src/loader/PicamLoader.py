import os
import nibabel as nib
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import numpy as np

from .BaseLoader import BaseDataset

class PicamDataset(BaseDataset):
    def __init__(self, opt, train=1):
        self.opt = opt
        if not train:
            opt.dataroot = opt.dataroot.replace('train.txt', 'test.txt')
        print(opt.dataroot)

        self.img_paths, self.labels, self.age, self.sex, self.mmse = self.__get_from_txt(opt.dataroot)
        # self.transform = transforms.Compose([
        # 				transforms.ToTensor(),
        # 				])
        self.transform = transforms.ToTensor()
        self.aal = self.get_aal()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        pve_path = self.img_paths[index].replace('bet.nii.gz', 'bet_pveseg.nii.gz') # 
        
        pve = nib.load(os.path.join('dataset', pve_path)).get_data()
        idx = (pve == 0)
        img = nib.load(os.path.join('dataset', img_path)).get_data()
        img[idx] = 0
        if self.check_nan(img):
            print(img_path)
            self.replace_nan(img)

        img = self.transform(img) # torch to tensor will permute the axis, need to move it back
        img = img.permute((1,2,0))
        template = self.get_mask(img)
        img = img.unsqueeze(0)
        template = template.unsqueeze(0)
        return img, template, label, img_path

    def name(self):
        return 'PicamDataset'

    def __get_from_txt(self, dataroot):
        print(dataroot)
        img_paths, labels = [], []
        ages, sexs, mmses = [], [], []
        with open(dataroot, 'r') as tf:
            lines = tf.readlines()
            for line in lines:
                img_path, label, age, sex, mmse = line.strip().split(' ')
                label = int(label)
                img_paths.append(img_path)
                labels.append(label)
                ages.append(float(age))
                s = 1 if sex == 'M' else 0
                sexs.append(s)
                mmses.append(float(mmse))
        return img_paths, labels, ages, sexs, mmses

    def resize(self, img, a,b,c):
        img = img.float().unsqueeze(0).unsqueeze(0)
        img = F.interpolate(img, size=(a,b,c), mode='trilinear')
        img = img.squeeze(0).squeeze(0)
        return img
    
    def check_nan(self, img):
        nan = np.isnan(img)
        if len(nan[nan==True]) > 0:
            return True
        else:
            return False
    
    def replace_nan(self, img):
        return np.nan_to_num(img)
    
    def get_aal(self):
        aal = nib.load('AAL3v1_1mm.nii.gz').get_data()
        temp = np.zeros((1, 217, 181))
        aal = np.concatenate((temp, aal), axis=0)
        temp = np.zeros((182, 1, 181))
        aal = np.concatenate((temp, aal), axis=1)
        temp = np.zeros((182, 218, 1))
        aal = np.concatenate((temp, aal), axis=2)
        return torch.from_numpy(aal)
    
    def get_mask(self, imgs):
        template = torch.zeros(imgs.shape)
        for i in range(170):
            idx = (self.aal == i)
            template[idx] = imgs[idx]
        
        return template