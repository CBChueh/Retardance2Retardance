
import os
import numpy as np
import math

import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor

import PSProcessing as PSP

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadfromFolder(Dataset):
    def __init__(self, imgs_dir, params, Trueaugmentation=False, testmode=False):
        self.imgs_dir=imgs_dir
        self.p=params
        fileend=".bin"
        Specfile="ource"
        if Trueaugmentation:
            self.imgs=['Augmen_'+f for f in os.listdir(imgs_dir) if f.endswith(fileend) and Specfile not in f and "DOP" not in f]
        else:
            self.imgs=[f for f in os.listdir(imgs_dir) if f.endswith(fileend) and Specfile not in f and "DOP" not in f] 
        self.imgs=self.imgs[:16]
        
        if testmode:
            # find unique target bin, each target bin find match its corresponding source bin
            seen=set()
            imguniq=[]
            self.imgs.sort()
            for img in self.imgs:
                strstart=img.rfind('Bin')
                strpart=img[:strstart]
                if strpart not in seen:
                    imguniq.append(img)
                    seen.add(strpart)
            self.imgs=imguniq


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        ImgName=str(self.imgs[index])
        if ImgName.startswith('Augmen_'):
            ImgName=ImgName[7:]
            TrueAug=True
        else:
            TrueAug=False
        img_path = os.path.join(self.imgs_dir, ImgName)
        img_path2 = os.path.join(self.imgs_dir, ImgName)

        strstart=img_path2.rfind('_Bin')
        strend=img_path2.find('.bin')
        strpart=img_path2[strstart+1:strend]

        img_path2=img_path2.replace(strpart,'Source')

        if len(img_path)>200:
            print('too long')
        if not os.path.exists(img_path2):
            img_path2=img_path2.replace('Source','source')
            if not os.path.exists(img_path2):
                print('Does not exist '+ img_path2)
                print(img_path)

        source = self._read_image(img_path2)
        target = self._read_image(img_path)
        if TrueAug:
            V=PSP.makeRot3x3(math.pi*2*(torch.rand(1,3)*2-1))
            source=PSP.RotateR(source,V)
            target=PSP.RotateR(target,V)

        return source, target
    
    
    
    def _read_image(self,img_path):
        tempimg_array=np.fromfile(img_path,count=7*512*1024,dtype='float32').reshape((7,512,1024), order='C').transpose(2,1,0)
        img_array=np.zeros((1024,512,7),dtype="float32")

        # Normalizated OCT intensity
        img_array[:,:,0]=np.clip((tempimg_array[:,:,0]-55)/55,0,1)
        img_array[:,:,1:]=tempimg_array[:,:,1:] # Ret (rotation) vector preserve original range
        if math.isnan(img_array.sum()):
            print('I got u!!')
        return ToTensor()(img_array)
    
    