from argparse import ArgumentParser
from Network import Network
from MultiEpochsDataLoader import MultiEpochsDataLoader,LoadfromFolder
from torch.utils.data import DataLoader
import os
import json

def parse_args():
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')
    parser.add_argument('-ted','--test-dir',help='test images loc', default = './test/')
    parser.add_argument('--load-ckpt',help='load check point loc', default = './runs/R2R_Pretrained/R2R_Pretrained.pt')
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('-ME','--use-multi-epochs-loader',help='use multiple epochs loaders for speed up',action='store_true')
    return parser.parse_args()
    

if __name__=='__main__':
    params=parse_args()
    params.addtext= ('R2R')
    
    jsonName = os.path.join(os.path.dirname(params.load_ckpt),'R2R_stats.json')
    with open(jsonName,'r') as fd:
         j=json.load(fd)
    print(params)
    Test_Image_sets=LoadfromFolder(params.test_dir,params=params, testmode=True)
    params.trainable=False
    n2n=Network(params)
    if params.use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    else:
        loader_class=DataLoader
    Test_Image_loader = loader_class(Test_Image_sets,1,shuffle=False,persistent_workers=True,num_workers=20)
    n2n.load_model(params.load_ckpt)
    print('Test datasets: '+str(len(Test_Image_sets.imgs)))
    print(params)
    n2n.test(Test_Image_loader)

    print('Hello World!')