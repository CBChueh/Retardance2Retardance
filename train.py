from argparse import ArgumentParser
from torch.utils.data import DataLoader
from Network import Network
from MultiEpochsDataLoader import MultiEpochsDataLoader, LoadfromFolder


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-td','--train-dir',help='train images loc', default = './train/')
    parser.add_argument('-vd','--valid-dir',help='valid images loc', default = './valid/') 
    parser.add_argument('-lr','--learning-rate',help='model learning define',default=0.001,type=float)  # 0.001
    parser.add_argument('-aa','--alpha',help='portion for loss_2nd',default=0.01,type=float)
    parser.add_argument('-b','--batch-size',help='image size per batch', default=16,type=int) # 50
    parser.add_argument('-ne','--nb-epochs',help='Number of epochs',default=100,type=int)
    parser.add_argument('--cuda',help='To use cuda',action='store_true')
    parser.add_argument('--Trueaugmentation',help='To use augmentation and double dataset',action='store_true')
    parser.add_argument('-a','--adam',help='adam parameters [betas,epsilon]',nargs='+',default=[0.9,0.99,1e-8])
    parser.add_argument('-tb','--trainable',help='do you want to train with this dataset',default=True,type=bool)
    parser.add_argument('-ME','--use-multi-epochs-loader',help='use multiple epochs loaders for speed up',action='store_true')

    return parser.parse_args()

if __name__=='__main__':
    params=parse_args()
    print(params)

    params.addtext= ('-R2R' 
                     +'-'+str(params.nb_epochs) 
                     +'-w-'+str(params.alpha))
    print(params.addtext)
    Train_Image_sets=LoadfromFolder(params.train_dir,params=params,Trueaugmentation=params.Trueaugmentation)
    Valid_Image_sets=LoadfromFolder(params.valid_dir,params=params,Trueaugmentation=params.Trueaugmentation)
    if params.use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader
    else:
        loader_class=DataLoader
    Train_Image_loader = loader_class(Train_Image_sets,params.batch_size,shuffle=True,persistent_workers=True,num_workers=8) #30
    Valid_Image_loader = loader_class(Valid_Image_sets,params.batch_size,shuffle=True,persistent_workers=True,num_workers=8,drop_last=True) # 20
    print('Train dataset pairs: '+str(len(Train_Image_sets.imgs)))
    print('Valid dataset pairs: '+str(len(Valid_Image_sets.imgs)))

    n2n=Network(params)
    n2n.train(Train_Image_loader,Valid_Image_loader)

    print('Hello world!')
    