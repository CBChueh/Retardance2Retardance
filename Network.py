import os
import time
import numpy as np
import json

import torch
import torch.nn as nn
from torchsummary import summary
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import PSProcessing as PSP
from MultiEpochsDataLoader import LoadfromFolder

class Network(object):
    def __init__(self,params):
        self.pam=params
        self._compile()

    def _compile(self):
        self.model=R2R()
        if self.pam.trainable:
            self.optim=Adam(self.model.parameters(),
                            lr=self.pam.learning_rate,
                            betas=self.pam.adam[:2],
                            eps=self.pam.adam[2])
            self.schedular=lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.pam.nb_epochs/4,factor=0.5)
        self.loss = nn.L1Loss()
        self.loss2=nn.MSELoss()
        self.use_cuda = torch.cuda.is_available() and self.pam.cuda
        if self.use_cuda:
            self.model=self.model.cuda()
        if self.pam.trainable:
            self.loss=self.loss.cuda()
            self.loss2=self.loss2.cuda()

    def train(self,train_Datasets,val_Datasets):
        self.model.train(True)
        alpha=self.pam.alpha
        import socket
        from datetime import datetime
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        runName = "./runs"
        logDir = os.path.join(runName,current_time+"_"+self.pam.addtext)
        writer_train = SummaryWriter(log_dir=logDir+'_train')
        writer_valid = SummaryWriter(log_dir=logDir+'_valid')
        ckptDir=writer_train.log_dir
        stats=vars(self.pam)
        num_batches = len(train_Datasets)
        print('Number of batches: '+str(num_batches))
        interupted =False
        summary(self.model,input_size=(7,1024,512),batch_size=num_batches)

        stats['train_loss']=[]
        stats['valid_loss']=[]

        for epoch in range(self.pam.nb_epochs):
            Loss_train_batch_val=0
            Loss_train_batch_sum=0
            Time_batch_sum=0
            Time_epoch_start=time.time()
            for batch_idx, (source_cpu, target_cpu) in enumerate(train_Datasets):
                Time_batch_start=time.time()
                if self.use_cuda:
                    source=source_cpu.cuda()
                    target=target_cpu.cuda()
                Output=self.model(source)
                Loss_train_batch=self.LossCalcu(Output,target)
                Loss_train_batch_val=Loss_train_batch.item()
                Loss_train_batch_sum+=Loss_train_batch_val
                self.optim.zero_grad(set_to_none=True)
                Loss_train_batch.backward()
                self.optim.step()

                dec=int(np.ceil(np.log10(self.pam.nb_epochs)))
                print('\rEpoch{:>{dec}d} / {:>{dec}d} '.format(epoch+1,self.pam.nb_epochs,dec=str(dec)),end=' ')
                print(self.progressbar(epoch + 1,self.pam.nb_epochs),end=', ')
                dec = int(np.ceil(np.log10(num_batches)))
                print('Batch {:>{dec}d} /  {:>{dec}d}'.format(batch_idx + 1, num_batches,dec=str(dec)),end=' ')
                print(self.progressbar(batch_idx+1,num_batches),end=', ')
                Time_batch_end=time.time()
                Time_batch=Time_batch_end-Time_batch_start
                Time_batch_sum+=Time_batch
                print('Train-[L: {:>1.3f}]'.format(Loss_train_batch_val), end='       \r')

            if self.use_cuda:
                source = source.detach()
                target = target.detach()
            Time_epoch=time.time()-Time_epoch_start


            print('\nValidating model!',end='\r')
            Time_valid_start=time.time()
            Loss_valid, output_source_denoised=self.eval(val_Datasets)
            Time_valid=time.time()-Time_valid_start

            Loss_train_batch_avg=Loss_train_batch_sum/num_batches

            print('Epoch {:d} / {:d} Train loss_batch avg: {:>1.5f} | Valid loss: {:>1.5f} | Train time: {:>1.4f} | Valid time: {:>1.4f} | Estimated remaining time (min): {:>1.4f}'
                .format(epoch + 1, self.pam.nb_epochs,
                        Loss_train_batch_avg, Loss_valid,
                        Time_epoch,Time_valid,(Time_epoch+Time_valid)*(self.pam.nb_epochs-epoch-1)/60))
            stats['train_loss'].append(Loss_train_batch_avg)
            stats['valid_loss'].append(Loss_valid)
            if epoch % 5 == 0:
                self.save_model(stats,epoch,ckptDir,epoch+1==self.pam.nb_epochs or interupted)

            writer_train.add_scalar("Loss_tot/epoch",Loss_train_batch_avg,epoch)
            writer_valid.add_scalar("Loss_tot/epoch",Loss_valid,epoch)


            if epoch % 10 == 0:
                e,h,w = np.shape(output_source_denoised)
                extraImg=torch.ones([h,w],dtype=torch.float32).cuda()
                    
                writer_valid.add_image('Total/Source_Target_Output',torch.cat((torch.cat((output_source_denoised[0,:,:],extraImg),-2),
                                                    (torch.cat((output_source_denoised[1,:,:],output_source_denoised[4,:,:]),-2)+1)/2,
                                                    (torch.cat((output_source_denoised[2,:,:],output_source_denoised[5,:,:]),-2)+1)/2,
                                                    (torch.cat((output_source_denoised[3,:,:],output_source_denoised[6,:,:]),-2)+1)/2),-1), epoch, dataformats='HW')
        print('Training finished!')
        writer_train.flush()
        writer_valid.flush()


    def eval(self, val_Datasets):
        self.model.train(False)
        loss_val_sum=0
        num_batches=len(val_Datasets)
        with torch.no_grad(): # Disable gradient computation and reduce memory consumption.
            for batch_idx, (source_cpu, target_cpu) in enumerate(val_Datasets):
                if self.use_cuda:
                    source = source_cpu.cuda()
                    target = target_cpu.cuda()

                Output = self.model(source[:,:7,:,:])
                Loss_eval_batch=self.LossCalcu(Output,target)
                    
                loss_val_sum+=Loss_eval_batch.item()
                dec = int(np.ceil(np.log10(num_batches)))
                print('Eval Batch {:>{dec}d} /  {:>{dec}d}'.format(batch_idx + 1, num_batches,dec=str(dec)),end=' ')
                print(self.progressbar(batch_idx+1,num_batches),end='\r')
                if batch_idx == 0:
                    Prev_source_denoised=Output[0,:,:,:]
        if self.use_cuda:
            source = source.detach()
            target = target.detach()
        loss_val_avg=loss_val_sum/num_batches
        return loss_val_avg, Prev_source_denoised

    def test(self, test_Datasets):

        self.model.train(False)
        Writerdir=self.pam.load_ckpt[:self.pam.load_ckpt.find('_train')]+'_test'
        n=0
        while os.path.isdir(Writerdir+ '-' + str(n)):
            n+=1
        Writerdir=Writerdir+'-'+str(n)

        writer = SummaryWriter(log_dir=Writerdir)
        if 'alph' not in self.pam :
            self.pam.alpha=0
        stats = vars(self.pam)
        stats['test_loss']=[]
        stats['test_AnoErr']=[]

        denoised_dir=self.CreateSavingFolder()

        num_batches=len(test_Datasets)
        batch_time_sum=0
        with torch.no_grad(): # Disable gradient computation and reduce memory consumption.
            for batch_idx, (source, target) in enumerate(test_Datasets):
                batch_time_start=time.time()
                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                Output = self.model(source[:,:7,:,:])

                Loss_test_batch=self.LossCalcu(Output,target)

                writer.add_scalar("Loss/batch",Loss_test_batch.item(),batch_idx)
                stats['test_loss'].append(Loss_test_batch.item())

                img_name = test_Datasets.dataset.imgs[batch_idx]
                fname = os.path.splitext(img_name)[0]
                self.save_results(source,os.path.join(denoised_dir, f'{fname}-source'))
                self.save_results(target,os.path.join(denoised_dir, f'{fname}-target'))
                self.save_results(Output,os.path.join(denoised_dir, f'{fname}-denoised'))

                dec = int(np.ceil(np.log10(num_batches)))
                print('Test data {:>{dec}d} /  {:>{dec}d}'.format(batch_idx + 1, num_batches,dec=str(dec)),end=' ')
                print(self.progressbar(batch_idx+1,num_batches),end=', ')
                batch_time_end=time.time()
                batch_time_new=batch_time_end-batch_time_start
                batch_time_sum+=batch_time_new
                print('Time: {:>1.4f} | Estimated remaining time (min): {:>1.4f}'
                    .format(batch_time_new,(batch_time_new)*(num_batches-batch_idx-1)/60),end='\t\t\r')

        fname_dict='{}/R2R_stats.json'.format(writer.log_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)
        print('Testing finished!')
        writer.flush()

    def LossCalcu(self,Output,target):
        Loss_batch_2nd=self.Loss_2nd(Output)
        Loss_batch_1st=self.Loss_1st(Output,target)
        return Loss_batch_1st+Loss_batch_2nd*self.pam.alpha

    def Loss_1st(self,Output,target):
        w=torch.tensor([1,1,2,2,1,2,1],device=torch.device('cuda'))[None,:,None,None]
        return self.loss(Output*w,target[:,:Output.shape[1]]*w)
    
    def Loss_2nd(self,k):
        RTR=PSP.RMatInnerProd(k[:,1:,:,:],k[:,1:,:,:])
        Ivar=RTR.diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)[:,:,:,None,None]/3*torch.eye(3,device=torch.device('cuda'))
        return self.loss2(RTR,Ivar)

    def progressbar(self, ind, total):
        dec = int(np.ceil(np.log10(total)))
        bar_size = 8
        progress = ind / total
        fill = int(np.ceil(progress * bar_size))
        return('[{}{}]'.format('=' * fill + '>', ' ' * (bar_size - fill),dec=str(dec)))
    
    def save_results(self,Img, Name):
        with open(Name+'.bin', "ab") as f:
            for data in Img.detach().cpu().numpy():
                data.tofile(f)

    def save_model(self,stats,epoch,ckpt_dir,END=False):
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)

        if not END:
            fname_model='{}/R2R-epoch-{}.pt'.format(ckpt_dir,epoch+1)
        else:
            fname_model='{}/R2R-epoch-lastest.pt'.format(ckpt_dir)

        print('Saving ckpt --> {}'.format(fname_model))
        torch.save(self.model.state_dict(),fname_model)
        fname_dict='{}/R2R_stats.json'.format(ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

    def load_model(self, ckpt_fname):
        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

        print(self.model)
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

    def CreateSavingFolder(self):
        [head,tail]=os.path.split(self.pam.load_ckpt)
        [head, tail0]=os.path.split(head)
        Output_dir=os.path.join('./Results/','Results_'+tail0+'_'+tail)
        n=0
        while os.path.isdir(Output_dir+ '-' + str(n)):
            n+=1
        Output_dir=Output_dir+'-'+str(n)
        print(Output_dir)
        os.makedirs(Output_dir)
        return Output_dir

class R2R(nn.Module):
    def __init__(self):
        super(R2R,self).__init__()
        self._block1 = nn.Sequential(
            nn.Conv2d(7, 48, 7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))

        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))

        self._block6 = nn.Sequential(
            nn.Conv2d(96 + 7, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 7, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder                                       # in_ch * 512
        pool1 = self._block1(x)                         # 48 * 256
        pool2 = self._block2(pool1)                     # 48 * 128
        pool3 = self._block2(pool2)                     # 48 * 064

        # Decoder
        upsample4 = self._block3(pool3)               # 48 * 128
        concat4 = torch.cat((upsample4, pool2), dim=-3) # 96 * 128
        upsample3 = self._block4(concat4)               # 96 * 256
        concat3 = torch.cat((upsample3, pool1), dim=-3) # 144 * 256
        upsample2 = self._block5(concat3)               # 96 * 512
        concat1 = torch.cat((upsample2, x), dim=-3)     # 100 * 512

        # Final activation
        return self._block6(concat1)                    # out_ch * 512
