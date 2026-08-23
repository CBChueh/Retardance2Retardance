import torch
import torch.linalg as LA
import math
from scipy import signal

def makeRot(input):
    # input.shape = [batch size, ch, width, height]
    # permute and add dim for matrix proc in torch 
    if input.ndim==4:
        input=torch.permute(input,(0,2,3,1))

    omega=input[...,1:].detach().clone()
    if omega.shape[-1]<3:
        temp=torch.zeros(omega.shape[:3],device=torch.device('cuda'))
        temp=temp[...,None]
        omega=torch.cat((omega,temp),3)

    omega=omega[...,None]

    ret=torch.norm(omega,dim=-2,keepdim=True)
    omega=omega/ret

    if input.ndim<3:
        q=omega[0,:,None]
        u=omega[1,:,None]
        v=omega[2,:,None]
    else:
        q=omega[...,0,:,None]
        u=omega[...,1,:,None]
        v=omega[...,2,:,None]

    w=torch.zeros(q.shape,device=torch.device('cuda'))

    I=torch.eye(3,device=torch.device('cuda'))

    K=torch.cat((torch.cat((w,-v,u),-2),torch.cat((v,w,-q),-2),torch.cat((-u,q,w),-2)),-1)
    KK=torch.matmul(omega,torch.transpose(omega,-1,-2))-I
    Out=K*torch.sin(ret)+KK*(1-torch.cos(ret))+I

    return Out

def makeRot3x3(input):
    # input.shape = [q,u,v]
    # permute and add dim for matrix proc in torch 
    omega=input.detach().clone()
    omega=omega[...,None]
    ret=torch.norm(omega,dim=-2,keepdim=True)
    omega=omega/ret
    q=omega[...,0,:,None]
    u=omega[...,1,:,None]
    v=omega[...,2,:,None]
    w=torch.zeros(q.shape)
    I=torch.eye(3)
    K=torch.cat((torch.cat((w,-v,u),-2),torch.cat((v,w,-q),-2),torch.cat((-u,q,w),-2)),-1)
    KK=torch.matmul(omega,torch.transpose(omega,-1,-2))-I
    Out=K*torch.sin(ret)+KK*(1-torch.cos(ret))+I
    return Out[0,:,:]

def RotateR(input,V):
    A=torch.tensor([[1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,0],
                    [0,0,-1,0,0,0],
                    [0,0,0,0,-1,0],
                    [0,0,0,0,0,1]],dtype=torch.float).transpose(1,0)
    D=torch.tensor([[1,0,0],
                    [0,1,0],
                    [0,0,-1]],dtype=torch.float)
    
    r=input[1:,:,:].detach().clone().permute((1,2,0))
    Rot=torch.matmul(r,A).reshape(input.shape[1],input.shape[2],3,3)
    Rout=torch.matmul(torch.matmul(torch.matmul(torch.matmul(D,V.transpose(1,0)),D),Rot),V)
    return torch.cat((input[0,:,:][None,:,:],Rout[:,:,0,:].permute(2,0,1),Rout[:,:,1,1:].permute(2,0,1),Rout[:,:,-1,-1][None,:,:]),0)

def jVecInnerProd(j1,j2):
    j1n=j1/j1.pow(2).sum(1).sqrt()[:,None,:,:]
    j2n=j2/j2.pow(2).sum(1).sqrt()[:,None,:,:]
    return (1-(j1n*j2n).sum(1).abs()).mean()
def RMatRecover(R):
    # batch * 6 * width * height
    return torch.cat((R[:,0:3,:,:],R[:,1,:,:][:,None,:,:],R[:,3:5,:,:],-R[:,2,:,:][:,None,:,:],-R[:,4,:,:][:,None,:,:],R[:,5,:,:][:,None,:,:]),1)

def RMatInnerProd(R1,R2):
    # batch * 6 * width * height
    MSize=R1.size()
    NewMSize=(MSize[0],3,3,MSize[2],MSize[3])
    R1=torch.permute(RMatRecover(R1).reshape(NewMSize),(0,3,4,1,2))
    R2=torch.permute(RMatRecover(R2).reshape(NewMSize),(0,3,4,1,2))
    return torch.matmul(R1.transpose(-1,-2),R2)

def RotErr(R1,R2):
    err_matrix=torch.transpose(R1,-2,-1)-R2
    err= torch.mean(LA.matrix_norm(err_matrix,ord=2,dim=(-2,-1)))
    return err


def FilterKernal(fwx_i,fwz_i):
   fwx=torch.tensor(fwx_i)
   nx = (((fwx*1.5).round()-1)/2)
   nxx=torch.linspace(-nx,nx,(fwx*1.5).round().int())*2*torch.tensor(2).log().sqrt()/fwx
   fwz=torch.tensor(fwz_i)
   nz = (((fwz*1.5).round()-1)/2)
   nzz=torch.linspace(-nz,nz,(fwz*1.5).round().int())*2*torch.tensor(2).log().sqrt()/fwz
   h=torch.exp(-nxx**2)[None,:]*torch.exp(-nzz**2)[:,None]
   return h

def DOPCal(input):
    # Input: int + Stokes vec without spectral bin
    # Output: DOP
    h=FilterKernal(fwx_i=12,fwz_i=5)
    S1f=torch.tensor(signal.convolve(input,h[None],mode='same'))
    QUVf=S1f[1:,:,:].pow(2).sum(0).sqrt()
    return (QUVf/S1f[0,:,:])
