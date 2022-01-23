import torch
import numpy as np


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def clip_tensor(inp,min,max):
    result = (inp >= min)*inp + (inp<min)*min
    result = (result <= max)*result + (result>max)*max
    return result


def nptotorch(x):
    if isinstance(x,torch.Tensor):
        return x
    elif isinstance(x,np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    else: torch.tensor(x)

def check_dict_valid(typ, type_dict):
    if not(typ in type_dict): 
        raise ValueError(typ + ' is not supported\n\
                            Supported are: '+ str([types for types in type_dict]))


def FD_Central_CoefficientMatrix(c:list,meshx:int,periodic:bool=False):
    '''
    c is list of FD coefficient
    e.g. for 1st derivative with 2nd accuracy central difference:
    c=[-0.5,0] 
    '''
    if 2*len(c)-1>=meshx: raise ValueError
    acc = len(c)   
    
    tmp=[]
    c.reverse()
    for i in range(acc):
        x = torch.cat((torch.cat((torch.zeros((i,meshx-i)),
                                    c[i]*torch.eye(meshx-i)),dim=0),
                                    torch.zeros((meshx,i))
                                    ),dim=1)
        tmp.append(x)
    re=tmp[0]
    for k in tmp[1:]:
        re+=k+k.T

    if periodic:
        re[:acc,-acc:]=re[acc:2*acc,:acc]
        re[-acc:,:acc]=re[:acc,acc:2*acc]
    return re

def FD_upwind_CoefficientMatrix(c:list,meshx:int,periodic:bool=False):
    '''
    c is list of Backward FD coefficient
    e.g. for 1st derivative with 1st accuracy:
    c=[-1,1] 
    '''
    if len(c)>=meshx: raise ValueError
    acc = len(c)   
    
    tmp=[]
    
    c.reverse()
    for i in range(acc):
        x = torch.cat((torch.cat((torch.zeros((i,meshx-i)),
                                    c[i]*torch.eye(meshx-i)),dim=0),
                                    torch.zeros((meshx,i))
                                    ),dim=1)
        tmp.append(x)

    bre=tmp[0]
    fre=-tmp[0]
    
    for k in tmp[1:]:
        fre+=-k.T
        bre+=k

    if periodic:
        fre[-acc:,:acc]=fre[:acc,acc:2*acc]
        bre[:acc,-acc:]=bre[acc:2*acc,:acc]
    return fre,bre
