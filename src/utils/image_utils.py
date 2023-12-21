import numpy as np
import torch

def gamma(img):
    return img ** (1/2.2)

def gamma_inv(img):
    return img ** 2.2

def ACESToneMapping(img):
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14

    # img *= adapted_lum
    return (img * (A * img + B)) / (img * (C * img + D) + E)

def ToneMapping(img):
    img = ACESToneMapping(img)
    return img.clip(min=0., max=1.)

def yCbCr2rgb(input_im : torch.Tensor)->torch.Tensor:
    batch_size,_,height,width = input_im.size()
    mat = torch.tensor([[1.164, 1.164, 1.164],
                        [0, -0.392, 2.017],
                        [1.596, -0.813, 0]], device=input_im.device)
    bias = torch.tensor([-16.0/255.0, -128.0/255.0, -128.0/255.0], device=input_im.device)

    batch_size,_,height,width = input_im.size()

    out = input_im.permute(0, 2, 3, 1).reshape(-1, 3)

    out = out + bias

    out = out.mm(mat)

    return out.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)
 
def rgb2yCbCr(input_im : torch.Tensor)->torch.Tensor:
    mat = torch.tensor([[0.257, -0.148, 0.439],
                        [0.504, -0.291, -0.368],
                        [0.098, 0.439, -0.071]], device=input_im.device)
    bias = torch.tensor([16.0/255.0, 128.0/255.0, 128.0/255.0], device=input_im.device)

    batch_size,_,height,width = input_im.size()

    out = input_im.permute(0, 2, 3, 1).reshape(-1, 3)

    out = out.mm(mat)

    out = out + bias

    return out.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2)

def psnr(target : torch.Tensor, ref : torch.Tensor):
    mse = torch.mean((target - ref) ** 2)
    return 10 * torch.log10(1.0 / mse)

def hdr_psnr(target : torch.Tensor, ref : torch.Tensor):
    return psnr(ToneMapping(target), ToneMapping(ref))