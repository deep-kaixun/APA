
# import torchvision
import torch.nn as nn
# import torch
import scipy.stats as st
# import timm
import torch.nn.functional as F
import torchvision
import torch
import os
import timm
from efficientnet_pytorch import EfficientNet
# import torch
import cv2
import numpy as np
import PIL.Image
import torch
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
## simple wrapper model to normalize an input image
class WrapperModel(nn.Module):
    def __init__(self, model, mean, std,resize=False):
        super(WrapperModel, self).__init__()
        self.mean = torch.Tensor(mean)
        self.model=model
        self.resize=resize
        self.std = torch.Tensor(std)
    def forward(self, x):
        return self.model((x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None])


def load_model(model):
    home_path = 'model_ckpt/'
    if model == 'ResNet50':
        net = torchvision.models.resnet50()
        net.load_state_dict(torch.load(os.path.join(home_path, 'resnet50-0676ba61.pth')))
    elif model == 'mnv2':
        net = torchvision.models.mobilenet_v2()
        net.load_state_dict(torch.load(os.path.join(home_path, 'mobilenet_v2-b0353104.pth')))
    elif model == 'DenseNet161':
        # https://download.pytorch.org/models/densenet161-8d451a50.pth
        net = torchvision.models.densenet161(pretrained=True)
        # net.load_state_dict(torch.load(os.path.join(home_path, 'densenet161-8d451a50.pth')))
    elif model == 'ResNet152':
        # https://download.pytorch.org/models/resnet152-394f9c45.pth
        net = torchvision.models.resnet152()
        net.load_state_dict(torch.load(os.path.join(home_path, 'resnet152-394f9c45.pth')))
    elif model == 'EF-b7':
        # net = EfficientNet.from_name('efficientnet-b7')
        # loaded_state_dict = torch.load(os.path.join(home_path, 'checkpoints/efficientnet-b7-dcc49843.pth'))
        # net.load_state_dict(loaded_state_dict, strict=True)
        net= EfficientNet.from_pretrained("efficientnet-b7")
    elif model == 'vgg19':
        # https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
        net = torchvision.models.vgg19()
        net.load_state_dict(torch.load(os.path.join(home_path, 'vgg19-dcbb9e9d.pth')))
    elif model == 'inception_v3':
        net = torchvision.models.inception_v3(pretrained=True)
        # net.load_state_dict(torch.load('/youtu-pangu-public/omenzychen/Working/TransferablePatch/checkpoints/inception_v3_google-1a9a5a14.pth'))
    elif model == 'mvit':
        net = timm.create_model('mobilevit_s', pretrained=True, num_classes=1000)
        # net.load_state_dict(torch.load('/youtu-pangu-public/omenzychen/Working/Text2ImageAttack/third_party/mobilevit_s-38a5a959.pth'))
    elif model == 'pvtv2':
        net = timm.create_model('pvt_v2_b2_li', pretrained=True, num_classes=1000)
        # state_dict = torch.load('/youtu-pangu-public/omenzychen/Working/TransferablePatch/checkpoints/pvt_v2_b2_li.pth')
    elif model == 'swint':
        net = timm.create_model('swin_base_patch4_window7_224.ms_in22k_ft_in1k', pretrained=True, num_classes=1000)
    elif model == 'vit':
        net = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True, num_classes=1000)
    return net




def resize_image(input_image, resolution, interpolation=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    if interpolation is None:
        interpolation = cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA
    img = cv2.resize(input_image, (W, H), interpolation=interpolation)
    return img


def vertical_shift(x):
    _, _, w, _ = x.shape
    step = np.random.randint(low = 0, high=w, dtype=np.int32)
    return x.roll(step, dims=2)

def horizontal_shift(x):
    _, _, _, h = x.shape
    step = np.random.randint(low = 0, high=h, dtype=np.int32)
    return x.roll(step, dims=3)

def vertical_flip( x):
    return x.flip(dims=(2,))


def random_size(x):
    img_size=x.shape[-1]
    rnd = int(np.random.randint(int(img_size*0.9),img_size,size=1)[0])
    return F.interpolate(x.clone(), size=(rnd,rnd))

def horizontal_flip(x):
    return x.flip(dims=(3,))

def rotate180(x):
    return x.rot90(k=2, dims=(2,3))
    
def scale(x):
    return torch.rand(1)[0] * x


    

def padding(x_ori,x_resize):
    ori_size=x_ori.shape[-1]
    resize_size=x_resize.shape[-1]
    h_rem = ori_size-resize_size
    w_rem = ori_size-resize_size
    pad_top = int(np.random.randint(0, h_rem,size=1)[0])
    pad_bottom = h_rem - pad_top
    pad_left = int(np.random.randint(0, w_rem,size=1)[0])
    pad_right = w_rem - pad_left
    X_out = F.pad(x_resize,(pad_left,pad_top,pad_right,pad_bottom),
                      mode='constant', value=0)
    return X_out
    
    
def brightness(x,p=0.5):
    return p*x

def add_noise(x):
    return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)

def gkern(kernel_size=3, nsig=3):
    x = np.linspace(-nsig, nsig, kernel_size)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return torch.from_numpy(stack_kernel.astype(np.float32)).to('cuda')

def ori_trans(x_in):
    img_size=x_in.shape[-1]
    x_resize=random_size(x_in)
    x_out=padding(x_in,x_resize)
    #x_out=brightness(x_out)
    x_out=F.interpolate(x_out, size=(img_size,img_size))
    return x_out

def sia(x, choice=-1):
    _, _, w, h = x.shape
    op = [vertical_shift, horizontal_shift, vertical_flip, horizontal_flip, rotate180, scale, add_noise]
    num_block = 3
    y_axis = [0,] + np.random.choice(list(range(1, h)), num_block-1, replace=False).tolist() + [h,]
    x_axis = [0,] + np.random.choice(list(range(1, w)), num_block-1, replace=False).tolist() + [w,]
    y_axis.sort()
    x_axis.sort()
        
    x_copy = x.clone()
    for i, idx_x in enumerate(x_axis[1:]):
        for j, idx_y in enumerate(y_axis[1:]):
            chosen = choice if choice >= 0 else np.random.randint(0, high=len(op), dtype=np.int32)
            x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

    return x_copy







