import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 0 = id of the gpu
# To know which id have to be set on the previous line, on Windows 10 run "nvidia-smi" on CMD to check all the installed GPUs data, like
# the id, driver version and CUDA version 

# Check if your GPU driver and CUDA is enabled and is accessible by PyTorch
# TO USE CUDA SELECT A GPU ON THE EXECUTION ENVIRONMENT, NOT A TPU
import torch
#print(torch.version.cuda) # the CUDA version must be printed
#print(torch.cuda.is_available()) # True must be printed (if False press on the Restart button at the top of the notebook)
#print(torch.cuda.current_device())

from fastai.vision.all import *
from fastai.vision import *
from fastai.metrics import accuracy
from torch.nn import L1Loss
import cv2
from skimage.util import montage
from matplotlib.image import BboxImage
import numpy as np

ds_path = '../CNN_model/Datasets/ServerMIA_KaggleVinBigData/train_val/' # path is related to where "effusion_classification_covid_v2.ipynb" is located (it's the notebook that uses this library)


data = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # CategoryBlock = label
    get_items=get_image_files,
    get_y= parent_label, # parent_label = the folders names of the images are the labels
    item_tfms=Resize(128), # used to make all images from the dataset have the same size
    splitter=RandomSplitter(valid_pct=0.1, seed=42), # split training/validation; parameter 0.1 means there will be 10% of validation images 
    batch_tfms= [*aug_transforms(do_flip=False, size=(120,160)), Normalize.from_stats(*imagenet_stats)] 
)

path_dl = Path(ds_path)
Path.BASE_PATH = path_dl
path_dl.ls().sorted()

dls = data.dataloaders(path_dl, path=path_dl, bs = 64) # bs: how many samples per batch to load 

def setupEffusionLearner():
    learn = cnn_learner(dls, resnet50, metrics=accuracy)
    return learn.load('effusion_classification_stage-1')