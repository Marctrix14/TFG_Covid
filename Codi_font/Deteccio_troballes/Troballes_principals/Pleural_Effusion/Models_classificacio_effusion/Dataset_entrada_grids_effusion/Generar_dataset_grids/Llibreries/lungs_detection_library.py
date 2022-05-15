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

ds_path = '../../../../Lungs_detection/Datasets/roi_detection_subset/train_val/' # path is related to where "effusion_classification_covid_v2" is located (it's the notebook that uses this library)

# Get the annotations associated to an image 
def img2txt_name(imgPath): # imgPath is a string that contains the path of an image
    return imgPath.replace('.png', '.txt')

# Generate the bounding boxes of an image with Path "imgPath"
#   "imgPath" is a Path and not a filename because the DataBlock object used next gets the paths 
#   of all the images of the dataset & stores them in a list named "get_image_files"
#   For each image of the dataset, it's path is passed as parameter in get_bboxes to generate the target bounding boxes 
def get_bboxes(imgPath):
    
    img = PILImage.create(str(imgPath))

    # Get the annotations of the bounding boxes of the lungs of the rx image with Path "f"
    fullAnnot = np.genfromtxt(img2txt_name(str(imgPath)))

    bboxes = np.zeros((2,4))

    for i in range(len(fullAnnot)):
        cx = int(fullAnnot[i][1]*img.size[0]) 
        cy = int(fullAnnot[i][2]*img.size[1])
        
        w = int(fullAnnot[i][3]*img.size[0])
        h = int(fullAnnot[i][4]*img.size[1])
        
        bbox = np.zeros(4)
        bbox[0] = float(cx-w/2.0) # minx 
        bbox[1] = float(cy-h/2.0) # miny
        bbox[2] = float(cx+w/2.0) # maxX
        bbox[3] = float(cy+h/2.0) # maxY

        bboxes[i] = bbox

    return bboxes

# Cell
class NoLabelBBoxLabeler(Transform):
    """ Bounding box labeler with no label """
    def setups(self, x): noop
    def decode (self, x, **kwargs):
        self.bbox,self.lbls = None,None
        return self._call('decodes', x, **kwargs)

    def decodes(self, x:TensorBBox):
        self.bbox = x
        return self.bbox if self.lbls is None else LabeledBBox(self.bbox, self.lbls)

'''
def custom_bb_pad(samples, pad_idx=0):
    "Function that collect `samples` of bboxes and adds padding with `pad_idx`."
    # 'samples' is a list that contains a tuple with 2 elements: a TensorImage & a TensorBBox. TensorBBox size is (2,4)
    #   TensorImage size is [3, width, height]
    max_len = max([len(s[1]) for s in samples]) # s[1] is a TensorBBox. max_len equals to 2 (number of bboxes associated to a TensorBBox)
    def _f(img,bbox): # img is a TensorImage, bbox is a TensorBBox
        bbox = torch.cat([bbox,bbox.new_zeros(max_len-bbox.shape[0], 4)])        
        return img,bbox
    return [_f(*s) for s in samples] # _f function receives a TensorImage as first parameter (img) and a TensorBBox as second parameter (bbox)
'''
'''
CustomBboxBlock = TransformBlock(type_tfms=TensorBBox.create, 
                             item_tfms=[PointScaler, NoLabelBBoxLabeler], dls_kwargs = {'before_batch': custom_bb_pad})               
'''

CustomBboxBlock = TransformBlock(type_tfms=TensorBBox.create, 
                             item_tfms=[PointScaler, NoLabelBBoxLabeler])   

class BBoxReshape(DisplayedTransform):
    "Normalize/denorm batch of `TensorImage`"
    parameters,order = L(),100
    def __init__(self): 
        noop

    def setups(self, dl:DataLoader):
        noop

    def encodes(self, x:TensorBBox): return torch.reshape(x,(x.shape[0],8))
    def decodes(self, x:TensorBBox): return torch.reshape(x,(x.shape[0],2,4))


# Documentation for DataBlock 
# -> (DataBlock creation) https://docs.fast.ai/tutorial.datablock.html#Bounding-boxes
# -> (Split train/val data) https://docs.fast.ai/data.transforms.html#FuncSplitter 
data = DataBlock(
    blocks=(ImageBlock, CustomBboxBlock), # ImageBlock means type of inputs are images; BBoxBlock & BBoxLblBlock = type of targets are BBoxes & their labels
    get_items=get_image_files,
    n_inp=1, # number of inputs; it's 1 because the only inputs are the rx images (ImageBlock)
    get_y= get_bboxes,
    splitter = FuncSplitter(lambda img: Path(img).parent.name == 'valid'), # split items by result of func (True for validation, False for training set).
    batch_tfms= [*aug_transforms(do_flip=False, size=(120,160)), Normalize.from_stats(*imagenet_stats), BBoxReshape] 
)

path_dl = Path(ds_path)
Path.BASE_PATH = path_dl
path_dl.ls().sorted()

dls = data.dataloaders(path_dl, path=path_dl, bs = 64) # bs: how many samples per batch to load 

class LungDetector(nn.Module):
    def __init__(self, arch=models.resnet18): # resnet18 has 18 lineal layers and it's the default arch if none arch is set as parameter
        super().__init__() 
        self.cnn = create_body(arch) # cut off the body of a typically pretrained arch
        self.head = create_head(num_features_model(self.cnn), 8)

    # NOTE: What does forward function mean?    
    def forward(self, im): # NOTE: what does im mean?
        x = self.cnn(im) # NOTE: why im is passed as parameter to cnn?
        x = self.head(x)
        # NOTE: Understand what the following line does?, what is x.sigmoid_()?
        return 2 * (x.sigmoid_() - 0.5)

def loss_fn(preds, targs):
    return L1Loss()(preds, targs.squeeze())

def setupLungsDetectionLearner():
    learn = Learner(dls, LungDetector(arch=models.resnet50).cuda(), loss_func=loss_fn) # LungDetector.cuda() to work the model on GPU
    return learn.load('roi_detector_stage-1')