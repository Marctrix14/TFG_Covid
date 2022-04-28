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

path = '../../../Lungs_detection/Datasets/roi_detection_subset/train_val/' # path is related to where "effusion_classification_covid_v2" is located (it's the notebook that uses this library)

# Get the annotations associated to an image 
def img2txt_name(f):
    f = os.path.basename(f)
    return path + f'{str(f)[:-4]}.txt'

def get_bboxes(f):
    
    img = PILImage.create(path+f.name)

    # Get the annotations of the bounding boxes of the lungs of the rx image with path "f"
    fullAnnot = np.genfromtxt(img2txt_name(f))

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

def custom_bb_pad(samples, pad_idx=0):
    "Function that collect `samples` of bboxes and adds padding with `pad_idx`."
    #samples = [(s[0], *clip_remove_empty(*s[1:])) for s in samples] # s[0] is a tuple of TensorImage & TensorBbox, TensorBbox size is (2,4)
    max_len = max([len(s[1]) for s in samples]) # equals to 4 (number of bbox coordinates)
    def _f(img,bbox):
        bbox = torch.cat([bbox,bbox.new_zeros(max_len-bbox.shape[0], 4)])        
        return img,bbox
    return [_f(*s) for s in samples]


CustomBboxBlock = TransformBlock(type_tfms=TensorBBox.create, 
                             item_tfms=[PointScaler, NoLabelBBoxLabeler], dls_kwargs = {'before_batch': custom_bb_pad})               

class BBoxReshape(DisplayedTransform):
    "Normalize/denorm batch of `TensorImage`"
    parameters,order = L(),100
    def __init__(self): 
        noop

    def setups(self, dl:DataLoader):
        noop

    def encodes(self, x:TensorBBox): return torch.reshape(x,(x.shape[0],8))
    def decodes(self, x:TensorBBox): return torch.reshape(x,(x.shape[0],2,4))


data = DataBlock(
    blocks=(ImageBlock, CustomBboxBlock), # ImageBlock means type of inputs are images; BBoxBlock & BBoxLblBlock = type of targets are BBoxes & their labels
    get_items=get_image_files,
    n_inp=1, # number of inputs; it's 1 because the only inputs are the rx images (ImageBlock)
    get_y= get_bboxes,
    splitter = RandomSplitter(0.1), # split training/validation; parameter 0.1 means there will be 10% of validation images 
    batch_tfms= [*aug_transforms(do_flip=False, size=(120,160)), Normalize.from_stats(*imagenet_stats), BBoxReshape] 
)

path_dl = Path(path)
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

def setupLungsLearner():
    learn = Learner(dls, LungDetector(arch=models.resnet50).cuda(), loss_func=loss_fn) # LungDetector.cuda() to work the model on GPU
    return learn.load('roi_detector_stage-1')