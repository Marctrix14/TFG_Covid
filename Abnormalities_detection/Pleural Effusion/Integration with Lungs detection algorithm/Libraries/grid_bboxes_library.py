from fastai.vision.all import *
import cv2
import numpy as np


def gridBbox(bbox, imgRes, imgOri, nRows, nCols):
    # Param:
    #   bbox: minx miny maxX maxY
    #   grids: numpy array that stores each generated grid from bbox  
    #   TODO: params nRows & nCols must be removed after optimizing the code 

    # NOTE: Example of bbox: [[array([ 87.5,  70. , 240.5, 458. ])], ['0']]
    # NOTE: Coordinates have decimals, they must be truncated 
    # NOTE: For the moment, make all grids the same size 
    # NOTE: Don't take into account nGrids to be a prime number 
    # NOTE: Number of grids must be greater than one 

    # Paint vertical lines of bboxLeft 
    #image_res[:, int(bboxLeft[0])] = (255, 0, 0) 
    #image_res[:, int(bboxLeft[2])] = (255, 0, 0) 

    # Paint vertical lines of bboxRight 
    #image_res[:, int(bboxRight[0])] = (255, 0, 0)  
    #image_res[:, int(bboxRight[2])] = (255, 0, 0) 

    # Paint horizontal lines
    # NOTE: For the moment, just use bboxLeft. In the future, should use both bbox
    # and check which has the horizontal line more up/down
    #image_res[int(bboxRight[1]), int(bboxLeft[0]): int(bboxRight[2])] = (255, 0, 0) 
    #image_res[int(bboxRight[3]), int(bboxLeft[0]): int(bboxRight[2])] = (255, 0, 0) 

    # Paint borders of the bbox 
    # x axis of imgRes (openCV format) refers to the columns
    # y axis refers to the rows 
    # to access a pixel of imgRes we should do imgRes[y,x] as we would access a value of a matrix
    # Do the same when accessing a pixel of imgOri
    # More info at: https://pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/

    # Vertical borders 
    imgRes[int(bbox[1]): int(bbox[3]), int(bbox[0])] = (255, 0, 0) # (R,G,B)
    imgRes[int(bbox[1]): int(bbox[3]), int(bbox[2])] = (255, 0, 0) 
    # Horizontal borders 
    imgRes[int(bbox[1]), int(bbox[0]) : int(bbox[2])] = (255, 0, 0) 
    imgRes[int(bbox[3]), int(bbox[0]) : int(bbox[2])] = (255, 0, 0) 

    # TODO: param grids length should be nGrids, not the nRows * nCols 
    # Get the number of grids for the bbox
    # nGrids = len(grids)  
    
    # TODO: Get the number of rows and columns 
    # nRows * nCols = nGrids
    # nRows has to be greater or equal than nCols because 
    # the bboxes are taller than wide 
    # NOTE: For the moment, nRows & nCols are given as params 
    #nRows = 0
    #nCols = 0
    
    # Get the size of each grid 
    gridHeight = (bbox[3] - bbox[1]) / nRows # (maxY - minY) / nRows
    gridWidth = (bbox[2] - bbox[0]) / nCols # (maxX - minX) / nCols

    grids = np.zeros(nRows * nCols, object) 
    
    # Obtain the grids and paint their borders in red 
    x = bbox[0] # minx
    y = bbox[1] # miny
    for nGrids in range(nRows*nCols): # nIterations = number of grids
        grids[nGrids] = imgOri[int(y): int(y + gridHeight), int(x): int(x + gridWidth)]        
        x = x + gridWidth
        # check if x (column index) has reached the right border of the bbox        
        if (x < bbox[2]): # check if x < maxX
            # paint vertical line of the added grid 
            imgRes[int(bbox[1]):int(bbox[3]), int(x)] = (255, 0, 0) 
        # right border reached 
        else: 
            x = bbox[0] # set x to minX 
            # paint horizontal line 
            y = y + gridHeight
            imgRes[int(y), int(bbox[0]): int(bbox[2])] = (255, 0, 0) 

    return grids