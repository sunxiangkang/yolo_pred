import numpy as np
from home.ol.yolov3 import config

def sortBboxes(bboxes,scores,classes):
    index=np.argsort(-scores)
    classes=classes[index][:config.topK]
    scores=scores[index][:config.topK]
    bboxes=bboxes[index][:config.topK]
    return bboxes,scores,classes

def calcIOU(box1,boxes2):
    box1=np.transpose(box1)
    boxes2=np.transpose(boxes2)

    xMin=np.maximum(box1[0],boxes2[0])
    yMin=np.maximum(box1[1],boxes2[1])
    xMax=np.minimum(box1[2],boxes2[2])
    yMax=np.minimum(box1[3],boxes2[3])

    w=np.maximum(xMax-xMin,0)
    h=np.maximum(yMax-yMin,0)

    inter=w*h
    s1=(box1[2]-box1[0])*(box1[3]-box1[1])
    s2=(boxes2[2]-boxes2[0])*(boxes2[3]-boxes2[1])
    IOU=inter/(s1+s2-inter)

    return IOU

def nms(boxes,scores,classes):
    boxes,scores,classes=sortBboxes(boxes,scores,classes)
    keepBoxes=np.ones(scores.shape,np.bool)
    for i in range(scores.size-1):
        if keepBoxes[i]:
            IOU=calcIOU(boxes[i],boxes[(i+1):])
            keepIndexes=np.logical_or(IOU<config.nmsThreshold,classes[i]!=classes[(i+1):])
            keepBoxes[(i+1):]=np.logical_and(keepBoxes[(i+1):],keepIndexes)
    indexes=np.where(keepBoxes)

    return boxes[indexes],scores[indexes],classes[indexes]