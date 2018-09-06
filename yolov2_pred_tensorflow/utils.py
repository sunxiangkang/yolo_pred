import tensorflow as tf
import numpy as np
import config
import cv2
import colorsys
from collections import OrderedDict

def PreProcessImg(img,imgSize=(416,416)):
    imgC=np.copy(img).astype(np.float32)
    imgRGB=cv2.cvtColor(imgC,cv2.COLOR_BGR2RGB)
    imgResize=cv2.resize(imgRGB,imgSize)
    imgNorm=imgResize/255.0
    imgB=np.expand_dims(imgNorm,axis=0)
    return imgB

def RestoreVariabels(variables):
    funDict={}
    for index in range(len(config.trainedModelWeightsNames)):
        funDict.update({config.trainedModelWeightsNames[index][0]:variables[index]})
    saver=tf.train.Saver(funDict)
    return saver

def PostPrecess(bboxes,objProbs,classProbs,imgShape=(416,416),thres=0.5):
    bboxes=bboxes.reshape([-1,4])
    bboxes[:,0:1]*=imgShape[1]
    bboxes[:,1:2]*=imgShape[0]
    bboxes[:,2:3]*=imgShape[1]
    bboxes[:,3:4]*=imgShape[0]

    bboxesMinMax=[0,0,imgShape[1]-1,imgShape[0]-1]
    bboxes=CutBboxes(bboxes,bboxesMinMax)
    #objProbs.shape:[batchSize,W,H,每个anchor属于前景还是后景的prob]
    #reshape后:第一行的第一个的第一个anchor
    #第一行的第一列的最后一个anchor
    #第一行第二列的第一个anchor
    #......
    objProbs=np.reshape(objProbs,[-1])
    classProbs=np.reshape(classProbs,[len(objProbs),-1])
    classMaxIndexes=np.argmax(classProbs,axis=1)
    classProbs=classProbs[np.arange(len(classProbs)),classMaxIndexes]

    scores=objProbs*classProbs
    keepIndexes=scores>thres
    classMaxIndexes=classMaxIndexes[keepIndexes]
    scores=scores[keepIndexes]
    bboxes=bboxes[keepIndexes]

    bboxes,scores,classMaxIndex=SortBboxes(bboxes,scores,classMaxIndexes)
    bboxes,scores,classMaxIndex=NMS(bboxes,scores,classMaxIndex)

    return bboxes,scores,classMaxIndex

def CutBboxes(bboxes,bboxesMinMax):
    bboxes=np.copy(bboxes).transpose()
    bboxesMinMax=np.transpose(bboxesMinMax)

    bboxes[0]=np.maximum(bboxes[0],bboxesMinMax[0])
    bboxes[1]=np.maximum(bboxes[1],bboxesMinMax[1])
    bboxes[2]=np.minimum(bboxes[2],bboxesMinMax[2])
    bboxes[3]=np.minimum(bboxes[3],bboxesMinMax[3])

    bboxes=np.transpose(bboxes)
    return bboxes

def SortBboxes(bboxes,scores,classes,topK=400):
    index=np.argsort(-scores)
    classes=classes[index][:topK]
    scores=scores[index][:topK]
    bboxes=bboxes[index][:topK]
    return bboxes,scores,classes

def CalcIOU(bboxes1,bboxes2):
    bboxes1=np.transpose(bboxes1)
    bboxes2=np.transpose(bboxes2)

    yMin=np.maximum(bboxes1[0],bboxes2[0])
    xMin=np.maximum(bboxes1[1],bboxes2[1])
    yMax=np.minimum(bboxes1[2],bboxes2[2])
    xMax=np.minimum(bboxes1[3],bboxes2[3])

    H=np.maximum(yMax-yMin,0.)
    W=np.maximum(xMax-xMin,0.)

    inter=H*W
    S1=(bboxes1[2]-bboxes1[0])*(bboxes1[3]-bboxes1[1])
    S2=(bboxes2[2]-bboxes1[0])*(bboxes2[3]-bboxes2[1])
    IOU=inter/(S1+S2-inter)
    return IOU

#经过SortBboxes之后，classes,scores,bboxes已经是按照scores大小排序
def NMS(bboxes,scores,classes,thres=0.45):
    keepedBboxes=np.ones(scores.shape,dtype=np.bool)
    for i in range(scores.size-1):
        if keepedBboxes[i]:
            overlaps=CalcIOU(bboxes[i],bboxes[(i+1):])
            keepIndexes=np.logical_or(overlaps<thres,classes[i]!=classes[(i+1):])
            keepedBboxes[(i+1):]=np.logical_and(keepedBboxes[(i+1):],keepIndexes)
    indexes=np.where(keepedBboxes)
    return bboxes[indexes],scores[indexes],classes[indexes]

def DrawDetections(img,bboxes,scores,clsIndexes,labels,thres=0.3):
    hsv_tuples = [(x / float(len(labels)), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    imgC=img.copy()
    H,W,_=img.shape
    for index,bbox in enumerate(bboxes):
        if scores[index]<thres:
            continue
        clsIndex=clsIndexes[index]
        cv2.rectangle(imgC,(bbox[0],bbox[1]),(bbox[2],bbox[3]),colors[clsIndex],8)
        mess='%s:%.3f'%(labels[clsIndex],scores[index])
        if bbox[1] < 20:
            text_loc = (int(bbox[0]) + 2, int(bbox[1]) + 15)
        else:
            text_loc = (int(bbox[0]), int(bbox[1]) - 10)
        cv2.putText(imgC, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX,1e-3*H,(255,255,255))
    return imgC