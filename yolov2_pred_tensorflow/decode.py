import tensorflow as tf
import numpy as np

def DecodeModel(modelOutput,numClass=20,anchors=None):
    _,H,W,D=modelOutput.get_shape().as_list()
    numAnchors=len(anchors)
    #[x,y,w,h,gtProb,classes(80个)]*5(个anchor)=425
    detectionResult=tf.reshape(modelOutput,[-1,H*W,numAnchors,numClass+5])

    xyOffset=tf.nn.sigmoid(detectionResult[:,:,:,0:2])
    whOffset=tf.exp(detectionResult[:,:,:,2:4])
    objProbs=tf.nn.sigmoid(detectionResult[:,:,:,4])
    classProbs=tf.nn.softmax(detectionResult[:,:,:,5:])

    heightIndex=np.arange(H)
    widthIndex=np.arange(W)
    xOff,yOff=np.meshgrid(heightIndex,widthIndex)
    xOff=np.reshape(xOff,[1,-1,1])
    yOff=np.reshape(yOff,[1,-1,1])

    bboxX=(xOff+xyOffset[:,:,:,0])/W
    bboxY=(yOff+xyOffset[:,:,:,1])/H
    bboxW=(anchors[:,0]*whOffset[:,:,:,0])/W
    bboxH=(anchors[:,1]*whOffset[:,:,:,1])/H

    bboxes=tf.stack([bboxX-bboxW/2,bboxY-bboxH/2,bboxX+bboxW/2,bboxY+bboxH/2],axis=3)

    return bboxes,objProbs,classProbs