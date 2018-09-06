import numpy as np
from home.ol.yolov3 import config

def filterBoxes(bboxes,objProbs,classProbs):
    objProbs=np.reshape(objProbs,(-1,))
    classProbs=np.reshape(classProbs,[objProbs.shape[0],-1])
    classIndexes=np.argmax(classProbs,axis=-1)
    classProbs=classProbs[np.arange(classProbs.shape[0]),classIndexes]

    scores=objProbs*classProbs
    keepIndexes=scores>config.scoreThreshold

    bboxes=bboxes[keepIndexes]
    scores=scores[keepIndexes]
    classIndexes=classIndexes[keepIndexes]

    return bboxes,scores,classIndexes