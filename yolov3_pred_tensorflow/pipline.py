import tensorflow as tf
import numpy as np
import cv2
from home.ol.yolov3 import getmodelout
from home.ol.yolov3 import preProcessImg
from home.ol.yolov3 import config
from home.ol.yolov3 import decode
from home.ol.yolov3 import filterBoxes
from home.ol.yolov3 import nms
from home.ol.yolov3 import drawdetections

def predict(img,pbGraph):
    sess=tf.Session()
    inputImg=preProcessImg.preProcessImg(img,config.inputShape[0])
    #out1:(1, 13, 13, 3, 85),out2:(1, 26, 26, 3, 85),out3:(1, 52, 52, 3, 85)
    out=getmodelout.getmodelout(sess,pbGraph,1)

    anchorMask=[[6,7,8],[3,4,5],[0,1,2]]
    numclass=len(config.classes)
    anchors=np.array(config.anchors,np.int32)
    boxesA = [];scoresA = [];classesA = []

    for i in range(len(out)):
        feature=out[i]
        decodeOut=decode.decode(feature,anchors[anchorMask[i]],numclass,img.shape[:2],config.inputShape)
        bboxes,objProbs,classProbs=sess.run(decodeOut,feed_dict={"input_1:0":inputImg})
        #(507, 4),(1, 169, 3),(1, 169, 3, 80)
        #(2028, 4),(1, 676, 3),(1, 676, 3, 80)
        #(8112, 4),(1, 2704, 3),(1, 2704, 3, 80)
        bboxes,scores,classes=filterBoxes.filterBoxes(bboxes,objProbs,classProbs)
        boxesA.extend(bboxes)
        scoresA.extend(scores)
        classesA.extend(classes)

    boxesA=np.array(boxesA)
    scoresA=np.array(scoresA)
    classesA=np.array(classesA)
    boxesOut,scoresOut,classesOut=nms.nms(boxesA,scoresA,classesA)

    imgC=drawdetections.drawDetections(img,boxesOut,scoresOut,classesOut)

    cv2.imshow("res",imgC)
    cv2.waitKey()


if __name__=="__main__":
    imgPath = './demo/person.jpg'
    pbGraph = './model/yolo3/saved_model.pb'
    img = cv2.imread(imgPath)
    predict(img,pbGraph)