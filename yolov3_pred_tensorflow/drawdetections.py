import cv2
import colorsys
from home.ol.yolov3 import config


def drawDetections(img,boxes,scores,classes):
    hsv_tuples = [(x / float(len(config.classes)), 1., 1.) for x in range(len(config.classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    imgC=img.copy();H,W,_=img.shape
    for index,bbox in enumerate(boxes):
        clsIndex=classes[index]
        cv2.rectangle(imgC,(bbox[0],bbox[1]),(bbox[2],bbox[3]),colors[clsIndex],8)
        mess='%s:%.3f'%(config.classes[clsIndex],scores[index])
        if bbox[1] < 20:
            textLoc = (int(bbox[0]) + 2, int(bbox[1]) + 15)
        else:
            textLoc = (int(bbox[0]), int(bbox[1]) - 10)
        cv2.putText(imgC, mess, textLoc, cv2.FONT_HERSHEY_SIMPLEX,1e-3*H,(255,255,255))
    return imgC