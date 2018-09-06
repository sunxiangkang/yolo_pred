import numpy as np
import tensorflow as tf


def mapToInputimg(bboxX, bboxY, bboxW, bboxH,imgShape,inputShape):
    """
    decode出来的bboxX*416 是映射到标准input上的GT框
    但是原图是经过缩放和平移才转换成标准input
    所以，应该先平移回去
    再乘以416 映射到标准input
    再乘以imgShape/newImgshape(缩放比例)映射回原图

    :parameter
    :imgShape：读入的原始的图片大小
    :inputShape：将要映射到的标准的大小(416*416)
    """
    newShape=np.round(imgShape*np.min(inputShape/imgShape))
    offset=(inputShape-newShape)/2./inputShape

    #------深坑------
    scale=inputShape/newShape
    #------深坑------

    bboxX=(bboxX-offset[1])*scale[1]
    bboxY=(bboxY-offset[0])*scale[0]
    bboxW*=scale[1]
    bboxH*=scale[0]
    bboxes = tf.stack([bboxX - bboxW / 2, bboxY - bboxH / 2, bboxX + bboxW / 2, bboxY + bboxH / 2], axis=3)

    bboxes = tf.reshape(bboxes,[-1,4])
    temp=np.array(np.tile([imgShape[1],imgShape[0]],2))
    bboxes*=temp

    return bboxes


def decode(feature,anchors,numClasses,imgShape,inputShape):
    [_, H, W, N,D] = feature.get_shape().as_list()
    imgShape = np.array(imgShape, np.float32)
    inputShape = np.array(inputShape[0], np.float32)
    numAnchors=len(anchors)
    # [x,y,w,h,gtProb,classes(80个)]*3(个anchor)=255
    detectionResult = tf.reshape(feature, [-1, H * W, numAnchors, numClasses + 5])

    xyOffset = tf.nn.sigmoid(detectionResult[:, :, :, 0:2])
    whOffset = tf.exp(detectionResult[:, :, :, 2:4])
    objProbs = tf.nn.sigmoid(detectionResult[:, :, :, 4])
    classProbs = tf.nn.softmax(detectionResult[:, :, :, 5:])

    heightIndex = np.arange(H)
    widthIndex = np.arange(W)
    xOff, yOff = np.meshgrid(heightIndex, widthIndex)
    xOff = np.reshape(xOff, [1, -1, 1])
    yOff = np.reshape(yOff, [1, -1, 1])

    bboxX = (xOff + xyOffset[:, :, :, 0]) / W
    bboxY = (yOff + xyOffset[:, :, :, 1]) / H
    bboxW = (anchors[:, 0] * whOffset[:, :, :, 0]) / inputShape[1]
    bboxH = (anchors[:, 1] * whOffset[:, :, :, 1]) / inputShape[0]

    bboxes=mapToInputimg(bboxX, bboxY, bboxW, bboxH, imgShape, inputShape)

    return bboxes, objProbs, classProbs