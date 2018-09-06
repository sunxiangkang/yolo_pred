import utils
import cv2
import darknet
import tensorflow as tf
import numpy as np
import decode
import config


def Predict(imgPath,inputSize=(416,416)):
    img=cv2.imread(imgPath)
    imgShape=img.shape[:2]
    imgB=utils.PreProcessImg(img,inputSize)
    inputHolder=tf.placeholder(tf.float32,[None,inputSize[0],inputSize[1],3])
    out=darknet.DarkNet19(inputHolder)
    outDecoded=decode.DecodeModel(out,numClass=len(config.classNames),anchors=np.array(config.anchors))

    variables=tf.global_variables()
    saver=utils.RestoreVariabels(variables)
    ckpt=tf.train.get_checkpoint_state(config.modelPath)
    with tf.Session() as sess:
        saver.restore(sess,ckpt.model_checkpoint_path)
        bboxes,objProbs,classProbs=sess.run(outDecoded,feed_dict={inputHolder:imgB})

    bboxes,scores,classMaxIndex=utils.PostPrecess(bboxes,objProbs,classProbs,imgShape)
    imgC=utils.DrawDetections(img,bboxes,scores,classMaxIndex,config.classNames)
    #cv2.namedWindow("res",0)
    cv2.imshow("res",imgC)
    cv2.waitKey()


if __name__=='__main__':
    import sys
    import os
    sys.path.append(os.getcwd())
    imgPath=r'.\demo\person.jpg'
    Predict(imgPath)