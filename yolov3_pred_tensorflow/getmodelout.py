import tensorflow as tf
import cv2
from home.ol.yolov3 import preProcessImg
from home.ol.yolov3 import config
import numpy as np


def getmodelout(sess,pbGraph,inputImg):
    with sess as sess:
        output_graph_def = tf.GraphDef()
        with open(pbGraph, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        res1=sess.graph.get_tensor_by_name('reshape_1/Reshape:0')
        res2=sess.graph.get_tensor_by_name('reshape_2/Reshape:0')
        res3=sess.graph.get_tensor_by_name('reshape_3/Reshape:0')

        #res1,res2,res3=sess.run([res1,res2,res3],feed_dict={"input_1:0":inputImg})
    return res1,res2,res3

if __name__=='__main__':
    sess=tf.Session()
    pbGraph='./model/yolo3/saved_model.pb'
    imgPath = './demo/person.jpg'
    img = cv2.imread(imgPath)
    inputImg = preProcessImg.preProcessImg(img, config.inputShape[0])
    out1,out2,out3=getmodelout(sess,pbGraph,inputImg)
