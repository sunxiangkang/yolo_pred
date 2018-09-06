from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
from home.ol.yolov2 import config
from home.ol.yolov2 import darknet


sess=tf.Session()
#获取训练好的模型的变量名称
def GetWeightsName(modelPath):
    ckpt = tf.train.get_checkpoint_state(modelPath)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    return var_to_shape_map

#获取定义的模型的变量名称
def GetModelWeightsName():
    x=tf.placeholder(tf.float32,[1, 416, 416, 3])
    modelout=darknet.DarkNet19(x)
    for var in tf.global_variables():
        print(var)

if __name__=='__main__':
    #weightsNames=GetWeightsName(config.modelPath)
    #print(weightsNames)
    GetModelWeightsName()