import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from functools import partial

def NetArgScope(stddev,isTraining,alpha=0.1,decay=0.98,batchNormScale=True,epsilon=0.001):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                        activation_fn=partial(tf.nn.leaky_relu,alpha=alpha),
                        stride=1,
                        padding='SAME',
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training':isTraining,
                            'decay':decay,
                            'scale':batchNormScale,
                            'epsilon':epsilon,
                            'updates_collections':tf.GraphKeys.UPDATE_OPS}) as sc:
        return sc

def darknet53(input,stddev=0.1,isTraining=False):
    netScope=NetArgScope(stddev,isTraining)
    with slim.arg_scope(netScope) as scope:
        net=slim.conv2d(input,32,[3,3],scope='conv1_1')
        net=slim.conv2d(net,64,[3,3],stride=2,scope='conv1_2')
        shortcut=net
        net=slim.conv2d(net,32,[1,1],scope="conv1_3")
        net=slim.conv2d(net,64,[3,3],activation_fn=None,scope="conv1_res")
        net=tf.nn.relu(net+shortcut,name='relu1_res')

        net=slim.conv2d(net,128,[3,3],stride=2,scope='conv2_1')
        shortcut=net
        for i in range(2):
            net=slim.conv2d(net,64,[1,1],scope='conv2_{}'.format(i+2))
            net=slim.conv2d(net,128,[3,3],activation_fn=None,scope="conv2_res{}".format(i+2))
            net=tf.nn.relu(net+shortcut,name='relu2_res{}'.format(i+2))

        net=slim.conv2d(net,256,[3,3],stride=2,scope='conv3_1')
        shortcut=net
        for i in range(8):
            net=slim.conv2d(net,128,[1,1],scope='conv3_{}'.format(i+2))
            net=slim.conv2d(net,256,[3,3],activation_fn=None,scope='conv3_res{}'.format(i+2))
            net=tf.nn.relu(net+shortcut,name='relu3_res{}'.format(i+2))
        preScale3=net

        net=slim.conv2d(net,512,[3,3],stride=2,scope='conv4_1')
        shortcut=net
        for i in range(8):
            net=slim.conv2d(net,256,[1,1],scope='conv4_{}'.format(i+2))
            net=slim.conv2d(net,512,[3,3],activation_fn=None,scope='conv4_res{}'.format(i+2))
            net=tf.nn.relu(net+shortcut,name='relu4_res{}'.format(i+2))
        preScale2=net

        net=slim.conv2d(net,1024,[3,3],stride=2,scope='conv5_1')
        shortcut=net
        for i in range(4):
            net=slim.conv2d(net,512,[1,1],scope='conv5_{}'.format(i+2))
            net=slim.conv2d(net,1024,[3,3],activation_fn=None,scope='conv5_res{}'.format(i+2))
            net=tf.nn.relu(net+shortcut,name='relu5_res{}'.format(i+2))
        preScale1=net

    #(1, 13, 13, 1024),(1, 26, 26, 512),(1, 52, 52, 256)
    return preScale1,preScale2,preScale3


def Get2x(net,prenet):
    net2x=tf.image.resize_images(net,[2*tf.shape(net)[1],2*tf.shape(net)[2]])
    net2xConcat=tf.concat([net2x,prenet],3)
    return net2xConcat


def Scale(preScale1,preScale2,preScale3,stddev=0.1,isTraining=False):
    sc=NetArgScope(stddev=stddev,isTraining=isTraining)
    with slim.arg_scope(sc) as scope:
        net=slim.conv2d(preScale1,512,[1,1],scope='scale1_conv1')   #53
        net=slim.conv2d(net,1024,[3,3],scope='scale1_conv2')    #54
        net = slim.conv2d(net, 512, [1, 1], scope='scale1_conv3')   #55
        net = slim.conv2d(net, 1024, [3, 3], scope='scale1_conv4')  #56
        net=slim.conv2d(net,512,[1,1],scope='scale1_conv5')     #57
        shortcut=net
        netT=slim.conv2d(shortcut,1024,[3,3],scope='scale1_conv6')  #58

        #80分类*[x,y,w,h,back ground/ground truth]*3
        scale1=slim.conv2d(netT,255,[1,1],
                           activation_fn=None,
                           normalizer_fn=None,
                           scope='yoloS1')   #59--问题出在这

        net=slim.conv2d(net,256,[1,1],scope='scale2_conv1')     #60
        net=Get2x(net,preScale2)
        net=slim.conv2d(net,256,[1,1],scope='scale2_conv2')     #61
        net=slim.conv2d(net,512,[3,3],scope='scale2_conv3')     #62
        net=slim.conv2d(net,256,[1,1],scope='scale2_conv4')     #63
        net=slim.conv2d(net,512,[3,3],scope='scale2_conv5')     #64
        net=slim.conv2d(net,256,[1,1],scope='scale2_conv6')     #65
        shortcut=net
        netT=slim.conv2d(shortcut,512,[3,3],scope='scale2_conv7')   #66
        scale2=slim.conv2d(netT,255,[1,1],
                           activation_fn=None,
                           normalizer_fn=None,
                           scope='yoloS2')       #67--还有这

        net=slim.conv2d(net,128,[1,1],scope='scale3_conv1')     #68
        net=Get2x(net,preScale3)

        for i in range(3):
             net=slim.conv2d(net,128,[1,1],scope='scale3_conv{}'.format(i*2+2))     #69 71 73
             net=slim.conv2d(net,256,[3,3],scope='scale3_conv{}'.format(i*2+3))     #70 72 74
        scale3=slim.conv2d(net,255,[1,1],
                           activation_fn=None,
                           normalizer_fn=None,
                           scope='yoloS3')    #73--还有这

        #(1, 13, 13, 255),(1, 26, 26, 255),(1, 52, 52, 255)
        return scale1,scale2,scale3


if __name__=='__main__':
    x = tf.random_normal([1, 416, 416, 3])
    pre1,pre2,pre3=darknet53(x)
    yoloout=Scale(pre1,pre2,pre3)
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(yoloout[0]).shape)
        print(sess.run(yoloout[1]).shape)
        print(sess.run(yoloout[2]).shape)