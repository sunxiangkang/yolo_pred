import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from functools import partial


def NetArgScope(stddev,isTraining,alpha=0.1,decay=0.98,batchNormScale=True,epsilon=0.001):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                        activation_fn=partial(tf.nn.leaky_relu,alpha=alpha),
                        padding='VALID',
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training':isTraining,
                            'decay':decay,
                            'scale':batchNormScale,
                            'epsilon':epsilon,
                            'updates_collections':tf.GraphKeys.UPDATE_OPS}) as sc:
        return sc

#输出dstsize*dstsize*425
def DarkNet19(input,lastChannels=425,stddev=0.1,isTraining=False):
    scope=NetArgScope(stddev,isTraining)
    with slim.arg_scope(scope):
        net=tf.pad(input,[[0,0],[1,1],[1,1],[0,0]],name='pad1')
        net=slim.conv2d(net,32,[3,3],stride=1,scope='conv1')
        net=slim.max_pool2d(net,[2,2],2,scope='maxPool1')

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad2')
        net=slim.conv2d(net,64,[3,3],1,scope='conv2')
        net=slim.max_pool2d(net,[2,2],2,scope='maxPool2')

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad3_1')
        net=slim.conv2d(net,128,[3,3],stride=1,scope='conv3_1')
        net=slim.conv2d(net,64,[1,1],stride=1,scope='conv3_2')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad3_2')
        net=slim.conv2d(net,128,[3,3],stride=1,scope='conv3_3')
        net=slim.max_pool2d(net,[2,2],stride=2,scope='maxPool3')

        net=tf.pad(net,[[0,0],[1,1],[1,1],[0,0]],name='pad4_1')
        net=slim.conv2d(net,256,[3,3],stride=1,scope='conv4_1')
        net=slim.conv2d(net,128,[1,1],stride=1,scope='conv4_2')
        net=tf.pad(net,[[0,0],[1,1],[1,1],[0,0]],name='pad4_2')
        net=slim.conv2d(net,256,[3,3],stride=1,scope='conv4_3')
        net=slim.max_pool2d(net,[2,2],stride=2,scope='maxPool4')

        net=tf.pad(net,[[0,0],[1,1],[1,1],[0,0]],name='pad5_1')
        net=slim.conv2d(net,512,[3,3],stride=1,scope='conv5_1')
        net=slim.conv2d(net,256,[1,1],stride=1,scope='conv5_2')
        net=tf.pad(net,[[0,0],[1,1],[1,1],[0,0]],name='pad5_2')
        net=slim.conv2d(net,512,[3,3],stride=1,scope='conv5_3')
        net=slim.conv2d(net,256,[1,1],stride=1,scope='conv5_4')
        net=tf.pad(net,[[0,0],[1,1],[1,1],[0,0]],name='pad5_3')
        net=slim.conv2d(net,512,[3,3],stride=1,scope='conv5_5')
        shortcut=net
        net=slim.max_pool2d(net,[2,2],stride=2,scope='maxPool5')

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad6_1')
        net=slim.conv2d(net,1024,[3,3],stride=1,scope='conv6_1')
        net=slim.conv2d(net,512,[1,1],stride=1,scope='conv6_2')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad6_2')
        net=slim.conv2d(net,1024,[3,3],stride=1,scope='conv6_3')
        net=slim.conv2d(net,512,[1,1],stride=1,scope='conv6_4')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad6_3')
        net=slim.conv2d(net,1024,[3,3],stride=1,scope='conv6_5')

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad7_1')
        net=slim.conv2d(net,1024,[3,3],stride=1,scope='conv7_1')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad7_2')
        net=slim.conv2d(net,1024,[3,3],stride=1,scope='conv7_2')

        shortcut=slim.conv2d(shortcut,64,[1,1],stride=1,scope='conv_shortcut')
        shortcut=tf.space_to_depth(shortcut,2,name='shortcut_spacetodepth')
        net=tf.concat([shortcut,net],axis=-1,name='concat')

        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad8')
        net=slim.conv2d(net,1024,[3,3],stride=1,scope='conv8')

        output=slim.conv2d(net,lastChannels,[1,1],normalizer_fn=None,activation_fn=None,scope='conv_dec')

        return output


if __name__=='__main__':
    x = tf.random_normal([1, 416, 416, 3])
    model_output = DarkNet19(x)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init=tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(model_output).shape)  # (1,13,13,425)