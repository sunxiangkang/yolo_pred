import tensorflow as tf
import numpy as np
import cv2

# leaky_relu激活函数
def LeakyRelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

class Yolo(object):
    def __init__(self,weightsFile,verbose=True):
        self.verbose=verbose    # 后面程序打印描述功能的标志位

        # 检测超参数
        self.S=7    # cell数量
        self.B=2    # 每个网格的边界框数
        self.classes=["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]
        self.C=len(self.classes)    # 类别数

        self.xOffset=np.transpose(np.reshape(np.array([np.arange(self.S)]*self.S*self.B),
                                             [self.B,self.S,self.S]),[1,2,0])
        self.yOffset=np.transpose(self.xOffset,[1,0,2])
        self.thresholld=0.2 # 类别置信度分数阈值
        self.iouThreshold=0.4   # IOU阈值，小于0.4的会过滤掉
        self.maxOutputSize=10   # NMS选择的边界框的最大数量

        self.sess=tf.Session()
        self._BuildNet()    # 【1】搭建网络模型(预测):模型的主体网络部分，这个网络将输出[batch,7*7*30]的张量
        self._BuildDetector()   # 【2】解析网络的预测结果：先判断预测框类别，再NMS
        self._LoadWeights(weightsFile)  # 【3】导入权重文件
        #self.DetectFromFile(imgFile=inputImage)  # 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。


    # 【1】搭建网络模型(预测):模型的主体网络部分，这个网络将输出[batch,7*7*30]的张量
    def _BuildNet(self):
        # 打印状态信息
        #if self.verbose:
            #print("Start to build the network ...")

        # 输入、输出用占位符，因为尺寸一般不会改变
        self.images=tf.placeholder(tf.float32,[None,448,448,3])

        #定义网络模型
        with tf.variable_scope("yolo"):
            net=self.__ConvLayer(self.images,2,64,7,2)
            net=self.__MaxPoolLayer(net,3,2,2)
            net=self.__ConvLayer(net,4,192,3,1)
            net=self.__MaxPoolLayer(net,5,2,2)
            net = self.__ConvLayer(net,6, 128, 1, 1)
            net = self.__ConvLayer(net, 7, 256, 3, 1)
            net = self.__ConvLayer(net, 8, 256, 1, 1)
            net = self.__ConvLayer(net, 9, 512, 3, 1)
            net = self.__MaxPoolLayer(net, 10, 2, 2)
            net = self.__ConvLayer(net, 11, 256, 1, 1)
            net = self.__ConvLayer(net, 12, 512, 3, 1)
            net = self.__ConvLayer(net, 13, 256, 1, 1)
            net = self.__ConvLayer(net, 14, 512, 3, 1)
            net = self.__ConvLayer(net, 15, 256, 1, 1)
            net = self.__ConvLayer(net, 16, 512, 3, 1)
            net = self.__ConvLayer(net, 17, 256, 1, 1)
            net = self.__ConvLayer(net, 18, 512, 3, 1)
            net = self.__ConvLayer(net, 19, 512, 1, 1)
            net = self.__ConvLayer(net, 20, 1024, 3, 1)
            net = self.__MaxPoolLayer(net, 21, 2, 2)
            net = self.__ConvLayer(net, 22, 512, 1, 1)
            net = self.__ConvLayer(net, 23, 1024, 3, 1)
            net = self.__ConvLayer(net, 24, 512, 1, 1)
            net = self.__ConvLayer(net, 25, 1024, 3, 1)
            net = self.__ConvLayer(net, 26, 1024, 3, 1)
            net = self.__ConvLayer(net, 28, 1024, 3, 2)
            net = self.__ConvLayer(net, 29, 1024, 3, 1)
            net = self.__ConvLayer(net, 30, 1024, 3, 1)
            net = self.__Flatten(net)
            net = self.__FcLayer(net, 33, 512, activation=LeakyRelu)
            net = self.__FcLayer(net, 34, 4096, activation=LeakyRelu)
            net = self.__FcLayer(net, 36, self.S * self.S * (self.B * 5 + self.C))

        # 网络输出，[batch,7*7*30]的张量
        self.predicts = net

    def __ConvLayer(self,x,id,numFilter,filterSize,stride):
        #获取通道数
        inChannels=x.get_shape().as_list()[-1]
        # 均值为0标准差为0.1的正态分布，初始化权重w；shape=行*列*通道数*卷积核个数
        scopeName='conv_'+str(id)
        with tf.variable_scope(scopeName):
            weights=tf.Variable(tf.truncated_normal([filterSize,filterSize,inChannels,numFilter],
                                                    mean=0.0,stddev=0.1),name='weights')
            bias=tf.Variable(tf.zeros([numFilter,]),name='biases')    # 列向量

            # padding, 注意: 不用padding="SAME",否则可能会导致坐标计算错误
            padSize=filterSize//2   # 除法运算，保留商的整数部分
            padMat=np.array([[0,0],[padSize,padSize],[padSize,padSize],[0,0]])
            xPad=tf.pad(x,padMat)
            conv=tf.nn.conv2d(xPad,weights,strides=[1,stride,stride,1],padding='VALID')
            output=LeakyRelu(tf.nn.bias_add(conv,bias))

        #if self.verbose:
            #print("Layer %d:type=conv,numFilters=%d,filterSize:%d,stride=%d,outputShape=%s"
                  #%(id,numFilter,filterSize,stride,str(output.get_shape())))
        return output

    # 池化层：x输入；id：层数索引；pool_size：池化尺寸；stride：步长
    def __MaxPoolLayer(self,x,id,poolSize,stride):
        output=tf.layers.max_pooling2d(inputs=x,pool_size=poolSize,strides=stride,padding='SAME')

        #if self.verbose:
            #print('Layer%d:type=MaxPool,poolSize=%d,stride=%d,outputshape=%s'
                  #%(id,poolSize,stride,str(output.get_shape())))
        return output

    # 扁平层：因为接下来会连接全连接层，例如[n_samples, 7, 7, 32] -> [n_samples, 7*7*32]
    def __Flatten(self,x):
        tranX=tf.transpose(x,[0,3,1,2]) # [batch,行,列,通道数channels] -> [batch,通道数channels,列,行]
        #求积函数product()
        nums=np.product(x.get_shape().as_list()[1:])   # 计算的是总共的神经元数量，第一个表示batch数量所以去掉
        # [batch,通道数channels,列,行] -> [batch,通道数channels*列*行],-1代表自适应batch数量
        return tf.reshape(tranX,[-1,nums])

    # 全连接层：x输入；id：层数索引；num_out：输出尺寸；activation：激活函数
    def __FcLayer(self,x,id,numOut,activation=None):
        numIn=x.get_shape().as_list()[-1]    # 通道数/维度
        # 均值为0标准差为0.1的正态分布，初始化权重w；shape=行*列*通道数*卷积核个数
        scopeName='fc_'+str(id)
        with tf.variable_scope(scopeName):
            weights=tf.Variable(tf.truncated_normal(shape=[numIn,numOut],mean=0.0,stddev=0.1),name='weights')
            bias=tf.Variable(tf.zeros(shape=[numOut]),name='biases')   # 列向量
            output=tf.nn.xw_plus_b(x,weights,bias)

        # 正常全连接层是leak_relu激活函数；但是最后一层是liner函数
        if activation:
            output=activation(output)

        #if self.verbose:
            #print('Layer%d:type=Fc,numOut=%d,outputShape=%s'
                  #%(id,numOut,str(output.get_shape())))
        return output

    # 【2】解析网络的预测结果：先判断预测框类别，再NMS
    def _BuildDetector(self):
        self.width=tf.placeholder(tf.float32,name='imgW')
        self.height=tf.placeholder(tf.float32,name='imgH')

        # 网络回归[batch,7*7*30]：
        idx1=self.S*self.S*self.C
        idx2=idx1+self.S*self.S*self.B

        # 1.类别概率[:,:7*7*20]  20维
        classPorb=tf.reshape(self.predicts[0,:idx1],[self.S,self.S,self.C])
        # 2.置信度[:,7*7*20:7*7*(20+2)]  2维
        confs=tf.reshape(self.predicts[0,idx1:idx2],[self.S,self.S,self.B])
        # 3.边界框[:,7*7*(20+2):]  8维 -> (x,y,w,h)
        boxes=tf.reshape(self.predicts[0,idx2:],[self.S,self.S,self.B,4])

        # 将x，y转换为相对于图像左上角的坐标
        # w，h的预测是平方根乘以图像的宽度和高度
        #self.xOffset.shape=[self.S,self.S,self.B](第ij个点第【1，2】个盒的偏移量)
        #/self.S*self.width:将特征点映射回原图
        boxes=tf.stack([(boxes[:,:,:,0]+tf.constant(self.xOffset,dtype=tf.float32))/self.S*self.width,
                        (boxes[:,:,:,1]+tf.constant(self.yOffset,dtype=tf.float32))/self.S*self.height,
                        tf.square(boxes[:,:,:,2])*self.width,
                        tf.square(boxes[:,:,:,3])*self.height
                        ],axis=3)
        # 类别置信度分数：[S,S,B,1]*[S,S,1,C]=[S,S,B,类别置信度C]
        #当前bounding box中含有object的置信度Pr(Object)
        #当前bounding box预测目标位置的准确性IOU(pred|truth)
        scores=tf.expand_dims(confs,-1)*tf.expand_dims(classPorb,2)

        scores=tf.reshape(scores,[-1,self.C])   # [S*S*B, C]
        boxes=tf.reshape(boxes,[-1,4])  # [S*S*B, 4]

        # 只选择类别置信度最大的值作为box的类别、分数
        boxClass=tf.argmax(scores,axis=1)   # 边界框box的类别
        boxCLassScores=tf.reduce_max(scores,axis=1) # 边界框box的分数

        # 利用类别置信度阈值self.threshold，过滤掉类别置信度低的
        filterMask=boxCLassScores>self.thresholld
        scores=tf.boolean_mask(boxCLassScores,filterMask)
        boxes=tf.boolean_mask(boxes,filterMask)
        boxClass=tf.boolean_mask(boxClass,filterMask)

        # NMS (不区分不同的类别)
        # 中心坐标+宽高box (x, y, w, h) -> xmin=x-w/2 -> 左上+右下box (xmin, ymin, xmax, ymax)，因为NMS函数是这种计算方式
        _boxes=tf.stack([boxes[:,0]-0.5*boxes[:,2],
                         boxes[:,1]-0.5*boxes[:,3],
                         boxes[:,0]+0.5*boxes[:,2],
                         boxes[:,1]+0.5*boxes[:,3]],axis=1)
        nmsIndices=tf.image.non_max_suppression(_boxes,scores,self.maxOutputSize,self.iouThreshold)

        self.scores=tf.gather(scores,nmsIndices)
        self.boxes=tf.gather(boxes,nmsIndices)
        self.boxClasses=tf.gather(boxClass,nmsIndices)

    # 【3】导入权重文件
    def _LoadWeights(self,weightsFile):
        #if self.verbose:
            # 打印状态信息
            #print("Start to load weights file from %s."%weightsFile)

        # 导入权重
        saver=tf.train.Saver()  # 初始化
        saver.restore(self.sess,weightsFile)    # saver.restore导入/saver.save保存

    # 【4】从预测输入图片，并可视化检测边界框、将obj的分类结果和坐标保存成txt。
    def __DetectFromImage(self,img):
        """Do detection given a cv image"""
        imgH,imgW,_=img.shape
        imgResized=cv2.resize(img,(448,448))
        imgRGB=cv2.cvtColor(imgResized,cv2.COLOR_BGR2RGB)
        imgResizedNp=np.asarray(imgRGB) #将结构数据转化为ndarray。
        imgs=np.zeros((1,448,448,3),dtype=np.float32)
        imgs[0]=(imgResizedNp/255.0)*2.0-1.0
        scores,boxes,boxClasses=self.sess.run([self.scores,self.boxes,self.boxClasses],
                                              feed_dict={self.images:imgs,self.width:imgW,self.height:imgH})
        return scores,boxes,boxClasses

    def __ShowResult(self,img,results,imshow=True,detectedBoxesFile=None,detectedImgFile=None):
        """Show the detection boxes"""
        imgCp=img.copy()
        f=open(detectedBoxesFile,'w') if detectedBoxesFile else False
        # draw boxes
        for i in range(len(results)):
            x=int(results[i][1])
            y=int(results[i][2])
            w=int(results[i][3])//2
            h=int(results[i][4])//2
            if self.verbose:
                #print('class:%s,[x,y,w,h]=[%d,%d,%d,%d],confidence=%f'
                      #%(results[i][0],x,y,w,h,results[i][-1]))
                # 中心坐标 + 宽高box(x, y, w, h) -> xmin = x - w / 2 -> 左上 + 右下box(xmin, ymin, xmax, ymax)
                cv2.rectangle(imgCp, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                # 在边界框上显示类别、分数(类别置信度)
                cv2.rectangle(imgCp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)  # puttext函数的背景
                cv2.putText(imgCp, results[i][0] + ' : %.2f' % results[i][5], (x - w + 5, y - h - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                # 保存obj检测结果为txt文件
        if f:
            f.write(results[i][0] + ',' + str(x) + ',' + str(y) + ',' +str(w) + ',' + str(h) + ',' + str(results[i][5]) + '\n')
        if imshow:
            cv2.imshow('YOLO_small detection', imgCp)
            cv2.waitKey()
        if detectedImgFile:
            cv2.imwrite(detectedImgFile, imgCp)
        if f:
            f.close()

    # image_file是输入图片文件路径；
    # deteted_boxes_file="boxes.txt"是最后坐标txt；detected_image_file="detected_image.jpg"是检测结果可视化图片
    def DetectFromFile(self,imgFile,imshow=True,
                        detectedBoxesFile='boxes.txt',detectedImgFile='detectedImg.jpg'):
        img=cv2.imread(imgFile)
        imgH,imgW,_=img.shape
        scores,boxes,boxesClasses=self.__DetectFromImage(img)
        predictBoxes=[]
        #print('--检测到个数：--',len(scores))
        for i in range(len(scores)):
            #预测框数据为：[概率, x, y, w, h, 类别置信度
            predictBoxes.append((self.classes[boxesClasses[i]],boxes[i,0],
                                boxes[i,1],boxes[i,2],boxes[i,3],scores[i]))
        self.__ShowResult(img,predictBoxes,imshow,detectedBoxesFile,detectedImgFile)


if __name__=='__main__':
    weightsPath='./weights/YOLO_small.ckpt'
    imgPath='./examples/person.jpg'
    yoloNet = Yolo(weightsFile=weightsPath)

    yoloNet.DetectFromFile(imgPath)