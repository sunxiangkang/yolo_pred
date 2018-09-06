# YoLoV3-Pred #
## 说明 ##
+ preprocessimg.py:对图片进行预处理，使符合网络输入要求
+ darknet53.py:yolov3的特征提取网络(darknet-53)和物体检测网络
+ getmodelout.py:加载训练好的二值化pb模型，获得模型输出
+ decode.py:对模型输出进行解码
+ filterboxes.py:对解码后的boxes进行筛选，选出scores大于设定阈值的boxes
+ nms.py:对符合条件的boxes进行非极大值抑制
+ drawdetections.py:画出boxes
+ pipline.py:组合各个功能模块
+ config.py:保存网络运行的配置
+ model:保存了官网下载的weights转换到Keras模型的程序和Keras模型转换到tensorflow的二进制pb模型的程序，以及pb模型的tensorName
## 运行方式 ##
+ 下载训[练好的模型](https://pan.baidu.com/s/1hobB8P947ODJTKGdl7kUvQ)解压，将pb模型放至./model/yolo3/saved_model.pb
+ 运行"python pipline.py"
## 参考 ##
[xiaochus](https://github.com/xiaochus/YOLOv3)

[qqwweee](https://github.com/qqwweee/keras-yolo3)

[IronMastiff](https://github.com/IronMastiff/YOLOv3_tensorflow)

[IronMastiff博客](https://blog.csdn.net/IronMastiff/article/details/79940118)

ps:三位大牛的项目都有点小问题（小弟的拙见），有兴趣的同学可以自己采坑，一起交流。