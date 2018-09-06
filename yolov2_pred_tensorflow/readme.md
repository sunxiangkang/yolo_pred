# YoLoV2-Predict #
## 说明 ##
+ config.py:保存配置
+ darknet.py:通过tensorflow的slim模块，定义了darknet网络
+ decode.py:对网络输出进行解码
+ utils:用到的一些工具函数
+ pipline.py:主文件
+ demo:demo图片
+ weights:包含checkpoint_dir和weightsname.py。其中checkpoint_dir保存下载的训练好的模型，weightsname.py用于获得下载的训练好的模型和定义的模型的变量的名称。如果使用别的训练好的模型，可以使用weightsname.py获取变量名称，然后更改config.py
## 运行方式 ##
+ 下载[训练好的模型](https://pan.baidu.com/s/1wQaO0DLTsM14DF-HdrbS9g)，解压到./weights/checkpoint_dir/
+ 运行"python pipline.py"
## 参考 ##
[yolo2-pytorch-master](https://github.com/longcw/yolo2-pytorch)

