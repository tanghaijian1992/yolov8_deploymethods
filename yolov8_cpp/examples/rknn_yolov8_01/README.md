该rknn模型为yolov8n官方的模型结构，没有做量化，训练识别椅子圆盘类别。

1.该例子为一个ros工程，作用是读取ros图像话题并做椅子圆盘的识别。  

2.然后需要手动修改一下读取rknn的路径，默认接收图像话题为/device_0/rgb/rgb_raw ,可以使用example根目录下的 rosbag.bag 来播放获取，可实时读取图像话题并进行推理

3.可以直接  catkin make， 编译完成后则可运行 roslaunch yolov8_rknn_ros rknn_engine.launch, 即可实时读取图像话题并进行推理

