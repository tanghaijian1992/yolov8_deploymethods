该rknn模型为  https://blog.csdn.net/zhangqian_1/article/details/145401547?spm=1001.2014.3001.5502  中的第二种部署方法，有做量化，识别椅子圆盘类

1.该例子为一个ros工程，作用是读取ros图像话题并做椅子圆盘的识别。  

2.然后需要手动修改一下读取rknn的路径，默认接收图像话题为/device_0/rgb/rgb_raw ,可以使用example根目录下的 rosbag.bag 来播放获取，可实时读取图像话题并进行推理

3.可以直接  catkin make， 编译完成后则可运行 roslaunch yolov8_rknn_ros rknn_engine.launch, 即可实时读取图像话题并进行推理

