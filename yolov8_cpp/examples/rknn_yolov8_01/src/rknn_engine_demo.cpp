#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>


#define NMS_THRESH 0.45
#define OBJECT_THRESH 0.1
#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640

using namespace std;
using namespace cv;

// 定义类别
const vector<string> CLASSES = {"plate_base"};
const int CLASS_NUM = CLASSES.size();
const int STRIDES[3] = {8, 16, 32};

rknn_context ctx;
int ret;
rknn_input inputs[1];
rknn_output outputs[1];

// 计算 anchors 数量
const int ANCHORS = (INPUT_WIDTH / STRIDES[0]) * (INPUT_HEIGHT / STRIDES[0]) +
                    (INPUT_WIDTH / STRIDES[1]) * (INPUT_HEIGHT / STRIDES[1]) +
                    (INPUT_WIDTH / STRIDES[2]) * (INPUT_HEIGHT / STRIDES[2]);

struct DetectBox {
    int classId;
    float score;
    float xmin, ymin, xmax, ymax;

    DetectBox(int classId, float score, float xmin, float ymin, float xmax, float ymax)
        : classId(classId), score(score), xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}
};

// 计算IOU
float IOU(DetectBox &box1, DetectBox &box2) {
    float xmin = max(box1.xmin, box2.xmin);
    float ymin = max(box1.ymin, box2.ymin);
    float xmax = min(box1.xmax, box2.xmax);
    float ymax = min(box1.ymax, box2.ymax);

    float interWidth = max(0.0f, xmax - xmin);
    float interHeight = max(0.0f, ymax - ymin);
    float interArea = interWidth * interHeight;

    float area1 = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin);
    float area2 = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin);

    return interArea / (area1 + area2 - interArea);
}

// 非极大值抑制（NMS）
vector<DetectBox> NMS(vector<DetectBox> &detectResult) {
    vector<DetectBox> predBoxes;

    // 置信度排序
    sort(detectResult.begin(), detectResult.end(),
         [](const DetectBox &a, const DetectBox &b) { return a.score > b.score; });

    for (size_t i = 0; i < detectResult.size(); i++) {
        if (detectResult[i].classId == -1) continue;
        predBoxes.push_back(detectResult[i]);

        for (size_t j = i + 1; j < detectResult.size(); j++) {
            if (detectResult[j].classId == detectResult[i].classId) {
                if (IOU(detectResult[i], detectResult[j]) > NMS_THRESH) {
                    detectResult[j].classId = -1;
                }
            }
        }
    }
    return predBoxes;
}

// 后处理
vector<DetectBox> postprocess(float *output, int img_h, int img_w) {
    vector<DetectBox> detectResult;
    float scale_h = float(img_h) / INPUT_HEIGHT;
    float scale_w = float(img_w) / INPUT_WIDTH;

    for (int i = 0; i < ANCHORS; i++) {
        int cls_index = 0;
        float cls_value = -1;
        for (int cl = 0; cl < CLASS_NUM; cl++) {
            float val = output[i + (4 + cl) * ANCHORS];
            if (val > cls_value) {
                cls_value = val;
                cls_index = cl;
            }
        }

        if (cls_value > OBJECT_THRESH) {
            float cx = output[i];
            float cy = output[i + ANCHORS];
            float cw = output[i + 2 * ANCHORS];
            float ch = output[i + 3 * ANCHORS];

            float xmin = (cx - 0.5 * cw) * scale_w;
            float ymin = (cy - 0.5 * ch) * scale_h;
            float xmax = (cx + 0.5 * cw) * scale_w;
            float ymax = (cy + 0.5 * ch) * scale_h;

            detectResult.emplace_back(cls_index, cls_value, xmin, ymin, xmax, ymax);
        }
    }
    return NMS(detectResult);
}

// 加载并运行 RKNN 推理
vector<DetectBox> run_rknn_inference(Mat &image) {
    // 设置输入
    Mat img_resized;
    resize(image, img_resized, Size(INPUT_WIDTH, INPUT_HEIGHT));
    cvtColor(img_resized, img_resized, COLOR_BGR2RGB);

    inputs[0].index = 0;
    inputs[0].buf = img_resized.data;
    inputs[0].size = INPUT_WIDTH * INPUT_HEIGHT * 3;
    inputs[0].pass_through = false;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret != 0) {
        cerr << "rknn_inputs_set failed: " << ret << endl;
        exit(-1);
    }

    // 推理
    ret = rknn_run(ctx, nullptr);
    if (ret != 0) {
        cerr << "rknn_run failed: " << ret << endl;
        exit(-1);
    }

    // 获取输出
    outputs[0].want_float = true;
    ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
    if (ret != 0) {
        cerr << "rknn_outputs_get failed: " << ret << endl;
        exit(-1);
    }

    // 解析输出
    vector<DetectBox> results = postprocess((float *)outputs[0].buf, image.rows, image.cols);

    return results;
}

cv::Mat convertImageMsgToMatManual(const sensor_msgs::ImageConstPtr& msg) {
    // 验证图像数据有效性
    if (msg->data.empty()) {
        throw std::runtime_error("Empty image data");
    }

    // 创建 OpenCV 矩阵
    cv::Mat mat;
    
    // 根据编码格式解析图像数据
    if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
        mat = cv::Mat(msg->height, msg->width, CV_8UC3, 
                     const_cast<uchar*>(&msg->data[0])).clone();
        std::cout<<"encoding type bgr8!"<<std::endl;
        // cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB); // 可选：转换为RGB
    }
    else if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
        mat = cv::Mat(msg->height, msg->width, CV_8UC3, 
                     const_cast<uchar*>(&msg->data[0])).clone();
    }
    else if (msg->encoding == sensor_msgs::image_encodings::MONO8) {
        mat = cv::Mat(msg->height, msg->width, CV_8UC1, 
                     const_cast<uchar*>(&msg->data[0])).clone();
    }
    else if (msg->encoding == sensor_msgs::image_encodings::BAYER_RGGB8) {
        // 示例：拜耳阵列解码（需根据实际需求实现）
        cv::Mat bayer_mat(msg->height, msg->width, CV_8UC1, 
                         const_cast<uchar*>(&msg->data[0]));
        cv::cvtColor(bayer_mat, mat, cv::COLOR_BayerBG2BGR); // 转换为BGR
    }
    else {
        throw std::runtime_error("Unsupported encoding: " + msg->encoding);
    }

    // 处理步长（stride）不一致的情况
    if (msg->step != msg->width * mat.elemSize()) {
        cv::Mat tmp_mat(msg->height, msg->width, mat.type(),
                       const_cast<uchar*>(&msg->data[0]), msg->step);
        mat = tmp_mat.clone();
    }

    return mat;
}

class YoloV8RKNN {
public:
    YoloV8RKNN() {

        // 订阅 ROS 话题
        sub = nh.subscribe("/device_0/rgb/rgb_raw", 1, &YoloV8RKNN::imageCallback, this);

        std::string model_path = "/home/robot/yolov8_deploymethods/yolov8_cpp/examples/rknn_yolov8_01/yolov8_plate.rknn";            //********需要改为自己电脑的路径 */
        void* model_ptr = (void*)model_path.c_str();  // 推荐
        ret = rknn_init(&ctx, model_ptr, 0, 0, NULL);
        if (ret != 0) {
            cerr << "rknn_init failed: " << ret << endl;
            exit(-1);
        }
    }

    ~YoloV8RKNN() {
        // 释放 RKNN 资源
        rknn_outputs_release(ctx, 1, outputs);
        rknn_destroy(ctx);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber sub;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        // cv_bridge::CvImagePtr cv_ptr;
        // try {
        //     cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        // } catch (cv_bridge::Exception& e) {
        //     ROS_ERROR("cv_bridge 错误: %s", e.what());
        //     return;
        // }

        // cv::Mat input_img = cv_ptr->image;
        // // cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
        // ROS_INFO("image size: %d %d", input_img.rows, input_img.cols);

        // imwrite("/home/robot/rknn_ws/test_write.jpg", input_img);
        // Mat image = imread("/home/robot/rknn_ws/test_write.jpg");


        // 获取开始时间点
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat image = convertImageMsgToMatManual(msg);

        // cv::resize(input_img, input_img, cv::Size(640, 640));
        vector<DetectBox> predictions = run_rknn_inference(image);
        ROS_INFO("objects size: %d", predictions.size());

        // 画出检测框
        for (const auto &box : predictions) {
            rectangle(image, Point(box.xmin, box.ymin), Point(box.xmax, box.ymax), Scalar(255, 0, 0), 2);
            putText(image, CLASSES[box.classId] + to_string(box.score), Point(box.xmin, box.ymin - 5),
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }

        // imwrite("/home/robot/rknn_ws/test_rknn_result.jpg", image);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "myFunction 运行时间: " << duration.count() << " 毫秒" << std::endl;
        cout << "Detection complete!" << endl;
        cv::waitKey(600);
    }
};

int main(int argc, char** argv) {
    // cout << "Running RKNN Model..." << endl;

    // // 读取图像
    // Mat image = imread("/home/robot/rknn_ws/test.jpg");
    // if (image.empty()) {
    //     cerr << "Failed to load image!" << endl;
    //     return -1;
    // }

    // vector<DetectBox> predictions = run_rknn_inference(image);

    // // 画出检测框
    // for (const auto &box : predictions) {
    //     rectangle(image, Point(box.xmin, box.ymin), Point(box.xmax, box.ymax), Scalar(255, 0, 0), 2);
    //     putText(image, CLASSES[box.classId] + to_string(box.score), Point(box.xmin, box.ymin - 5),
    //             FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    // }

    // imwrite("/home/robot/rknn_ws/test_rknn_result.jpg", image);
    // cout << "Detection complete!" << endl;

    ros::init(argc, argv, "yolov8_rknn");
    YoloV8RKNN node;
    ros::spin();

    return 0;
}
