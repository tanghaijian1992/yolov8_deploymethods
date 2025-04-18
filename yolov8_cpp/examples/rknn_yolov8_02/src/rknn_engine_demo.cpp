#include <iostream>
#include <vector>
#include <cmath>
#include "rknn_api.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <fstream>


#include <im2d.h>
#include "rga.h"
#include "RgaUtils.h"
#undef INTER_LINEAR  // 去掉 RGA 中宏定义

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>


#define ONNX_MODEL "/home/robot/rknn_ws/last0303.onnx"
#define RKNN_MODEL "/home/robot/rknn_ws/last0303_3.rknn"
#define DATASET "/home/yjh/thj/data/plate_base/images/images_list.txt"

#define NMS_THRESH 0.45
#define OBJECT_THRESH 0.1
#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640

using namespace std;
using namespace cv;


std::vector<std::string> CLASSES = {"plate_base"};
int class_num = CLASSES.size();
int head_num = 3;
const float nms_thresh = 0.3f;
const float object_thresh = 0.1f;

std::vector<int> strides = {8, 16, 32};
std::vector<std::vector<int>> map_size = { {80, 80}, {40, 40}, {20, 20} };

const int input_height = 640;
const int input_width = 640;

rknn_context ctx;
int ret;
const int output_num = 2;
rknn_output outputs[output_num];

// 计算 anchor 总数
int anchors = (input_height / strides[0] * input_width / strides[0] +
               input_height / strides[1] * input_width / strides[1] +
               input_height / strides[2] * input_width / strides[2]);

// 全局网格数据（如果后续需要）
std::vector<float> meshgrid;
void GenerateMeshgrid() {
    for (int index = 0; index < head_num; index++) {
        for (int i = 0; i < map_size[index][0]; i++) {
            for (int j = 0; j < map_size[index][1]; j++) {
                meshgrid.push_back(j + 0.5f);
                meshgrid.push_back(i + 0.5f);
            }
        }
    }
}

// 检测框结构体
struct DetectBox {
    int classId;
    float score;
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    DetectBox(int _classId, float _score, float _xmin, float _ymin, float _xmax, float _ymax)
        : classId(_classId), score(_score), xmin(_xmin), ymin(_ymin), xmax(_xmax), ymax(_ymax) {}
};

// 计算 IOU
float IOU(float xmin1, float ymin1, float xmax1, float ymax1,
          float xmin2, float ymin2, float xmax2, float ymax2) {
    float xmin = std::max(xmin1, xmin2);
    float ymin = std::max(ymin1, ymin2);
    float xmax = std::min(xmax1, xmax2);
    float ymax = std::min(ymax1, ymax2);
    float innerWidth = std::max(0.0f, xmax - xmin);
    float innerHeight = std::max(0.0f, ymax - ymin);
    float innerArea = innerWidth * innerHeight;
    float area1 = (xmax1 - xmin1) * (ymax1 - ymin1);
    float area2 = (xmax2 - xmin2) * (ymax2 - ymin2);
    float total = area1 + area2 - innerArea;
    return innerArea / total;
}

// Sigmoid 函数
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// 非极大值抑制（NMS）
std::vector<DetectBox> NMS(std::vector<DetectBox>& detectResult) {
    std::vector<DetectBox> predBoxs;
    // 按 score 从大到小排序
    std::sort(detectResult.begin(), detectResult.end(),
              [](const DetectBox& a, const DetectBox& b) {
                  return a.score > b.score;
              });
    for (size_t i = 0; i < detectResult.size(); i++) {
        if (detectResult[i].classId == -1)
            continue;
        predBoxs.push_back(detectResult[i]);
        for (size_t j = i + 1; j < detectResult.size(); j++) {
            if (detectResult[i].classId == detectResult[j].classId) {
                float iou = IOU(detectResult[i].xmin, detectResult[i].ymin,
                                detectResult[i].xmax, detectResult[i].ymax,
                                detectResult[j].xmin, detectResult[j].ymin,
                                detectResult[j].xmax, detectResult[j].ymax);
                if (iou > nms_thresh) {
                    detectResult[j].classId = -1;
                }
            }
        }
    }
    return predBoxs;
}

// 后处理：解析输出，生成检测框
std::vector<DetectBox> postprocess(const std::vector<std::vector<float>>& outputs, int image_h, int image_w) {
    std::cout << "postprocess ..." << std::endl;
    float scale_h = static_cast<float>(image_h) / input_height;
    float scale_w = static_cast<float>(image_w) / input_width;

    std::vector<DetectBox> detectResult;
    const std::vector<float>& output_cls = outputs[0];
    const std::vector<float>& output_reg = outputs[1];

    std::cout<<"output_cls: ";
    for (int i = 0; i < anchors; i++) {
        int cls_index = 0;
        float cls_value = -1.0f;
        for (int cl = 0; cl < class_num; cl++) {
            float val = output_cls[i + cl * anchors];
            // std::cout<<val<<"  ";
            if (val > cls_value) {
                cls_value = val;
                cls_index = cl;
            }
        }
        if (cls_value > object_thresh) {
            float cx = output_reg[i + 0 * anchors];
            float cy = output_reg[i + 1 * anchors];
            float cw = output_reg[i + 2 * anchors];
            float ch = output_reg[i + 3 * anchors];

            float xmin = (cx - 0.5f * cw) * scale_w;
            float ymin = (cy - 0.5f * ch) * scale_h;
            float xmax = (cx + 0.5f * cw) * scale_w;
            float ymax = (cy + 0.5f * ch) * scale_h;

            detectResult.emplace_back(cls_index, cls_value, xmin, ymin, xmax, ymax);
        }
    }
    std::cout<<std::endl;
    std::cout << "detectResult: " << detectResult.size() << std::endl;
    std::vector<DetectBox> predBox = NMS(detectResult);
    return predBox;
}

static float DeQnt2F32(int8_t qnt, int zp, float scale)
{
    return ((float)qnt - (float)zp) * scale;
}

// RKNN 推理封装函数
std::vector<std::vector<float>> export_rknn_inference(const cv::Mat& img) {
    // 准备输入数据
    // 假设 img 为 RGB 格式，尺寸为 input_width x input_height，且数据类型为 CV_8UC3
    // 若需要归一化处理，可参考下述转换（本例中仅做归一化处理示例）
    cv::Mat input_img;
    img.convertTo(input_img, CV_8UC3, 1.0 );

    // 设置输入 tensor（具体参数根据模型要求调整）
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = input_img.data;
    inputs[0].size = input_img.total() * input_img.elemSize();
    inputs[0].pass_through = false;
    // 如果模型输入为UINT8型
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    // auto t0 = std::chrono::high_resolution_clock::now();

    ret = rknn_inputs_set(ctx, 1, inputs);
    if (ret < 0) {
        std::cerr << "Set input failed!" << std::endl;
        exit(ret);
    }

    // auto t1 = std::chrono::high_resolution_clock::now();
    // 运行模型
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        std::cerr << "Run model failed!" << std::endl;
        exit(ret);
    }

    // auto t2 = std::chrono::high_resolution_clock::now();

    // 获取输出（假设有两个输出）
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < output_num; i++) {
        outputs[i].want_float = false;
        outputs[i].is_prealloc = false;
    }
    ret = rknn_outputs_get(ctx, output_num, outputs, nullptr);
    if (ret < 0) {
        std::cerr << "Get outputs failed!" << std::endl;
        exit(ret);
    }

    // auto t3 = std::chrono::high_resolution_clock::now();

    rknn_tensor_attr output_attrs[output_num];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < output_num; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[0]), sizeof(rknn_tensor_attr));
    // if (output_attrs[0].type == RKNN_TENSOR_UINT8) {
    //     // uint8 处理
    //     std::cout<<"type: uint8"<<std::endl;
    // } else if (output_attrs[0].type == RKNN_TENSOR_INT8) {
    //     // int8 处理
    //     std::cout<<"type: int8"<<std::endl;
    // }

    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < output_num; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
        std::cout<<"  scale: "<<output_attrs[i].scale<<"  zp: "<<output_attrs[i].zp;
    }
    
    // 对所有数据做反量化处理
    std::vector<std::vector<float>> ret_outputs;
    for (int i = 0; i < output_num; i++) {
        int output_size = outputs[i].size / sizeof(int8_t);
        int8_t* output_ptr = reinterpret_cast<int8_t*>(outputs[i].buf);
        std::vector<int8_t> out_data(output_ptr, output_ptr + output_size);
        std::vector<float> conv_buff;
        for (size_t j = 0; j < out_data.size(); j++)
        {
            float conv_val = DeQnt2F32(out_data[j], out_zps[i], out_scales[i]);
            // float conv_val = out_data[j];
            conv_buff.push_back(conv_val);
            if (i == 0 && conv_val!=0)
            {
                // std::cout<<" origin data: "<<static_cast<int>(out_data[j])<<" conv data:"<<conv_val;
            }
        }
        ret_outputs.push_back(conv_buff);
    }

    // auto t4 = std::chrono::high_resolution_clock::now();


    // std::cout << "Preprocess: " << (std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)).count() << " ms\n";
    // std::cout << "Inference : " << (std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)).count()  << " ms\n";
    // std::cout << "OutputGet : " << (std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2)).count()  << " ms\n";
    // std::cout << "反量化: " << (std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3)).count()  << " ms\n";

    return ret_outputs;
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

        // 创建 RKNN 上下文
        std::string model_path = "/home/robot/yolov8_deploymethods/yolov8_cpp/examples/rknn_yolov8_02/yolov8_plate.rknn";      //需要替换为自己的目录路径

        // 读取模型文件到内存
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file.good()) {
            std::cerr << "Error opening model file: " << model_path << std::endl;
            exit(-1);
        }
        std::streamsize model_size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> model_data(model_size);
        if (!file.read(model_data.data(), model_size)) {
            std::cerr << "Error reading model file" << std::endl;
            exit(-1);
        }
        
        // 使用模型数据初始化
        ret = rknn_init(&ctx, model_data.data(), model_size, RKNN_FLAG_PRIOR_HIGH, nullptr);
        if (ret < 0) {
            std::cerr << "RKNN init failed!" << std::endl;
            exit(ret);
        }

    }

    ~YoloV8RKNN() {
        // 释放输出
        rknn_outputs_release(ctx, output_num, outputs);
        rknn_destroy(ctx);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber sub;

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {


        // 获取开始时间点
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat orig_img = convertImageMsgToMatManual(msg);

        //使用rga对图像进行resize和convert格式
        int src_width = orig_img.cols;
        int src_height = orig_img.rows;

        // 2. 目标尺寸
        int dst_width = input_width;
        int dst_height = input_height;

        // 3. 准备 src buffer（im2d 接口要求分配物理连续内存，测试中普通 Mat 也能用）
        im_rect src_rect = { 0, 0, src_width, src_height };
        im_rect dst_rect = { 0, 0, dst_width, dst_height };

        // 4. 创建目标 Mat（BGR）
        cv::Mat dst_img(dst_height, dst_width, CV_8UC3);

        // 5. 设置 src/dst 的 rga buffer
        rga_buffer_t src_buf = wrapbuffer_virtualaddr(
            orig_img.data, src_width, src_height, RK_FORMAT_BGR_888);
        rga_buffer_t dst_buf = wrapbuffer_virtualaddr(
            dst_img.data, dst_width, dst_height, RK_FORMAT_BGR_888);

        // 6. 执行 resize 操作
        ret = imresize_t(src_buf, dst_buf, 0, 0, 1, 1);
        if (ret != IM_STATUS_SUCCESS) {
            std::cerr << "RGA resize failed: " << imStrError(ret) << std::endl;
            return ;
        }

        std::cout << "RGA resize success." << std::endl;



        cv::Mat resized(dst_height, dst_width, CV_8UC3);
        rga_buffer_t src_convert = wrapbuffer_virtualaddr(dst_img.data, dst_width, dst_height, RK_FORMAT_BGR_888);
        rga_buffer_t dst_convert = wrapbuffer_virtualaddr(resized.data, dst_width, dst_height, RK_FORMAT_RGB_888);

        // 4. 执行格式转换（BGR -> RGB）
        // int ret = imconvert(src_convert, dst_convert, NULL);
        ret = imcvtcolor(src_convert, dst_convert, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
        if (ret != IM_STATUS_SUCCESS) {
            std::cerr << "RGA convert failed: " << imStrError(ret) << std::endl;
            return ;
        }

        std::cout << "RGA BGR -> RGB success." << std::endl;

        // 调用 RKNN 推理
        std::vector<std::vector<float>> outputs = export_rknn_inference(resized);

        // 后处理得到检测框
        std::vector<DetectBox> predbox = postprocess(outputs, src_height, src_width);
        std::cout << "Detected boxes: " << predbox.size() << std::endl;

        // 绘制检测结果
        for (size_t i = 0; i < predbox.size(); i++) {
            int xmin = static_cast<int>(predbox[i].xmin);
            int ymin = static_cast<int>(predbox[i].ymin);
            int xmax = static_cast<int>(predbox[i].xmax);
            int ymax = static_cast<int>(predbox[i].ymax);
            int classId = predbox[i].classId;
            float score = predbox[i].score;
            cv::rectangle(orig_img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(255, 0, 0), 2);
            std::string title = CLASSES[classId] + " " + std::to_string(score);
            cv::putText(orig_img, title, cv::Point(xmin, ymin), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 0, 255), 2);
        }
        // cv::imwrite("/home/robot/yolov8_deploymethods/yolov8_cpp/examples/rknn_yolov8_02/test_rknn_result_noconcat_qua.jpg", orig_img);
        // std::cout << "Result saved to ./test_rknn_result_noconcat.jpg" << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "myFunction 运行时间: " << duration.count() << " 毫秒" << std::endl;
        cout << "Detection complete!" << endl;
        // cv::waitKey(600);
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
