#ifndef _MODEL_H_
#define _MODEL_H_

#include <string>
#include <vector>
#include <opencv2/dnn.hpp>

typedef struct {
    std::vector<int> ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<int> indices;
} detection_struct;

class Model {
    private:
        // private members
        std::string cfg;
        std::string weights;
        std::string model_path;
        std::string class_path;
        cv::dnn::Net net;
        std::vector<std::string> classes;
        std::vector<std::string> output_layers;

    public:
        // public members
        Model(std::string cfg = "", std::string weights = "", std::string class_path = "");
        Model(std::string model_path = "", std::string class_path = "");
        ~Model();
        void print();
        void set_cfg(std::string cfg);
        void set_weights(std::string weights);
        void set_class_path(std::string class_path);
        void set_classes();
        void set_net();
        void set_output_layers();
        std::string get_cfg();
        std::string get_weights();
        std::string get_class_path();
        cv::dnn::Net get_net();
        std::vector<std::string> get_classes();
        std::vector<std::string> get_output_layers();
        detection_struct detect_objects(cv::Mat frame);
        cv::Mat draw_boxes(cv::Mat frame, detection_struct detection_results);
};
#endif