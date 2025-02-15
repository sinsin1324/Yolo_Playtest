#include "model.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <fstream>

Model::Model(std::string cfg, std::string weights, std::string class_path)
    : cfg(cfg), weights(weights), class_path(class_path) {
    set_net();
    set_classes();
    set_output_layers();
    std::cout << "Model created with config & weights" << std::endl;
}

Model::Model(std::string model_path, std::string class_path)
    : model_path(model_path), class_path(class_path) {
    set_net();
    set_classes();
    set_output_layers();
    std::cout << "Model created with pytorchq" << std::endl;
}

Model::~Model() {
    std::cout << "Model destroyed" << std::endl;
}

void Model::print() {
    std::cout << "Model: " << this->cfg << ", " << this->weights << std::endl;
}

void Model::set_cfg(std::string cfg) {
    this->cfg = cfg;
}

void Model::set_weights(std::string weights) {
    this->weights = weights;
}

void Model::set_class_path(std::string class_path) {
    this->class_path = class_path;
}

void Model::set_net() {
    std::cout << "Setting network..." << std::endl;
    this->net = cv::dnn::readNetFromDarknet(this->cfg, this->weights);
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // Use GPU if available: DNN_TARGET_OPENCL
    std::cout << "Network set" << std::endl;
}

void Model::set_classes() {
    std::cout << "Setting classes..." << std::endl;
    std::ifstream file(this->class_path);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return;
    }
    while (std::getline(file, line)) {
        this->classes.push_back(line);
    }
    std::cout << "Classes set" << std::endl;
}

void Model::set_output_layers() {
    std::cout << "Setting output layers..." << std::endl;
    std::vector<int> layer_ids = this->net.getUnconnectedOutLayers();
    std::vector<std::string> layer_names = this->net.getLayerNames();
    
    for (int i : layer_ids) {
        this->output_layers.push_back(layer_names.at(i-1));
    }
    std::cout << "Output layers set" << std::endl;
}

std::string Model::get_cfg() {
    return this->cfg;
}

std::string Model::get_weights() {
    return this->weights;
}

std::string Model::get_class_path() {
    return this->class_path;
}

cv::dnn::Net Model::get_net() {
    return this->net;
}

std::vector<std::string> Model::get_classes() {
    return this->classes;
}

std::vector<std::string> Model::get_output_layers() {
    return this->output_layers;
}

detection_struct Model::detect_objects(cv::Mat frame) {
    // create blob from current frame (input to dnn)
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // forward pass
    std::vector<cv::Mat> network_output;
    net.forward(network_output, this->output_layers);

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // process network output
    for (auto& output : network_output) {
        for (int i = 0; i < output.rows; i++) {
            cv::Mat row = output.row(i);
            // data = [center_x, center_y, width, height, confidence, class_1, class_2, ..., class_n]
            float* data = output.ptr<float>(i);
            float confidence = data[4];
            // if (confidence > 0.5) {
            //     std::cout << "Confidence: " << confidence << std::endl;
            //     std::cout << "Class: " << std::distance(data+5, std::max_element(data + 5, data + output.cols)) << std::endl;
            // }
            if (confidence> 0.5) {
                // convert center_x, center_y, width, height to pixel values (they are normalized)
                float centerX = data[0] * frame.cols;
                float centerY = data[1] * frame.rows;
                float width = data[2] * frame.cols;
                float height = data[3] * frame.rows;

                // calculate starting x and y coordinates
                int starting_x = static_cast<int>(centerX - width/2);
                int starting_y = static_cast<int>(centerY - height/2);

                // get class with highest confidence score and where to draw boxes in the frame
                auto highest_class_score = std::max_element(data + 5, data + output.cols);
                classIds.push_back(std::distance(data + 5, highest_class_score));
                confidences.push_back(*highest_class_score);
                boxes.push_back(cv::Rect(starting_x, starting_y, width, height));
            }
        }
    }

    // apply non-maximum suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
    
    detection_struct results = {classIds, 
                                confidences, 
                                boxes,
                                indices};
    return results;
}

cv::Mat Model::draw_boxes(cv::Mat frame, detection_struct detection_results) {
    for (int id : detection_results.indices) {
        cv::Rect box = detection_results.boxes.at(id);
        cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 2);
        std::string label = cv::format("%.2f", detection_results.confidences[id]) + " " + this->classes[detection_results.ids[id]];
        cv::putText(frame, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
    return frame;
}