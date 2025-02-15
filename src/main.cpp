#include "live_feed.h"
#include "model.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

void image_detection(cv::Mat frame, Model model) {
    detection_struct detection_results;
    std::string image_path = cv::samples::findFile("data/bicycle_cars_people.jpeg");
    frame = cv::imread(image_path);
    detection_results = model.detect_objects(frame);
    frame = model.draw_boxes(frame, detection_results);
    cv::imshow("Detected Objects", frame);
    cv::waitKey(0);
}

void video_detection(cv::Mat frame, Model model, LiveFeed live_feed) {
    detection_struct detection_results;
    live_feed.warmup_cam();
    while (true) {
        frame = live_feed.get_frame();
        if (frame.empty()) {
            std::cerr << "Error, blank frame captured" << std::endl;
        } else {
            // call function to detect objects with dnn
            detection_results = model.detect_objects(frame);
            // call function to draw boxes on frame
            frame = model.draw_boxes(frame, detection_results);
            cv::imshow("Detected Objects", frame);
        }
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}

int main() {
    // video feed from webcam
    LiveFeed live_feed;
    cv::Mat frame;

    // Model model("cfg/yolov3-tiny.cfg", "weights/yolov3-tiny.weights", "cfg/coco.names");
    Model model("cfg/yolov4-tiny.cfg", "weights/yolov4-tiny.weights", "cfg/coco.names");
    model.print();

    // image_detection(frame, model);
    video_detection(frame, model, live_feed);

    return 0;
}
