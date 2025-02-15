#include "live_feed.h"
#include <iostream>

LiveFeed::LiveFeed() 
    : capture(0) {
    if (!this->capture.isOpened()) {
        std::cerr << "Error opening camera" << std::endl;
    } else {
        std::cout << "Camera opened" << std::endl;
    }
    std::cout << "LiveFeed created" << std::endl;
}

LiveFeed::~LiveFeed() {
    this->capture.release();
    cv::destroyAllWindows();
    std::cout << "LiveFeed destroyed" << std::endl;
}

void LiveFeed::start() {
    std::cout << "Live feed started, press 'q' to end feed" << std::endl;
    while (true) {
        this->capture >> this->frame;
        if (this->frame.empty()) {
            std::cerr << "Error, blank frame captured" << std::endl;
            break;
        }
        cv::imshow("Live Feed", this->frame);
        char c = static_cast<char>(cv::waitKey(1));
        if (c == 'q') {
            break;
        }
    }

    std::cout << "Live feed ended" << std::endl;
}

cv::Mat LiveFeed::get_frame() {
    this->capture >> this->frame;
    return this->frame;
}

void LiveFeed::warmup_cam() {
    std::cout << "Warming up camera..." << std::endl;
    cv::Mat frame;
    for (int i = 0; i < 20; i++) {
        this->capture >> frame;
    }
    std::cout << "Camera warmed up" << std::endl;
}