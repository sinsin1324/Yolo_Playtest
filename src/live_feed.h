#ifndef _LIVE_FEED_H_
#define _LIVE_FEED_H_

#include <string>
#include <opencv2/opencv.hpp>

class LiveFeed {
    private:
        cv::VideoCapture capture;
        cv::Mat frame;
    public:
        LiveFeed();
        ~LiveFeed();
        void start();
        cv::Mat get_frame();
        void warmup_cam();
};

#endif