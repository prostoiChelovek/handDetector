//
// Created by prostoichelovek on 29.01.19.
// https://github.com/osamazhar/MouseCursorKalmanFilter
//

#ifndef HANDDETECTOR_FILTER_H
#define HANDDETECTOR_FILTER_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

class Filter {
public:
    unsigned int F_type = CV_32F;

    int stateSize, measSize;
    KalmanFilter KF;

    Filter() = default;

    Filter(int stateSize_, int measSize_,
           Scalar processNoiseCov, Scalar measurementNoiseCov, Scalar errorCovPost, int contrSize = 0);

    vector<float> predict();

    void update(vector<float> newData);

private:
    double ticks = 0;
    Mat meas;

};


#endif //HANDDETECTOR_FILTER_H
