#ifndef HANDDETECTOR_UTILS_H
#define HANDDETECTOR_UTILS_H

#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

float getAngle(Point a, Point b, Point c);

double getDist(Point a, Point b);

void mask_morph(Mat &mask);

Mat deleteBg(Mat img, Mat &out, Ptr<BackgroundSubtractor> bgs, bool learn = false,
             int thresh_sens_val = 20);

#endif //HANDDETECTOR_UTILS_H
