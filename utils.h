#ifndef HANDDETECTOR_UTILS_H
#define HANDDETECTOR_UTILS_H

#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1);

double getDist(Point a, Point b);

void mask_morph(Mat &mask);

#endif //HANDDETECTOR_UTILS_H
