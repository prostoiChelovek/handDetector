#ifndef HANDDETECTOR_UTILS_H
#define HANDDETECTOR_UTILS_H

#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;

float getAngle(Point s, Point f, Point e);

double getDist(Point a, Point b);

void mask_morph(Mat &mask);

#endif //HANDDETECTOR_UTILS_H
