//
// Created by mr_blaze on 18.10.18.
//

#ifndef HANDDETECTOR_FINGER_H
#define HANDDETECTOR_FINGER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.h"

using namespace cv;
using namespace std;

class Finger {
public:
    Point ptStart, ptEnd, ptFar;

    double angle, inAngle, length;
    float depth;
    bool shouldCheckAngles = true;

    int minDepth = 11;
    bool ok = false;

    // hand`s
    Rect boundingBox;

    Finger();

    Finger(const Vec4i &v, const vector<Point> &cnt, const Rect &boundingBox_,
           bool shouldCheckAngles_ = true);

    bool checkDepth();

    bool checkAngles();

    bool check();

    void getPoints(vector<Point> cnt, Vec4i v);

    // For use it, firstly call getPoints()
    void countAngles();

    void draw(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2,
              Scalar fingertipColor = Scalar(0, 0, 0));
};


#endif //HANDDETECTOR_FINGER_H
