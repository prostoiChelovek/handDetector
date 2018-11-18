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
    Point ptStart; // fingertip
    Point ptEnd, ptFar;

    float depth;

    float maxAngle = 95;
    int minDepth = 11;

    bool ok = false;

    bool shouldCheckAngles = true;
    bool shouldCheckDist = false; // check distance between ptStart, ptFar and ptEnd;

    Rect boundingBox; // hand`s

    Finger();

    Finger(const Vec4i &v, const vector<Point> &cnt, const Rect &boundingBox_,
           bool shouldCheckAngles_ = true, bool shouldCheckDist_ = true);

    bool checkDepth();
    bool checkAngles();

    bool checkDists(); // check distance between ptStart, ptFar and ptEnd;
    bool check();

    void getPoints(vector<Point> cnt, Vec4i v);

    void draw(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2,
              Scalar fingertipColor = Scalar(0, 0, 0));
};


#endif //HANDDETECTOR_FINGER_H
