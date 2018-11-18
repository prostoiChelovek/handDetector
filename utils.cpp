#include "utils.h"

using namespace cv;

// https://picoledelimao.github.io/blog/2015/11/15/fingertip-detection-on-opencv/
float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1) {
    float dist1 = sqrt((px1 - cx1) * (px1 - cx1) + (py1 - cy1) * (py1 - cy1));
    float dist2 = sqrt((px2 - cx1) * (px2 - cx1) + (py2 - cy1) * (py2 - cy1));
    float Ax, Ay;
    float Bx, By;
    float Cx, Cy;
    Cx = cx1;
    Cy = cy1;
    if (dist1 < dist2) {
        Bx = px1;
        By = py1;
        Ax = px2;
        Ay = py2;
    } else {
        Bx = px2;
        By = py2;
        Ax = px1;
        Ay = py1;
    }
    float Q1 = Cx - Ax;
    float Q2 = Cy - Ay;
    float ptStart = Bx - Ax;
    float ptEnd = By - Ay;
    float A = acos((ptStart * Q1 + ptEnd * Q2) / (sqrt(ptStart * ptStart + ptEnd * ptEnd) * sqrt(Q1 * Q1 + Q2 * Q2)));
    A = A * 180 / CV_PI;
    return A;
}

double getDist(Point a, Point b) {
    return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}

void mask_morph(Mat &mask) {
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
    erode(mask, mask, erodeElement);
    erode(mask, mask, erodeElement);
    dilate(mask, mask, dilateElement);
    dilate(mask, mask, dilateElement);
}