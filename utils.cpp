#include "utils.h"

using namespace cv;

double getDist(Point a, Point b) {
    return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}

float getAngle(Point a, Point b, Point c) {
    float l1 = getDist(b, a);
    float l2 = getDist(b, c);
    float dot = (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y);
    float angle = acos(dot / (l1 * l2));
    angle = angle * 180 / CV_PI;
    return angle;
}

void mask_morph(Mat &mask) {
    Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
    erode(mask, mask, erodeElement);
    erode(mask, mask, erodeElement);
    dilate(mask, mask, dilateElement);
    dilate(mask, mask, dilateElement);
}

Mat deleteBg(Mat img, Mat &out, Ptr<BackgroundSubtractor> bgs, bool learn, int thresh_sens_val) {
    Mat fgMask, grayscale, threshDiff;
    bgs->apply(img, fgMask, learn);

    threshold(fgMask, threshDiff, thresh_sens_val, 255, THRESH_BINARY);
    mask_morph(threshDiff);

    Mat res;
    img.copyTo(res, threshDiff);
    out = res;

    return threshDiff;
}