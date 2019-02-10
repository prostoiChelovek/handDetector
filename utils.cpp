#include "utils.h"

using namespace cv;

double getDist(Point a, Point b) {
    return sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
}

float getAngle(Point s, Point f, Point e) {
    float l1 = getDist(f, s);
    float l2 = getDist(f, e);
    float dot = (s.x - f.x) * (e.x - f.x) + (s.y - f.y) * (e.y - f.y);
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