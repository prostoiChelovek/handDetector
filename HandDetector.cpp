#include "HandDetector.h"

using namespace cv;
using namespace std;

HandDetector::HandDetector() = default;

bool HandDetector::loadCascade(String path) {
    if (cascadePath.empty())
        cascadePath = path;
    cascade.load(cascadePath);
    cascadeLoaded = !cascade.empty();
    return !cascade.empty();
}

void HandDetector::findHandsContours(Mat img) {
    hands.clear();
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    for (const auto &contour : contours) {
        Hand h(contour, shouldCheckSize, shouldCheckAngles);
        h.getBr();
        if (h.checkSize())
                hands.push_back(h);
    }
}

Mat HandDetector::deleteBg(Mat img, Mat bg, Mat &out) {
    Mat deltaImg;
    absdiff(img, bg, deltaImg);

    Mat grayscale, threshDiff;
    cvtColor(deltaImg, grayscale, CV_BGR2GRAY);
    threshold(grayscale, threshDiff, bgs_thresh_sens_val, 255, THRESH_BINARY);
    mask_morph(threshDiff);

    Mat res;
    img.copyTo(res, threshDiff);
    out = res;
    return threshDiff;
}

Mat HandDetector::detectHands_range(Mat img, Scalar lower, Scalar upper) {
    Mat mask;
    inRange(img, lower, upper, mask);
    threshold(mask, mask, range_thresh_sens_val, 255, THRESH_BINARY);
    mask_morph(mask);
    if (blur_range) {
        blur(mask, mask, range_blur_ksize);
        threshold(mask, mask, range_thresh_sens_val, 255, THRESH_BINARY);
    }
    findHandsContours(mask);
    return mask;
}

void HandDetector::detectHands_Cascade(Mat img) {
    if (!cascadeLoaded)
        return;
    cvtColor(img, img, COLOR_BGR2GRAY);
    vector<Rect> rects;
    cascade.detectMultiScale(img, rects, 1.1,
                             2, 0 | CASCADE_SCALE_IMAGE);
    for (Rect &r : rects) {
        Mat i = img(r);
        Mat thresh;
        threshold(i, thresh, cascade_thresh_sens_val, 255, THRESH_BINARY);
        findHandsContours(thresh);
    }
}

void HandDetector::getFingers() {
    for (Hand &h : hands) {
        h.getFingers();
        checkHands();
    }
}

void HandDetector::getHigherFingers() {
    for (Hand &h : hands) {
        h.getHigherFinger();
    }
}

void HandDetector::getFarthestFingers() {
    for (Hand &h : hands) {
        h.getFarthestFinger();
    }
}

void HandDetector::checkHands() {
    int i = 0;
    for (Hand &h : hands) {
        if (!h.checkSize() || h.fingers.empty())
            hands.erase(hands.begin() + i);
        i++;
    }
}

void HandDetector::getCenters() {
    for (Hand &h : hands) {
        h.getCenter();
    }
}

void HandDetector::drawHands(Mat &img, const Scalar color, int thickness) {
    int i = 0;
    for (Hand &h : hands) {
        h.draw(img, color, thickness);
        drawContours(img, contours, i, color);
        i++;
    }
}