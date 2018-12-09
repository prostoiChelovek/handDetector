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
    threshold(img, img, thresh_sens_val, 255, THRESH_BINARY);
    findContours(img, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    for (auto &contour : contours) {
        Hand h(contour, shouldCheckSize, shouldCheckAngles);
        if (h.checkSize())
                hands.push_back(h);
    }
}

Mat HandDetector::deleteBg(Mat img, Mat bg, Mat &out) {
    Mat deltaImg;
    absdiff(img, bg, deltaImg);

    Mat grayscale, threshDiff;
    cvtColor(deltaImg, grayscale, CV_BGR2GRAY);
    threshold(grayscale, threshDiff, thresh_sens_val, 255, THRESH_BINARY);
    mask_morph(threshDiff);

    Mat res;
    img.copyTo(res, threshDiff);
    out = res;
    return threshDiff;
}

Mat HandDetector::detectHands_range(Mat img, Scalar lower, Scalar upper) {
    hands.clear();
    Mat mask;
    inRange(img, lower, upper, mask);
    threshold(mask, mask, thresh_sens_val, 255, THRESH_BINARY);
    mask_morph(mask);
    if (shouldBlur)
        blur(mask, mask, blurKsize);
    findHandsContours(mask);
    return mask;
}

void HandDetector::detectHands_Cascade(Mat img) {
    if (!cascadeLoaded)
        return;
    hands.clear();
    cvtColor(img, img, COLOR_BGR2GRAY);
    vector<Rect> rects;
    cascade.detectMultiScale(img, rects, 1.1,
                             2, 0 | CASCADE_SCALE_IMAGE);
    for (Rect &r : rects) {
        findHandsContours(img(r));
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