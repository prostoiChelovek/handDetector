#include "HandDetector.h"

using namespace cv;
using namespace std;

void HandDetector::findHandsContours(Mat img) {
    threshold(img, img, thresh_sens_val, 255, THRESH_BINARY);
    findContours(img, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    for (auto &contour : contours) {
        Hand h(contour, shouldCheckSize, shouldCheckAngles);
        if (h.checkSize())
            hands.push_back(h);
    }
}

bool HandDetector::loadCascade(String path) {
    if (cascadePath.empty())
        cascadePath = path;
    cascade.load(cascadePath);
    cascadeLoaded = !cascade.empty();
    return !cascade.empty();
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
        ShortHand shH = h.getSame(lastHands);
        h.getFingersIndexes(shH.fingers);
    }
    checkHands();
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

void HandDetector::initFilters() {
    for (Hand &h : hands) {
        ShortHand shH = h.getSame(lastHands);
        if (shH.filtersIndex == -1 && h.filtersIndex == -1) {
            filters.emplace_back(vector<Filter>{});
            h.filtersIndex = filters.size() - 1;
            vector<Filter> &fltr = filters[filters.size() - 1];
            for (int i = 0; i < h.maxFingers; i++) {
                fltr.emplace_back(8, 6, processNoiseCov, measurementNoiseCov, errorCovPost);
            }
        } else if (shH.filtersIndex != -1) {
            h.filtersIndex = shH.filtersIndex;
        }
    }
}

void HandDetector::updateFilters() {
    for (Hand &h : hands) {
        if (h.filtersIndex >= filters.size())
            continue;
        h.updateFilters(filters[h.filtersIndex]);
    }
}

void HandDetector::stabilize() {
    for (Hand &h : hands) {
        if (h.filtersIndex >= filters.size())
            continue;
        h.stabilizeFingers(filters[h.filtersIndex]);
    }
}

void HandDetector::updateLast() {
    lastHands.clear();
    for (const Hand &h : hands) {
        ShortHand shH = {h.border, h.filtersIndex};
        for (const Finger &f : h.fingers) {
            ShortFinger shF = {
                    f.ptStart,
                    f.ptEnd,
                    f.ptFar,
                    f.index
            };
            shH.fingers.emplace_back(shF);
        }
        lastHands.emplace_back(shH);
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
