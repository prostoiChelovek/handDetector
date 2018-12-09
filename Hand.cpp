#include "Hand.h"

using namespace cv;
using namespace std;

Hand::Hand(vector<Point> contour_, bool shouldCheckSize_, bool shouldCheckAngles_,
           bool shouldCheckDists_) {
    contour = move(contour_);
    moment = moments(contour);
    area = moment.m00;
    border = boundingRect(contour);
    shouldCheckSize = shouldCheckSize_;
    shouldCheckAngles = shouldCheckAngles_;
    shouldCheckDists = shouldCheckDists_;
}

bool Hand::checkSize() {
    if (shouldCheckSize) {
        ok = (area > areaLimits.width && area < areaLimits.height) &&
             (maxAspectRatio != -1 && border.height / border.width <= maxAspectRatio &&
              border.width / border.height <= maxAspectRatio);
    } else
        ok = true;
    return ok;
}

void Hand::removeCloseFingertips() {
    if (!shouldCheckDists)
        return;
    for (int i = 0; i < fingers.size(); i++) {
        for (int j = 0; j < fingers.size(); j++) {
            if (i != j && getDist(fingers[i].ptStart, fingers[j].ptStart) <= minFTDist)
                fingers.erase(fingers.begin() + i);
        }
    }
}

void Hand::getCenter() {
    center = Point(moment.m10 / area, moment.m01 / area);
}

void Hand::getFingers() {
    if (contour.empty()) {
        ok = false;
        return;
    }
    convexHull(contour, hull, false);
    convexHull(contour, hullI, false);
    convexityDefects(contour, hullI, defects);

    for (const Vec4i &v : defects) {
        Finger f = Finger(v, contour, border, shouldCheckAngles);
        if (f.ok)
            fingers.push_back(f);
    }
    removeCloseFingertips();
    if (maxFingers != -1 && fingers.size() > maxFingers)
        ok = false;
}

void Hand::getHigherFinger() {
    Point higher(fingers[0].ptStart.x, fingers[0].ptStart.y);
    Finger hf = fingers[0];
    for (Finger &f : fingers) {
        if (f.ptStart.y < higher.y) {
            higher = f.ptStart;
            hf = f;
        }
    }
    higherFinger = hf;
}

void Hand::getFarthestFinger() {
    double farthest = -1;
    double dist;
    Finger ff = fingers[0];
    for (Finger &f : fingers) {
        dist = getDist(f.ptFar, f.ptEnd);
        if (dist > farthest) {
            farthest = dist;
            ff = f;
        }
    }
    farthestFinger = ff;
}

void Hand::drawFingers(Mat &img, Scalar color, int thickness) {
    for (Finger &f : fingers) {
        f.draw(img, color, thickness);
        circle(img, higherFinger.ptStart, 10, Scalar(0, 0, 255), thickness);
        circle(img, farthestFinger.ptStart, 10, Scalar(0, 255, 255), thickness);
    }
}

void Hand::draw(Mat &img, Scalar color, int thickness) {
    drawFingers(img, color, thickness);
    rectangle(img, border, color, thickness);
    if (center.x != -1)
        circle(img, center, 5, color, thickness);
}
