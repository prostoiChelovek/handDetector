#include "Hand.h"

using namespace cv;
using namespace std;

Hand::Hand(vector<Point> contour_, bool shouldCheckSize_, bool shouldCheckAngles_) {
    contour = move(contour_);
    moment = moments((Mat) contour);
    area = moment.m00;
    shouldCheckSize = shouldCheckSize_;
    shouldCheckAngles = shouldCheckAngles_;
}

bool Hand::checkSize() {
    if (shouldCheckSize)
        ok = (area > areaLimits.width && area < areaLimits.height);
    else
        ok = true;
    return ok;
}

void Hand::getBr() {
    border = boundingRect(contour);
}

void Hand::getCenter() {
    center = Point(moment.m10 / area, moment.m01 / area);
}

void Hand::getFingers() {
    if (!contour.empty()) {
        convexHull(contour, hull, false);
        convexHull(contour, hullI, false);
        convexityDefects(contour, hullI, defects);
        for (const Vec4i &v : defects) {
            Finger f = Finger(v, contour, border, shouldCheckAngles);
            if (f.ok)
                fingers.push_back(f);
        }
    } else
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

// for use it, first call getCenter
void Hand::getFarthestFinger() {
    double farthest = -1;
    double dist;
    Finger ff = fingers[0];
    for (Finger &f : fingers) {
        dist = getDist(center, f.ptStart);
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
        // circle(img, higherFinger.ptStart, 10, Scalar(0, 0, 255), thickness);
        circle(img, farthestFinger.ptStart, 10, Scalar(0, 255, 255), thickness);
    }
}

void Hand::draw(Mat &img, Scalar color, int thickness) {
    drawFingers(img, color, thickness);
    getBr();
    rectangle(img, border, color, thickness);
    if (center.x != -1)
        circle(img, center, 5, color, thickness);
}
