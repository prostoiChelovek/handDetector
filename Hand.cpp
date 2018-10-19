#include "Hand.h"

using namespace cv;
using namespace std;

Hand::Hand(vector<Point> contour_) {
    contour = move(contour_);
    moment = moments((Mat) contour);
    area = moment.m00;
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
            Finger f = Finger(v, contour, border);
            if (f.ok)
                fingers.push_back(f);
        }
    } else
        ok = false;
}

void Hand::gethigherFinger() {
    Point higher(fingers[0].ptStart.x, fingers[0].ptStart.y);
    Finger hf = fingers[0];
    for (Finger &f : fingers) {
        if (f.ptStart.y < higher.y) {
            higher = f.ptStart;
            hf = f;
        }
    }
    if (higher.x != -1) {
        higherFinger = hf;
    }
}

void Hand::drawFingers(Mat &img, Scalar color, int thickness) {
    for (Finger &f : fingers) {
        f.draw(img, color, thickness);
        circle(img, higherFinger.ptStart, 10, Scalar(0, 0, 255), thickness);
    }
}

void Hand::draw(Mat &img, Scalar color, int thickness) {
    drawFingers(img, color, thickness);
    getBr();
    rectangle(img, border, color, thickness);
    if (center.x != -1)
        circle(img, center, 5, color, thickness);
}
