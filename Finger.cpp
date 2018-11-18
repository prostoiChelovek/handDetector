#include "Finger.h"

Finger::Finger() = default;

Finger::Finger(const Vec4i &v, const vector<Point> &cnt, const Rect &boundingBox_,
               bool shouldCheckAngles_) {
    boundingBox = boundingBox_;
    depth = v[3] / 256;

    shouldCheckAngles = shouldCheckAngles_;

    getPoints(cnt, v);
    countAngles();

    check();
}

bool Finger::checkDepth() {
    ok = depth >= minDepth;
    return ok;
}

bool Finger::checkAngles() {
    if (shouldCheckAngles)
        ok = (angle > -30 && angle < 160 && abs(inAngle) > 20 && abs(inAngle) < 120 &&
              length > 0.1 * boundingBox.height);
    else
        ok = true;
    return ok;
}

bool Finger::check() {
    ok = checkDepth() && checkAngles();
    return ok;
}

void Finger::getPoints(vector<Point> cnt, Vec4i v) {
    int startidx = v[0];
    ptStart = Point(cnt[startidx]);
    int endidx = v[1];
    ptEnd = Point(cnt[endidx]);
    int faridx = v[2];
    ptFar = Point(cnt[faridx]);
}

// For use it, firstly call getPoints()
void Finger::countAngles() {
    Point center = Point(boundingBox.x + boundingBox.width / 2,
                         boundingBox.y + boundingBox.height / 2);
    angle = atan2(center.y - ptStart.y, center.x - ptStart.x) * 180 / CV_PI;
    inAngle = innerAngle(ptStart.x, ptStart.y, ptEnd.x, ptEnd.y, ptFar.x, ptFar.y);
    length = sqrt(pow(ptStart.x - ptFar.x, 2) + pow(ptStart.y - ptFar.y, 2));
}

void Finger::draw(Mat &img, Scalar color, int thickness, Scalar fingertipColor) {
    line(img, ptStart, ptEnd, color, thickness);
    line(img, ptStart, ptFar, color, thickness);
    line(img, ptEnd, ptFar, color, thickness);
    circle(img, ptFar, 5, color, thickness);
    circle(img, ptStart, 10, fingertipColor, thickness);
}