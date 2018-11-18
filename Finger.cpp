#include "Finger.h"

Finger::Finger() = default;

Finger::Finger(const Vec4i &v, const vector<Point> &cnt, const Rect &boundingBox_,
               bool shouldCheckAngles_, bool shouldCheckDist_) {
    boundingBox = boundingBox_;
    depth = v[3] / 256;

    shouldCheckAngles = shouldCheckAngles_;
    shouldCheckDist = shouldCheckDist_;

    getPoints(cnt, v);

    check();
}

bool Finger::checkDepth() {
    ok = depth >= minDepth;
    return ok;
}

bool Finger::checkAngles() {
    if (!shouldCheckAngles)
        return true;

    ok = (getAngle(ptStart, ptFar, ptEnd) < maxAngle);
    return ok;
}

bool Finger::checkDists() {
    if (!shouldCheckDist)
        return true;

    static int minDist = boundingBox.height / 5;

    ok = (getDist(ptStart, ptFar) > minDist && getDist(ptEnd, ptFar) > minDist);
    return ok;
}

bool Finger::check() {
    ok = checkDepth() && checkAngles() && checkDists();
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

void Finger::draw(Mat &img, Scalar color, int thickness, Scalar fingertipColor) {
    line(img, ptStart, ptEnd, color, thickness);
    line(img, ptStart, ptFar, color, thickness);
    line(img, ptEnd, ptFar, color, thickness);
    circle(img, ptFar, 5, color, thickness);
    circle(img, ptStart, 10, fingertipColor, thickness);
}