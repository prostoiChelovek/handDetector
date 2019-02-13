#include "Finger.h"

ShortFinger::ShortFinger() {
    index = -1;
    hndAngle = -1;
}

ShortFinger::ShortFinger(const Point &ptStart, Point ptEnd, Point ptFar, int hndAngle, int index)
        : ptStart(ptStart), ptEnd(ptEnd), ptFar(ptFar), hndAngle(hndAngle), index(index) {}

bool ShortFinger::operator==(const ShortFinger &b) const {
    return index == b.index &&
           ptStart == b.ptStart &&
           ptFar == b.ptFar &&
           ptEnd == b.ptEnd;
}

bool ShortFinger::operator!=(const ShortFinger &b) const {
    return !(operator==(b));
}


Finger::Finger() = default;

Finger::Finger(const Vec4i &defect, const vector<Point> &cnt, const Rect &boundingBox,
               bool shouldCheckAngles, bool shouldCheckDist, int index)
        : boundingBox(boundingBox), shouldCheckAngles(shouldCheckAngles),
          shouldCheckDist(shouldCheckDist), index(index) {
    depth = defect[3] / 256;
    getPoints(cnt, defect);
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

    int minDist = boundingBox.height / 5;
    int maxDist = minDist * 3;

    int d1 = getDist(ptStart, ptFar);
    int d2 = getDist(ptEnd, ptFar);
    int d3 = getDist(ptStart, ptEnd);

    ok = (d1 > minDist && d2 > minDist)
         && (d3 < maxDist);
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

ShortFinger Finger::getSame(const vector<ShortFinger> &fingers) {
    int minDist = -1;
    ShortFinger res;
    for (const ShortFinger &fn : fingers) {
        int crDist = abs((getDist(ptFar, ptStart) + getDist(ptEnd, ptStart))
                         - (getDist(fn.ptFar, fn.ptStart) + getDist(fn.ptEnd, fn.ptStart)));
        if (hndAngle != -1 && fn.hndAngle != -1)
            crDist += abs(hndAngle - fn.hndAngle) * 5;

        if (minDist == -1 || crDist < minDist) {
            minDist = crDist;
            res = fn;
        }
    }
    return res;
}

void Finger::updateFilter(Filter &f) {
    f.update(vector<float>{
            ptStart.x, ptStart.y,
            ptEnd.x, ptEnd.y,
            ptFar.x, ptFar.y,
    });
}

void Finger::stabilize(Filter &f) {
    vector<float> prd = f.predict();
    ptStart.x = prd[0];
    ptStart.y = prd[1];
    ptEnd.x = prd[2];
    ptEnd.y = prd[3];
    ptFar.x = prd[4];
    ptFar.y = prd[5];
}

void Finger::draw(Mat &img, Scalar color, int thickness, Scalar fingertipColor) {
    line(img, ptStart, ptEnd, color, thickness);
    line(img, ptStart, ptFar, color, thickness);
    line(img, ptEnd, ptFar, color, thickness);
    circle(img, ptFar, 5, color, thickness);
    circle(img, ptStart, 10, fingertipColor, thickness);
    putText(img, to_string(index), Point(ptStart.x + 10, ptStart.y - 10),
            CV_FONT_HERSHEY_COMPLEX_SMALL, 0.6, fingertipColor);
}
