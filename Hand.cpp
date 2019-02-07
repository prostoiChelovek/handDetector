#include <algorithm>
#include "Hand.h"

using namespace cv;
using namespace std;

ShortHand::ShortHand() {
    filtersIndex = -1;
}

ShortHand::ShortHand(const Rect &border) : border(border) {
    filtersIndex = -1;
}

ShortHand::ShortHand(const Rect &border, int filtersIndex) : border(border), filtersIndex(filtersIndex) {}

bool ShortHand::operator==(const ShortHand &b) {
    if (filtersIndex != b.filtersIndex)
        return false;
    if (border != b.border)
        return false;
    if (fingers.size() != b.fingers.size())
        return false;
    for (int i = 0; i < fingers.size(); i++) {
        if (fingers[i] != b.fingers[i])
            return false;
    }
    return true;
}


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
        Finger f(v, contour, border, shouldCheckAngles);
        if (f.ok)
            fingers.push_back(f);
    }
    removeCloseFingertips();
    if (maxFingers != -1 && fingers.size() > maxFingers) {
        fingers.erase(fingers.begin() + maxFingers - 1);
    }
}

void Hand::getHigherFinger() {
    int maxY = -1;
    Finger hf;
    for (Finger &f : fingers) {
        if (maxY == -1 || f.ptStart.y < maxY) {
            maxY = f.ptStart.y;
            hf = f;
        }
    }
    higherFinger = hf;
}

void Hand::getFarthestFinger() {
    double farthest = -1;
    double dist;
    Finger ff;
    for (Finger &f : fingers) {
        dist = getDist(f.ptFar, f.ptEnd);
        if (dist > farthest) {
            farthest = dist;
            ff = f;
        }
    }
    farthestFinger = ff;
}

ShortHand Hand::getSame(const vector<ShortHand> &hands) const {
    int minDiff = NULL;
    ShortHand res;
    for (const ShortHand &h : hands) {
        int diff = 0;
        diff += getDist(Point(h.border.x, h.border.y),
                        Point(border.x, border.y));
        diff += abs(h.border.width - border.width);
        diff += abs(h.border.height - border.height);
        diff += (filtersIndex == h.filtersIndex ? -25 : 25);
        if (minDiff == NULL || diff < minDiff) {
            minDiff = diff;
            res = h;
        }
    }
    return res;
}

void Hand::getFingersIndexes(const vector<ShortFinger> &lastFingers_) {
    int i = 0;
    vector<ShortFinger> lastFingers = lastFingers_;
    vector<int> indexes;
    for (Finger &f : fingers) {
        ShortFinger shF = f.getSame(lastFingers);
        if (shF.index == -1) {
            f.index = i;
        } else {
            f.index = shF.index;
        }
        if (find(indexes.begin(), indexes.end(), f.index) != indexes.end()) {
            sort(indexes.begin(), indexes.end());
            for (int j = 0; j < maxFingers; j++) {
                if (find(indexes.begin(), indexes.end(), j) == indexes.end()) {
                    f.index = j;
                    break;
                }
            }
        }
        indexes.emplace_back(f.index);
        lastFingers.erase(remove(lastFingers.begin(), lastFingers.end(), shF), lastFingers.end());
        i++;
    }
}

void Hand::updateFilters(vector<Filter> &filters) {
    for (Finger &f : fingers) {
        if (f.index >= filters.size())
            continue;
        f.updateFilter(filters[f.index]);
    }
}

void Hand::stabilizeFingers(vector<Filter> &filters) {
    for (Finger &f : fingers) {
        if (f.index >= filters.size())
            continue;
        f.stabilize(filters[f.index]);
    }
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
        circle(img, center, 3, color, CV_FILLED);
}