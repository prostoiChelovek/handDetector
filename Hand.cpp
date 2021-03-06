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


Hand::Hand(vector<Point> contour, bool shouldCheckSize, bool shouldCheckAngles,
           bool shouldGetLast, int maxAngle, bool shouldCheckDists)
        : contour(contour), shouldCheckSize(shouldCheckSize), shouldCheckAngles(shouldCheckAngles),
          shouldGetLast(shouldGetLast), maxAngle(maxAngle), shouldCheckDists(shouldCheckDists) {
    moment = moments(contour);
    area = moment.m00;
    border = boundingRect(contour);
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

void Hand::getFingers(const vector<ShortFinger> &lastFingers) {
    if (contour.empty()) {
        ok = false;
        return;
    }
    convexHull(contour, hull, false);
    convexHull(contour, hullI, false);
    convexityDefects(contour, hullI, defects);

    for (const Vec4i &v : defects) {
        Finger f(v, contour, border, maxAngle, shouldCheckAngles);
        if (f.ok)
            fingers.push_back(f);
    }
    sort(fingers.begin(), fingers.end(), [](const Finger &a, const Finger &b) {
        return a.ptStart.y < b.ptStart.y;
    });
    removeCloseFingertips();
    getCenter();
    getFingersIndexes(lastFingers);
    if (maxFingers != -1 && fingers.size() > maxFingers)
        fingers.resize(maxFingers);

    if (shouldGetLast) {
        if (maxFingers != -1 && fingers.size() < maxFingers) {
            for (const Finger &f : fingers) {
                if (maxFingers != -1 && fingers.size() >= maxFingers)
                    break;
                auto fnd = find_if(fingers.begin(), fingers.end(), [f](const Finger &b) {
                    return getDist(f.ptEnd, b.ptStart) <= 50;
                });
                if (fnd != fingers.end())
                    continue;

                Finger nf = f;
                nf.ptStart = f.ptEnd;
                nf.ptEnd = f.ptStart;
                nf.index = -1;
                fingers.emplace_back(nf);
            }
        }
    }
}

void Hand::getFingersIndexes(const vector<ShortFinger> &lastFingers) {
    using fingersIt = vector<Finger>::iterator;
    vector<pair<float, fingersIt>> angles;

    for (auto f = fingers.begin(); f < fingers.end(); f++) {
        float angle = getAngle(Point(0, center.y), center, (*f).ptFar);
        angles.emplace_back(angle, f);
    }
    sort(angles.begin(), angles.end(),
         [](const pair<float, fingersIt> &a, const pair<float, fingersIt> &b) {
             return a.first < b.first;
         });
    for (auto &a : angles) {
        if ((*a.second).ptFar.y > center.y)
            a.first = -a.first;
    }

    // maybe, it is not the best way, but it works and i`m tired
    vector<int> existInds;
    for (const auto &a : angles) {
        Finger &f = *a.second;
        f.hndAngle = a.first;
        ShortFinger lastF = f.getSame(lastFingers);
        if (lastF.index != -1) {
            if (find(existInds.begin(), existInds.end(), lastF.index) != existInds.end()) {
                f.index = -1;
                continue;
            }
            f.index = lastF.index;
            existInds.emplace_back(f.index);
        }
    }

    int i = maxFingers - 1;
    for (auto &a : angles) {
        Finger &f = *a.second;
        if (f.index != -1)
            continue;
        for (; i >= 0; i--) {
            if (find(existInds.begin(), existInds.end(), i) == existInds.end()) {
                f.index = i;
                existInds.emplace_back(i);
                break;
            }
        }
    }
    sort(fingers.begin(), fingers.end(), [](const Finger &a, const Finger &b) {
        return a.hndAngle < b.hndAngle;
    });
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
        dist = getDist(f.ptStart, f.ptFar);
        if (dist > farthest) {
            farthest = dist;
            ff = f;
        }
    }
    farthestFinger = ff;
}

ShortHand Hand::getSame(const vector<ShortHand> &hands) const {
    int minDiff = -1;
    ShortHand res;
    for (const ShortHand &h : hands) {
        int diff = 0;
        diff += getDist(Point(h.border.x, h.border.y),
                        Point(border.x, border.y));
        diff += abs(h.border.width - border.width);
        diff += abs(h.border.height - border.height);
        diff += (filtersIndex == h.filtersIndex ? -25 : 25);
        if (minDiff == -1 || diff < minDiff) {
            minDiff = diff;
            res = h;
        }
    }
    return res;
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
    rectangle(img, border, color, thickness);
    if (center.x != -1)
        circle(img, center, 3, color, LineTypes::FILLED);
    drawFingers(img, color, thickness);
}