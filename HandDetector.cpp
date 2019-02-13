#include "HandDetector.h"

using namespace cv;
using namespace std;

void HandDetector::findHandsContours(Mat img) {
    threshold(img, img, thresh_sens_val, 255, THRESH_BINARY);
    mask_morph(img);
    if (shouldBlur)
        blur(img, img, blurKsize);
    threshold(img, img, thresh_sens_val, 255, THRESH_BINARY);
    findContours(img, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    for (auto &contour : contours) {
        Hand h(contour, shouldCheckSize, shouldCheckAngles, shouldGetLast);
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

Mat HandDetector::detectHands_range(Mat img, Scalar lower, Scalar upper) {
    hands.clear();
    Mat mask;
    inRange(img, lower, upper, mask);
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
        ShortHand shH = h.getSame(lastHands);
        h.getFingers(shH.fingers);
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
    vector<ShortHand> lastHands_ = lastHands;
    vector<int> indexes;
    for (Hand &h : hands) {
        ShortHand shH = h.getSame(lastHands_);
        if ((shH.filtersIndex == -1 && h.filtersIndex == -1) || shH.filtersIndex >= filters.size()) {
            filters.emplace_back(vector<Filter>{});
            h.filtersIndex = filters.size() - 1;
            vector<Filter> &fltr = filters[h.filtersIndex];
            for (int i = 0; i < h.maxFingers; i++) {
                fltr.emplace_back(8, 6, processNoiseCov, measurementNoiseCov, errorCovPost);
            }
        } else if (shH.filtersIndex != -1) {
            h.filtersIndex = shH.filtersIndex;
        }
        lastHands_.erase(remove(lastHands_.begin(), lastHands_.end(), shH), lastHands_.end());
        if (find(indexes.begin(), indexes.end(), h.filtersIndex) != indexes.end()) {
            sort(indexes.begin(), indexes.end());
            h.filtersIndex = -1;
        }
        if (h.filtersIndex != -1)
            indexes.emplace_back(h.filtersIndex);
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
    vector<ShortHand> lastHands_ = lastHands;
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
        ShortHand same = h.getSame(lastHands_);
        auto el = find(lastHands.begin(), lastHands.end(), same);
        if (el != lastHands.end()) {
            *el = same;
            lastHands_.erase(remove(lastHands_.begin(), lastHands_.end(), same), lastHands_.end());
        } else
            lastHands.emplace_back(shH);
    }

    for (ShortHand h : lastHands_) {
        auto el = find(lastHands.begin(), lastHands.end(), h);
        ShortHand &vH = *el;
        if (h.nfFrames < maxNFFrames) {
            vH.nfFrames++;
            *el = vH;
        } else {
            lastHands.erase(el);
            if (vH.filtersIndex != -1) {
                filters.erase(filters.begin() + vH.filtersIndex);
                for (int i = vH.filtersIndex; i < filters.size(); i++) {
                    auto elInd = find_if(lastHands.begin(), lastHands.end(),
                                         [&i](const ShortHand &obj) { return obj.filtersIndex == i; });
                    ShortHand &fndEl = *elInd;
                    fndEl.filtersIndex--;
                    *elInd = fndEl;
                }
            }
        }
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
