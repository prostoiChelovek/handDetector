#ifndef HANDDETECTOR_HAND_H
#define HANDDETECTOR_HAND_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

#include "Finger.h"
#include "Filter.h"

using namespace cv;
using namespace std;

struct ShortHand {
    ShortHand();

    explicit ShortHand(const Rect &border);

    ShortHand(const Rect &border, int filtersIndex);

    bool operator==(const ShortHand &b);

    Rect border;
    int filtersIndex;
    vector<ShortFinger> fingers;
    // number of frames where it were not found
    int nfFrames = 0;
};


class Hand {
public:
    vector<Point> contour;
    vector<Point> hull;
    vector<int> hullI;
    vector<Vec4i> defects;
    Rect border;
    vector<Finger> fingers;

    Moments moment;
    Point center = Point(-1, -1);

    double area;
    Size areaLimits = Size(50 * 50, 800 * 800);

    // set to -1 to not check
    int maxFingers = 5;
    int maxAspectRatio = 4;

    int minFTDist = 10; // min distance between fingertips

    Finger higherFinger;
    Finger farthestFinger;

    bool ok = false;

    bool shouldCheckSize = true;
    bool shouldCheckAngles = true;
    bool shouldCheckDists = true; // check distance between fingertips
    bool shouldGetLast = true;

    // index of filters in handDetector`s vector
    int filtersIndex = -1;

    explicit Hand(vector<Point> contour, bool shouldCheckSize = true,
                  bool shouldCheckAngles = true, bool shouldGetLast = true,
                  bool shouldCheckDists = true);

    bool checkSize();

    void removeCloseFingertips(); // remove fingertips that are too close to each other

    void getCenter();

    void getFingers();

    void getHigherFinger();
    void getFarthestFinger();

    ShortHand getSame(const vector<ShortHand> &hands) const;

    void getFingersIndexes();
    void updateFilters(vector<Filter> &filters);
    void stabilizeFingers(vector<Filter> &filters);

    void drawFingers(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2);
    void draw(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2);
};


#endif //HANDDETECTOR_HAND_H
