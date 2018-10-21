#ifndef HANDDETECTOR_HAND_H
#define HANDDETECTOR_HAND_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "Finger.h"

using namespace cv;
using namespace std;

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
    Finger higherFinger;
    Finger farthestFinger;
    bool ok = false;
    bool shouldCheckSize = true;
    bool shouldCheckAngles = true;

    explicit Hand(vector<Point> contour_, bool shouldCheckSize_ = true,
                  bool shouldCheckAngles_ = true);

    bool checkSize();

    void getBr();

    void getCenter();

    void getFingers();

    void getHigherFinger();

    void getFarthestFinger();

    void drawFingers(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2);

    void draw(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2);
};


#endif //HANDDETECTOR_HAND_H
