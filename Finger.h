#ifndef HANDDETECTOR_FINGER_H
#define HANDDETECTOR_FINGER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "Filter.h"
#include "utils.h"

using namespace cv;
using namespace std;

struct ShortFinger {
    ShortFinger();

    ShortFinger(const Point &ptStart, Point ptEnd, Point ptFar, int hndAnggle = -1, int index = -1);

    bool operator==(const ShortFinger &b) const;

    bool operator!=(const ShortFinger &b) const;

    Point ptStart; // fingertip
    Point ptEnd;
    Point ptFar;
    int index;
    int hndAngle;
};

class Finger {
public:
    Point ptStart; // fingertip
    Point ptEnd, ptFar;

    float depth;

    float maxAngle = 125;
    int minDepth = 11;

    bool ok = false;

    bool shouldCheckAngles = true;
    bool shouldCheckDist = true; // check distance between ptStart, ptFar and ptEnd;

    int index = -1;

    int hndAngle = -1;

    Rect boundingBox; // hand`s

    Finger();

    Finger(const Vec4i &defect, const vector<Point> &cnt, const Rect &boundingBox,
           bool shouldCheckAngles = true, bool shouldCheckDist = true, int index = -1);

    bool checkDepth();
    bool checkAngles();
    bool checkDists(); // check distance between ptStart, ptFar and ptEnd;
    bool check();

    void getPoints(vector<Point> cnt, Vec4i v);

    ShortFinger getSame(const vector<ShortFinger> &fingers);

    void updateFilter(Filter &f);
    void stabilize(Filter &f);

    void draw(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2,
              Scalar fingertipColor = Scalar(0, 0, 0));
};


#endif //HANDDETECTOR_FINGER_H
