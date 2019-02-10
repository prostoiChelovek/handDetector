#ifndef HANDDETECTOR_HANDDETECTOR_H
#define HANDDETECTOR_HANDDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "Hand.h"
#include "Filter.h"

using namespace cv;
using namespace std;

class HandDetector {
public:
    vector<Hand> hands;
    vector<vector<Point>> contours;

    int thresh_sens_val = 20;
    Size blurKsize = Size(10, 10);

    // detectHands_Cascade
    String cascadePath;
    CascadeClassifier cascade;
    bool cascadeLoaded = false;

    // Filter
    Scalar processNoiseCov = Scalar::all(1e-2);
    Scalar measurementNoiseCov = Scalar::all(1e-2);
    Scalar errorCovPost = Scalar::all(.3);
    vector<vector<Filter>> filters;

    bool shouldBlur = true;
    bool shouldCheckSize = true;
    bool shouldCheckAngles = true;

    // number of frames on which the hand was not found so that it was deleted
    int maxNFFrames = 100;

    vector<ShortHand> lastHands;

    HandDetector() = default;

    void findHandsContours(Mat img);

    bool loadCascade(String path);

    Mat detectHands_range(Mat img, Scalar lower, Scalar upper);
    void detectHands_Cascade(Mat img);

    void getFingers();

    void getHigherFingers();
    void getFarthestFingers();

    void getCenters();

    void checkHands();

    void initFilters();

    void updateFilters();

    void stabilize();

    void updateLast();

    void drawHands(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2);

};


#endif //HANDDETECTOR_HANDDETECTOR_H
