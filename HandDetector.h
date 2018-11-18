#ifndef HANDDETECTOR_HANDDETECTOR_H
#define HANDDETECTOR_HANDDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "Hand.h"

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

    bool shouldBlur = true;
    bool shouldCheckSize = false;
    bool shouldCheckAngles = true;

    HandDetector();

    bool loadCascade(String path);

    void findHandsContours(Mat img);

    Mat deleteBg(Mat img, Mat bg, Mat &out);

    Mat detectHands_range(Mat img, Scalar lower, Scalar upper);

    void detectHands_Cascade(Mat img);

    void getFingers();

    void getHigherFingers();
    void getFarthestFingers();

    void getCenters();

    void checkHands();

    void drawHands(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2);
};


#endif //HANDDETECTOR_HANDDETECTOR_H
