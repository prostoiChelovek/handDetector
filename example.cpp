#include <opencv2/opencv.hpp>

#include "HandDetector.h"

using namespace std;
using namespace cv;

int main()
{
    HandDetector hd;
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open video capture" << endl;
        return EXIT_FAILURE;
    }

    Mat frame, img, blur, imgYCrCb, mask;

    Scalar lower = Scalar(0, 135, 90);
    Scalar upper = Scalar(255, 230, 150);

    Ptr<BackgroundSubtractor> bgs = createBackgroundSubtractorMOG2();
    bool bgs_learn = true;
    int bgs_learnNFrames = 100;

    while (cap.isOpened()) {
        cap >> frame;
        flip(frame, frame, 0);
        flip(frame, frame, 1);

        deleteBg(frame, img, bgs, bgs_learn, hd.thresh_sens_val);
        cvtColor(img, imgYCrCb, COLOR_BGR2YCrCb);
        mask = hd.detectHands_range(imgYCrCb, lower, upper);
        hd.getFingers();

        hd.initFilters();
        hd.updateFilters();
        hd.stabilize();

        hd.getHigherFingers();
        hd.getFarthestFingers();

        hd.drawHands(frame, Scalar(255, 0, 100), 2);

        imshow("hands", frame);
        imshow("mask", mask);

        hd.updateLast();

        if (bgs_learnNFrames > 0)
            bgs_learnNFrames--;
        else if (bgs_learn) {
            bgs_learn = false;
            cout << "Background learned" << endl;
        }

        char key = waitKey(1);
        if (key != -1) {
            switch (key) {
                case 'q': // exit
                    cout << "exit" << endl;
                    return EXIT_SUCCESS;
                case 'b': // change bg
                    bgs_learnNFrames = 100;
                    bgs_learn = true;
                    cout << "Starting learning background." << endl;
                    break;
                default:
                    printf("Key pressed: %c\n", key);
                    break;
            }
        }
    }
    return EXIT_SUCCESS;
}
