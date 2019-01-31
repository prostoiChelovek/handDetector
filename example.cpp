#include <opencv2/opencv.hpp>

#include "HandDetector.h"

using namespace std;
using namespace cv;

int main()
{
    HandDetector hd;
    VideoCapture cap(0);

    Mat frame, img, blur, bg, imgYCrCb, mask;

    Scalar lower = Scalar(0, 135, 90);
    Scalar upper = Scalar(255, 230, 150);

    cap >> bg;

    while (cap.isOpened()) {
        cap >> frame;
        flip(frame, frame, 0);
        flip(frame, frame, 1);

        hd.deleteBg(frame, bg, img);
        cvtColor(img, imgYCrCb, COLOR_BGR2YCrCb);
        mask = hd.detectHands_range(imgYCrCb, lower, upper);
        hd.getFingers();

        hd.initFilters();
        hd.updateFilters();
        hd.stabilize();

        hd.getCenters();
        hd.getHigherFingers();
        hd.getFarthestFingers();

        hd.drawHands(frame, Scalar(255, 0, 100), 2);

        imshow("hands", frame);
        imshow("mask", mask);

        hd.updateLast();

        char key = waitKeyEx(1);
        if (key != -1) {
            switch (key) {
                case 'q': // exit
                    cout << "exit" << endl;
                    return EXIT_SUCCESS;
                case 'b': // change bg
                    cap >> bg;
                    flip(bg, bg, 0);
                    flip(bg, bg, 1);
                    cout << "Background changed." << endl;
                    break;
                default:
                    printf("Key presed: %c\n", key);
                    break;
            }
        }
    }
    return EXIT_SUCCESS;
}
