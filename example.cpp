#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "handDetector.hpp"

using namespace std;
using namespace cv;

#define Q_KEY 1048689
#define B_KEY 1048674

int main()
{
	HandDetector hd;
	VideoCapture cap(1);

	Mat frame, img, blur, bg, imgHSV, mask;

	Scalar lower = Scalar(91, 0, 0);
	Scalar upper = Scalar(255, 255, 255);

	cap >> bg;

	while (cap.isOpened())
	{
		cap >> frame;
		flip(frame, frame, 0);
		flip(frame, frame, 1);

		hd.deleteBg(frame, bg, img);
		cvtColor(img, imgHSV, COLOR_BGR2HSV);
		mask = hd.detectHands_range(imgHSV, lower, upper);
		hd.getFingers();
		hd.getCenters();
		hd.getHigherFingers();
		hd.drawHands(frame, Scalar(255, 0, 100), 2);

		imshow("hands", frame);
		imshow("mask", mask);

		int key = waitKeyEx(1);
		if (key != -1)
			printf("Key presed: %i\n", key);
		switch (key)
		{
		case Q_KEY: // exit
			return EXIT_SUCCESS;
		case B_KEY: // change bg
			cap >> bg;
			flip(bg, bg, 0);
			flip(bg, bg, 1);
			cout << "Background changed." << endl;
			break;
		}
	}
	return EXIT_SUCCESS;
}
