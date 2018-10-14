#ifndef HadDetector_HPP
#define HandDetector_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "finger.hpp"

using namespace cv;
using namespace std;

class Hand
{
 public:
	vector<Point> contour;
	Vec4i hierarchy;
	vector<Point> hull;
	vector<int> hullI;
	vector<Vec4i> defects;
	Rect border;
	vector<Finger> fingers;
	Mat img;
	Moments moment;
	Point center = Point(-1, -1);
	double area;
	Size areaLimits = Size(50 * 50, 800 * 800);
	Finger higherFinger;
	bool ok = false;

	Hand(vector<Point> contour_, Vec4i hierarchy_)
	{
		contour = contour_;
		hierarchy = hierarchy_;
		moment = moments((Mat)contour);
		area = moment.m00;
	}

	bool checkSize()
	{
		ok = (area > areaLimits.width && area < areaLimits.height);
		return ok;
	}

	void br()
	{
		border = boundingRect(contour);
	}

	void getCenter()
	{
		center = Point(moment.m10 / area, moment.m01 / area);
	}

	void getFingers()
	{
		if (contour.size() != 0)
		{
			convexHull(contour, hull, false);
			convexHull(contour, hullI, false);
			convexityDefects(contour, hullI, defects);
			for (const Vec4i &v : defects)
			{
				Finger f = Finger(v, contour, border);
				if (f.ok)
					fingers.push_back(f);
			}
		}
		else
			ok = false;
	}

	void gethigherFinger()
	{
		Point higher(fingers[0].ptStart.x, fingers[0].ptStart.y);
		Finger hf = fingers[0];
		for (Finger &f : fingers)
		{
			if (f.ptStart.y < higher.y)
			{
				higher = f.ptStart;
				hf = f;
			}
		}
		if (higher.x != -1)
		{
			higherFinger = hf;
		}
	}

	void drawFingers(Mat &img, Scalar color = Scalar(255, 0, 100),
						  Scalar hftColor = Scalar(0, 0, 255), int thickness = 2)
	{
		for (Finger &f : fingers)
		{
			f.draw(img, color, thickness);
			circle(img, higherFinger.ptStart, 10, Scalar(0, 0, 255), thickness);
		}
	}
	void draw(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2)
	{
		drawFingers(img, color, thickness);
		br();
		rectangle(img, border, color, thickness);
		if (center.x != -1)
			circle(img, center, 5, color, thickness);
	}
};

class HandDetector
{
 public:
	vector<Hand> hands;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// detectHands_range
	int range_thresh_sens_val = 100;
	bool blur_range = true;
	Size range_blur_ksize = Size(10, 10);

	int bgs_thresh_sens_val = 20;

	// detectHands_Cascade
	String cascadePath;
	CascadeClassifier cascade;
	bool cascadeLoaded = false;
	int cascade_thresh_sens_val = 20;

	HandDetector() {}

	void mask_morph(Mat &mask)
	{
		Mat erodeElement = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat dilateElement = getStructuringElement(MORPH_RECT, Size(8, 8));
		erode(mask, mask, erodeElement);
		erode(mask, mask, erodeElement);
		dilate(mask, mask, dilateElement);
		dilate(mask, mask, dilateElement);
	}

	bool loadCascade(String path)
	{
		if (cascadePath.empty())
			cascadePath = path;
		cascade.load(cascadePath);
		cascadeLoaded = !cascade.empty();
		return !cascade.empty();
	}

	void findHandsContours(Mat img)
	{
		hands.clear();
		findContours(img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size(); i++)
		{
			Hand h(contours[i], hierarchy[i]);
			h.br();
			for (Hand hnd : hands)
			{
				if (contourArea(h.contour) < contourArea(hnd.contour) &&
					 (h.border.x > hnd.border.x && h.border.x < hnd.border.x + hnd.border.width) &&
					 (h.border.y > hnd.border.y && h.border.y < hnd.border.y + hnd.border.height))
				{
					h.area = -1;
					break;
				}
			}
			if (h.area != -1)
			{
				if (h.checkSize())
					hands.push_back(h);
			}
		}
	}

	Mat deleteBg(Mat img, Mat bg, Mat &out)
	{
		Mat deltaImg;
		absdiff(img, bg, deltaImg);

		Mat grayscale, threshDiff;
		cvtColor(deltaImg, grayscale, CV_BGR2GRAY);
		threshold(grayscale, threshDiff, bgs_thresh_sens_val, 255, THRESH_BINARY);
		mask_morph(threshDiff);

		Mat res;
		img.copyTo(res, threshDiff);
		out = res;
		return threshDiff;
	}

	Mat detectHands_range(Mat img, Scalar lower, Scalar upper)
	{
		Mat mask;
		inRange(img, lower, upper, mask);
		threshold(mask, mask, range_thresh_sens_val, 255, THRESH_BINARY);
		mask_morph(mask);
		if (blur_range)
		{
			blur(mask, mask, range_blur_ksize);
			threshold(mask, mask, range_thresh_sens_val, 255, THRESH_BINARY);
		}
		findHandsContours(mask);
		return mask;
	}

	void detectHands_Cascade(Mat img)
	{
		if (!cascadeLoaded)
			return;
		cvtColor(img, img, COLOR_BGR2GRAY);
		vector<Rect> rects;
		cascade.detectMultiScale(img, rects, 1.1,
										 2, 0 | CASCADE_SCALE_IMAGE);
		for (Rect &r : rects)
		{
			Mat i = img(r);
			Mat thresh;
			threshold(i, thresh, cascade_thresh_sens_val, 255, THRESH_BINARY);
			findHandsContours(thresh);
		}
	}

	void getFingers()
	{
		for (Hand &h : hands)
		{
			h.getFingers();
			checkHands();
		}
	}

	void getHigherFingers()
	{
		for (Hand &h : hands)
		{
			h.gethigherFinger();
		}
	}

	void checkHands()
	{
		int i = 0;
		for (Hand &h : hands)
		{
			if (!h.checkSize() || h.fingers.size() == 0)
				hands.erase(hands.begin() + i);
			i++;
		}
	}

	void getCenters()
	{
		for (Hand &h : hands)
		{
			h.getCenter();
		}
	}

	void drawHands(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2)
	{
		int i = 0;
		for (Hand &h : hands)
		{
			h.draw(img, color, thickness);
			drawContours(img, contours, i, color);
			i++;
		}
	}
};

#endif // !HandDetector_HPP