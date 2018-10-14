#ifndef HD_FINGER_HPP
#define HD_FINGER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "utils.hpp"

using namespace cv;
using namespace std;

class Finger
{
 public:
	Point ptStart, ptEnd, ptFar;
	Vec4i v;
	vector<Point> cnt;

	double angle, inAngle, length;
	float depth;

	int minDepth = 11;
	bool ok = false;

	// hand`s
	Rect boundingBox;

	Finger() {}

	Finger(const Vec4i &v_, const vector<Point> &cnt_, const Rect &boundingBox_)
	{
		v = v_;
		cnt = cnt_;
		boundingBox = boundingBox_;
		depth = v[3] / 256;

		getPoints();
		countAngles();

		check();
	}

	bool checkDepth()
	{
		ok = depth >= minDepth;
		return ok;
	}

	bool checkAngles()
	{
		ok = (angle > -30 && angle < 160 && abs(inAngle) > 20 && abs(inAngle) < 120 && length > 0.1 * boundingBox.height);
		return ok;
	}

	bool check()
	{
		checkDepth();
		checkAngles();
		return ok;
	}

	void getPoints()
	{
		int startidx = v[0];
		ptStart = Point(cnt[startidx]);
		int endidx = v[1];
		ptEnd = Point(cnt[endidx]);
		int faridx = v[2];
		ptFar = Point(cnt[faridx]);
	}

	// For use it, firstly call getPoints()
	void countAngles()
	{
		Point center = Point(boundingBox.x + boundingBox.width / 2,
									boundingBox.y + boundingBox.height / 2);
		angle = atan2(center.y - ptStart.y, center.x - ptStart.x) * 180 / CV_PI;
		inAngle = innerAngle(ptStart.x, ptStart.y, ptEnd.x, ptEnd.y, ptFar.x, ptFar.y);
		length = sqrt(pow(ptStart.x - ptFar.x, 2) + pow(ptStart.y - ptFar.y, 2));
	}

	void draw(Mat &img, Scalar color = Scalar(255, 0, 100), int thickness = 2,
				 Scalar fingertipColor = Scalar(0, 0, 0))
	{
		line(img, ptStart, ptEnd, color, thickness);
		line(img, ptStart, ptFar, color, thickness);
		line(img, ptEnd, ptFar, color, thickness);
		circle(img, ptFar, 5, color, thickness);
		circle(img, ptStart, 10, fingertipColor, thickness);
	}
};

#endif // !HD_FINGER_HPP
