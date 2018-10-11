#ifndef HadDetector_HPP
#define HandDetector_HPP
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// https://picoledelimao.github.io/blog/2015/11/15/fingertip-detection-on-opencv/
float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{
   float dist1 = sqrt((px1 - cx1) * (px1 - cx1) + (py1 - cy1) * (py1 - cy1));
   float dist2 = sqrt((px2 - cx1) * (px2 - cx1) + (py2 - cy1) * (py2 - cy1));
   float Ax, Ay;
   float Bx, By;
   float Cx, Cy;
   //find closest point to C
   //printf("dist = %lf %lf\n", dist1, dist2);
   Cx = cx1;
   Cy = cy1;
   if (dist1 < dist2)
   {
      Bx = px1;
      By = py1;
      Ax = px2;
      Ay = py2;
   }
   else
   {
      Bx = px2;
      By = py2;
      Ax = px1;
      Ay = py1;
   }
   float Q1 = Cx - Ax;
   float Q2 = Cy - Ay;
   float ptStart = Bx - Ax;
   float ptEnd = By - Ay;
   float A = acos((ptStart * Q1 + ptEnd * Q2) / (sqrt(ptStart * ptStart + ptEnd * ptEnd) * sqrt(Q1 * Q1 + Q2 * Q2)));
   A = A * 180 / CV_PI;
   return A;
}

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
   Point higherFingerip = Point(-1, -1);
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

   void getHigherFingerip()
   {
      Point higher(fingers[0].ptStart.x, fingers[0].ptStart.y);
      for (Finger &f : fingers)
      {
         if (f.ptStart.y < higher.y)
         {
            higher = f.ptStart;
         }
      }
      if (higher.x != -1)
      {
         higherFingerip = higher;
      }
   }

   void drawFingers(Mat &img, Scalar color = Scalar(255, 0, 100),
                    Scalar hftColor = Scalar(0, 0, 255), int thickness = 2)
   {
      for (Finger &f : fingers)
      {
         f.draw(img, color, thickness);
         circle(img, higherFingerip, 10, Scalar(0, 0, 255), thickness);
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

   void getHigherFingerstips()
   {
      for (Hand &h : hands)
      {
         h.getHigherFingerip();
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

#endif // !FaceDetector_HPP