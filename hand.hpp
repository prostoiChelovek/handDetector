#ifndef HD_HAND_HPP
#define HD_HAND_HPP

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

#endif // !HD_HAND_HPP