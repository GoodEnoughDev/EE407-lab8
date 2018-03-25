#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string> 
#include <sstream>
#include <stdio.h>
#include <deque>
#include <opencv2/cudacodec.hpp>

using namespace cv;
using namespace std;

extern int H_MIN;
extern int H_MAX;
extern int S_MIN;
extern int S_MAX;
extern int V_MIN;
extern int V_MAX;

void searchForMovement(cv::Mat thresholdImage, cv::Mat &cameraFeed, cv::Rect roi);

// Lab 5 functions
deque<Point2f> flattenDeque(deque<Point2f> inputDeque);
deque<Point2f> calculateProjection(deque<Point2f> positions, cv::Rect roi, bool x_increasing);
void drawProjection(cv::Mat &cameraFeed, deque<Point2f> collisions);
float leastSquaresFitSlope(deque<Point2f> positions);
void clearPositions();
void createTrackbars(string windowName);
void on_trackbar( int, void* );
void findCorners(cv::Mat src_gray, cv::Mat &cameraFeed);
void createTrackbars();
void searchForPaddles(cv::Rect roi, cv::Mat thresholdImage, cv::Mat &cameraFeed);
