#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string> 
#include <sstream>
#include <stdio.h>
#include <deque>
#include <opencv2/cudacodec.hpp>
//#include <helpers.h>
#include <cmath>  

using namespace cv;
using namespace std;

#define max_position_array_length 10

int theObject[2] = {0,0};
int x_queue[2] = {0,0};
int y_queue[2] = {0,0};
int vQx[5] = {0,0,0,0,0};
int vQy[5] = {0,0,0,0,0};
float velocity[2] = {0,0};
    
deque<Point2f> positions;
int x[3] = {0,0,0};
int y[3] = {0,0,0};

//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS=50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20*20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;


//-----------------------------------------------------------------------------------------------------------------
// int to string helper function
//-----------------------------------------------------------------------------------------------------------------
string intToString(int number)
{
    //this function has a number input and string output
    std::stringstream ss;
    ss << number;
    return ss.str();
}


//-----------------------------------------------------------------------------------------------------------------
// Clear the positions deque
//-----------------------------------------------------------------------------------------------------------------
void clearPositions()
{
	positions.clear();
}


//-----------------------------------------------------------------------------------------------------------------
// print a deque
//-----------------------------------------------------------------------------------------------------------------
void printDeque(deque<Point2f> points)
{
    for(int i = 0; i < points.size(); i++)
    {
        std::cout << "X: " << points.at(i).x << "\tY: " << points.at(i).y << std::endl;
    }
	std::cout << "" << std::endl;
}


//-----------------------------------------------------------------------------------------------------------------
// Calculate the slope based on a deque of positions
//-----------------------------------------------------------------------------------------------------------------
float leastSquaresFitSlope(deque<Point2f> positions)
{
    int length = positions.size();
    float x_sum = 0;
    float y_sum = 0;
		
    for(int i = 0; i < length; i++)
    {	
        x_sum += positions.at(i).x;
        y_sum += positions.at(i).y;
    }
    float x_mean = x_sum/length;
    float y_mean = y_sum/length;
    float num = 0;
    float den = 0;
		
    for(int i = 0; i < length; i++)
    {
	    num += ((positions.at(i).x - x_mean) * (positions.at(i).y - y_mean));
	    den += ((positions.at(i).x - x_mean) * (positions.at(i).x - x_mean));
    }
    // return the slope for our line of best fit
    float slope = num / den;
    return slope;
}


//-----------------------------------------------------------------------------------------------------------------
// Flatten deque into a "line" from deltas between position points
//-----------------------------------------------------------------------------------------------------------------
deque<Point2f> flattenDeque(deque<Point2f> inputDeque)
{
    deque<Point2f> flatDeque;
    int length = inputDeque.size();
    int x = 0;
    int y = 0;
    for(int i = 1; i < length; i++)
    {
		x += abs(inputDeque.at(i).x - inputDeque.at(i-1).x);
		y += abs(inputDeque.at(i).y - inputDeque.at(i-1).y);
		flatDeque.push_back(Point2f(x,y));
    }
    return flatDeque;
}


//-----------------------------------------------------------------------------------------------------------------
// Draw the line segments for our projection
//-----------------------------------------------------------------------------------------------------------------
void drawProjection(cv::Mat &cameraFeed, deque<Point2f> collisions)
{
	for(int i = 1; i < collisions.size(); i++)
    {
		line(cameraFeed,collisions.at(i-1),collisions.at(i),Scalar(255, 0, 0),2);
	}
}


//-----------------------------------------------------------------------------------------------------------------
// Trackbar call back function
//-----------------------------------------------------------------------------------------------------------------
void on_trackbar( int, void* )
{//This function gets called whenever a
    // trackbar position is changed
}


//-----------------------------------------------------------------------------------------------------------------
// Create trackbar
//-----------------------------------------------------------------------------------------------------------------
void createTrackbars(string windowName)
{
    //create window for trackbars
    namedWindow(windowName,0);
    //create memory to store trackbar name on window
    char TrackbarName[50];
    sprintf( TrackbarName, "H_MIN", H_MIN);
    sprintf( TrackbarName, "H_MAX", H_MAX);
    sprintf( TrackbarName, "S_MIN", S_MIN);
    sprintf( TrackbarName, "S_MAX", S_MAX);
    sprintf( TrackbarName, "V_MIN", V_MIN);
    sprintf( TrackbarName, "V_MAX", V_MAX);
    //create trackbars and insert them into window
    //3 parameters are: the address of the variable that is changing when the trackbar is moved(eg.H_LOW),
    //the max value the trackbar can move (eg. H_HIGH), 
    //and the function that is called whenever the trackbar is moved(eg. on_trackbar)
    //                                  ---->    ---->     ---->      
    createTrackbar( "H_MIN", windowName, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", windowName, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", windowName, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", windowName, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", windowName, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", windowName, &V_MAX, V_MAX, on_trackbar );
}




//-----------------------------------------------------------------------------------------------------------------
// Search for paddles
//-----------------------------------------------------------------------------------------------------------------
void searchForPaddles(cv::Rect roi, cv::Mat thresholdImage, cv::Mat &cameraFeed){
    
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::Rect paddle = cv::Rect(0,0,0,0);
    cv::Mat temp;

    thresholdImage.copyTo(temp);
    
    cv::Mat temp_roi = temp(roi);
    cv::findContours(temp_roi, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    if(contours.size() > 0)
    {
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size() - 1));
        paddle = boundingRect(largestContourVec.at(0));
    }

    // Draw roi around paddle area
    rectangle(cameraFeed, roi, Scalar(0, 0, 255), 2);

    paddle.x = paddle.x + roi.x;
    paddle.y = paddle.y + roi.y;

    // Draw box around paddle
    if(contours.size() > 0)
    {
        line(cameraFeed, Point(paddle.x, paddle.y), Point(paddle.x + paddle.width, paddle.y), Scalar(0, 0, 255), 2);
        line(cameraFeed, Point(paddle.x, paddle.y), Point(paddle.x, paddle.y + paddle.height), Scalar(0, 0, 255), 2);
        line(cameraFeed, Point(paddle.x + paddle.width,paddle.y),Point(paddle.x + paddle.width,paddle.y + paddle.height),Scalar(0, 0, 255), 2);
        line(cameraFeed, Point(paddle.x, paddle.y + paddle.height),Point(paddle.x + paddle.width, paddle.y + paddle.height), Scalar(0, 0, 255), 2);
        putText(cameraFeed, "(" + intToString(paddle.x) + "," + intToString(paddle.y) + ")", Point(paddle.x,paddle.y), 1, 1,Scalar(255, 0, 0), 2);
    }
}


//-----------------------------------------------------------------------------------------------------------------
// Find Corners
//-----------------------------------------------------------------------------------------------------------------
void findCorners(cv::Mat src_gray, cv::Mat &cameraFeed){
    
    cv::Mat dst, dst_norm;
    dst = cv::Mat::zeros(src_gray.size(), CV_32FC1);

    // Detector parameters
    int blockSize = 2;
    int apertureSize = 7;
    int thresh = 200;
    double k = 0.04;
    
    // Detecting Corners
    cv::cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    // Normalizing
    cv::normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    // Drawing circle around corner
    for(int j = 0; j < dst_norm.rows; j++){
        for(int i = 0; i < dst_norm.cols; i++){
            if((int) dst_norm.at<float>(j,i) > thresh){
                cv::circle(cameraFeed,Point(i,j),5,Scalar(0,255,0),2,8,0);
            }
        }
    }
}


//----------------------------------------------------------------------------------------------------------------
// Calculate the line segments for our projection
//-----------------------------------------------------------------------------------------------------------------
deque<Point2f> calculateProjection(deque<Point2f> positions, cv::Rect roi, bool x_increasing)
{
	deque<Point2f> projectedPath;
	projectedPath.push_back(positions.at(positions.size() - 1));
    
	if(positions.size() > 2)
    {
    	// process the deque into a straight line
    	deque<Point2f> flatDeque = flattenDeque(positions);
    	//printDeque(flatDeque);
    	// perform least square fit
    	float slope = leastSquaresFitSlope(flatDeque);
		if(slope <= 0)
        {
			return projectedPath;
		}
    	//std::cout << "Slope: " << slope << "\tlength: " << positions.size() <<std::endl;
    	// Calculate wall collisions
		Point2f currentPosition = positions.at(positions.size() - 1);
		bool xDir = x_increasing;
		int xMin = roi.x;
		int xMax = roi.x + roi.width;
		int yMin = roi.y;
		int yMax = roi.y + roi.height;
		bool searching = true;

		while(searching)
        {			
			if(xDir)
            {
				float right_collision_height = (xMax * slope) + (currentPosition.y - (slope * currentPosition.x));
				if(right_collision_height > 0)
                {					
					if(right_collision_height < yMax)
                    {
						Point2f collision = Point2f(xMax, right_collision_height);
						projectedPath.push_back(collision);
						currentPosition = collision;
						xDir = !xDir;
		    		}
					else
                    {
						searching = false;
						int x_prediction = int((yMax - (currentPosition.y - (slope * currentPosition.x))) / slope);
						projectedPath.push_back(Point2f(x_prediction, yMax));
					}
				}
				else
                {
					searching = false;
				}
			}
			else
            {
				float m = -1 * slope;
				float x = xMin;
				float b = (currentPosition.y - (m * currentPosition.x));
				float left_collision_height = (m * x) + b;
				std::cout << "Y: " << left_collision_height << "\tslope: " << slope << "\txDir: " << xDir <<std::endl;
				if((left_collision_height > yMin))
                {
					if(left_collision_height < yMax)
                    {
						Point2f collision = Point2f(xMin, left_collision_height);
						projectedPath.push_back(collision);
						currentPosition = collision;
						xDir = !xDir;
		    		}
					else
                    {
						searching = false;
						int x_prediction = int((yMax - b)/m);
						if((x_prediction > xMin) && (x_prediction < xMax)){
							projectedPath.push_back(Point2f(x_prediction, yMax));
						}					
					}
				}
				else
                {
					searching = false;
				}
			}
		}
		printDeque(projectedPath);
	}
    return projectedPath;
}


//-----------------------------------------------------------------------------------------------------------------
// Search for Moving Object
//-----------------------------------------------------------------------------------------------------------------
void searchForMovement(cv::Mat thresholdImage, cv::Mat &cameraFeed, cv::Rect roi)
{
    bool objectDetected = false;
    int xpos, ypos;

    //bounding rectangle of the object, we will use the center of this as its position.
    cv::Rect objectBoundingRectangle = cv::Rect(0,0,0,0);

    //these two vectors needed for output of findContours
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

    cv::Mat temp;

    thresholdImage.copyTo(temp);

#ifdef TEST_LIVE_VIDEO
    //find contours of filtered image using openCV findContours function
    cv::findContours(temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours
#else
    cv:Rect roi1(roi.x, roi.y, roi.width, roi.height);
    cv::Mat roi_temp = temp(roi1); 
    //find contours of filtered image using openCV findContours function
    cv::findContours(roi_temp,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );// retrieves external contours
#endif

    //if contours vector is not empty, we have found some objects
    if(contours.size()>0)
	   objectDetected = true;
    else 
	   objectDetected = false;
 
    if(objectDetected)
    {
        //the largest contour is found at the end of the contours vector
        //we will simply assume that the biggest contour is the object we are looking for.
        vector< vector<Point> > largestContourVec;
        largestContourVec.push_back(contours.at(contours.size()-1));

        //make a bounding rectangle around the largest contour then find its centroid
        //this will be the object's final estimated position.
        objectBoundingRectangle = boundingRect(largestContourVec.at(0));

        xpos = objectBoundingRectangle.x+objectBoundingRectangle.width/2;
        ypos = objectBoundingRectangle.y+objectBoundingRectangle.height/2;
 
        //update the objects positions by changing the 'theObject' array values
        theObject[0] = xpos , theObject[1] = ypos;
    }

#ifdef TEST_LIVE_VIDEO
    x[0] = x[1];
	x[1] = x[2];
	x[2] = theObject[0];
    y[0] = y[1];
	y[1] = y[2];
	y[2] = theObject[1];
#else
	x[0] = x[1];
	x[1] = x[2];
	x[2] = theObject[0] + roi.x;
    y[0] = y[1];
	y[1] = y[2];
	y[2] = theObject[1] + roi.y;
#endif

    // Determine if the positions deque should be cleared
    bool x_increasing = ((x[2] - x[0]) >= 0);
    bool y_increasing = ((y[2] - y[0]) >= 0);

    // Insert the current position into the deque
    if(y_increasing && !(y[2] == y[1] == y[0]))
    {
        positions.push_back(Point2f(x[2],y[2]));
        deque<Point2f> path = calculateProjection(positions, roi, x_increasing);
		if(positions.size() >= 2)
        {		
			drawProjection(cameraFeed, path);
		}
    }
    else
    {
		positions.clear();
    }

    circle(cameraFeed, Point(x[2], y[2]), 10, cv::Scalar(0, 255, 0), 2); 

    //draw our ROI
    line(cameraFeed,Point(roi.x,roi.y),Point(roi.x,roi.y + roi.height),Scalar(255, 0, 0),2);
    line(cameraFeed,Point(roi.x,roi.y + roi.height),Point(roi.x + roi.width,roi.y + roi.height),Scalar(255, 0, 0),2);
    line(cameraFeed,Point(roi.x,roi.y),Point(roi.x + roi.width,roi.y),Scalar(255, 0, 0),2);
    line(cameraFeed,Point(roi.x + roi.width,roi.y),Point(roi.x + roi.width,roi.y + roi.height),Scalar(255,0,0),2);
 
    //write the position of the object to the screen
    putText(cameraFeed,"(" + intToString(x[2])+","+intToString(y[2])+")",Point(x[2],y[2]),1,1,Scalar(255,0,0),2);
}
