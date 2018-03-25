//-----------------------------------------------------------------------------------------------------------------
// File:   example2.cpp
// Author: Allan A. Douglas
//-----------------------------------------------------------------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string> 
#include <sstream>
#include <stdio.h>
#include <helpers.h>
#include <deque>
#include <opencv2/cudacodec.hpp>

// Select Video Source
// The MP4 demo uses a ROI for better tracking of the moving object
//#define TEST_LIVE_VIDEO

using namespace cv;
using namespace std;



//-----------------------------------------------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------------------------------------------
int main() 
{
    // OpenCV frame matrices
    cv::Mat frame0, frame1, result, frame0_warped, frame1_warped, threshold, HSV, HSV2, threshold_top, threshold_bot, threshold_all;

    cv::cuda::GpuMat gpu_frame0, gpu_frame0_warped, gpu_frame1, gpu_frame1_warped, gpu_grayImage0, gpu_grayImage1, gpu_differenceImage, gpu_thresholdImage;

    int toggle, frame_count;

    // ROI coordinates
    int ball_roi_x = 520;
    int ball_roi_y = 95;
    int ball_roi_w = 305;
    int ball_roi_h = 525;

    int top_paddle_roi_x = 520;
    int top_paddle_roi_y = 66;
    int top_paddle_roi_w = 305;
    int top_paddle_roi_h = 22;

    int bot_paddle_roi_x = 520;
    int bot_paddle_roi_y = 624;
    int bot_paddle_roi_w = 305;
    int bot_paddle_roi_h = 22;

    const string windowName = "Original Image";
    const string windowName1 = "HSV Image";
    const string windowName2 = "Thresholded Image";
    const string windowName3 = "After Morphological Operations";
    const string trackbarWindowName = "Trackbars";

    // ROI for finding the ball
    cv::Rect roi(ball_roi_x, ball_roi_y, ball_roi_w, ball_roi_h);

    cv::Rect roi_top(top_paddle_roi_x, top_paddle_roi_y, top_paddle_roi_w, top_paddle_roi_h);
    cv::Rect roi_bot(bot_paddle_roi_x, bot_paddle_roi_y, bot_paddle_roi_w, bot_paddle_roi_h);

    createTrackbars(trackbarWindowName);

#ifdef TEST_LIVE_VIDEO
    // Camera video pipeline
    std::string pipeline = "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
#else
    // MP4 file pipeline
    std::string pipeline = "filesrc location=/home/nvidia/labs/lab_3/pong_video.mp4 ! qtdemux name=demux ! h264parse ! omxh264dec ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
#endif
    std::cout << "Using pipeline: " << pipeline << std::endl;
 
    // Create OpenCV capture object, ensure it works.
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) 
    {
        std::cout << "Connection failed" << std::endl;
        return -1;
    }
    
    // Input Quadilateral or Image plane coordinates
    Point2f inputQuad[4];
    
    // Output Quadilateral or world plane coordinates
    Point2f outputQuad[4];

    // Lambda Matrix
    cv::Mat lambda(2, 4, CV_32FC1);

    // OpenCV coordinate system is based on rows and then columns
    inputQuad[0] = Point2f(520, 80);
    inputQuad[1] = Point2f(880, 77);
    inputQuad[2] = Point2f(923, 672);
    inputQuad[3] = Point2f(472, 655);

    // The 4 points where the mapping is to be done, from top-left in clockwise order
    outputQuad[0] = Point2f(437, 0);
    outputQuad[1] = Point2f(842, 0);
    outputQuad[2] = Point2f(842, 719);
    outputQuad[3] = Point2f(437, 719);

    // Get the Perspective Transform Matrix i.e. lambda
    lambda = cv::getPerspectiveTransform(inputQuad, outputQuad);

    // Capture the first frame with GStreamer
    cap >> frame0;
    
    // Upload to GPU memory
    gpu_frame0.upload(frame0);

    // Warp perspective
    cv::cuda::warpPerspective(gpu_frame0, gpu_frame0_warped, lambda, gpu_frame0.size());

    gpu_frame0_warped.download(frame0_warped);

    // Convert the frames to gray scale (monochrome)
    cv::cuda::cvtColor(gpu_frame0,gpu_grayImage0,cv::COLOR_BGR2GRAY);

    // Initialize 
    toggle = 0;
    frame_count = 0;

    while (true) 
    {
        if (toggle == 0) 
        {
            cap >> frame1;
            gpu_frame1.upload(frame1);
            cv::cuda::warpPerspective(gpu_frame1, gpu_frame1_warped, lambda, gpu_frame1.size());
            gpu_frame1_warped.download(frame1_warped);
            cv::cuda::cvtColor(gpu_frame1_warped,gpu_grayImage1,cv::COLOR_BGR2GRAY);
            toggle = 1;
        } 
        else 
        {
            cap >> frame0;
            gpu_frame0.upload(frame0);
            cv::cuda::warpPerspective(gpu_frame0, gpu_frame0_warped, lambda, gpu_frame0.size());
            gpu_frame0_warped.download(frame0_warped);
            cv::cuda::cvtColor(gpu_frame0_warped,gpu_grayImage0,cv::COLOR_BGR2GRAY);
            toggle = 0;
	}
 
	// Compute the absolte value of the difference
	cv::cuda::absdiff(gpu_grayImage0, gpu_grayImage1, gpu_differenceImage);

	// Threshold the difference image
    cv::cuda::threshold(gpu_differenceImage, gpu_thresholdImage, 50, 255, cv::THRESH_BINARY);

    gpu_thresholdImage.download(result);

	// Find the location of any moving object and show the final frame
	if (toggle == 0) 
    {
        //searchForMovement(result,frame0_warped, roi);
                
        cv::line(frame0, inputQuad[0], inputQuad[1], Scalar(0, 255, 0), 2);
        cv::line(frame0, inputQuad[1], inputQuad[2], Scalar(0, 255, 0), 2);
        cv::line(frame0, inputQuad[2], inputQuad[3], Scalar(0, 255, 0), 2);
        cv::line(frame0, inputQuad[3], inputQuad[0], Scalar(0, 255, 0), 2);

        cvtColor(frame0_warped,HSV,COLOR_BGR2HSV);
        cvtColor(frame0,HSV2,COLOR_BGR2HSV);

        //inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold);
        inRange(HSV,Scalar(77,206,202),Scalar(256,256,256),threshold_top);
        inRange(HSV,Scalar(0,0,252),Scalar(256,98,256),threshold_bot);
        inRange(HSV2,Scalar(0,0,92),Scalar(256,256,256),threshold_all);

        //searchForPaddles(roi_top,threshold_top,frame0_warped);
        //searchForPaddles(roi_bot,threshold_bot,frame0_warped);
        findCorners(threshold_all,frame0);
                
        imshow("Original", frame0);
        //imshow("Frame", frame0_warped);
        imshow(windowName2,threshold_all);
        //imshow(windowName,frame0_warped);
        //imshow(windowName1,HSV);

        waitKey(30);
	}
	else 
    {
        //searchForMovement(result,frame1_warped, roi);

        cv::line(frame1, inputQuad[0], inputQuad[1], Scalar(0, 255, 0), 2);
        cv::line(frame1, inputQuad[1], inputQuad[2], Scalar(0, 255, 0), 2);
        cv::line(frame1, inputQuad[2], inputQuad[3], Scalar(0, 255, 0), 2);
        cv::line(frame1, inputQuad[3], inputQuad[0], Scalar(0, 255, 0), 2);	

        cvtColor(frame1_warped,HSV,COLOR_BGR2HSV);
        cvtColor(frame1,HSV2,COLOR_BGR2HSV);

        //inRange(HSV,Scalar(H_MIN,S_MIN,V_MIN),Scalar(H_MAX,S_MAX,V_MAX),threshold;
        inRange(HSV,Scalar(77,206,202),Scalar(256,256,256),threshold_top);
        inRange(HSV,Scalar(0,0,252),Scalar(256,98,256),threshold_bot);
        inRange(HSV2,Scalar(0,0,92),Scalar(256,256,256),threshold_all);

        //searchForPaddles(roi_top,threshold_top,frame1_warped);
        //searchForPaddles(roi_bot,threshold_bot,frame1_warped);
        findCorners(threshold_all,frame1);

        imshow("Original", frame1);
        //imshow("Frame", frame1_warped);
        imshow(windowName2,threshold_all);
        //imshow(windowName,frame1_warped);
        //imshow(windowName1,HSV);

        waitKey(30);
    }

        frame_count++;

        cv::waitKey(1); //needed to show frame
    }
}
