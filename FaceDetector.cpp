#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;

String face_cascade_name = "haarcascade_frontalface_default.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String window_name = "Face Detection with Haar Cascades";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
double detectedFaceArea;
RNG rng(12345);

void detectAndDisplay(Mat frame);

int main(int argc, const char** argv)
{
	Mat cameraFrame;
	int cameraNumber = 0;
	VideoCapture camera;

	if (!face_cascade.load(face_cascade_name)) { printf("Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("Error loading\n"); return -1; };
	camera.open(cameraNumber);

	if (!camera.isOpened()) { printf("Could not access the camera or video\n"); return -1; };

	camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	detectedFaceArea = 0;

	while (true)
	{
		camera >> cameraFrame;

		if (!cameraFrame.empty()) detectAndDisplay(cameraFrame);
		else { printf("No captured frame"); break; };

		int c = waitKey(10);
	}

	return 0;
}

void detectAndDisplay(Mat frame)
{
	int faceCounter = 0, eyeCounter = 0;
	double faceArea;
	vector <Rect> faces;
	Mat frame_gray;
	 
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point pt1 = Point(faces[i].x, faces[i].y + faces[i].height);
		Point pt2 = Point(faces[i].x + faces[i].width, faces[i].y);
		faceArea = faces[i].area();

		if (faceArea > detectedFaceArea)
		{
			rectangle(frame, pt1, pt2, Scalar(255, 0, 255), 4, 8, 0);

			Mat faceROI = frame_gray(faces[i]);
			vector <Rect> eyes;
			vector <Point> eyeCenters;

			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point pte1 = Point(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y + eyes[j].height);
				Point pte2 = Point(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y);
				rectangle(frame, pte1, pte2, Scalar(255, 0, 0), 4, 8, 0);

				int centerX = ((faces[i].x + eyes[j].x) + (faces[i].x + eyes[j].x + eyes[j].width)) / 2;
				int centerY = ((faces[i].y + eyes[j].y + eyes[j].height) + (faces[i].y + eyes[j].y)) / 2;

				Point center = Point(centerX, centerY);
				eyeCenters.push_back(center);
				circle(frame, center, 3, Scalar(0, 0, 255), 4, 8, 0);

				eyeCounter++;
				if (eyeCounter == 2)
				{
					int centerX = (eyeCenters.at(0).x + eyeCenters.at(1).x) / 2;
					int centerY = (eyeCenters.at(0).y + eyeCenters.at(1).y) / 2;
					Point center = Point(centerX, centerY);
					circle(frame, center, 3, Scalar(0, 255, 0), 4, 8, 0);

					break;
				}
			}

			faceCounter++;
			if (faceCounter == 1) break;
			detectedFaceArea = faceArea;
		}
	}

	imshow(window_name, frame);
}