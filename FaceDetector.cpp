#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <queue>

using namespace std;
using namespace cv;

String face_cascade_name = "haarcascade_frontalface_default.xml";
String eyes_cascade_name = "haarcascade_eye.xml";
String window_name = "Face Detection with Haar Cascades";
FILE *file;

const int kFastEyeWidth = 50;
const int eyePercentTop = 25;
const int eyePercentSide = 13;
const int eyePercentHeight = 30;
const int eyePercentWidth = 35;
const double kGradientThreshold = 50.0;
const int kWeightBlurSize = 5;
const bool kEnableWeight = false;
const float kWeightDivisor = 150.0;
const bool kEnablePostProcess = true;
const double kPostProcessThreshold = 0.97;
const bool kPlotVectorField = false;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
RNG rng(12345);

void detectAndDisplay(Mat frame);
int findBiggestFace(vector <Rect> faces);
Point findEyeCenter(Mat face, Rect eye, string debugWindow);
void scaleToFastSize(const cv::Mat &src, cv::Mat &dst);
Mat computeMatXGradient(const cv::Mat &mat);
Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);
void testPossibleCentersFormula(int x, int y, unsigned char weight, double gx, double gy, cv::Mat &out);
Mat floodKillEdges(cv::Mat &mat);
bool inMat(Point p, int rows, int cols);
Point unscalePoint(Point p, Rect origSize);
bool floodShouldPushPoint(const Point &np, const Mat &mat);
void printAngles(Point center, Point pupil, bool faceGot, bool bothEyesGot);

int main(int argc, const char** argv)
{
	Mat cameraFrame;
	int cameraNumber = 0;
	VideoCapture camera;

	//printf("OpenCV: %s", getBuildInformation().c_str());

	/*
	file = fopen("some location to connect arduino", "w");
	if (file == NULL)
	{
		cout << "usb port unable to open" << endl;
		return 1;
	}
	cout << "Usb port opened" << endl;
	*/

	if (!face_cascade.load(face_cascade_name)) { printf("Error loading\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)) { printf("Error loading\n"); return -1; };
	camera.open(cameraNumber);

	// External camera should be used instead of computer's camera
	if (!camera.isOpened()) { printf("Could not access the camera or video\n"); return -1; };

	camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);


	while (true)
	{
		camera >> cameraFrame;

		if (!cameraFrame.empty()) detectAndDisplay(cameraFrame);
		else { printf("No captured frame"); break; };

		int c = waitKey(10);
	}

	return 0;
}

int findBiggestFace(vector <Rect> faces)
{
	int maxArea = 0, maxIndex = 0;
	double area;

	for (int i = 0; i < faces.size(); i++)
	{
		area = faces[i].width * faces[i].height;
		if (area > maxArea) maxIndex = i;
	}

	return maxIndex;
}

void detectAndDisplay(Mat frame)
{
	vector <Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	size_t i = findBiggestFace(faces);
	Point center, pupilLoc, pupil;
	bool faceFound = false, bothEyesFound = false;

	if (faces.size() != 0)
	{
		faceFound = true;
		Point pt1 = Point(faces.at(i).x, faces.at(i).y + faces.at(i).height);
		Point pt2 = Point(faces.at(i).x + faces.at(i).width, faces.at(i).y);

		// Purple box is generated for captured face
		rectangle(frame, pt1, pt2, Scalar(255, 0, 255), 4, 8, 0);
		Mat faceROI = frame_gray(faces[i]);

		vector <Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		size_t eyeCounter = 2, eyesGenerated = 0;
		if (eyes.size() < 2) eyeCounter = eyes.size();

		for (size_t j = 0; j < eyeCounter; j++)
		{
			pupil = findEyeCenter(faceROI, eyes[j], "pupil");

			Point pte1 = Point(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y + eyes[j].height);
			Point pte2 = Point(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y);

			int centerX = faces[i].x + eyes[j].x + pupil.x;
			int centerY = faces[i].y + eyes[j].y + pupil.y;

			if (centerY < (faces[i].y + faces[i].height / 2))
			{
				eyesGenerated++;
				// Blue boxes generated for eye regions
				rectangle(frame, pte1, pte2, Scalar(255, 0, 0), 4, 8, 0);
				pupilLoc = Point(centerX, centerY);

				// Draws a circle at the pupils
				circle(frame, pupilLoc, 3, Scalar(0, 0, 255), 4, 8, 0);

				// Draws a circle at the center of eye boxes
				center = Point((pte1.x + pte2.x) / 2, (pte1.y + pte2.y) / 2);
				circle(frame, center, 3, Scalar(0, 255, 0), 4, 8, 0);
			}
		}

		if (eyesGenerated == 2) bothEyesFound = true;
	}

	printAngles(center, pupilLoc, faceFound, bothEyesFound);
	imshow(window_name, frame);
}

void printAngles(Point center, Point pupil, bool faceGot, bool bothEyesGot)
{
	if (bothEyesGot)
	{
		int firstAngle = abs(center.x - pupil.x);
		int secondAngle = abs(center.y - pupil.y);
		string firstA, secondA;
		string tiltA = "00", tiltS = "+";
		string firstS = "+", secondS = "+";

		// Angle thresholds need more testing
		if (firstAngle < 8) firstA = "00";
		else if (firstAngle < 15) firstA = "20";
		else firstA = "40";

		if (secondAngle < 8) secondA = "00";
		else if (secondAngle < 15) secondA = "20";
		else secondA = "40";

		if (center.x > pupil.x) firstS = "-";
		if (center.y > pupil.y) secondS = "-";

		// Output should be printed to the driver for arduino instead of terminal
		cout << firstS + firstA + secondS + secondA + tiltS + tiltA + "$\n";
		//fprintf(file, "%s", firstS + firstA + secondS + secondA + tiltS + tiltA + "$");
	}
	// Face got but eyes are missing
	else if (faceGot)
	{
		cout << "###$\n";
		//fprintf(file, "%s", "###$");
	}
	// Nothing got
	else 
	{
		cout << "!!!$\n";
		//fprintf(file, "%s", "!!!$");
	}

	//fflush(file);
}

Point findEyeCenter(Mat face, Rect eye, string debugWindow) 
{
	Mat eyeROIUnscaled = face(eye);
	Mat eyeROI;

	scaleToFastSize(eyeROIUnscaled, eyeROI);
	
	// Find the gradient
	Mat gradientX = computeMatXGradient(eyeROI);
	Mat gradientY = computeMatXGradient(eyeROI.t()).t();
	// Normalize and threshold the gradient
	// compute all the magnitudes
	Mat mags = matrixMagnitude(gradientX, gradientY);
	//compute the threshold
	double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
	
	//normalize
	for (int y = 0; y < eyeROI.rows; ++y) 
	{
		double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		const double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < eyeROI.cols; ++x) 
		{
			double gX = Xr[x], gY = Yr[x];
			double magnitude = Mr[x];
			if (magnitude > gradientThresh) 
			{
				Xr[x] = gX / magnitude;
				Yr[x] = gY / magnitude;
			}
			else 
			{
				Xr[x] = 0.0;
				Yr[x] = 0.0;
			}
		}
	}

	// Create a blurred and inverted image for weighting
	Mat weight;
	GaussianBlur(eyeROI, weight, cv::Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
	for (int y = 0; y < weight.rows; ++y) {
		unsigned char *row = weight.ptr<unsigned char>(y);
		for (int x = 0; x < weight.cols; ++x) {
			row[x] = (255 - row[x]);
		}
	}

	// Run the algorithm
	Mat outSum = Mat::zeros(eyeROI.rows, eyeROI.cols, CV_64F);
	// for each possible center

	for (int y = 0; y < weight.rows; ++y) 
	{
		const unsigned char *Wr = weight.ptr<unsigned char>(y);
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		for (int x = 0; x < weight.cols; ++x) 
		{
			double gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0)
			{
				continue;
			}
			testPossibleCentersFormula(x, y, Wr[x], gX, gY, outSum);
		}
	}
	// scale all the values down, basically averaging them
	double numGradients = (weight.rows*weight.cols);
	Mat out;
	outSum.convertTo(out, CV_32F, 1.0 / numGradients);

	// Find the maximum point
	Point maxP;
	double maxVal;
	minMaxLoc(out, NULL, &maxVal, NULL, &maxP);
	// Flood fill the edges
	if (kEnablePostProcess) 
	{
		Mat floodClone;
		//double floodThresh = computeDynamicThreshold(out, 1.5);
		double floodThresh = maxVal * kPostProcessThreshold;
		threshold(out, floodClone, floodThresh, 0.0f, THRESH_TOZERO);
		
		Mat mask = floodKillEdges(floodClone);
		
		// redo max
		minMaxLoc(out, NULL, &maxVal, NULL, &maxP, mask);
	}
	return unscalePoint(maxP, eye);
}

void scaleToFastSize(const Mat &src, Mat &dst) 
{
	resize(src, dst, cv::Size(kFastEyeWidth, (((float)kFastEyeWidth) / src.cols) * src.rows));
}

Mat computeMatXGradient(const cv::Mat &mat) 
{
	Mat out(mat.rows, mat.cols, CV_64F);

	for (int y = 0; y < mat.rows; ++y) {
		const uchar *Mr = mat.ptr<uchar>(y);
		double *Or = out.ptr<double>(y);

		Or[0] = Mr[1] - Mr[0];
		for (int x = 1; x < mat.cols - 1; ++x) {
			Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
		}
		Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
	}

	return out;
}

Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY)
{
	Mat mags(matX.rows, matX.cols, CV_64F);

	for (int y = 0; y < matX.rows; ++y) {
		const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
		double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < matX.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
		}
	}
	return mags;
}

double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor)
{
	Scalar stdMagnGrad, meanMagnGrad;
	meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
}

void testPossibleCentersFormula(int x, int y, unsigned char weight, double gx, double gy, Mat &out)
{
	// for all possible centers
	for (int cy = 0; cy < out.rows; ++cy) {
		double *Or = out.ptr<double>(cy);
		for (int cx = 0; cx < out.cols; ++cx) {
			if (x == cx && y == cy) {
				continue;
			}
			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx*gx + dy*gy;
			dotProduct = std::max(0.0, dotProduct);
			// square and multiply by the weight
			if (kEnableWeight) {
				Or[cx] += dotProduct * dotProduct * (weight / kWeightDivisor);
			}
			else {
				Or[cx] += dotProduct * dotProduct;
			}
		}
	}
}

Mat floodKillEdges(Mat &mat)
{
	rectangle(mat, Rect(0, 0, mat.cols, mat.rows), 255);

	Mat mask(mat.rows, mat.cols, CV_8U, 255);
	queue<Point> toDo;
	toDo.push(Point(0, 0));
	while (!toDo.empty()) {
		Point p = toDo.front();
		toDo.pop();
		if (mat.at<float>(p) == 0.0f) {
			continue;
		}
		// add in every direction
		Point np(p.x + 1, p.y); // right
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x - 1; np.y = p.y; // left
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x; np.y = p.y + 1; // down
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x; np.y = p.y - 1; // up
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		// kill it
		mat.at<float>(p) = 0.0f;
		mask.at<uchar>(p) = 0;
	}
	return mask;
}

bool floodShouldPushPoint(const Point &np, const Mat &mat)
{
	return inMat(np, mat.rows, mat.cols);
}

bool inMat(Point p, int rows, int cols) 
{
	return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

Point unscalePoint(Point p, Rect origSize) 
{
	float ratio = (((float)kFastEyeWidth) / origSize.width);
	float x = round(p.x / ratio);
	float y = round(p.y / ratio);
	return Point(x, y);
}