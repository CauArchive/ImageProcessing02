#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

void setLabel(Mat& image, string str, vector<Point> contour) {
	int fontface = FONT_HERSHEY_SIMPLEX;
	double scale = 0.5;
	int thickness = 1;
	int baseline = 0;
	Size text = getTextSize(str, fontface, scale, thickness, &baseline);
	Rect r = boundingRect(contour);
	Point pt(r.x + ((r.width - text.width) / 2),
		r.y + ((r.height + text.height) / 2));
	rectangle(image, pt + Point(0, baseline),
		pt + Point(text.width, -text.height), CV_RGB(200, 200, 200),
		FILLED);
	putText(image, str, pt, fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
}

void convertColorToGray(Mat& image, Mat& gray) {
	int rows = image.rows, cols = image.cols;
	gray.create(image.size(), CV_8UC1);
	if (image.isContinuous() && gray.isContinuous()) {
		cols = rows * cols;
		rows = 1;
	}
	for (int row = 0; row < rows; row++) {
		uchar* pointer_row = image.ptr<uchar>(row);
		uchar* pointer_row_gray = gray.ptr<uchar>(row);
		for (int col = 0; col < cols; col++) {
			pointer_row_gray[col] =
				(uchar)(0.299f * pointer_row[0] + 0.587f * pointer_row[1] +
					0.114f * pointer_row[2]);
			pointer_row += 3;
		}
	}
}

void convertGrayToBinary(Mat& gray, Mat& binary, int thresholdValue, int maxValue) {
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
}

void findContoursFromBinary(Mat& binary, vector<vector<Point> >& contours) {
	findContours(binary, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
}

int main(int, char**) {
	Mat img, img_result, img_gray;
	// Image Load
	img = imread("2.png", IMREAD_COLOR);
	if (img.empty()) {
		cout << "No Image Found\n";
		return -1;
	}
	// Convert Image to GrayScale
	convertColorToGray(img, img_gray);
	// Convert Image to Binary Image
	Mat img_binary;
	convertGrayToBinary(img_gray, img_binary, 30, 255);
	// Find Contours of Images
	vector<vector<Point> > contours;
	findContoursFromBinary(img_binary, contours);
	// Approximate Contours
	vector<Point2f> approx;
	img_result = img.clone();
	for (size_t i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), approx,
			arcLength(Mat(contours[i]), true) * 0.02, true);
		if (fabs(contourArea(Mat(approx))) < 120000)  // Detect Image with Area < 120000)
		{
			int size = approx.size();
			// If approximated Contour size is over 3, it is a triangle
			if (size == 3) setLabel(img_result, "triangle", contours[i]);
			// If approximated Contour size is over 4, it is a rectangle
			else if (size == 4 && isContourConvex(Mat(approx)))
				setLabel(img_result, "rectangle", contours[i]);
			else if (size <= 7 && isContourConvex(Mat(approx)))
				setLabel(img_result, to_string(approx.size()), contours[i]);
			// Else decide it is circle
			else
				setLabel(img_result, "circle", contours[i]);
		}
	}
	imshow("input", img);
	imshow("result", img_result);
	waitKey(0);
	return 0;
}
