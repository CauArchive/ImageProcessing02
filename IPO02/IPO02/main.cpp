#include <iostream>

#include "opencv2/core/core_c.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

const static char* INPUT = "Input img";
const static char* RESULT = "Result img";
const static char* BG = "Background img";
const static int FRAME_DELAY = 10;

void convertColorToGray(Mat& image, Mat& gray) {
  int rows = image.rows, cols = image.cols;
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

void convertGrayToBinary(Mat& gray, Mat& binary, int thresholdValue,
                         int maxValue) {
  int rows = gray.rows, cols = gray.cols;
  if (gray.isContinuous() && binary.isContinuous()) {
    cols = rows * cols;
    rows = 1;
  }
  for (int row = 0; row < rows; row++) {
    uchar* pointer_row = gray.ptr<uchar>(row);
    uchar* pointer_row_binary = binary.ptr<uchar>(row);
    for (int col = 0; col < cols; col++) {
      if (pointer_row[col] > thresholdValue) {
        pointer_row_binary[col] = maxValue;
      } else {
        pointer_row_binary[col] = 0;
      }
    }
  }
}

void CustomBlur(Mat& image, Mat& result, int size) {
  int rows = image.rows, cols = image.cols;
  if (image.isContinuous() && result.isContinuous()) {
    cols = rows * cols;
    rows = 1;
  }
  for (int row = 0; row < rows; row++) {
    uchar* pointer_row = image.ptr<uchar>(row);
    uchar* pointer_row_result = result.ptr<uchar>(row);
    for (int col = 0; col < cols; col++) {
      int sum = 0;
      for (int i = -size / 2; i <= size / 2; i++) {
        for (int j = -size / 2; j <= size / 2; j++) {
          if (row + i >= 0 && row + i < rows && col + j >= 0 &&
              col + j < cols) {
            sum += pointer_row[(row + i) * cols + col + j];
          }
        }
      }
      pointer_row_result[col] = sum / (size * size);
    }
  }
}

void bgSub(Mat& src) {
  convertColorToGray(src, src);
  // cvtColor(src, src, COLOR_RGB2GRAY);
  // medianBlur(src, src, 7);
  CustomBlur(src, src, 7);
  convertGrayToBinary(src, src, 50, 255);
  Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
  morphologyEx(src, src, 3, element);
  morphologyEx(src, src, 2, element);
}

int main() {
  VideoCapture cap("sample.mp4");

  if (!cap.isOpened()) {
    printf("Can't open the video");
    return -1;
  }

  Mat frame, inrange_out;
  cap.read(frame);
  Mat background = frame.clone();
  // Mat background = imread("BG_abs.jpg", 1);
  while (cap.read(frame)) {
    imshow(INPUT, frame);
    Mat diff;
    absdiff(frame, background, diff);
    bgSub(diff);
    imshow(RESULT, diff);
    waitKey(FRAME_DELAY);
  }
  destroyAllWindows();
  return 0;
}