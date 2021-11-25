#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
    cv::Mat image = cv::imread("BG_abs.jpg");
    cv::namedWindow("Original Image");
    cv::imshow("Original Image", image);

    // 입력 영상에 대한 부분 전경/배경 레이블을 지정하는 방법
    // 전경 객체 내부를 포함하는 내부 직사각형을 정의
    cv::Rect rectangle(10, 300, 580, 180);
    // 경계 직사각형 정의
    // 직사각형 밖의 화소는 배경으로 레이블링

   // 입력 영상과 자체 분할 영상 외에 cv::grabCut 함수를 호출할 때
   // 이 알고리즘에 의해 만든 모델을 포함하는 두 행렬의 정의가 필요
    cv::Mat result; // 분할 (4자기 가능한 값)
    cv::Mat bgModel, fgModel; // 모델 (초기 사용)
    cv::grabCut(image,    // 입력 영상
        result,    // 분할 결과
        rectangle,   // 전경을 포함하는 직사각형
        bgModel, fgModel, // 모델
        5,     // 반복 횟수
        cv::GC_INIT_WITH_RECT); // 직사각형 사용
       // cv::CC_INT_WITH_RECT 플래그를 이용한 경계 직사각형 모드를 사용하도록 지정

     // cv::GC_PR_FGD 전경에 속할 수도 있는 화소(직사각형 내부의 화소 초기값)
     // cv::GC_PR_FGD와 동일한 값을 갖는 화소를 추출해 분할한 이진 영상을 얻음
    cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
    // 전경일 가능성이 있는 화소를 마크한 것을 가져오기
    cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
    // 결과 영상 생성
    image.copyTo(foreground, result);
    // 배경 화소는 복사되지 않음

    cv::namedWindow("Result");
    cv::imshow("Result", result);

    cv::namedWindow("Foreground");
    cv::imshow("Foreground", foreground);

    cv::waitKey(0);

    return 0;
}