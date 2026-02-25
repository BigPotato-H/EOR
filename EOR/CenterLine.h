#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "DataSet.h"

using namespace std;


class CenterLine
{
public:
	//static void ThinSubiteration(cv::Mat & pSrc, cv::Mat & pDst,int label);
	//static void normalizeLetter(cv::Mat & inputarray, cv::Mat & outputarray);
	static void getCenterLine(const cv::Mat&g_srcImage, LINE_VEC& ll, int mask_type = 0);
	//static void recoverImageSize(cv::Mat& img);
};
