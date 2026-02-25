#include "CenterLine.h"
#include <io.h>
#include <direct.h>
#include <vector>
#include "DataSet.h"

using namespace std;
using namespace cv;
#define CENTER_LINE 1

void ThinSubiteration(Mat & pSrc, Mat & pDst, int label) {
	int rows = pSrc.rows;
	int cols = pSrc.cols;
	pSrc.copyTo(pDst);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (pSrc.at<uchar>(i, j) == 1.0f) {
				/// get 8 neighbors
				/// calculate C(p)
				int neighbor0 = (int)pSrc.at<uchar>(i - 1, j - 1);
				int neighbor1 = (int)pSrc.at<uchar>(i - 1, j);
				int neighbor2 = (int)pSrc.at<uchar>(i - 1, j + 1);
				int neighbor3 = (int)pSrc.at<uchar>(i, j + 1);
				int neighbor4 = (int)pSrc.at<uchar>(i + 1, j + 1);
				int neighbor5 = (int)pSrc.at<uchar>(i + 1, j);
				int neighbor6 = (int)pSrc.at<uchar>(i + 1, j - 1);
				int neighbor7 = (int)pSrc.at<uchar>(i, j - 1);
#if 0
				int C = int(~neighbor1 & (neighbor2 | neighbor3)) +
					int(~neighbor3 & (neighbor4 | neighbor5)) +
					int(~neighbor5 & (neighbor6 | neighbor7)) +
					int(~neighbor7 & (neighbor0 | neighbor1));
				if (C == 1) {
					/// calculate N
					int N1 = int(neighbor0 | neighbor1) +
						int(neighbor2 | neighbor3) +
						int(neighbor4 | neighbor5) +
						int(neighbor6 | neighbor7);
					int N2 = int(neighbor1 | neighbor2) +
						int(neighbor3 | neighbor4) +
						int(neighbor5 | neighbor6) +
						int(neighbor7 | neighbor0);
					int N = min(N1, N2);
					if ((N == 2) || (N == 3)) {
						/// calculate criteria 3
						switch (label)
						{
						case 1:
						{
							int c3 = (neighbor1 | neighbor2 | ~neighbor4) & neighbor3;
							if (c3 == 0)
							{
								pDst.at<uchar>(i, j) = 0.0f;
							}
						}
						break;
						case 2:
						{
							int E = (neighbor5 | neighbor6 | ~neighbor0) & neighbor7;
							if (E == 0)
							{
								pDst.at<uchar>(i, j) = 0.0f;
							}
						}
						break;
						default:
							break;
						}
					}
				}
#else
				int C = int(~neighbor0 & neighbor1) + int(~neighbor1 & neighbor2) + int(~neighbor2 & neighbor3)
					+ int(~neighbor3 & neighbor4) + int(~neighbor4 & neighbor5) + int(~neighbor5 & neighbor6)
					+ int(~neighbor6 & neighbor7) + int(~neighbor7 & neighbor0);
				if (C == 1)
				{
					int count = 0;
					count = int(neighbor0) + int(neighbor1) + int(neighbor2) + int(neighbor3)
						+ int(neighbor4) + int(neighbor5) + int(neighbor6) + int(neighbor7);
					if (2 <= count && count <= 6)
					{
						switch (label)
						{
						case 1:
						{
							if ((int(neighbor2) == 0) || (int(neighbor4) == 0) || (int(neighbor0) == 0 && int(neighbor6) == 0))
							{
								pDst.at<uchar>(i, j) = 0.0f;
							}
						}
						break;
						case 2:
						{
							if ((int(neighbor0) == 0) || (int(neighbor6) == 0) || (int(neighbor2) == 0 && int(neighbor4) == 0))
							{
								pDst.at<uchar>(i, j) = 0.0f;
							}
						}
						break;
						default:
							break;
						}
					}
				}
#endif
			}
		}
	}
}

void normalizeLetter(const Mat & inputarray, Mat & outputarray) 
{
	vector<cv::Point> points_vec;

	bool bDone = false;
	int rows = inputarray.rows;
	int cols = inputarray.cols;
	inputarray.copyTo(outputarray);
	/// pad source
	Mat p_enlarged_src = Mat(rows + 2, cols + 2, CV_8U);
	//将周围扩大的区域置0
	for (int i = 0; i < (rows + 2); i++) {
		p_enlarged_src.at<uchar>(i, 0) = 0.0f;
		p_enlarged_src.at<uchar>(i, cols + 1) = 0.0f;
	}
	for (int j = 0; j < (cols + 2); j++) {
		p_enlarged_src.at<uchar>(0, j) = 0.0f;
		p_enlarged_src.at<uchar>(rows + 1, j) = 0.0f;
	}

	cv::Mat temp;
	inputarray.convertTo(temp, CV_8U);
	int threshold = cv::threshold(temp, temp, 0, 255, CV_THRESH_OTSU);

	for (int i = 0; i < rows; i++) 
	{
		for (int j = 0; j < cols; j++) 
		{
			if (inputarray.at<uchar>(i, j) > threshold) 
			{
				p_enlarged_src.at<uchar>(i + 1, j + 1) = 1.0f;
			}
			else
			{
				p_enlarged_src.at<uchar>(i + 1, j + 1) = 0.0f;
			}
		}
	}

	/// start to thin
	Mat p_thinMat1 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_thinMat2 = Mat::zeros(rows + 2, cols + 2, CV_32FC1);
	Mat p_cmp = Mat::zeros(rows + 2, cols + 2, CV_8UC1);

	while (bDone != true) {
		/// sub-iteration 1
		ThinSubiteration(p_enlarged_src, p_thinMat1,1);

		/// sub-iteration 2
		ThinSubiteration(p_thinMat1, p_thinMat2,2);

		/// compare
		compare(p_enlarged_src, p_thinMat2, p_cmp, CV_CMP_EQ);	//比较输入的src1和src2中的元素，真为255，否则为0
	
		int num_non_zero = countNonZero(p_cmp);					//返回灰度值不为0的像素数
		if (num_non_zero == (rows + 2) * (cols + 2)) {
			bDone = true;
		}
		/// copy
		p_thinMat2.copyTo(p_enlarged_src);
	}

	// copy result
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			outputarray.at<uchar>(i, j) = p_enlarged_src.at<uchar>(i + 1, j + 1);
			//if (outputarray.at<uchar>(i, j) != (0, 0))
			//{
			//	line_vec.push_back(cv::Point(j, i));
			//}
		}
	}
	
}


void CenterLine::getCenterLine(const cv::Mat&g_srcImage, LINE_VEC& ll, int mask_type)
{
//	recoverImageSize(g_srcImage);
	cv::Mat g_dstImage;

#if CENTER_LINE
	normalizeLetter(g_srcImage, g_dstImage);

	//cv::Sobel(g_dstImage, g_dstImage, CV_8UC1, 0, 1, 3);

	auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 5));
	cv::dilate(g_dstImage, g_dstImage, kernel);

#else
	g_dstImage = g_srcImage;
#endif

	//cv::namedWindow("out", WINDOW_NORMAL);
	//cv::imshow("out", g_dstImage);
	//cv::waitKey(0);
	LINE_VEC contours;
	cv::findContours(g_dstImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	ll.swap(contours);
	return;
	//for (const std::vector<cv::Point>& contour : contours)
	//{
	//	std::vector<cv::Point> centralLine;
	//	double epsilon = 0.01 * cv::arcLength(contour, true); // Adjust the epsilon value as needed
	//	cv::approxPolyDP(contour, centralLine, epsilon, true);
	//	ll.push_back(centralLine);
	//}



}

