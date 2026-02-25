#pragma once
#include "opencv2/opencv.hpp"

class MutualInfo
{
public:
	MutualInfo();
	~MutualInfo();

	float calcMutualInformation(const cv::Mat &img1, const cv::Mat &img2);
	float calcMutualInformation(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &mask);

	float mutual_information(cv::Mat ref, cv::Mat flt);
private:
	float calcImageEntropy(const cv::Mat &img);
	float calcCombineEntropy(const cv::Mat &img1, const cv::Mat &img2);
};
