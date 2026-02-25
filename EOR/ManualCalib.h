// ManualCalib.cpp : 定义控制台应用程序的入口点。

// for std
// for opencv
#pragma once
#include "DataSet.h"

class ManualCalib
{
public:
	ManualCalib();
	~ManualCalib();

	void process(vector<double>& camPose, const string& folder_path, const vector<CalibSpace::Point3d2d>&ref_pt_vec);

	void processKitti(vector<double>& camPose, const string& folder_path);
	void initCamera(cv::Mat& intrisicMat, cv::Mat_<double>& distCoeffs);
	bool solvePnP(const vector<CalibSpace::Point3d2d>& ref_pt_vec, const cv::Mat& intrisicMat,
		const cv::Mat_<double> & distCoeffs, cv::Mat& rVec, cv::Mat& tVec, vector<int>& inliers, bool ransac = false);

	void rt2camPose(const cv::Mat& rVec, const cv::Mat& tVec, vector<double>& camPose);
};

