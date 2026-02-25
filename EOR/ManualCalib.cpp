// ManualCalib.cpp : 定义控制台应用程序的入口点。
// for std
#include <iostream>
// for opencv
#include "ManualCalib.h"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
//#include "opencv2/ximgproc.hpp"
//#include "opencv2/line_descriptor.hpp"
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>

//#include <boost/concept_check.hpp>

#include <iostream>
#include<fstream>
#include <glog/logging.h>


#include "Optimize.h"
#include "KMath/TransRotation.h"
#include "DataIO.h"
#include "TestCalib.h"

ManualCalib::ManualCalib()
{
}

ManualCalib::~ManualCalib()
{
}

void ManualCalib::process(vector<double>& camPose, const string& folder_path, const vector<CalibSpace::Point3d2d>&ref_pt_vec)
{
// 	OptimizeCeres::optimize(ref_pt_vec, camPose);
// 	OptimizeCeres::evaluate(ref_pt_vec, camPose);

	cv::Mat intrisicMat;
	cv::Mat_<double> distCoeffs;
	initCamera(intrisicMat, distCoeffs);

	cv::Mat rvec;
	cv::Mat tvec;
	vector<int> inliers;
	solvePnP(ref_pt_vec, intrisicMat, distCoeffs, rvec, tvec, inliers);
	rt2camPose(rvec, tvec, camPose);
//	OptimizeCeres::evaluate(ref_pt_vec, camPose);

	vector<CalibSpace::PointXYZI> pc(ref_pt_vec.size());
	transform(ref_pt_vec.begin(), ref_pt_vec.end(), pc.begin(), [](const auto& p3d2d)->CalibSpace::PointXYZI {
		CalibSpace::PointXYZI xyzi;
		xyzi.x = p3d2d.p3d.x;
		xyzi.y = p3d2d.p3d.y;
		xyzi.z = p3d2d.p3d.z;
		return xyzi; });

	string img_path = folder_path + "image/1617846647005897.jpg";
	cv::Mat camera_img = cv::imread(img_path);

	Calib cb;
	cb.projectPointCloud2Image(pc, camPose, camera_img);
//	cb.projectPointCloud2Image(pc, intrisicMat, distCoeffs, rvec, tvec, camera_img);

	string out_path = folder_path + "proj/calib1617846647005897.jpg";
	cv::imwrite(out_path, camera_img);

	LOG(INFO) << "******";
	for (int i = 0; i < 6; i ++)
	{
		LOG(INFO) << camPose[i];
	}
	return;
}

void ManualCalib::processKitti(vector<double>& camPose, const string& folder_path)
{
	string file_path = folder_path + "refpoints.txt";
	vector<CalibSpace::Point3d2d> ref_pt_vec;
//	DataIO::readRefPoints(file_path, ref_pt_vec);

//	OptimizeCeres::optimize(ref_pt_vec, camPose);
//	OptimizeCeres::evaluate(ref_pt_vec, camPose);

	cv::Mat intrisicMat;
	cv::Mat_<double> distCoeffs;
	initCamera(intrisicMat, distCoeffs);

	cv::Mat rvec;
	cv::Mat tvec;
	vector<int> inliers;
	solvePnP(ref_pt_vec, intrisicMat, distCoeffs, rvec, tvec, inliers);
	rt2camPose(rvec, tvec, camPose);
	OptimizeCeres::evaluate(ref_pt_vec, camPose);

	vector<CalibSpace::PointXYZI> pc;
	string lidar_path = folder_path + "velodyne_points/txt/0000000036.txt";
//	DataIO::readPointCloud(lidar_path, pc);

	string img_path = folder_path + "image_00/data/0000000036.png";
	cv::Mat camera_img = cv::imread(img_path);

	//	cv::Mat img;
	Calib cb;
	cb.projectPointCloud2Image(pc, camPose, camera_img);

	//cv::Mat add_img;
	//cv::addWeighted(camera_img, 0.5, img, 0.5, 0, add_img);
	string out_path = folder_path + "image_00/proj/calib0000000036.png";
	cv::imwrite(out_path, camera_img);

	LOG(INFO) << "******";
	for (int i = 0; i < 6; i++)
	{
		LOG(INFO) << camPose[i];
	}
	return;
}

void ManualCalib::initCamera(cv::Mat& intrisicMat, cv::Mat_<double>& distCoeffs)
{
	intrisicMat = cv::Mat(3, 3, cv::DataType<double>::type); // Intrisic matrix
	distCoeffs = cv::Mat_<double>(1, 5, cv::DataType<double>::type);   // Distortion vector
	intrisicMat.at<double>(0, 0) = CalibSpace::FX;
	intrisicMat.at<double>(1, 0) = 0;
	intrisicMat.at<double>(2, 0) = 0;

	intrisicMat.at<double>(0, 1) = 0;
	intrisicMat.at<double>(1, 1) = CalibSpace::FY;
	intrisicMat.at<double>(2, 1) = 0;

	intrisicMat.at<double>(0, 2) = CalibSpace::CX;
	intrisicMat.at<double>(1, 2) = CalibSpace::CY;
	intrisicMat.at<double>(2, 2) = 1;

	distCoeffs << 0.0, 0.0, 0.0, 0.0, 0.0;
}

bool ManualCalib::solvePnP(const vector<CalibSpace::Point3d2d>& ref_pt_vec,
	const cv::Mat& intrisicMat, const cv::Mat_<double> & distCoeffs, cv::Mat& rVec, cv::Mat& tVec, vector<int>& inliers, bool ransac)
{
	if (ref_pt_vec.size() < 3)
	{
		return false;
	}
	vector<cv::Point3d> d3_pts(ref_pt_vec.size());
	vector<cv::Point2d> d2_pts(ref_pt_vec.size());

	transform(ref_pt_vec.begin(), ref_pt_vec.end(), d3_pts.begin(), [](const auto& p3d2d)->cv::Point3d {
		return p3d2d.p3d; });

	transform(ref_pt_vec.begin(), ref_pt_vec.end(), d2_pts.begin(), [](const auto& p3d2d)->cv::Point2d{
		return p3d2d.p2d; });
	if (ransac)
	{
		return cv::solvePnPRansac(d3_pts, d2_pts, intrisicMat, distCoeffs, rVec, tVec, false, 100, 8.0, 0.99, inliers);
	}
	else
	{
		return cv::solvePnP(d3_pts, d2_pts, intrisicMat, distCoeffs, rVec, tVec);
	}

	//LOG(INFO) << rVec;
	//LOG(INFO) << tVec;

	//vector<cv::Point2f> prj_points;
	//cv::projectPoints(d3_pts, rVec, tVec, intrisicMat, distCoeffs, prj_points);
	//LOG(INFO) << d2_pts;
	//LOG(INFO) << prj_points;
	//vector<cv::Point2f> dif_points(d2_pts.size());
	//for (int i = 0; i < d2_pts.size(); i++)
	//{
	//	dif_points[i] = d2_pts[i] - prj_points[i];
	//}
	//LOG(INFO) << dif_points;
}

void ManualCalib::rt2camPose(const cv::Mat& rVec, const cv::Mat& tVec, vector<double>& camPose)
{
	camPose[0] = tVec.at<double>(0);
	camPose[1] = tVec.at<double>(1);
	camPose[2] = tVec.at<double>(2);

//	Eigen::Vector3d c = { camPose[3] , camPose[4] , camPose[5] };
//	cout << c << endl;
//	cout << rVec << endl;
	cv::Mat R;
	cv::Rodrigues(rVec, R);
//	cout << R << endl;

	Eigen::Matrix3f ER;
	cv::cv2eigen(R, ER);
//	cout << ER << endl;
	Eigen::Vector3f euler = ER.eulerAngles(0, 1, 2);
//	cout << euler << endl;

//	Eigen::Matrix3f er;
//	TransRotation::eigenEuler2RotationMatrix(euler, er);
//	cout << er << endl;
//	cv::Mat rrrVec;
//	cv::Rodrigues(R, rrrVec);
//	cout << rrrVec << endl;

	camPose[3] = euler[2];
	camPose[4] = euler[1];
	camPose[5] = euler[0];

	//Eigen::Vector3f cr(camPose[5], camPose[4], camPose[3]);
	//Eigen::Matrix3f CR;
	//TransRotation::eigenEuler2RotationMatrix(cr, CR);
	//cout << CR << endl;
}

///////////////////////////////////////////ransac///////////////////////////////////////////////////////
