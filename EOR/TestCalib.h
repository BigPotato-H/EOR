// Calib.cpp : 定义控制台应用程序的入口点。

// for std
// for opencv
#pragma once
#ifndef Calib_H
#define Calib_H
//#include "RansacSolver.h"
#include "DataSet.h"
#include <set>
#include <opencv2/flann/miniflann.hpp>

//#include "Kalman.h"

enum SolveMethod {
	NONE = -1,
	ICP = 0,
	JC = 1
};

class Calib
{
public:
	Calib();
	~Calib();

	void process(vector<double>& camPose, SolveMethod type = ICP);
	void setCameraImage(const cv::Mat& img);

	void calcBlurImage(const vector<vector<cv::Point>>& line_vec, cv::Mat& blur_mat);
	void calcBlurImage(const map<int, LINE_VEC>& line_vec_map, cv::Mat& blur_mat);

	void calcDistanceTransformImage(const map<int, LINE_VEC>& line_vec_map,
		cv::Mat& blur_mat);

	void extractImageRoadLines(const string& folder_path, const string& img_na,
		const cv::Mat& camera_img, vector<vector<cv::Point>>& line_vec);
	void extractImageVerticalLines(const string& folder_path, const string& img_na, const cv::Mat& camera_img, vector<vector<cv::Point>>& line_vec);
	void extractImageDeepLearningLines(const string& folder_path, const string& img_na,
		const cv::Mat& camera_img, vector<vector<cv::Point>>& line_vec);
	
	void extractImageDeepLearningMultiLines(const string& folder_path,
		const string& img_na,
		map<int, LINE_VEC>& lines_map);

	void extractImageDeepLearningMultiLinesXml(const string& folder_path, 
		const string& img_na,
		const set<ObjectClassification>& oc_set, 
		map<int, LINE_VEC>& lines_map);


	void projectPointCloud2Image(const vector<CalibSpace::PointXYZI>& pc, const vector<double>& camPose, cv::Mat&img);

	void projectPointCloud2Image(const vector<CalibSpace::PointXYZI>& pc, const cv::Mat& intrisicMat, const cv::Mat& distCoeffs,
		const cv::Mat& rVec, const cv::Mat& tVec, cv::Mat&img);

	void inversePerspectiveMapping(const cv::Mat& src, const cv::Mat& warpmat_src2ipm, cv::Mat& dest);
	void regist3D2DLines(const vector<vector<cv::Point3f>>& xyz_line_vec, const vector<vector<cv::Point>>& line_vec, vector<double>& camPose);
	bool buildKDTree(const map<int, Point_VEC>& ij_linevec, map<int, cv::Mat>& src_map);
	void releaseKDTree();
	bool isValid(const vector<double>&  cam, const vector<double>&  base, const vector<double>&  diff);
	double iterateClosestPoint2d3d(const map<int, Point3f_VEC>& xyz_vec,
		const map<int, LINE_VEC>& lines_map,
		const map<int, LINE_VEC>& camera_lines_map,
		vector<double>& camPose, const vector<double>& diff_threshold);

	void iterateClosestPoint2d3d(const Point3f_VEC& xyz_vec, const Point_VEC& ij_vec, vector<double>& camPose);
	
	//void iterateClosestPoint3d3d(const map<int, HDObject_VEC>& local_local_hdobj_vec,
	//	const map<int, LINE_VEC>& lines_map, vector<double>& camPose);

	void iterateJC(const map<int, Point3f_VEC>& xyz_vec_map, const cv::Mat& blur_mat, vector<double>& camPose);

	double iterateDistanceTransform(const map<int, Point3f_VEC>& xyz_vec_map, 
		const map<int, LINE_VEC>& lines_map, 
		vector<double>& camPose);

	double iterateCrossEntropy(const map<int, Point3f_VEC >& xyz_vec_map,
		const map<int, LINE_VEC>& lines_map, 
		const float& pitch,
		vector<double>& camPose);
	//	void iterateClosestPoint(const vector<vector<cv::Point3f>>& xyz_line_vec, const vector<vector<cv::Point>>& line_vec, double* camPose);
	bool findCorrespondPoints(const vector<double>& camRot, const vector<cv::Point3f>& xyz_vec, const set<int>& xyz_s_e_indices, const vector<cv::Point>& ij_vec, const set<int>& s_e_indices, vector<CalibSpace::Point3d2d>& p3d2ds, float D);
	bool findCorrespondPoints(const vector<double>& camRot, 
		const map<int, Point3f_VEC>& xyz_vec, 
		const map<int, Point_VEC>& ij_vec, 
		vector<CalibSpace::Point3d2d>& p3d2ds, 
		float D,
		int tm);
	bool findCorrespondPoints(const vector<double>& camRot, const Point3f_VEC& xyz_vec, const Point_VEC& ij_vec, vector<CalibSpace::Point3d2d>& p3d2ds, float D);
	
	bool optimizePnP(const vector<CalibSpace::Point3d2d>& ref_vec, 
		vector<double>& camPose, 
		vector<int>& inliers, 
		bool ransac = false);

	bool optimizePanoPnP(const vector<CalibSpace::Point3d2d>& ref_vec, 
		vector<double>& camPose, 
		vector<int>& inliers, 
		bool ransac = false);

	void getValid3dPoints(const vector<vector<cv::Point3f>>& xyz_line_vec, vector<cv::Point3f>& xyz_vec);
	void getValid2dPoints(const vector<vector<cv::Point>>& line_vec, vector<cv::Point>& ij_vec);
	void regist3D2DJC(const vector<vector<cv::Point3f>>& xyz_line_vec, const vector<vector<cv::Point>>& line_vec, vector<double>& camPose);

// 	void kalmanInitial(const CamPose& cp, HN::KalmanFilter* kf);
// 	Eigen::VectorXd kalmanTracePredict(const CamPose& last_cp, const CamPose& cp, HN::KalmanFilter* kf);
	void perspectiveMappingPoints(vector<vector<cv::Point>>& line_pts, const cv::Mat& warpmat_ipm2src);
	template<typename T>
	void projectObjects2Image(const vector<vector<T>>& obj_vec,
		const vector<double>& camPose, cv::Mat& img);

	void initCamera(cv::Mat& intrisicMat, cv::Mat_<double>& distCoeffs);
	bool solvePnP(const vector<CalibSpace::Point3d2d>& ref_pt_vec,
		const cv::Mat& intrisicMat, const cv::Mat_<double>& distCoeffs, 
		cv::Mat& rVec, cv::Mat& tVec, vector<int>& inliers, bool ransac);
	void rt2camPose(const cv::Mat& rVec, const cv::Mat& tVec, vector<double>& camPose);


private:
	size_t read3dVectorData(const string& file_path, vector<vector<cv::Point3f>>& line_vec);

	size_t calc2dVectorData(const cv::Mat& img, vector<vector<cv::Point>>& line_vec);

private:

	cv::Mat m_camera_img;
	map<int, cv::Scalar> m_color_map;
};

template<typename T>
void Calib::projectObjects2Image(const vector<vector<T>>& obj_vec,
	const vector<double>& camPose, cv::Mat&img)
{
	vector<cv::Point2d> imagePoints;
	vector<cv::Point3d> objectPoints(pc.size());
	transform(pc.begin(), pc.end(), objectPoints.begin(), [](const auto& p)->cv::Point3d {
		return cv::Point3d(p.x, p.y, p.z);
	});

	int i = 0;
	for_each(pc.begin(), pc.end(), [&](const auto& p) {
		double d3[3] = { p.x, p.y, p.z };
		double d2[2];
		if (OptimizeCeres::convertPoint3dTo2d(camPose, d3, d2))
		{
			cv::drawMarker(img, cv::Point(d2[0], d2[1]), cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 10);
		}
		i++;
	});

}
#endif