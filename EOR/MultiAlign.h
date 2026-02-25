#pragma once
#ifndef MultiAlign_H
#define MultiAlign_H

#include "DataSet.h"
#include "TestCalib.h" 
#include "DataIO.h"
#include "KF/FusionEKF.h"


#ifdef IMAGEMATCH_EXPORTS2
#define IMAGEMATCH_API   __declspec(dllexport)
#else
#define IMAGEMATCH_API __declspec(dllimport)
#endif


#include <string>

//#define DS_LOG //delicate system car online
#define BiSeNet 0 //reasoning by bisenet or dlink 
#define  READ_MASK 1  //read mask or xml
#define CONFIG_PARA_TBL "config_para_tbl"
#define CONFIG_PARA_RECON_TBL "config_para_recon_tbl"



enum RegMethod
{
	RegMethod_NONE = 0,//0 none
	RegMethod_EOR = 1, // enhanced object registration
	RegMethod_LMR = 2,  // lane marking only registration
	RegMethod_MOR = 3, // multi object registration
	RegMethod_NONE_GT = 5,
};

class HDMapData
{
public:
	HDMapData() {};
	~HDMapData() {};

	HDObject_VEC ld_vec;
	HDObject_VEC hd_obj_vec;
	HDSignBoard_VEC hd_isb_vec;
	HDObject_VEC junction_vec;
};


class IMAGEMATCH_API MultiAlign
{
public:
	MultiAlign();
	~MultiAlign();


	void process(int step = 0);

	bool preprocess(vector<RAW_INS>& ins_vec,
		HDMapData& hdobj_data,
		const std::string& _dataPath,
		const std::string& _midPath,
		const int& band, const string& para_tbl_na = CONFIG_PARA_TBL,
		string car_id = "");

	void preprocess(int step, int method);
	void jitterCamPose(map<string, CamPose>& pose_map);

	void groudTruthCamPose(map<string, CamPose>& pose_map);
	void clearFiles();
	void calcEgoHeightWithTrajectoriesAndLaneMarkings();
	void processRegistHDAndMSSImages(int run_type = 0);
	//void processRegistLocalMapAndSingleImage(RAW_INS& ins, HDMapData& hdobj_data,
	//	std::map<int/*类别*/, std::vector<std::vector<cv::Point>>/*轮廓点集*/> percep_contours);

	//void processBuildPosGraphConstraints();
	//void processPoseGraphOptimize();

	bool isInIntersection(CamPose& cp);
	void kalmanFilterInitialize(const vector<double>& pose);
	bool kalmanFilterEstimate(CamPose& cp);
	void processRelativePose();
	
	void cutPartImage();

	void inversePerspectiveAdjustImageLines(const string& img_na, map<int, LINE_VEC>& lines_map);

protected:

	bool initConfig(const string& para_tbl_na,const string& car_id);
	bool initDBConfig(const vector<string>& v);
	bool initCalibSpace();

	void updateCamPose(vector<double>& campos);
	int getImageIndex(const string& img_na);

	
	void transformCoord(vector<RAW_INS>& ins_vec);
	void transformCoord(HDObject_VEC& _data);
	void getImageCenter(cv::Point2f& c);
	void horizonalAssignX(vector<double>& interval_vec, vector<int>& x_vec);
	void horizonalAssignLocalHDMapX(vector<double>& interval_vec, const RAW_INS& ins, vector<int>& x_vec);
	void splitLaneContours(LINE_VEC& contoursf);
	void buildTimeIndex(cv::Mat& source, const vector<RAW_INS>& ins_vec);
	
	bool readTrajecotry(vector<RAW_INS>& ins_vec);
	void readHDMap(bool keep_inside_intersection = false);
	
	//bool getLocalHDMap(const vector<RAW_INS>& ins_vec, const int& idx, HDObject_VEC& local_local_hdobj_vec);

	bool getLocalHDMap(const RAW_INS& ins, map<int, HDObject_VEC>& local_local_hdobj_vec);
	bool calcEORWeight(const string& img_na, OptimizePara& op);
	bool runSpatialFeatures(const vector<double>& gt_campos,
		const RAW_INS& ins, const CamPose& cp);
	void smooth();
	void buildPosGraphConstraints();

	bool isValid(const vector<double>&  cam, const vector<double>&  base, const vector<double>&  diff_threshold);
	void calcCurrentPoseVertex(const RAW_INS& ins, const vector<double>& camPose, Eigen::Vector3d& camera_in_world, Eigen::Vector4d& q);
	void recoverCurrentPoseVertex(const POSE& pose, const RAW_INS& ins, vector<double>& campose);
	bool calcRelativePoseEdge2d2d(const CamPose& cp1, const CamPose& cp2, Eigen::Matrix3f& R, Eigen::Vector3f& t);
	bool calcRelativePoseEdge3d2d(const CamPose& cp1, const CamPose& cp2, Eigen::Matrix3f& R, Eigen::Vector3f& t);
	bool calcImage2Image(const cv::Mat& img, const cv::Mat& img2, cv::Mat& homography_matrix);
	void transformLocalRelative2WorldRelative(Eigen::Matrix3f& er12, Eigen::Vector3f& et12, const CamPose& cp1);
	void saveG2OVertex(CamPose& cp);
	bool saveG2OEdge(const CamPose& cp1, const CamPose& cp2);
	
	//for car online registration with local map
	//bool registLocalHDMapImageLaneDividers(map<int, LINE_VEC>& lines_map,
	//	const vector<RAW_INS>& ins_vec, 
	//	const int& idx, 
	//	CamPose& cp);

	void undistortImage(const cv::Mat& src_img, cv::Mat& dest_img);
	//for read image semantic by self reading mask
	bool registHDImageLaneDividers(const vector<RAW_INS>& ins_vec,
		const int& idx,
		CamPose& cp);

	//for receive image semantic by para
	//bool registHDImageLaneDividers(const string& img_na, 
	//	const vector<RAW_INS>& ins_vec, 
	//	const int& idx, 
	//	std::map<int/*类别*/, std::vector<std::vector<cv::Point>>/*轮廓点集*/> lines_map, 
	//	CamPose& cp);

	//	void filterPoles(HDMapObject_VEC & obj_vec, const double& ins_z);
	
	float evaluteMutualInfo(const map<int, LINE_VEC>& lines_map_rgb, const map<int, LINE_VEC>& lines_map_hdmap);
	float evaluteSimilarity(const map<int, LINE_VEC>& lines_map_rgb, const map<int, LINE_VEC>& lines_map_hdmap);

	void buildImageObject(const string& img_na, HDObject_VEC& isb_vec /*const multimap<string, Mark>& marks_map*/);
	cv::Scalar getSignBoardClassifyColor(const HDSignBoard& sgb);
	cv::Scalar getHDMapObjectClassifyColor(const int& fcode);
	void getImageVec(const string& folder_path, vector<string>& img_na_vec);
	
	//	void manualCalib(const string& folder_path, const vector<RAW_INS>& ins_vec, double* camPose);
	bool matchImageWithLog(const string& img_na, double t0, double t_end, float time_res, int& idx);
	void splitJumpContours(map<int, LINE_VEC>& contours_map);
	void splitJumpSignBoard(vector<vector<cv::Point>> &plys);
	void cutPartImage(const vector<string> &img_na_vec);

	template<typename T> void getEgoMapContours(const map<int, vector<T>>& hdobj_vec, 
		const vector<double>& camPose,
		map<int, LINE_VEC>& contours_map,
		bool ipm = true);

	template<typename T>
	void transformHDMapFromWorld2Camera(const vector<T>& hdobj_vec, const RAW_INS& ins, const vector<double>& camPose, 
		map<HDMapPointID, cv::Point3f>& cam_pts_map);
	template<typename T> 
	void transformHDMapFromWorld2Image(const vector<T>& hdobj_vec, const POSE& pos, vector<vector<cv::Point>>& line_vec);
	template<typename T> 
	void transformHDMapFromWorld2Image2(const vector<T>& hdobj_vec, const CamPose& cp, const Eigen::Matrix3f& R, 
		const Eigen::Vector3f& Trans, map<int, LINE_VEC>& contours_map);
	ObjectClassification convHDMapType2ObjectClassification(int type);
	//template<typename T>
	//void getValid3dPoints(const vector<T>& hdobj_vec, const RAW_INS& ins, const cv::Point3f& tt, const cv::Vec3f& euler,
	//	vector<cv::Point3f>& xyz_vec, vector<vector<cv::Point>>& contours);

	//template<typename T> 
	//void getValid3dLines(const vector<T>& hdobj_vec, const RAW_INS& ins, const cv::Point3f& tt, const cv::Vec3f& euler, 
	//	vector<vector<cv::Point3f>>& xyz_vec, vector<vector<cv::Point>>& contours);

	template<typename T> void getValid3dLines(const vector<T>& hdobj_vec, const RAW_INS& ins, const vector<double>& camPose, vector<vector<cv::Point3f>>& xyz_vec, vector<vector<cv::Point>>& contours);
	template<typename T>
	void transformHDMapFromWorld2Ego(const vector<T>& hdobj_vec, const RAW_INS& ins, 
		map<int, vector<T>>& ego_hdobj_map);
	void transformInsFromWorld2Ego(const vector<RAW_INS>& ins_vec,  const int& idx,
		vector<RAW_INS>& local_ins_vec);
	void convInsFromWorld2Ego(const vector<RAW_INS>& ins_vec, const RAW_INS& ins, 
		vector<RAW_INS>& local_ins_vec);

	template<typename T>
	void transformEgo2Camera(const vector<T>& hdobj_vec, const vector<double>& camPose, map<int, Point3f_VEC>& xyz_vec_map);

	bool beyongImageRange(int type, Eigen::Vector3f lp);
	template<typename T> 
	void removeByTraceRange(vector<T>& local_local_hdobj_vec, const vector<RAW_INS>& ins_vec);
	
	void creatFolder(const string& folder_path);
	bool findMatchRect(multimap<string, Mark>::const_iterator low_img_rect, multimap<string, Mark>::const_iterator up_img_rect, const cv::Rect& sb_rect,cv::Rect & rect);

	void mergeImageLines(vector<vector<cv::Point2f>>& line_vec);
	template<typename T>
	void horizonalAssignHDMap(vector<T>& local_local_hdobj_vec, const vector<RAW_INS>& local_ins_vec);
	void inversePerspectiveHorizonalAssignImageLines(vector<vector<cv::Point2f>>& lines,
		map<int, vector<vector<cv::Point2f>>>& lines_map);

	void inversePerspectiveHorizonalAssignImageLines_DBSCAN(vector<vector<cv::Point2f>>& lines, 
		map<int, vector<vector<cv::Point2f>>>& lines_map);
	//	bool saveImage(const string& sub_folder, cv::Mat& img, vector<vector<cv::Point>>& contours);

	void saveImage(const string& sub_folder,
		const string& img_na, 
		const cv::Mat& img, 
		const cv::Scalar& sca,
		map<int, LINE_VEC>& lines_map);

	void saveImage(const string& sub_folder, 
		const string& img_na, 
		const cv::Mat& img, 
		const map<int, cv::Scalar>& type_color,
		map<int, LINE_VEC>& lines_map);


	template<typename PointT>
	void fitByOpenCV(std::vector<PointT>& img_shp, int n);


	void collectLocalIns(const vector<RAW_INS>& ins_vec,
		int idx,
		double len,
		vector<RAW_INS>& local_ins_vec,
		bool add = true,
		bool farward = true);

	bool analyzeRobustnessbyRandomSample(const vector<double>& gt_campos, 
		const RAW_INS& ins,
		const string& img_na);

	void analyzeSampleRobustness(const vector<double>& gt_campos, 
		const map<int, LINE_VEC>& lines_map, 
		const map<int, HDObject_VEC>& ego_hdobj_vec,
		const string& img_na);

	/*void analyzeSpatialDistribution(const vector<double>& gt_campos,
		const map<int, LINE_VEC>& lines_map,
		const map<int, HDObject_VEC>& ego_hdobj_vec,
		const CamPose& cp,
		const string& file_path);*/

protected:

	Calib m_calib;
	CapabilityCalculator m_cc;

	//cv::Point3f m_tt0;
	//cv::Vec3f m_euler0;
	//vector<double> m_camPose;
	//vector<double> m_diff_threshold;

/* 	vector<double> m_last_camPose;*/
	vector<RAW_INS> m_ins_vec;
	vector<cv::Point2f> m_trace_box;
	HDMapData m_hd_map;

	set<ObjectClassification> m_reg_oc_set;

	map<string, CamPose> m_pos_map;

	DataIO* m_io;

	FusionEKF* m_kf;

	RegMethod m_reg_method;
	string SUB_FOLDER_PLY;

	map<int, cv::Scalar> m_type_color;
	map<int, cv::Scalar> m_type_seg_color;
};

inline void findPeakPoints(const map<double, double>& value_p_map, vector<double>& v_vec, double merge_width)
{
	//map<T, double>peak_small_v_map;
	//auto itr = value_p_map.begin();
	//for (; itr != value_p_map.end(); itr++)
	//{
	//	auto last_itr = itr;
	//	auto next_itr = itr;
	//	next_itr++;
	//	last_itr--;
	//	//首
	//	if (itr == value_p_map.begin() && itr->second - next_itr->second <= 0)
	//	{
	//		peak_small_v_map.insert(*itr);
	//		continue;
	//	}
	//	//中
	//	if (itr->second - next_itr->second <= 0 &&
	//		itr->second - last_itr->second <= 0)
	//	{
	//		peak_small_v_map.insert(*itr);
	//	}
	//	//尾
	//	if (next_itr == value_p_map.end())
	//	{
	//		if (itr->second - last_itr->second <= 0)
	//		{
	//			peak_small_v_map.insert(*itr);
	//		}
	//		break;
	//	}
	//}
	if (value_p_map.size() == 0)
	{
		return;
	}

	if (value_p_map.size() == 1)
	{
		v_vec.push_back(value_p_map.begin()->first);
	}

	//	double avg_v = 0;
	map<double, double>peak_big_v_map;
	auto itr = value_p_map.begin();
	for (; itr != value_p_map.end(); itr++)
	{
		auto last_itr = itr;
		auto next_itr = itr;
		next_itr++;
		last_itr--;
		//		avg_v += itr->second;

		//首
		if (itr == value_p_map.begin() && itr->second - next_itr->second >= 0)
		{
			peak_big_v_map.insert(*itr);
			continue;
		}
		//中
		if (itr->second - next_itr->second >= 0 &&
			itr->second - last_itr->second >= 0)
		{
			peak_big_v_map.insert(*itr);
		}
		//尾
		if (next_itr == value_p_map.end())
		{
			if (itr->second - last_itr->second >= 0)
			{
				peak_big_v_map.insert(*itr);
			}
			break;
		}
	}
	//	avg_v = avg_v / value_p_map.size();

	itr = peak_big_v_map.begin();
	for (; itr != peak_big_v_map.end();)
	{
		auto last_itr = itr;
		auto next_itr = itr;
		next_itr++;
		last_itr--;
		//if (itr->second < avg_v)
		//{
		//	itr = peak_big_v_map.erase(itr);
		//	continue;
		//}
		//尾
		if (peak_big_v_map.size() == 1)
		{
			break;
		}

		if (next_itr == peak_big_v_map.end())
		{
			if (abs(itr->first - last_itr->first) < merge_width)
			{
				auto v = (itr->first + last_itr->first) / 2;
				auto vp = itr->second;
				peak_big_v_map.erase(itr);
				peak_big_v_map.erase(last_itr);
				peak_big_v_map.insert(make_pair(v, vp));
			}
			break;
		}
		//首
		if (abs(itr->first - next_itr->first) < merge_width)
		{
			auto v = (itr->first + next_itr->first) / 2;
			auto vp = itr->second;
			peak_big_v_map.erase(next_itr);
			itr = peak_big_v_map.erase(itr);
			peak_big_v_map.insert(make_pair(v, vp));
			itr--;
			continue;
		}
		else
		{
			itr++;
		}
	}

	for_each(peak_big_v_map.begin(), peak_big_v_map.end(), [&](const auto& p) {
		v_vec.push_back(p.first);
	});

	return;
}


template<typename PointT>
void MultiAlign::fitByOpenCV(std::vector<PointT>& img_shp, int n)
{
	//n次多项式
	if (img_shp.size() <= 2)
	{
		return;
	}
	//Number of key points  
	int N = img_shp.size();

	//构造矩阵X  
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(img_shp[k].y, i + j);
			}
		}
	}

	//构造矩阵Y  
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(img_shp[k].y, i) * img_shp[k].x;
		}
	}

	cv::Mat  A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A  
	cv::solve(X, Y, A, cv::DECOMP_LU);


	auto min_itr = min_element(img_shp.begin(), img_shp.end(), [](const auto& ele1, const auto& ele2)->bool {
		return ele1.y < ele2.y;
		});
	double start_y = min_itr->y;

	auto max_itr = max_element(img_shp.begin(), img_shp.end(), [](const auto& ele1, const auto& ele2)->bool {
		return ele1.y < ele2.y;
		});
	double end_y = max_itr->y;

	for (int i = 0; i < N; i++)
	{
		double y = start_y + (end_y - start_y) / N * i;
		img_shp[i].y = y;
		img_shp[i].x = A.at<double>(0, 0) +
			A.at<double>(1, 0) * y +
			A.at<double>(2, 0) * std::pow(y, 2);/*+
			A.at<double>(3, 0)*std::pow(y, 3)*/
	}

}
#endif