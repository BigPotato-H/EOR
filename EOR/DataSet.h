#pragma once
#ifndef DATASET_H
#define DATASET_H
//#include "mrpt/utils/types_math.h"
#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <map>
#include<Eigen/Core>
using namespace std;

typedef vector<cv::Point3f> Point3f_VEC;
typedef vector<cv::Point> Point_VEC;
typedef vector<vector<cv::Point>> LINE_VEC;

const string MuChengPano = "MuChengPano";
const string NanCePano = "NanCePano";
const string ZiCaiPano = "ZiCaiPano";
const string ZiCaiWide = "ZiCaiWide";

#define NA_TRACE_DB		("Trace_Info.db") 


enum CameraType {
	CAMERA_MSS_WIDE = 0,
	CAMERA_MSS_PANO,
	CAMERA_MSS_POLYWIDE,
	CAMERA_DS_WIDE
};

namespace CalibSpace{
	extern string data_item;
	extern int band;
	extern double CX;
	extern double CY;
	extern double FX;
	extern double FY;
	extern int IMG_WIDTH;
	extern int IMG_HEIGHT;
	extern 	cv::Mat intrisicMat;
	extern cv::Mat_<double> distCoeffs;
	extern CameraType camera_type;
	extern cv::Rect image_rect;
	extern vector<double> extrinsic_para;
	extern float ego_height;
	extern float camera_height;
	extern cv::Mat warpmat_src2ipm;
	extern cv::Mat warpmat_ipm2src;
	extern bool activate_flg;
	class Point3d2d
	{
	public:
		Point3d2d() {};
		Point3d2d(const cv::Point3d& _p3d, const cv::Point2d& _p2d) { p3d = _p3d; p2d = _p2d; type = 0; res = -1; }
		~Point3d2d() {};

		cv::Point3d p3d;
		cv::Point2d p2d;
		int type;//类型
		double res;
	};

	class PointXYZI
	{
	public:
		PointXYZI() {};
		PointXYZI(const cv::Point3d& pt, const double& _intensity) {
			x = pt.x;
			y = pt.y;
			z = pt.z;
			intensity = _intensity;
		};
		~PointXYZI() {};

		double x;
		double y;
		double z;
		double intensity;
	};	
	void TranslateAndRot(const cv::Point3d& pt, cv::Point3d& lpt, const cv::Point3d& t, const cv::Mat& r);
	void RotAndTranslate(const cv::Point3d& pt, cv::Point3d& pca, const cv::Point3d& t, const cv::Mat& r);
	void Camera2Image(const cv::Point3d& pca, cv::Point& pp);

	void EigenTranslateAndRot(const Eigen::Vector3d& pt, Eigen::Vector3f& lpt, const  Eigen::Vector3d& t, const  Eigen::Matrix3f& r);

	void initInversePerspectiveMappingMat(const vector<cv::Point2f>& corners,
		cv::Mat& warpmat_src2ipm,
		cv::Mat& warpmat_ipm2src);
	void pose6DoFToQuaternion(const vector<double>& pose,
		vector<double>& poseQuaternion);
	void QuaternionTopose6DoF(const Eigen::VectorXd& poseQuaternion,
		vector<double>& pose);
}


class RAW_INS
{
public:
	double time;
	string name;
	string sta_name;
	cv::Point3d point;
	cv::Point3d lonlat;
	double speedx;
	double speedy;
	double speedz;
	double roll;
	double pitch;
	double heading;

	Eigen::Matrix3d R;
	Eigen::Vector3d T;
};

class HDMapPointID
{
public:
	string obj_id;
	int v_id;

	bool operator<(const HDMapPointID& a) const
	{
		if (obj_id != a.obj_id)
		{
			return obj_id < a.obj_id;
		}
		else
		{
			return v_id < a.v_id;
		}
	}

};

class HDObject
{
public:
	HDObject() {};
	~HDObject() {};

	string obj_id;
	string prop_id;
	int type;
	int horizonal_idx;
	vector<cv::Point3d> shape;
	vector<cv::Point3d> shape_org;
	cv::Rect2d rect_3d;
	//---add 2d
	vector<cv::Point> ij_shape;
	cv::Rect rect;
};
typedef vector<HDObject> HDObject_VEC;

class HDSignBoard
{
public:
	HDSignBoard() {};
	~HDSignBoard() {};

	string obj_id;
	string prop_id;
	int fcode;
	uchar frame;
	uchar kind;
	uchar majortype;
	uchar subtype;
	string content;
	int width;
	int height;
	vector<cv::Point3d> shape_org;
	vector<cv::Point3d> shape;
	//---add 2d
	vector<cv::Point> ij_shape;
	cv::Rect rect;

	int type;
};
typedef HDSignBoard HDImageObject;
typedef vector<HDSignBoard> HDSignBoard_VEC;
typedef map<string, HDSignBoard> HDSignBoard_MAP;

//class ImageSignBoard
//{
//public:
//	ImageSignBoard() {};
//	~ImageSignBoard() {};
//
//	uint64 obj_id;
//	uchar frame;
//	uchar kind;
//	uchar majortype;
//	uchar subtype;
//	string content;
//	int width;
//	int height;
//	vector<cv::Point> shape;
//	cv::Rect rect;
//};
//typedef map<uint64, ImageSignBoard> ImageSignBoard_MAP;
//typedef vector<ImageSignBoard> ImageSignBoard_VEC;

inline string getFileName(const string& path)
{
	auto spos = path.rfind('\\');
	if (spos == string::npos)
	{
		spos = path.rfind('/');
	}
	if (spos == string::npos)
	{
		return "";
	}
	auto epos = path.rfind('.');
	string file_na = path.substr(spos + 1, epos - spos - 1);
	return file_na;
}


inline void getStationImageName(string& sta_name, string& name)
{
	auto rmv_end = name.find(".jpg");
	if (rmv_end != string::npos)
	{
		name = name.substr(0, rmv_end);
	}
	auto spos = name.rfind('@');
	if (spos == string::npos)
	{
		return;
	}
	sta_name = name.substr(0, spos);
	string img_na = name.substr(spos + 1, name.size() - spos - 1);
	name =  img_na;
}

enum HDMapFcode
{
	//HDMapFC_BLA = 1,
	//HDMapFC_BLB = 2,
	//HDMapFC_PROPP = 3,
	//HDMapFC_PROPG = 4,

	//HDMapFC_RV = 10, //道路向量
	//HDMapFC_RS = 11, //道路区间
	//HDMapFC_JUN = 12, //路口
	//HDMapFC_JG = 13, //路口组合
	//HDMapFC_RNW = 14, //道路接续网
	//HDMapFC_RCP = 15, //道路控制点
	//HDMapFC_RBN = 16, //道路向量边界连接点

	//HDMapFC_LD = 21, //车道边线
	//HDMapFC_LDC = 22, //车道轮廓线
	//HDMapFC_LV = 23, //车道向量
	//HDMapFC_LN = 24, //车道节点
	//HDMapFC_RSIDE = 25, //道路外侧线
	//HDMapFC_RPPT = 26, //道路属性点
	//HDMapFC_LPPT = 27, //车道属性点
	//HDMapFC_LNW = 28, //车道拓扑网
	//HDMapFC_LS = 29, //车道区间
	//HDMapFC_LCP = 30, //车道控制点

	//HDMapFC_GOBJ = 60, // 地物基础表
	HDMapFC_PML = 61, //地面印刷线
	HDMapFC_PMF = 62, //地面印刷面
	HDMapFC_LDL = 63, //车道分割线
	HDMapFC_WTURNL = 64, //待转区引导线
	HDMapFC_VSPDL = 65, //纵向减速标线
	HDMapFC_STOPL = 66, //停止线
	HDMapFC_STOPLL = 67, //停止让行线
	HDMapFC_HSPD = 68, //横向减速标线
	HDMapFC_SPDL = 69, //减速让行线
	HDMapFC_RTEETH = 70, //道路衔接部
	HDMapFC_DIRARR = 71, //方向箭头
	HDMapFC_TEXT = 72, //文字面
	HDMapFC_BELT = 73, //导流带
	HDMapFC_PEDSTRAIN = 74, //人形横道
	HDMapFC_PARKSITE = 75, //停车区
	HDMapFC_EMEAREA = 76, //紧急停车区
	HDMapFC_PEDWARN = 77, //前方人形横道提示
	HDMapFC_SAFELAND = 78, //安全岛
	HDMapFC_UNPARKING = 79, //禁止停车区
	HDMapFC_SIGNB = 80, //交通看板
	HDMapFC_SIGNL = 81, //交通信号灯
	HDMapFC_POLE = 82, //杆
	HDMapFC_PILLARI = 83, //支撑柱
	HDMapFC_PILLARO = 84, //支撑柱
	HDMapFC_BARRIER = 85, //道路隔离护栏
	HDMapFC_CURBE = 86, //道路路缘石


	HDMapFC_RGEO = 1000, //道路级几何
	HDMapFC_LGEO = 1001, //车道级几何
	HDMapFC_OGEO = 1002, //地物级几何
	HDMapFC_ADAS = 1003 //ADAS
};

enum ObjectClassification
{
	OC_unkown = 0,
	OC_road = 1,
	OC_dotted_lane,
	OC_solid_lane,
	OC_crosswalk,
	OC_diversion_zone,
	OC_stop_line,
	OC_transverse_deceleration_lane,
	OC_crosswalk_warning,
	OC_arrows,
	OC_lane = 13,
	OC_curbstone,
	OC_guardrail,
	OC_pole,
	OC_t_sign = 90,
	OC_car,
	OC_big_car
};

class POSE
{
public:
	int id;
	Eigen::Vector3d p;
	Eigen::Vector4d q;
};


struct CamPose
{
	string img_na;
	int idx;
	RAW_INS ins;
	vector<double> camPose;
	float regist_probability;
	double res;
	bool regist_flg;

	Eigen::Vector4d ins_q;
	Eigen::Vector4d q;

	POSE pos;

	Eigen::Matrix3d r;
	Eigen::Vector3d t;
};


class Mark
{
public:
	string file_name;
	string mark_type;
	cv::Rect2i rect;
};

//class  Object3D
//{
//public:
//	Object3D() {};
//	~Object3D() {};
//
//	string img_na;
//	string obj_id;
//	int type;
//	vector<cv::Point3d> shape;
//};

typedef HDObject Object3D;
typedef vector<Object3D> Object3D_VEC;



class OptimizePara
{
public:
	int occ_major;
	int dop_major;
	int long_major;
	map<int, float> weights;

	OptimizePara()
	{
		occ_major = OC_lane;
		dop_major = OC_lane;
		long_major = OC_lane;

		weights.clear();
		weights.insert(make_pair(OC_lane, 1));
		weights.insert(make_pair(OC_pole, 1));
		weights.insert(make_pair(OC_t_sign, 1));
	}
};

typedef  ObjectClassification FeatureID;
class SpatialFeature
{
public:
	FeatureID fid;
	int occupancy_count_3d;
	float occupancy_ratio_3d;
	float normal_entropy;
	float local_similarity;
	float dop_3d;

	int occupancy_count_2d;
	float occupancy_ratio_2d;
	float dop_2d;

	SpatialFeature()
	{
		dop_2d = 1;
		occupancy_count_2d = 0;
		occupancy_ratio_2d = 0;
		occupancy_count_3d = 0;
		occupancy_ratio_3d = 0;

		dop_3d = 1;
		normal_entropy = 0;
		local_similarity = 0;
	}
};



typedef map<FeatureID, SpatialFeature> StatisticData_MAP;//key is feature_type,data_type
#endif