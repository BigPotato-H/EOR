#include "MultiAlign.h"
#include "DataIO.h"

#include "DataIOArgoverse.h"


#include <opencv2/flann.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef CV_CONTRIB
#include <opencv2/xfeatures2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#endif

#include <opencv2/highgui.hpp>

#include<Eigen/Core>
#include<Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "TestCalib.h"
#include "OptimizeWeighted.h"

#include "HNMath/CoordinateTransform.h"
#include "HNMath/TransRotation.h"
#include "HNMath/GeometricAlgorithm2.h"

#include "HNString/HNString.h"
#include "HNString/EncodeStr.h"
#include "glog/logging.h"
#include "DataManager/XmlConfig.h"
#include <io.h>
#include <direct.h>
#include <fstream>

#include "pose_graph_3d/pose_graph_3d.h"
#include"HNFile/File.h"

#include <memory>
#include <numeric>

//#include "3DReconstructor.h"
//#include "MutualInfo.h"
//#include "ImageSimilarity.h"

#include "CapabilityCalculator.h"
#include <iostream>
#include <sys/stat.h>
///////////for 3d/////////////
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
//////////end

cv::flann::Index mss_kdtree;
cv::Rect LowerRect = cv::Rect(0, 600, 1920, 450);
#define VALID_SCORE 90
#define _BEV 0

const string& DATA_ARG("ARG");



string getPostfix(RegMethod rm)
{
	string post_fix = "";
	switch (rm)
	{
	case RegMethod_NONE:
		post_fix = "-0none";
		break;
	case RegMethod_EOR:
		post_fix = "-1eor";
		break;
	case RegMethod_LMR:
		post_fix = "-2lmr";
		break;
	case RegMethod_MOR:
		post_fix = "-3mor";
		break;
	case RegMethod_NONE_GT:
		post_fix = "-5gt";
		break;
	default:
		break;
	}

	return post_fix;
}

float get_rand()
{
	//rand() / double(RAND_MAX) 0~1的浮点数
	return 2.0 * rand() / double(RAND_MAX) - 1.0;
}

void randomJitter(vector<double>& camPose)
{
	int sz = 6;
	vector<double> ratio_vec = { 0.5 ,0.5 ,0.5, 0.04 ,0.04 ,0.04 };
	for (int i = 0; i < sz; i++)
	{
		camPose[i] += get_rand() * ratio_vec[i];
	}
}

bool directoryExists(const std::string& directoryPath) {
	struct stat info;
	if (stat(directoryPath.c_str(), &info) != 0) {
		return false;
	}
	return (info.st_mode & S_IFDIR) != 0;
}

bool createDirectory(const std::string& directoryPath) {
	int status = mkdir(directoryPath.c_str());
	return status == 0;
}

void createDirectoryIfNotExists(const std::string& directoryPath) {
	if (!directoryExists(directoryPath)) {
		if (createDirectory(directoryPath)) {
			std::cout << "Directory created: " << directoryPath << std::endl;
		}
		else {
			std::cout << "Failed to create directory: " << directoryPath << std::endl;
		}
	}
	else {
		std::cout << "Directory already exists: " << directoryPath << std::endl;
	}
}
#ifdef CV_CONTRIB
//rb特征检测与匹配程序，详细的注释在上一篇博客中
void findFeatureMatches(const cv::Mat& img_1, const cv::Mat& img_2,
	vector<cv::KeyPoint>& keypoints_1, vector<cv::KeyPoint>& keypoints_2, vector<cv::DMatch>& matches)
{
	//--初始化
	cv::Mat descriptors_1, descriptors_2;
//	cv::Ptr<cv::ORB> orb = cv::ORB::create();
	cv::Ptr<cv::xfeatures2d::SIFT> orb = cv::xfeatures2d::SIFT::create();// SURF特征检测类，Ptr 智能指针
	//-- 第1步：检测Oriented FAST角点位置
	orb->detect(img_1, keypoints_1);
	orb->detect(img_2, keypoints_2);

	//-- 第2步：根据角点位置计算BRIEF描述子
	orb->compute(img_1, keypoints_1, descriptors_1);
	orb->compute(img_2, keypoints_2, descriptors_2);

	//--第3步：对两幅图像中的BRIEF描述子进行匹配，使用汉明距离
	//vector<cv::DMatch> ngmatches;
	//cv::BFMatcher matcher(cv::NORM_HAMMING);
	//matcher.match(descriptors_1, descriptors_2, ngmatches);
	cv::BFMatcher matcher(cv::NORM_L2); //Brute - Force 匹配，参数表示匹配的方式，默认NORM_L2(欧几里得) ，NORM_L1(绝对值的和)
#if 0
	matcher.match(descriptors_1, descriptors_2, ngmatches);
//--第4步：匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	//找出所有匹配之间的最小距离和最大距离，即最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < descriptors_1.rows; i++) {
		double dist = ngmatches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//printf("-- Max dist ：%f \n", max_dist);
	//printf("-- Min dist : %f \n", min_dist);

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (ngmatches[i].distance <= max(2 * min_dist, 30.0))
		{
			matches.push_back(ngmatches[i]);
		}
	}
#else
	float knn_match_ratio = 0.6;
	vector<vector<cv::DMatch>> ngmatches;
	matcher.knnMatch(descriptors_1, descriptors_2, ngmatches, 2);

	for (size_t i = 0; i < ngmatches.size(); i++)
	{
		if (ngmatches[i][0].distance < knn_match_ratio * ngmatches[i][1].distance)
		{
			const auto& m = ngmatches[i][0];

			matches.push_back(m);
		}
	}
#endif
#if 0
	cv::Mat img_match = img_1.clone();
	cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	cv::imshow("match", img_match);
	cv::waitKey(0);
#endif
}
#endif

//对极约束函数
void poseEstimation2d2d(
	const vector<cv::KeyPoint> keypoints_1,
	const vector<cv::KeyPoint> keypoints_2,
	const vector<cv::DMatch> matches,
	cv::Mat& R, cv::Mat& t
)
{
	
	// 相机内参， TUM Freiburg2
	cv::Mat K = CalibSpace::intrisicMat;
	float focal_length = CalibSpace::FX;
	cv::Point2d principal_point(CalibSpace::CX, CalibSpace::CY);  // 光心，ＴＵＭ　ｄａｔａｓｅｔ标定值
																  //-- 对齐匹配的点对，并用.pt转化为像素坐标。把匹配点转换为简单的Point2f形式，
	vector<cv::Point> points1;//格式转换，point2f是oepncvpoint的一种数据类型，f就是float浮点的意思
	vector<cv::Point> points2;

	for (int i = 0; i < (int)matches.size(); i++)
	{
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);//queryIdx第一个图像索引,对应查询图像的特征描述子索引
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);//trainIdx第二个图像索引，对应训练图像的特征描述子索引
	}
	//push_back该函数将一个新的元素加到vector的最后面，位置为当前最后一个元素的下一个元素
	//-- 计算基础矩阵
	cv::Mat fundamental_matrix;//定义基础矩阵F，数据类型为Mat
	fundamental_matrix = cv::findFundamentalMat(points1, points2, /*CV_FM_8POINT*/CV_RANSAC);//调用OpenCV提供的基础矩阵计算函数findFundamentalMat，按照八点法进行计算，并返回一个4×4的F矩阵
//	cout << "fundamental_matrix is" << endl << fundamental_matrix << endl;

	//-- 计算本质矩阵
	               // 焦距TＵＭｄａｔａｓｅｔ标定值
	cv::Mat essential_matrix;                   //定义了一个本质矩阵E
	essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
	//调用OpenCV提供的本质矩阵计算函数findEssentialMat计算本质矩阵essential_matrix。由于计算本质矩阵E时需要提供归一化平面坐标，因此需要将像素坐标转化成归一化平面坐标，需要提供相机内参cu、cv与f。
//	cout << "essential_matrix is " << endl << essential_matrix << endl;

	//--计算单应矩阵
	cv::Mat homography_matrix;
	homography_matrix = cv::findHomography(points1, points2, cv::RANSAC);
//	cout << "homography_matrix is " << endl << homography_matrix << endl;

	//-- 从本质矩阵中恢复旋转矩阵和平移信息
	//通过OpenCV提供的R、t计算函数recoverPose计算R和t。由于函数默认使用本质矩阵进行解算，因此需要传入E。
	cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
//	cout << "R is " << endl << R << endl;
//	cout << "t is " << endl << t << endl;


}

//像素坐标系p到相机坐标系x
cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
	return cv::Point2d
	(
		(p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
		(p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
	);
}


void poseEstimation3d3d(
	vector<cv::Point3f>& pts1,
	vector<cv::Point3f>& pts2,
	Eigen::Matrix3f& R, Eigen::Vector3f& t
)
{
	cv::Point3f p1 = accumulate(pts1.begin(), pts1.end(), cv::Point3f()) / int(pts1.size());
	cv::Point3f p2 = accumulate(pts2.begin(), pts2.end(), cv::Point3f()) / int(pts2.size());
	
	for_each(pts1.begin(), pts1.end(), [&](auto& pt) {
		pt = pt - p1;
	});
	for_each(pts2.begin(), pts2.end(), [&](auto& pt) {
		pt = pt - p2;
	});
	

	// 计算 q1*q2^T
	int N = min(pts1.size(), pts2.size());
	Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
	for (int i = 0; i < N; i++)
	{
		W += Eigen::Vector3f(pts1[i].x, pts1[i].y, pts1[i].z) * Eigen::Vector3f(pts2[i].x, pts2[i].y, pts2[i].z).transpose();
	}
	//cout << "W=" << W << endl;

	// 对 W 进行 SVD 分解
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f U = svd.matrixU();
	Eigen::Matrix3f V = svd.matrixV();
	//cout << "U=" << U << endl;
	//cout << "V=" << V << endl;

	// 计算 R, t
	Eigen::Matrix3f R12 = U * (V.transpose());
	Eigen::Vector3f t12 = Eigen::Vector3f(p1.x, p1.y, p1.z) - R12 * Eigen::Vector3f(p2.x, p2.y, p2.z);

	R = R12;
	t = t12;
}

string getCarID(string sta_img_name)
{
//	sta_img_name = "TWX_09B_20191026_1026_1kbd_optim";

	auto spos = sta_img_name.find('_');
	if (spos == string::npos)
	{
		return "";
	}

	string car_id = sta_img_name.substr(0, spos);
	if (car_id == "TWX")
	{
		string car_string = sta_img_name;
		transform(car_string.begin(), car_string.end(), car_string.begin(), tolower);
		HNString::ReplaceA(car_string, "twx_09b", "0");
		HNString::ReplaceA(car_string, "kbd_optim", "0");
		HNString::ReplaceA(car_string, "kbd", "0");
		auto find_a = find_if(car_string.begin(), car_string.end(), [](const auto& s) ->bool{
			return !isdigit(s) && s != '_'; 
		});
		if (find_a != car_string.end())
		{
			auto spos = find_if(string::reverse_iterator(find_a), car_string.rbegin(), [](const auto& s)->bool {
				return s == '_';
			});

			auto epos = find_if(find_a, car_string.end(), [](const auto& s)->bool {
				return s == '_';
			});

			if (spos != car_string.rbegin() && epos != car_string.end())
			{
				string twx_car = "";
				copy(spos.base(), epos, back_inserter(twx_car));
				transform(twx_car.begin(), twx_car.end(), twx_car.begin(), toupper);
				car_id = twx_car;
			}
		}
	}
	return car_id;
}

string getSupplier(int _curSupplier)
{
	string sup_str = "";
	
	return sup_str;
}

void initLog(const string& sta_na)
{
	//创建日志
	google::InitGoogleLogging("a");
	FLAGS_minloglevel = 0;

	FLAGS_log_dir = "Logs";
	FLAGS_alsologtostderr = 1;
	google::SetLogDestination(google::GLOG_INFO, (FLAGS_log_dir + "\\log_").c_str());
	google::SetLogDestination(google::GLOG_WARNING, (FLAGS_log_dir + "\\log_").c_str());
}


void getTypeOutColorMap(map<int, cv::Scalar>& type_color_ref)
{
	type_color_ref.insert(make_pair(OC_road, cv::Scalar(128, 64, 128)));//road
	type_color_ref.insert(make_pair(OC_lane, cv::Scalar(255, 0, 0)));//lane
	type_color_ref.insert(make_pair(OC_curbstone, cv::Scalar(255, 0, 255)));//curb

	type_color_ref.insert(make_pair(OC_pole, cv::Scalar(0, 255, 0)));//pole
	type_color_ref.insert(make_pair(OC_t_sign, cv::Scalar(220, 220, 0)));//traffic sign
	type_color_ref.insert(make_pair(OC_crosswalk, cv::Scalar(128, 0, 255)));
}


void getTypeOutSegColorMap(map<int, cv::Scalar>& type_color_ref)
{
#if 0
	type_color_ref.insert(make_pair(OC_road, cv::Scalar(128, 64, 128)));//road
	type_color_ref.insert(make_pair(OC_lane, cv::Scalar(0, 0, 255)));//lane
	type_color_ref.insert(make_pair(OC_curbstone, cv::Scalar(255, 0, 255)));//curb

	type_color_ref.insert(make_pair(OC_pole, cv::Scalar(255, 128, 128)));//pole
	type_color_ref.insert(make_pair(OC_t_sign, cv::Scalar(0, 128, 255)));//traffic sign
#else
	auto clr = cv::Scalar(0, 0, 255);
	type_color_ref.insert(make_pair(OC_road, clr));//road
	type_color_ref.insert(make_pair(OC_lane, clr));//lane
	type_color_ref.insert(make_pair(OC_curbstone, clr));//curb

	type_color_ref.insert(make_pair(OC_pole, clr));//pole
	type_color_ref.insert(make_pair(OC_t_sign, clr));//traffic sign
#endif
}

void getAlignType(set<ObjectClassification>& reg_oc_set)
{
	reg_oc_set.clear();
	reg_oc_set.insert(OC_lane);
	reg_oc_set.insert(OC_pole);
	//reg_oc_set.insert(OC_arrows);
	reg_oc_set.insert(OC_t_sign);

	/*reg_oc_set.insert(OC_curbstone);
	reg_oc_set.insert(OC_guardrail);
	reg_oc_set.insert(OC_diversion_zone);
	reg_oc_set.insert(OC_stop_line);
	reg_oc_set.insert(OC_transverse_deceleration_lane);
	reg_oc_set.insert(OC_crosswalk_warning);
	reg_oc_set.insert(OC_t_sign);
	reg_oc_set.insert(OC_car);
	reg_oc_set.insert(OC_big_car);
	reg_oc_set.insert(OC_road);*/
}

void modifyFeatureCategories(int method, set<ObjectClassification>& reg_oc_set)
{
	if (method == RegMethod_LMR)
	{
		reg_oc_set = { OC_lane };
	}
	else
	{
		reg_oc_set = { OC_lane, OC_pole, OC_t_sign };
	}
}

bool doRegist(const map<int, HDObject_VEC>& ego_hdobj_vec,
	const map<int, LINE_VEC>& lines_map,
	RegMethod reg_method,
	StatisticData_MAP sd_map,
	vector<double>& campos)
{
	bool no_image_ll = false;
	for_each(lines_map.begin(), lines_map.end(), [&](const auto& l) {
		no_image_ll |= l.second.size() > 0;
		});
	if (!no_image_ll)
	{
		LOG(INFO) <<("no image lines...");
		return 0;
	}

	set<ObjectClassification> reg_oc_set;
	modifyFeatureCategories(reg_method, reg_oc_set);


	vector<double> pose = campos;
	//
	//resdual = m_calib.iterateCrossEntropy(xyz_vec_map, lines_map, ins.pitch, pose);
	//OptimizeCeresWeighted::optimizePoleGeometry(poles, pose);

	map<int, HDObject_VEC> regist_ego_hdobj_vec = ego_hdobj_vec;
	for (auto itr_type = regist_ego_hdobj_vec.begin(); itr_type != regist_ego_hdobj_vec.end(); )
	{
		const auto& oc = itr_type->first;
		auto find_type = find_if(reg_oc_set.begin(), reg_oc_set.end(), [&](const auto& type)->bool {
			return oc == type;
			});
		if (find_type == reg_oc_set.end())
		{
			itr_type = regist_ego_hdobj_vec.erase(itr_type);
		}
		else
		{
			itr_type++;
		}
	}

	map<int, LINE_VEC> regist_lines_map = lines_map;
	for (auto itr_type = regist_lines_map.begin(); itr_type != regist_lines_map.end(); )
	{
		const auto& oc = itr_type->first;
		auto find_type = find_if(reg_oc_set.begin(), reg_oc_set.end(), [&](const auto& type)->bool {
			return oc == type;
			});
		if (find_type == reg_oc_set.end())
		{
			itr_type = regist_lines_map.erase(itr_type);
		}
		else
		{
			itr_type++;
		}
	}

	switch (reg_method)
	{
	case RegMethod_EOR:
		OptimizeCeresWeighted::optimizeEnhancedObjectRegistration(regist_lines_map, regist_ego_hdobj_vec,
			pose, sd_map);
		break;
	case RegMethod_LMR:
		OptimizeCeresWeighted::optimizeLaneMaringOnlyRegistration(regist_lines_map, regist_ego_hdobj_vec,
			pose);
		break;
	case RegMethod_MOR:
		OptimizeCeresWeighted::optimizeMultiObjectRegistration(regist_lines_map, regist_ego_hdobj_vec,
			pose);
		break;
	default:
		break;
	}

	campos = pose;
	return true;
}


void calImage3DPointCloud(const Point_VEC& ij_vec,
	pcl::PointCloud<pcl::PointXYZ>& src)
{
	src.clear();

	for_each(ij_vec.begin(), ij_vec.end(), [&](const auto& p) {
		Eigen::Vector3d p3f;
		if (OptimizeCeresWeighted::convertCameraPoint2dTo3d((double)p.x, (double)p.y, p3f))
		{
			src.push_back(pcl::PointXYZ(p3f[0], p3f[1], p3f[2]));
		}
		});
}

void write_txt(const string& file_na, const string& first_line, pcl::PointCloud<pcl::PointXYZ>& src)
{
	ofstream ofs(file_na);
	ofs << first_line << endl;
	for (const auto& p : src)
	{
		string str = to_string(p.x) + "," + to_string(p.y) + "," + to_string(p.z) + "," + to_string(255);
		ofs << str << endl;
	}
	ofs.close();
}

void calNearestPointPairs(pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
	const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud_mid,
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree, double& error)
{
	double err = 0.0;
	std::vector<int>indexs(source_cloud->size());

//	#pragma omp parallel for reduction(+:err) //采用openmmp加速
	for (int i = 0; i < source_cloud->size(); ++i)
	{
		std::vector<int>index(1);
		std::vector<float>distance(1);
		kdtree->nearestKSearch(source_cloud->points[i], 1, index, distance);
		err = err + sqrt(distance[0]);
		indexs[i] = index[0];
	}

	pcl::copyPointCloud(*target_cloud, indexs, *target_cloud_mid);
	error = err / source_cloud->size();
}

void iterateClosestPoint3d3d(const map<int, HDObject_VEC>& local_local_hdobj_vec,
	const map<int, LINE_VEC>& lines_map, vector<double>& camPose)
{
	double init_z = camPose[2];
	//get lane marking points
	auto find_itr = lines_map.find(OC_lane);
	if (find_itr == lines_map.end())
	{
		return;
	}
	auto line_vec = find_itr->second;
	Point_VEC ij_vec;
	for_each(line_vec.begin(), line_vec.end(), [&](const auto& line) {
		copy(line.begin(), line.end(), back_inserter(ij_vec));
		});

	// the ego frame and camera frame have different axises
	//before registration, the axises should be uniformed
	pcl::PointCloud<pcl::PointXYZ> hd_ego_points;
	auto find_lane = local_local_hdobj_vec.find(OC_lane);
	if (find_lane == local_local_hdobj_vec.end())
	{
		return;
	}
	const auto& lanes = find_lane->second;

	for_each(lanes.begin(), lanes.end(), [&](const auto& obj) {
		auto pts = obj.shape;
	//	equalizPolyline(pts, 0.2);
		for_each(pts.begin(), pts.end(), [&](const auto& p) {
			hd_ego_points.push_back(pcl::PointXYZ(p.x, p.y, p.z));
			});
		});

	////axises change
	//Eigen::Vector3d ta;
	//Eigen::Matrix3d ra;
	//getPosRT(camPose, ta, ra);
	//Eigen::Matrix4d A; // Transformation matrices
	//A.setIdentity();
	//A.block(0, 0, 3, 3) = ra;
	//A.block(0, 3, 3, 1) = ta;

	pcl::PointCloud<pcl::PointXYZ> camera_img_points;
	calImage3DPointCloud(ij_vec, camera_img_points);

	pcl::PointCloud<pcl::PointXYZ> src_points;
	pcl::PointCloud<pcl::PointXYZ> target_points;

	Eigen::Matrix4f R_AXIS;
	R_AXIS << 0, -1, 0, 0,
		0, 0, -1, 0,
		1, 0, 0, 0,
		0, 0, 0, 1;
#if 0
	pcl::transformPointCloud(hd_ego_points, target_points, R_AXIS);
#else
	pcl::copyPointCloud(hd_ego_points, target_points);
#endif
	pcl::copyPointCloud(camera_img_points, src_points);

	//	pcl::transformPointCloud(hd_ego_points, src_points, R_AXIS);
	//	pcl::copyPointCloud(camera_img_points, target_points);

	//	write_txt("icp_src_points.txt", "x,y,z,r", src_points);
	//	write_txt("icp_tag_points.txt", "x,y,z,g", target_points);

		/////////////////////////

		// Create an instance of the IterativeClosestPoint class
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	// Set the parameters for ICP
	icp.setMaxCorrespondenceDistance(3.0);
	icp.setMaximumIterations(10);
	icp.setTransformationEpsilon(1e-3);
	icp.setEuclideanFitnessEpsilon(1e-3);


	Eigen::Matrix3f R_12 = Eigen::Matrix3f::Identity();
	Eigen::Vector3f T_12 = Eigen::Vector3f::Zero();
	Eigen::Matrix4f H_12 = Eigen::Matrix4f::Identity();
	//Eigen::Matrix4f H_12 = A.inverse().cast<float>();


	//set image points 3d to be the target and建立kd树
	//transform the hd points to image points
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	kdtree->setInputCloud(target_points.makeShared());

	double error = INT_MAX, score = INT_MAX;
	double min_error = 0.5;
	double epsilon = 0.1;
	int max_iters = 10;
	Eigen::Matrix4f H_final = H_12;
	int iters = 0;

	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_mid(new pcl::PointCloud<pcl::PointXYZ>());


	//开始迭代，直到满足条件
	while (error > min_error && iters < max_iters)
	{
		iters++;
		double last_error = error;
		//变换到最新
		pcl::transformPointCloud(src_points, src_points, H_12);
		//		write_txt("icp_mid_points.txt", "x,y,z,g", src_points);

				//计算最邻近点对
		calNearestPointPairs(src_points.makeShared(), target_points.makeShared(), target_cloud_mid, kdtree, error);
#if 1
		//计算点云中心坐标
		Eigen::Vector4f source_centroid, target_centroid_mid;
		pcl::compute3DCentroid(src_points, source_centroid);
		pcl::compute3DCentroid(*target_cloud_mid, target_centroid_mid);

		//去中心化
		Eigen::MatrixXf souce_cloud_demean, target_cloud_demean;
		pcl::demeanPointCloud(src_points, source_centroid, souce_cloud_demean);
		pcl::demeanPointCloud(*target_cloud_mid, target_centroid_mid, target_cloud_demean);

		//计算W=q1*q2^T
		Eigen::Matrix3f W = (souce_cloud_demean * target_cloud_demean.transpose()).topLeftCorner(3, 3);

		//SVD分解得到新的旋转矩阵和平移矩阵
		Eigen::JacobiSVD<Eigen::Matrix3f> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3f U = svd.matrixU();
		Eigen::Matrix3f V = svd.matrixV();

		if (U.determinant() * V.determinant() < 0)
		{
			for (int x = 0; x < 3; ++x)
				V(x, 2) *= -1;
		}

		R_12 = V * U.transpose();
		T_12 = target_centroid_mid.head(3) - R_12 * source_centroid.head(3);
		H_12 << R_12, T_12, 0, 0, 0, 1;
		H_final = H_12 * H_final; //更新变换矩阵

#else
		// Set the source and target point clouds
		icp.setInputSource(src_points.makeShared());
		icp.setInputTarget(target_cloud_mid);

		// Perform ICP with closest point iteration
		pcl::PointCloud<pcl::PointXYZ> aligned_source;
		Eigen::Matrix4f initial_transform = Eigen::Matrix4f::Identity();
		//Eigen::Matrix4f initial_transform = A.cast<float>();

		icp.align(aligned_source, initial_transform);

		// Get the optimized transformation matrix
		H_12 = icp.getFinalTransformation();
		H_final = H_12 * H_final;
#endif
	}

	H_final = H_final.inverse();

	Eigen::Matrix3f rotationMatrix = H_final.block<3, 3>(0, 0);
	// Extract translation vector
	Eigen::Vector3f translationVector = H_final.block<3, 1>(0, 3);
	// Convert rotation matrix to Euler angles
	Eigen::Vector3f euler = rotationMatrix.eulerAngles(0, 1, 2);
	camPose[0] = translationVector[0];
	camPose[1] = translationVector[1];
	camPose[2] = translationVector[2];
	camPose[3] = euler[0];
	camPose[4] = euler[1];
	camPose[5] = euler[2];

	//camPose[2] = init_z;
	camPose[2] = 0;
	return;
}


MultiAlign::MultiAlign()
{
	m_io = nullptr;
	getTypeOutColorMap(m_type_color);
	getTypeOutSegColorMap(m_type_seg_color);
}

MultiAlign::~MultiAlign()
{
}

void MultiAlign::creatFolder(const string& folder_path)
{
	string label_folder = folder_path + "label/";
	string ipm_folder = folder_path + "ipm/";
	string ply_folder = folder_path + SUB_FOLDER_PLY;
	string proj_folder = folder_path + "proj/";
	string sgn_file = folder_path + "signboard.csv";

	if (_access(label_folder.c_str(), 0) != 0)
	{
//		_mkdir(label_folder.c_str());
	}
	if (_access(ipm_folder.c_str(), 0) != 0)
	{
//		_mkdir(ipm_folder.c_str());
	}
	if (_access(ply_folder.c_str(), 0) != 0)
	{
		_mkdir(ply_folder.c_str());
	}
	if (_access(proj_folder.c_str(), 0) != 0)
	{
//		_mkdir(proj_folder.c_str());
	}

	if (_access(sgn_file.c_str(), 0) == 0)
	{
		remove(sgn_file.c_str());
	}
}

void test()
{
	/////////////////test///////////////////////////////////////
#if 0
	Eigen::Vector3f t(0.039385227318343589, 1.8493538418634632, 1.0628206246832461);
	Eigen::Matrix3f rot;
	/*rot << 0.99962856717097714, 0.017941197040605098, -0.020514413090539866,
			   0.018927414663347675, 0.084547718468097832, 0.99623964801447307,
			   0.019608178647207954, -0.99625789670676612, 0.084176734193895486,
			   0., 0., 0.;*/

	rot << 0.99881174595351840, 0.048647848367493106, -0.0029125580676849991, 
		-0.002694950329373550, 0.11480543964154222, 0.99338434066147552, 
		0.048660388284583675, -0.99219609849964918, 0.11480012514831139, 
		0., 0., 0.;
	Eigen::Vector3f rot_euler = rot.eulerAngles(0, 1, 2);
	rot_euler = rot_euler / M_PI * 180.0;
	auto a = rot_euler;
#endif

	//////////////////argoverse2/////////////////////01bb304d7bd835f8bbef7086b688e35e__Summer_2019
#if 0
	Eigen::Vector3f t(0.039385227318343589, 1.8493538418634632, 1.0628206246832461);
	Eigen::Matrix3f rot;

	t << 0.01169216, 1.43311425e+00, -1.63243801;
	rot << -0.00422135, -0.99999011, -0.00139741,
		2.84650700e-03, 1.38539718e-03, -9.99994989e-01,
		0.99998704, - 0.00422531,  0.00284063;

	Eigen::AngleAxisf a1(M_PI / 2.0, Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf a2(M_PI / 2.0, Eigen::Vector3f::UnitZ());
	Eigen::Matrix3f r_yz;
	r_yz = a2 * a1;
	rot = r_yz * rot;

	Eigen::Vector3f rot_euler = rot.eulerAngles(0, 1, 2);
	TransRotation::eigenEuler2RotationMatrix(rot_euler, rot);
	cout << rot << endl;

	rot_euler = rot_euler / M_PI * 180.0;
	cout << "-----------init euler:" << endl;
	cout << rot_euler << endl;
#endif

	//////////////////argoverse2/////////////////////05lBLQJs4ilyORCox6j9ndWAKZc31rs9__Autumn_2020
#if 0
	Eigen::Vector3f t;
	Eigen::Matrix3f rot;

	t << 0.00526012201267449, 1.38910362718449, -1.620563534437085;
	rot << 0.013183907298194436, 0.0021952433310273856, 0.9999106787583931,
		-0.9998995820826649, 0.0052265304379082445, 0.01317228645657409,
		-0.0051971472237219984, -0.9999839320140597, 0.0022639289820486352;

	Eigen::AngleAxisf a1(M_PI / 2.0, Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf a2(M_PI / 2.0, Eigen::Vector3f::UnitZ());
	Eigen::Matrix3f r_yz;
	r_yz = a2 * a1;
	rot = r_yz * rot;

	Eigen::Vector3f rot_euler = rot.eulerAngles(0, 1, 2);
	TransRotation::eigenEuler2RotationMatrix(rot_euler, rot);
	cout << rot << endl;

	rot_euler = rot_euler / M_PI * 180.0;
	cout << "-----------init euler:" << endl;
	cout << rot_euler << endl;
#endif


	//////////////////argoverse2/////////////////////01KuQJcTCSK5HQNK5QvqSXaiWOwdORtKk__Spring_2020
#if 0
	Eigen::Vector3f t;
	Eigen::Matrix3f rot;

	t << 0.00526012201267449, 1.38910362718449, -1.620563534437085;
	rot << 0.013183907298194436, 0.0021952433310273856, 0.9999106787583931,
		-0.9998995820826649, 0.0052265304379082445, 0.01317228645657409,
		-0.0051971472237219984, -0.9999839320140597, 0.0022639289820486352;

	Eigen::AngleAxisf a1(M_PI / 2.0, Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf a2(M_PI / 2.0, Eigen::Vector3f::UnitZ());
	Eigen::Matrix3f r_yz;
	r_yz = a2 * a1;
	rot = r_yz * rot;

	Eigen::Vector3f rot_euler = rot.eulerAngles(0, 1, 2);
	TransRotation::eigenEuler2RotationMatrix(rot_euler, rot);
	cout << rot << endl;

	rot_euler = rot_euler / M_PI * 180.0;
	cout << "-----------init euler:" << endl;
	cout << rot_euler << endl;
#endif
}

void undistortImages()
{
	if (MY_CONFIG.data_para.distort.size() == 5)
	{
		for (int i = 0; i < 5; i++)
		{
			CalibSpace::distCoeffs.at<double>(0, i) = MY_CONFIG.data_para.distort[i];
		}
	}

	if (CalibSpace::distCoeffs.at<double>(0, 0) == 0)
	{
		return;
	}
	vector<string> files;
//	HN_GENERAL::getAllFilesName(MY_CONFIG.img_path, ".PNG", files);
	HN_GENERAL::getAllFilesName(MY_CONFIG.img_path, ".JPG", files);
	for (const auto& file : files)
	{
		const auto& fna = MY_CONFIG.img_path + file;
		cv::Mat mat = cv::imread(fna);
		cv::Size image_size(mat.cols, mat.rows);
		//	cv::undistort(mat, dest, CalibSpace::intrisicMat, CalibSpace::distCoeffs);
		cv::Mat mapx = cv::Mat(image_size, CV_32FC1);
		cv::Mat mapy = cv::Mat(image_size, CV_32FC1);
		cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
		
		cv::initUndistortRectifyMap(CalibSpace::intrisicMat, CalibSpace::distCoeffs, R, CalibSpace::intrisicMat, image_size, CV_32FC1, mapx, mapy);
		cv::Mat dest = mat.clone();
		//另一种不需要转换矩阵的方式
		//undistort(imageSource,newimage,cameraMatrix,distCoeffs);
		cv::remap(mat, dest, mapx, mapy, cv::INTER_LINEAR);
		cv::imwrite(fna, dest);
	}
}

void MultiAlign::preprocess(int step, int method)
{
	test();

	//Eigen::Matrix<double, 3, 3> rotation_matrix;
	//vector<double> camPose = {0,0,0,2.8661089943370897,-0.02040965338001893,1.5536361797761182 };

	//OptimizeCeresWeighted::calcRotation(camPose, rotation_matrix);
	//cout << rotation_matrix;

	CalibSpace::band = 114;

	LOG(INFO) <<("initialize camera...");
	HN_GENERAL::read_xml_config();
	initCalibSpace();
	
#if 0
	m_io = new DataIO();
// 	string ply_path = "F:\\0hn\\4data\\whu_pc\\HD099_20181029_1\\objects";
// 	m_io->initDataPath(ply_path);
// 	m_io->initSQLPath(MY_CONFIG.sql[0]);
// 	m_io->importToDB(ply_path);

	//string ply_path = "F:\\0hn\\4data\\0image_match\\arg2\\objects\\";
	//string ply_path = "F:\\0hn\\4data\\0image_match\\arg1Ku\\objects\\";
	string ply_path = "F:\\0hn\\4data\\0image_match\\arg0c1\\objects\\";
	m_io->saveToCSV(ply_path);
	return;
#endif
	if (step==4)
	{
		return 	undistortImages();
	}
	//图像检测
// 	const string& folder_path = MY_CONFIG.data_path;
// 	const string& mid_folder_path = MY_CONFIG.mid_path;
// 	const string& detection_env = MY_CONFIG.detection_env;
// 	const string& gpu = MY_CONFIG.GPU_id;
// 
// 	creatFolder(mid_folder_path);
	//string json_path = ImageDetection::runDetection(mid_folder_path, detection_env, gpu);

	//m_io->readCamPosePara(mid_folder_path, m_pos_map);

	//string json_path = mid_folder_path + "marks.json";
	//multimap<string, Mark> marks_map;
	//m_io->readJson(json_path, marks_map);


	////for车端 test
	//MY_CONFIG.data_path = "D:\\0huinian\\data\\log\\20211013-24Hz\\";
	//MY_CONFIG.mid_path = "D:\\0huinian\\data\\log\\20211013-24Hz\\";
	//MY_CONFIG.img_path = "D:\\0huinian\\data\\log\\20211013-24Hz\\Image\\2\\";
	//MY_CONFIG.sql.push_back("hostaddr=172.16.100.113 port=9999 user=postgres password=pg$20200922 dbname=hdobj_hn");

	//MY_CONFIG.data_path = "D:\\0407limengqin\\";
	//MY_CONFIG.mid_path = "D:\\0407limengqin\\";
	//MY_CONFIG.img_path = "D:\\0407limengqin\\Image\\2\\";
//	MY_CONFIG.sql.push_back("hostaddr=172.16.100.113 port=9999 user=postgres password=pg$20200922 dbname=hdobj_hn");
	//////end

#ifdef DS_LOG
	//for 0407 only
//	m_io->renameImage(MY_CONFIG.img_path);
//	undistortImages();
#endif
	
	
	if (!readTrajecotry(m_ins_vec))
	{
		LOG(ERROR) << "无法读取%s..." << NA_TRACE_DB;
		return;
	}

	

	string para_tbl = CONFIG_PARA_TBL;
	bool keep_inside_intersection = false;//是否读取路口内的要素，重建精度分析需要
	if (step == 3)
	{
		para_tbl = CONFIG_PARA_RECON_TBL;		
		keep_inside_intersection = true;
	}
	
	readHDMap(keep_inside_intersection);
	//add 20230408 把方法配置放到这里来，不读配置了
	m_reg_method = RegMethod(method);
	SUB_FOLDER_PLY = "ply" + getPostfix(m_reg_method) + "\\";


	LOG(INFO) <<("regist method:[%d]", m_reg_method);

	if(!preprocess(m_ins_vec,m_hd_map,MY_CONFIG.data_path,MY_CONFIG.mid_path,CalibSpace::band, para_tbl))
	{
		return;
	}
}

void MultiAlign::process(int step)
{
	
}

bool readDataImageSize(const std::string & _dataPath, const std::string & _midPath, const string& img_name)
{
	//
	string img_path = "";

	if (CalibSpace::camera_type == CAMERA_MSS_WIDE)
	{
		img_path = _midPath + "/Image/2/";
	}
	else
	{
		img_path = _dataPath + "/Image/1/";
	}
	
	img_path = img_path + img_name + ".jpg";

	auto t = cv::imread(img_path);
	if (t.cols == 0 ||
		t.rows == 0) 
	{
		LOG(ERROR) <<("cannot read original image size: check images path...");
		return false;
	}

	CalibSpace::IMG_WIDTH = t.cols;
	CalibSpace::IMG_HEIGHT = t.rows;

	if (CalibSpace::camera_type == CAMERA_MSS_PANO)
	{
		CalibSpace::image_rect.width = 2048;
		CalibSpace::image_rect.height = 2048;

		int32_t wOff = (CalibSpace::IMG_WIDTH - CalibSpace::image_rect.width) / 2;
		wOff = wOff < 0 ? 0 : wOff;

		int32_t hOff = (CalibSpace::IMG_HEIGHT - CalibSpace::image_rect.height) / 2;
		hOff = hOff < 0 ? 0 : hOff;

		CalibSpace::image_rect.x = wOff;
		CalibSpace::image_rect.y = hOff;
	}
	else
	{
		CalibSpace::image_rect = cv::Rect(0, 0, CalibSpace::IMG_WIDTH, CalibSpace::IMG_HEIGHT);
	}

	LOG(INFO) <<("image size: %d, %d", CalibSpace::IMG_WIDTH, CalibSpace::IMG_HEIGHT);
	LOG(INFO) <<("image rect:%d, %d, %d, %d ",
		CalibSpace::image_rect.tl().x,
		CalibSpace::image_rect.tl().y, 
		CalibSpace::image_rect.width, 
		CalibSpace::image_rect.height);

	return true;
}


void relateInsTimeAndImageTime(vector<RAW_INS>& ins_vec)
{
	map<double, RAW_INS*> t_ins_map;
	for_each(ins_vec.begin(), ins_vec.end(), [&](RAW_INS& ins) {
		t_ins_map.insert(make_pair(ins.time, &ins));
	});
	vector<string> files;
	HN_GENERAL::getAllFilesName(MY_CONFIG.img_path, ".JPG", files);

	vector<RAW_INS> new_ins_vec;
	for (const auto& fna : files)
	{
		string fna_str = fna;
		HNString::ReplaceA(fna_str, ".JPG", "");
		double img_time = stod(fna_str) / 1000;
		auto high_itr = t_ins_map.upper_bound(img_time);
		auto low_itr = t_ins_map.lower_bound(img_time);

		map<double, RAW_INS*> diff_t_map;
		high_itr++;
		auto itr = low_itr;
		for (; itr != high_itr; itr++)
		{
			diff_t_map.insert(make_pair(abs(itr->first - img_time), itr->second));
		}
		if (diff_t_map.size() == 0)
		{
			continue;
		}
		auto& near_ins = diff_t_map.begin()->second;
		near_ins->name = "3K8E7_@" + fna_str;

		new_ins_vec.push_back(*near_ins);
	}

	ins_vec.swap(new_ins_vec);
}


void removeInvalidTracePoints(const string& _path, vector<RAW_INS>& ins_vec)
{
	ofstream of(_path + "ins.txt", ios::trunc);
	of << "x,y,z,name" << endl;

	auto itr_ins = ins_vec.begin();
	for (; itr_ins != ins_vec.end(); )
	{
		RAW_INS& ins = *itr_ins;
		getStationImageName(ins.sta_name, ins.name);
		string img_path = MY_CONFIG.img_path + ins.name + ".jpg";
		if (_access(img_path.c_str(), 0) != 0)
		{
			itr_ins = ins_vec.erase(itr_ins);
		}
		else
		{
			itr_ins++;
			string str = to_string(ins.lonlat.x) + "," + to_string(ins.lonlat.y) + "," + to_string(ins.lonlat.z) + "," + ins.name;
			of << str << endl;

		}
	}

	of.close();

}

bool MultiAlign::preprocess(vector<RAW_INS>& ins_vec, HDMapData& hdobj_data,
	const std::string & _dataPath,
	const std::string & _midPath,
	const int& band,
	const string& para_tbl_na,
	string car_id)
{
	MY_CONFIG.data_path = _dataPath + "/";
	MY_CONFIG.mid_path = _midPath + "/";
	creatFolder(MY_CONFIG.mid_path);

	if (ins_vec.size() == 0 ||
		hdobj_data.ld_vec.size() == 0)
	{
//		return;
	}


	string img_name = ins_vec[0].name;
	string sta_name = "";
	getStationImageName(sta_name, img_name);
	initLog(sta_name);

	//ins_vec.resize(10);

	//set
	m_hd_map.ld_vec.swap(hdobj_data.ld_vec);
	m_hd_map.hd_obj_vec.swap(hdobj_data.hd_obj_vec);

	CalibSpace::band = band;

	//	transformCoord(ins_vec);
	transformCoord(m_hd_map.ld_vec);
	transformCoord(m_hd_map.hd_obj_vec);

	//m_io->saveHDMap(MY_CONFIG.mid_path,m_hd_map.ld_vec);
	//m_io->saveHDMap(MY_CONFIG.mid_path,m_hd_map.hd_obj_vec);

	//用于识别对应的配置项

	LOG(INFO) <<("central band:%d", CalibSpace::band);
	LOG(INFO) <<("data path:");
	LOG(INFO) <<("%s", _dataPath.c_str());
	LOG(INFO) <<("mid path:");
	LOG(INFO) <<("%s", _midPath.c_str());
	LOG(INFO) <<("creat mid folder for registration...");
	LOG(INFO) <<("initialize config...");

	if (car_id == "")
	{
		car_id = getCarID(ins_vec[0].name);
	}
	
	initConfig(para_tbl_na, car_id);

	removeInvalidTracePoints(_midPath, ins_vec);
	if (ins_vec.size() == 0)
	{
		LOG(ERROR) <<("no valid ins points: check images path...");
		return false;
	}
	return readDataImageSize(_dataPath, _midPath, ins_vec[0].name);

	//string json_path = ImageDetection::runDetection(mid_folder_path, detection_env, gpu);

	//m_io->readCamPosePara(mid_folder_path, m_pos_map);

	//	string json_path = mid_folder_path + "marks.json";
	//	multimap<string, Mark> marks_map;
	//	m_io->readJson(json_path, marks_map);

}

#if 0
void MultiAlign::processBuildPosGraphConstraints()
{
	const string& folder_path = MY_CONFIG.mid_path;
	m_io->readCamPosePara(folder_path, m_pos_map);

	buildPosGraphConstraints();

}


void MultiAlign::processPoseGraphOptimize()
{
	const string& folder_path =  MY_CONFIG.mid_path;
	m_io->readCamPosePara(folder_path, m_pos_map);

	const string& opz_path = MY_CONFIG.mid_path + "pgo\\";
	_mkdir(opz_path.c_str());
#if 1

	PoseGraph3d::process(folder_path);

	map<int, POSE> poses_map;
	const string& file_path = "poses_optimized.txt";
	m_io->readPosesOptimized(file_path, poses_map);
#endif
	int sz = 0;
	auto itr_img = m_pos_map.begin();
	for (; itr_img != m_pos_map.end(); itr_img++, sz++)
	{
		string img_na = itr_img->first;
		
		if (img_na != "00000000-01-20190912112040075")
		{
			//	continue;
		}

		const auto& cp = itr_img->second;
		POSE pose = poses_map[sz];
// 		vector<double> camPose(6);
// 		recoverCurrentPoseVertex(pose, cp.ins, camPose);

		vector<vector<cv::Point>> contours;
		transformHDMapFromWorld2Image(m_hd_map.ld_vec, pose, contours);

		string ply_path = folder_path + SUB_FOLDER_PLY + img_na + ".jpg";
		cv::Mat t = cv::imread(ply_path);
		cv::polylines(t, contours, false, cv::Scalar(0, 128, 255), 2);

		ply_path = folder_path + "pgo/" + img_na + ".jpg";
		cv::imwrite(ply_path, t);
	}
}
#endif

bool MultiAlign::isInIntersection(CamPose& cp)
{
	const auto& junction_vec = m_hd_map.junction_vec;
	const auto& pt = cp.ins.point;
	cv::Point2f p2(pt.x, pt.y);
	auto check_junction_itr = find_if(junction_vec.begin(), junction_vec.end(), [&](const auto& _junc)->bool {
		if (_junc.rect_3d.contains(p2))
		{			
			vector<cv::Point2f> shape(_junc.shape.size());
			transform(_junc.shape.begin(), _junc.shape.end(), shape.begin(), [](const auto& pt)->cv::Point2f {
				return cv::Point2f(pt.x, pt.y);
				});
			return cv::pointPolygonTest(shape, p2, false) > 0;
		}
		else
		{
			return false;
		}
		});
	return check_junction_itr != junction_vec.end();
}

void  MultiAlign::kalmanFilterInitialize(const vector<double>& pose)
{
	m_kf = new FusionEKF(pose);
}

bool  MultiAlign::kalmanFilterEstimate(CamPose& cp)
{
	auto find_cp = m_pos_map.find(cp.img_na);
	if (find_cp == m_pos_map.end() ||
		find_cp == m_pos_map.begin())
	{
		return false;
	}
	find_cp--;
	auto& last_cp = find_cp->second;

	bool is_in_intersection = isInIntersection(cp);
	vector<double> kf_cp = m_kf->ProcessMeasurement(last_cp, is_in_intersection);
	/*Eigen::Matrix3d rr(Eigen::Quaterniond(Eigen::Vector4d(est[3], est[4], est[5], est[6])));
	Eigen::Vector3d euler = rr.eulerAngles(0, 1, 2);
	kf_cp[3] = euler[2];
	kf_cp[4] = euler[1];
	kf_cp[5] = euler[0];*/

#if 0
	const RAW_INS& ins = cp.ins;

	HDObject_VEC local_hdobj_vec;
	transformHDMapFromWorld2Ego(m_hd_map.ld_vec, ins, local_hdobj_vec);

	map<int, Point3f_VEC> xyz_vec_map;
	map<int, LINE_VEC> contours_map;
	transformEgo2Image(local_hdobj_vec, kf_cp, xyz_vec_map, contours_map);

	string ply_path = MY_CONFIG.mid_path + "/ipm/" + cp.img_na + ".png";
	cv::Mat t = cv::imread(ply_path);
	for_each(contours_map.begin(), contours_map.end(), [&](const auto& contours) {
		cv::polylines(t, contours.second, false, cv::Scalar(255, 128, 128), 2);
		});
	ply_path = MY_CONFIG.mid_path + "/kf/" + cp.img_na + ".jpg";
	cv::imwrite(ply_path, t);



	///////////////////////////original/////////////
	xyz_vec_map.clear();
	contours_map.clear();
	transformEgo2Image(local_hdobj_vec, cp.camPose, xyz_vec_map, contours_map);
	for_each(contours_map.begin(), contours_map.end(), [&](const auto& contours) {
		cv::polylines(t, contours.second, false, cv::Scalar(255, 255, 0), 3);
		});
	ply_path = MY_CONFIG.mid_path + "/kf/" + cp.img_na + ".jpg";
	cv::imwrite(ply_path, t);

	///////////////////////////
#endif

	cp.camPose.swap(kf_cp);

	return true;
}


void MultiAlign::processRelativePose()
{
	const string& folder_path =  MY_CONFIG.mid_path;
	m_io->readCamPosePara(folder_path, m_pos_map);

	/*for_each(m_pos_map.begin(), m_pos_map.end(), [](auto& _pair) {
		auto& cp = _pair.second;
		Eigen::Quaterniond q;
		Eigen::Quaterniond ins_q;
		TransRotation::eigenEulerAngle2Quaternion(cp.ins.heading, cp.ins.pitch, cp.ins.roll, ins_q);
		TransRotation::eigenEulerAngle2Quaternion(cp.camPose[3], cp.camPose[4], cp.camPose[5], q);
		cp.ins_q = ins_q.coeffs();
		cp.q = q.coeffs();
	});*/
	//vector<string> img_na_vec;
	//getImageVec(folder_path, img_na_vec);

	bool is_first_ok = false;
	int sz = 0;
	auto cp_itr = m_pos_map.begin();
	cp_itr++;
	for (; cp_itr != m_pos_map.end(); cp_itr++)
	{
		string img_na = cp_itr->first;
		if (img_na != "1634090985")
		{
	//		continue;
		}
		auto& cp = cp_itr->second;

		if (cp.regist_flg && cp.regist_probability >= VALID_SCORE)
		{
	//		continue;
		}
	
		auto last_cp_itr = cp_itr;

		last_cp_itr--;
		if (last_cp_itr == m_pos_map.end() ||
			!last_cp_itr->second.regist_flg ||
			last_cp_itr->second.regist_probability < VALID_SCORE)
		{
	//		continue;
		}
		const auto& last_cp = last_cp_itr->second;

		string file_path = MY_CONFIG.img_path + cp_itr->first + ".jpg";
		string last_file_path = MY_CONFIG.img_path + last_cp_itr->first + ".jpg";
		cv::Mat homography_matrix;
		cv::Mat img = cv::imread(last_file_path);
		cv::Mat img2 = cv::imread(file_path);

#if 0
		Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
		Eigen::Vector3f Trans = Eigen::Vector3f::Zero();
		//calcRelativePoseEdge2d2d(cp, last_cp, R, Trans);
		//Trans[2] = 5;
		//cout << cr << endl;

		map<int, LINE_VEC> contours_map;
		//transformHDMapFromWorld2Image2(m_hdobj_data.hdobj_ld_vec, last_cp, R, Trans, contours_map);
		//saveImage("opz\\", last_cp.img_na, img, cv::Scalar(0, 0, 0), contours_map);

		//transformHDMapFromWorld2Image2(m_hdobj_data.hdobj_ld_vec, cp, R, Trans, contours_map);
		//saveImage("opz\\", cp.img_na, img2, cv::Scalar(0, 0, 0), contours_map);
		

#else
		calcImage2Image(img, img2, homography_matrix);

		HDObject_VEC local_hdobj_vec;
		//transformHDMapFromWorld2Ego(m_hd_map.ld_vec, last_cp.ins, local_hdobj_vec);


		map<int, LINE_VEC> contours_map;
		//getEgoMapContours(local_hdobj_vec, last_cp.camPose, contours_map);

		//string ply_path = folder_path + "ply/" + img_na + ".jpg";
		//string ply_path_last = folder_path + "ply/" + last_cp.img_na + ".jpg";
		//cv::Mat t = cv::imread(ply_path);
		//cv::Mat t_last = cv::imread(ply_path_last);
		//cv::Mat t_last_cur = t_last * cr;
		//cv::addWeighted(t, 0.5, t_last_cur, 0.5, 0, t_last_cur);
		//cv::imwrite(ply_path, t_last_cur);

		string ply_path = MY_CONFIG.mid_path + SUB_FOLDER_PLY + cp_itr->first + ".jpg";
		cv::Mat ply_img = cv::imread(ply_path);
		for_each(contours_map.begin(), contours_map.end(), [&](auto& contours) {

		//	vector<vector<cv::Point>> cc;
			for_each(contours.second.begin(), contours.second.end(),[&](auto& line){

				vector<cv::Point2f> contf;
				for_each(line.begin(), line.end(), [&](const auto& p) {
					cv::Point2f pf = p;
					contf.push_back(pf);
				});

				cv::perspectiveTransform(contf, contf, homography_matrix);
				//for_each(line.begin(), line.end(), [&](const auto& p){
				//	cv::Point pf;
				//	//pf.x = p.x * cr.at<double>(0,0) + p.y * cr.at<double>(0, 1) + cr.at<double>(0, 2);
				//	//pf.y = p.x * cr.at<double>(1, 0) + p.y * cr.at<double>(1, 1) + cr.at<double>(1, 2);

				//	pf.x = p.x * cr.at<double>(0, 0) + p.y * cr.at<double>(1, 0) + cr.at<double>(2, 0);
				//	pf.y = p.x * cr.at<double>(0, 1) + p.y * cr.at<double>(1, 1) + cr.at<double>(2, 1);


				//	if (pf.x >= 0 && pf.x < CalibSpace::IMG_WIDTH &&
				//		pf.y >= 0 && pf.y < CalibSpace::IMG_HEIGHT)
				//	{
				//		contf.push_back(pf);
				//	}
				//});
				if (contf.size() > 0)
				{
					line.clear();
					for_each(contf.begin(), contf.end(), [&](const auto& pf) {
						cv::Point p = pf;
						line.push_back(pf);
					});
		//			cc.push_back(line);
				}
			});

	//		cv::polylines(ply_img, cc, false, cv::Scalar(0, 128, 255), 2);
		});
		saveImage("opz\\", img_na, ply_img, cv::Scalar(0, 128, 255), contours_map);
//		cv::imwrite(ply_path, ply_img);
#endif
	}
}
//void MultiAlign::manualCalib(const string& folder_path, const vector<RAW_INS>& ins_vec, double* camPose)
//{
//	double t0 = ins_vec.front().time;
//	double t_end = ins_vec.back().time;
//	float time_res = 1;
//	int idx = -1;
//	string img_na = "1617846647005897";
//	if (!matchImageWithLog(img_na, t0, t_end, time_res, idx))
//	{
//		return;
//	}
//	RAW_INS ins = ins_vec[idx];
//
//	vector<CalibSpace::Point3d2d> ref_pt_vec;
//	string file_path = folder_path + "refpoints.txt";
//	m_io->readRefPoints(file_path, ref_pt_vec);
//
//	cv::Vec3f r_euler(M_PI / 2, 0, ins.heading);
//	cv::Mat r;
////	TransRotation::eulerAngles2RotationMatrix(r_euler, r);
//	for_each(ref_pt_vec.begin(), ref_pt_vec.end(), [&](auto& p) {
//		cv::Point3d lp;
//		CalibSpace::TranslateAndRot(p.p3d, lp, ins.point, r);
//		p.p3d = lp;
//	});
//	ManualCalib mc;
//	mc.process(camPose, folder_path, ref_pt_vec);
//}

bool MultiAlign::matchImageWithLog(const string& img_na, double t0, double t_end, float time_res, int& idx)
{
	double img_ts = stoll(img_na)*1.0 / 1e6;

	if (img_ts < t0 - time_res ||
		img_ts > t_end + time_res)
	{
		return false;
	}
	float img_ts_shift = img_ts - t0;

	float d = 100;
	//预设knnSearch所需参数及容器
	int queryNum = 1;//用于设置返回邻近点的个数
	vector<float> vecQuery(1);//存放查询点的容器
	vector<int> vecIndex(queryNum);//存放返回的点索引
	vector<float> vecDist(queryNum);//存放距离
	cv::flann::SearchParams params(32);//设置knnSearch搜索参数S
	vecQuery[0] = img_ts_shift;
	mss_kdtree.knnSearch(vecQuery, vecIndex, vecDist, queryNum, params);

	d = sqrt(vecDist[0]);
	idx = vecIndex[0];
	// 1second
	if (d > time_res)
	{
		return false;
	}

	return true;
}

void MultiAlign::buildTimeIndex(cv::Mat& source, const vector<RAW_INS>& ins_vec)
{
	vector<double> time_vec;
	double t0 = ins_vec.front().time;
	for_each(ins_vec.begin(), ins_vec.end(), [&](const auto& ins) {
		time_vec.push_back(ins.time - t0);
	});

	source = cv::Mat(time_vec).reshape(1);
	source.convertTo(source, CV_32F);
	cv::flann::KDTreeIndexParams indexParams(1);
	mss_kdtree.build(source, indexParams);
}

int MultiAlign::getImageIndex(const string& img_na)
{
	int idx = -1;
	string t = img_na.substr(3, img_na.size() - 3);
	auto epos = t.find('_');
	if (epos == string::npos)
	{
		return idx;
	}
	string idx_str = t.substr(0, epos);
	idx = atoi(idx_str.c_str());
	return idx;
}

double getDiffAngle(const double& a)
{
	double b = a;
	if (a > M_PI / 6 * 5)
	{
		b = a - M_PI;
	}
	else if (a < -M_PI / 6 * 5)
	{
		b = a + M_PI;
	}
	return b;
}

Eigen::Vector3d calculateEulerAngleDifference(const Eigen::Vector3d& euler1, const Eigen::Vector3d& euler2)
{
	Eigen::Quaterniond q1 = Eigen::AngleAxisd(euler1(0), Eigen::Vector3d::UnitX())
		* Eigen::AngleAxisd(euler1(1), Eigen::Vector3d::UnitY())
		* Eigen::AngleAxisd(euler1(2), Eigen::Vector3d::UnitZ());
	Eigen::Quaterniond q2 = Eigen::AngleAxisd(euler2(0), Eigen::Vector3d::UnitX())
		* Eigen::AngleAxisd(euler2(1), Eigen::Vector3d::UnitY())
		* Eigen::AngleAxisd(euler2(2), Eigen::Vector3d::UnitZ());
	Eigen::Quaterniond q = q2.inverse() * q1;
	Eigen::Vector3d diff = q.toRotationMatrix().eulerAngles(0, 1, 2);

	return diff;
}


void getPosRT(const vector<double>& pos,
	Eigen::Vector3d& t,
	Eigen::Matrix3d& r)
{
	Eigen::Vector3d euler;
	for (int i = 0; i < 3; i++)
	{
		t[i] = pos[i];
	}
	for (int i = 0; i < 3; i++)
	{
		euler[i] = pos[i + 3];
	}
		
	TransRotation::eigenEuler2RotationMatrixd(euler, r);

//	cout << t << endl;
//	cout << euler << endl;
	euler = r.eulerAngles(0, 1, 2);
//	cout << euler << endl;

}

Eigen::Vector3d normalizeDiffEulerAngles(const Eigen::Vector3d& euler)
{
	Eigen::Vector3d normalizedEuler;

	for (int i = 0; i < 3; ++i)
	{
		normalizedEuler(i) = euler(i);

		if (abs(normalizedEuler(i) - M_PI) < 0.1 * M_PI)
		{
			normalizedEuler(i) -= M_PI;
		}
			
		if(abs(normalizedEuler(i) + M_PI) < 0.1 * M_PI)
		{
			normalizedEuler(i) += M_PI;
		}
	}

	return normalizedEuler;
}


void calSampleDiffPose(const vector<double>& sample, 
	const vector<double>& regist, 
	const vector<double>& gt,
	vector<double>& dif)
{
	Eigen::Matrix3d ra, rb, rg; // Your calculated rotation matrices
	Eigen::Vector3d ta, tb, tg; // Your calculated translation vectors

	getPosRT(sample, ta, ra);
	getPosRT(regist, tb, rb);
	getPosRT(gt, tg, rg);

	Eigen::Matrix4d A, B, E; // Transformation matrices

	// Convert ra and ta to transformation matrix A
	A.setIdentity();
	A.block(0, 0, 3, 3) = ra;
	A.block(0, 3, 3, 1) = ta;

	// Convert rb and tb to transformation matrix B
	B.setIdentity();
	B.block(0, 0, 3, 3) = rb;
	B.block(0, 3, 3, 1) = tb;

	// Convert rg and tg to transformation matrix E
	E.setIdentity();
	E.block(0, 0, 3, 3) = rg;
	E.block(0, 3, 3, 1) = tg;

	//Eigen::Matrix4d R_AXIS;
	//R_AXIS << 0, -1, 0, 0,
	//0, 0, -1, 0,
	//1, 0, 0, 0,
	//0, 0, 0, 1;
	//E = E * R_AXIS.inverse();


	// Calculate the differences
	//Eigen::Matrix4d diff = E.inverse() * A * B.inverse();
	Eigen::Matrix4d diff = E.inverse() * B * A;


	Eigen::Vector3d translation = diff.block<3, 1>(0, 3);

	// Extract Euler angles
	Eigen::Matrix3d rotationMatrix = diff.block<3, 3>(0, 0);
	Eigen::Vector3d euler = rotationMatrix.eulerAngles(0, 1, 2);

	euler = normalizeDiffEulerAngles(euler);
	//if (abs(euler[0] - M_PI) < 0.1 * M_PI ||
	//	abs(euler[0] + M_PI) < 0.1 * M_PI)
	//{
	//	euler = rotationMatrix.inverse().eulerAngles(0, 1, 2);
	//}

	dif.resize(6);
	dif[0] = translation[0];
	dif[1] = translation[1];
	dif[2] = translation[2];

	dif[3] = euler[0];
	dif[4] = euler[1];
	dif[5] = euler[2];
}

void calDiffPose(const vector<double>& gt, const vector<double>& regist,
	vector<double>& dif)
{
	Eigen::Matrix3d rb, rg; // Your calculated rotation matrices
	Eigen::Vector3d tb, tg; // Your calculated translation vectors

	getPosRT(regist, tb, rb);
	getPosRT(gt, tg, rg);

	Eigen::Matrix4d B, E; // Transformation matrices
	// Convert rb and tb to transformation matrix B
	B.setIdentity();
	B.block(0, 0, 3, 3) = rb;
	B.block(0, 3, 3, 1) = tb;

	// Convert rg and tg to transformation matrix E
	E.setIdentity();
	E.block(0, 0, 3, 3) = rg;
	E.block(0, 3, 3, 1) = tg;

	Eigen::Matrix4d diff = E * B.inverse();

	Eigen::Vector3d translation = diff.block<3, 1>(0, 3);

	// Extract Euler angles
	Eigen::Matrix3d rotationMatrix = diff.block<3, 3>(0, 0);
	
	Eigen::Vector3d euler = rotationMatrix.eulerAngles(0, 1, 2);
	euler = normalizeDiffEulerAngles(euler);
	//if (abs(euler[0] - M_PI) < 0.1 * M_PI ||
	//	abs(euler[0] + M_PI) < 0.1 * M_PI)
	//{
	//	euler = rotationMatrix.inverse().eulerAngles(0, 1, 2);
	//}
	dif.resize(6);
	dif[0] = translation[0];
	dif[1] = translation[1];
	dif[2] = translation[2];

	dif[3] = euler[0];
	dif[4] = euler[1];
	dif[5] = euler[2];

}

void calDiffDistanceAndAngle(const vector<double>& gt, const vector<double>& regist,
	vector<double>& dif	)
{
	Eigen::Matrix3d rb, rg; // Your calculated rotation matrices
	Eigen::Vector3d tb, tg; // Your calculated translation vectors

	getPosRT(regist, tb, rb);
	getPosRT(gt, tg, rg);

	Eigen::Matrix4d B, E; // Transformation matrices
	// Convert rb and tb to transformation matrix B
	B.setIdentity();
	B.block(0, 0, 3, 3) = rb;
	B.block(0, 3, 3, 1) = tb;

	// Convert rg and tg to transformation matrix E
	E.setIdentity();
	E.block(0, 0, 3, 3) = rg;
	E.block(0, 3, 3, 1) = tg;

	Eigen::Matrix4d diff = E.inverse() * B;

	Eigen::Vector3d translation = diff.block<3, 1>(0, 3);


	Eigen::Quaterniond q1(rb); // Convert R1 to quaternion
	Eigen::Quaterniond q2(rg); // Convert R2 to quaternion

	double angle = 2.0 * std::acos(std::abs(q1.dot(q2))); // Compute the angle between the quaternions
	double t = translation.norm();
	dif.push_back(t);
	dif.push_back(translation[0]);
	dif.push_back(translation[1]);
	dif.push_back(translation[2]);
	dif.push_back(angle);
}


void MultiAlign::jitterCamPose(map<string, CamPose>& pose_map)
{
	//如果存在就直接读
	string file_path = MY_CONFIG.mid_path + "jitter_pose.csv";
	if (_access(file_path.c_str(), 0) !=  0)
	{
		LOG(ERROR) << "无法读取jitter_pose.csv...";
		return;
	}
	m_io->readJitterCamPose(file_path, pose_map);
}

void MultiAlign::groudTruthCamPose(map<string, CamPose>& pose_map)
{
	//如果存在就直接读
	string file_path = MY_CONFIG.mid_path + "gt_pose.csv";
	if (_access(file_path.c_str(), 0) != 0)
	{
// 		ofstream os(MY_CONFIG.data_path + "gt_pose.csv", ios::trunc);
// 		os.close();

		const vector<RAW_INS>& ins_vec = m_ins_vec;
		vector<CamPose> pose_vec;

		for (const auto& ins : ins_vec)
		{
			CamPose cp;
			cp.img_na = ins.name;
			auto& campos = cp.camPose;
			updateCamPose(campos);

			
			{
				//arg的真值，
				//Eigen::Matrix3d 是column-major 
				//numpy 是row-major
				//<<输入默认是一行一行输入
				Eigen::Matrix3d r;
				//eigen的transpose不能赋值给自己
// 				r << -0.00402396249179951, -0.004602177956446962, 0.9999813136673714,
// 					-0.9999910700683226, 0.0013098367314269832, -0.004017973537547459,
// 					-0.0012913208261380316, -0.9999885520773697, -0.004607407593189922;
// 				Eigen::Matrix3d rt = r.transpose();
// 				Eigen::Vector3d t;
// 				t << 1.636028652685126, 0.005139524716952956, 1.434779494703929;
// 				t = -t;
// 				cp.r = rt;
// 				cp.t = rt * t;
				
				DataIOArgoverse* da = dynamic_cast<DataIOArgoverse*>(m_io);
				da->readEgo2Cam(cp.r, cp.t);                  
				//change ego  frame to camera frame
#if 1
				//角度都变成2pi以后优化很慢，gt就不变了，在后面局部需要的时候再变
				Eigen::Matrix4d E;
				E.setIdentity();
				E.block(0, 0, 3, 3) = cp.r;
				E.block(0, 3, 3, 1) = cp.t;

				Eigen::Matrix4d R_AXIS;
				R_AXIS << 0, -1, 0, 0,
					0, 0, -1, 0,
					1, 0, 0, 0,
					0, 0, 0, 1;
				E = E * R_AXIS.inverse();
				cp.r = E.block<3, 3>(0, 0);
				cp.t = E.block<3, 1>(0, 3);
#endif
				Eigen::Vector3d euler = cp.r.eulerAngles(0, 1, 2);

				cp.camPose[0] = cp.t[0];
				cp.camPose[1] = cp.t[1];
				cp.camPose[2] = cp.t[2];
				cp.camPose[3] = euler[0];
				cp.camPose[4] = euler[1];
				cp.camPose[5] = euler[2];

				/// 0c1 coorection
				if (MY_CONFIG.data_path.find("arg0c1") != string::npos)
				{
					campos[0] -= 0.09;
					campos[1] += 0.05;
					cp.camPose[2] -= 0.7;//////纵向真值感觉不对
				}

			}

			pose_vec.push_back(cp);
		}
		m_io->saveGTCamPose(file_path, pose_vec);

	}

	m_io->readGTCamPose(file_path, pose_map);
	
#if 0
	vector<CamPose> jitter_pose_vec;
	auto itr_cp = pose_map.begin();
	for (; itr_cp != pose_map.end(); itr_cp++)
	{
		auto jitter_cp = itr_cp->second;
		randomJitter(jitter_cp.camPose);
		jitter_pose_vec.push_back(jitter_cp);
	}
	string file_path_j = MY_CONFIG.mid_path + "jitter_pose.csv";
	m_io->saveJitterCamPose(file_path_j, jitter_pose_vec);
#endif
}

void MultiAlign::clearFiles()
{
	const string& mid_folder_path = MY_CONFIG.mid_path;

	string para_na = "reg" + getPostfix(m_reg_method) + ".csv";
	string para_na_diff = "reg" + getPostfix(m_reg_method) + "_diff.csv";

	const auto& para_file = mid_folder_path + para_na;
	if (_access(para_file.c_str(), 0) == 0)
	{
		ofstream os(para_file, ios::trunc);
		string head_line = "img_na,idx,iou,x,y,z,ax,ay,az";
		os << head_line << endl;
		os.close();
	}

	const auto& para_file_dif = mid_folder_path + para_na_diff;
	if (_access(para_file_dif.c_str(), 0) == 0)
	{
		ofstream os(para_file_dif, ios::trunc);
		string head_line = "img_na,idx,iou,x,y,z,ax,ay,az";
		os << head_line << endl;
		os.close();
	}

	const auto& sd_geo = mid_folder_path + "prob_distribution\\0_sd_geo.csv";
	if (_access(sd_geo.c_str(), 0) == 0)
	{
		ofstream os(sd_geo, ios::trunc);
		os.close();
	}



	/*const auto& sd_error_file = mid_folder_path + "spatial_features\\0_sd_error.csv";
	if (_access(sd_error_file.c_str(), 0) == 0)
	{
		ofstream os(sd_error_file, ios::trunc);
		string head_line = "img_na,idx,distance,lateral,vertical,longitudinal,angle,iou";
		os << head_line << endl;
		os.close();
	}

	string test_path = mid_folder_path + "0_sample_total.csv";
	if (_access(test_path.c_str(), 0) == 0)
	{
		ofstream os(test_path, ios::trunc);
		string head_line = "img_na,idx,se,x,y,z,ax,ay,az,iou";
		os << head_line << endl;
		os.close();
	}*/

	const auto& hdobj_file = mid_folder_path + "hdobj.csv";
	if (_access(hdobj_file.c_str(), 0) == 0)
	{
		remove(hdobj_file.c_str());
	}
}


void calcCameraHeightFromManualLanePoints(const string& data_path)
{
	
	/// <summary>
	/// ///arg2
	/// </summary>
	/// <param name="data_path"></param>
	Eigen::Vector3d p1(365, 1557, 1);
	Eigen::Vector3d p2(1381, 1527, 1);
	double w = 3.2;
	if (data_path.find("arg0c1") != string::npos)
	{
		p1= Eigen::Vector3d(349, 1429, 1);
		p2= Eigen::Vector3d(1110, 1429, 1);
		w = 3.2;

		
	}
	
	Eigen::Matrix3d K;
	K << CalibSpace::FX, 0, CalibSpace::CX,
		0, CalibSpace::FY, CalibSpace::CY,
		0, 0, 1;

	//camera frame
	// cp value  is the parameter coefficient of x, y in camera frame
	Eigen::Vector3d cp1 = K.inverse() * p1;
	Eigen::Vector3d cp2 = K.inverse() * p2;

	// set y fixed y1= y2
	// set lane width 3.5m to get camera height
	double alpha = cp2[0] / cp2[1] - cp1[0] / cp1[1];
	//double h = 3.5 / alpha - camera_height;
	double h = w / alpha;
	CalibSpace::camera_height = h;
}


bool getNearstShapePoint(const vector<cv::Point3d>& shape, const cv::Point3d& ins_point)
{
	if (shape.size() == 0)
	{
		return false;
	}

	map<double, cv::Point3d> dist_map;
	for_each(shape.begin(), shape.end(), [&](const auto& pt) {
		auto  p = pt - ins_point;
		double dis = p.ddot(p);
		dist_map[dis] = pt;
		});
	if (dist_map.size() == 0 || 
		dist_map.begin()->first >3.0)
	{
		return false;
	}
	cv::Point3d p = dist_map.begin()->second;
	CalibSpace::ego_height = ins_point.z - p.z;
	return true;
}

void MultiAlign::calcEgoHeightWithTrajectoriesAndLaneMarkings()
{
	const RAW_INS& ins = m_ins_vec[0];
	const auto& ins_point = ins.point;
	cv::Rect rect(ins_point.x - 10, ins_point.y - 10, 20, 20);
	vector<vector<cv::Point3d>> polyline_vec;
	auto itr_ld = m_hd_map.ld_vec.begin();
	for (; itr_ld != m_hd_map.ld_vec.end(); itr_ld++)
	{
		const auto& obj = *itr_ld;
		bool isIntersected = (obj.rect & rect).area() > 0;
		if (isIntersected)
		{
			polyline_vec.push_back(obj.shape);
		}
	}
	if (polyline_vec.size() == 0)
	{
		return;
	}

	map<double, cv::Point3d> dist_map;
	for (int i = 0; i < polyline_vec.size(); i++)
	{
		const auto& shape = polyline_vec[i];

		for_each(shape.begin(), shape.end(), [&](const auto& pt) {
			auto  p = pt - ins_point;
			double dis = p.ddot(p);
			dist_map[dis] = pt;
			});
	
	}
	
	if (	dist_map.begin()->first > 3.0)
	{
		return ;
	}
	
	cv::Point3d p = dist_map.begin()->second;
	CalibSpace::ego_height = ins_point.z - p.z;

}

void MultiAlign::processRegistHDAndMSSImages(int run_type/*0:default regist , 1:sample,2:sequence sdf*/)
{
	//add 20240218
	calcCameraHeightFromManualLanePoints(MY_CONFIG.data_path);
	calcEgoHeightWithTrajectoriesAndLaneMarkings();

	const vector<RAW_INS>& ins_vec = m_ins_vec;
	if (ins_vec.size() == 0)
	{
		return;
	}
	string para_na = "reg" + getPostfix(m_reg_method) + ".csv";
	string para_na_diff = "reg" + getPostfix(m_reg_method) + "_diff.csv";

	const string& folder_path = MY_CONFIG.data_path;
	const string& mid_folder_path = MY_CONFIG.mid_path;
	//序列配准
	if (run_type == 0)
	{
		clearFiles();
	}
	

	map<string, CamPose> gt_pose_map;
	groudTruthCamPose(gt_pose_map);

	map<string, HDSignBoard_VEC> img_sgn_vec_map;
	map<string, CamPose> init_pose_map;
	if (m_reg_method == RegMethod_NONE_GT)//ground truth
	{
		groudTruthCamPose(init_pose_map);
	}
	else 
	{
		jitterCamPose(init_pose_map);//抖动的初值
	}

	int total_sz = ins_vec.size();
	int sz = 0;
	int save_sz = 0;
	auto itr_ins = ins_vec.begin();
	for (; itr_ins != ins_vec.end(); itr_ins++, sz++)
//	for (; itr_ins != ins_vec.end(); itr_ins+= 10, sz += 10)
	{
		const RAW_INS& ins = *itr_ins;
		string img_na = itr_ins->name;
		string sta_na = "";
		getStationImageName(sta_na, img_na);
		string img_path = MY_CONFIG.img_path + img_na + ".jpg";
		if (_access(img_path.c_str(), 0) != 0)
		{
			continue;
		}


		if (img_na < "00000000-01-20181029120404740")
		{
		//	continue;
		}
		if (img_na < "1664268279")
		{
	//		continue;
		}
		if (img_na != "315973387899927214")
		{
		//	continue;
		}
		time_t stime = GetCurrentTime();
		LOG(INFO) <<("*****************************");
		string line = "";
		HNString::FormatA(line, "%d:...%s...%d/%d", sz, itr_ins->name.c_str(), (sz + 1), total_sz);
		LOG(INFO) << line;
		CamPose& cp = m_pos_map[img_na];
		cp.img_na = img_na;
		cp.ins = ins;
		cp.idx = sz;
		vector<double> campos(6);

		campos = init_pose_map[img_na].camPose;
		auto gt_campos = gt_pose_map[img_na].camPose;
		cp.camPose = campos;
		
		if (run_type == 1)//sample
		{
			// for random samples experiments comparing 
		//lane marking only and proposed method
			if (cp.img_na != "315967369899927220")
			{
				continue;
			}
			LOG(INFO) <<(".................random samples.................");
			analyzeRobustnessbyRandomSample(gt_campos, ins, cp.img_na);
			return;
		}
		

		if (run_type == 2)//.sequence spatial features
		{
			LOG(INFO) <<(".................sequence spatial features.................");
			//for calculation of sequence spatial features
			runSpatialFeatures(gt_campos, ins, cp);
			// 		if (sz > 5)
			// 		{
			// 			break;
			// 		}
			continue;
		}


		if (run_type == 1 ||
			run_type == 2)
		{
			continue;
		}

		if (!registHDImageLaneDividers(ins_vec, sz, cp))
		{
			//continue;
		}

		m_io->saveParas(mid_folder_path + para_na, cp);

		vector<double> dif_gt;
		auto save_delta_cp = cp;
		calDiffPose(gt_campos, cp.camPose, dif_gt);
		save_delta_cp.camPose = dif_gt;
		m_io->saveParas(mid_folder_path + para_na_diff, save_delta_cp);

		save_sz++;

		time_t etime = GetCurrentTime();
		LOG(INFO) <<("time: %.1f", (etime - stime) / 1000.0);

	}

	
	if (run_type == 2) //.sequence spatial features
	{
		string save_folder = MY_CONFIG.mid_path + "spatial_features\\";
		m_cc.saveSpatialDistributionDataCSV(save_folder);
	}
	

	LOG(INFO) <<(".................finish.................");
}


void MultiAlign::smooth()
{
	if (m_pos_map.size() < 2)
	{
		return;
	}

	auto itr_pos = m_pos_map.begin();
	auto itr_last = itr_pos;
	auto itr_next = itr_pos;
	for (; itr_pos != m_pos_map.end();itr_pos++)
	{
		//start
		itr_last--;
		itr_next++;
		if (itr_pos == m_pos_map.begin())
		{
			if (itr_pos->second.res > itr_next->second.res)
			{
				itr_pos->second.camPose = itr_next->second.camPose;
				continue;
			}
		}
		//end
		if (itr_next == m_pos_map.end())
		{
			if (itr_pos->second.res > itr_last->second.res)
			{
				itr_pos->second.camPose = itr_last->second.camPose;
				continue;
			}
		}
		//////
		if (itr_pos->second.res * 2 > itr_last->second.res + itr_next->second.res)
		{
			for (int i = 0; i< 5; i++)
			{
				itr_pos->second.camPose[i] = (itr_last->second.camPose[i] + itr_next->second.camPose[i]) * 0.5;
			}
			continue;
		}
	}
}


void MultiAlign::buildPosGraphConstraints()
{
	m_reg_oc_set.insert(OC_lane);

	const string& para_file = MY_CONFIG.data_path + "g2o";
	if (_access(para_file.c_str(), 0) == 0)
	{
		remove(para_file.c_str());
	}

	int sz = 0;
	int total_sz = m_pos_map.size();
	vector<int> indices;
	auto itr_img = m_pos_map.begin();
	for (; itr_img != m_pos_map.end(); itr_img++, sz++)
	{
		string img_na = itr_img->first;
		if (img_na < "1634090964")
		{
//			continue;
		}
		if (img_na > "1634090999")
		{
//			break;
		}
		auto& cp = itr_img->second;
		cp.idx = sz;

		LOG(INFO) <<("%d:...%s...%d/%d", sz, img_na.c_str(), (sz + 1), total_sz);

		saveG2OVertex(cp);
		if (itr_img != m_pos_map.begin())
		{
			auto last_itr_img = itr_img;
			last_itr_img--;

			bool be_edge = saveG2OEdge(last_itr_img->second, cp);
			/*if (cp.regist_flg && cp.regist_probability > VALID_SCORE &&be_edge)
			{
				indices.push_back(cp.idx);
			}*/

			
		}	
		indices.push_back(sz);
	}

	//ofstream of(MY_CONFIG.data_path + "idc", ios::trunc);
	//for_each(indices.begin(), indices.end(), [&](const auto& idx) {
	//	of << idx << endl;
	//});
	//of.close();
}

bool MultiAlign::isValid(const vector<double>&  cam, const vector<double>&  base, const vector<double>&  diff_threshold)
{
	for (int i = 0; i < 3; i++)
	{
		if (abs(cam[i] - base[i]) > diff_threshold[i])
		{
			return false;
		}
	}

	return true;
}

void MultiAlign::calcCurrentPoseVertex(const RAW_INS& ins, const vector<double>& camPose, Eigen::Vector3d& camera_in_world, Eigen::Vector4d& q)
{
	Eigen::Vector3d t = -Eigen::Vector3d(camPose[0], camPose[1], camPose[2]);
	Eigen::Vector3d euler(camPose[5], camPose[4], camPose[3]);//xyz
	Eigen::Matrix3d r;
	TransRotation::eigenEuler2RotationMatrixd(euler, r);

	Eigen::Vector3d T(ins.point.x, ins.point.y, ins.point.z);
	Eigen::Vector3d Euler(M_PI / 2 + ins.roll, ins.pitch, ins.heading);
	//cout << Euler << endl;
	Eigen::Matrix3d R;
	TransRotation::eigenEuler2RotationMatrixd(Euler, R);
	//cout << R << endl;
	//Eigen::Vector3f eee = R.eulerAngles(0, 1, 2);
	//cout << eee << endl;
	//Eigen::Matrix3f rrr;
	//cout << R << endl;

	Eigen::Vector3d camera2ego = r.inverse() * t;
	Eigen::Vector3d camera_in_ego = R.inverse() * camera2ego;

	for (int i = 0; i < 3; i++)
	{
		camera_in_world[i] = camera_in_ego[i] + T[i];
	}

//	Eigen::Matrix3d rr = R.inverse() * r;
//	Eigen::Matrix3d rr = R.inverse() * r.inverse();
	Eigen::Matrix3d rr = r * R;
	q = Eigen::Quaterniond(rr).coeffs();

	auto ee = Eigen::Matrix3d(Eigen::Quaterniond(q)).eulerAngles(0, 1, 2);
	ee = ee / M_PI * 180.0;
	return;
#if 0
	///////////
	Eigen::Vector3f ego_front(0, 0, 1);
	Eigen::Vector3f front_in_ego = R.inverse() * ego_front;
	Eigen::Vector3d ego_front_in_world;
	for (int i = 0; i < 3; i++)
	{
		ego_front_in_world[i] = front_in_ego[i] + T[i];
	}
	Eigen::Matrix3f front_to_camera = Eigen::Quaternionf::FromTwoVectors(front_in_ego, camera_in_ego).toRotationMatrix();
	Eigen::Vector3f front_to_camera_euler = front_to_camera.eulerAngles(0, 1, 2);
	front_to_camera_euler = front_to_camera_euler * 180.0 / M_PI;

	//Eigen::Vector3f ego_Euler = { (float)ins.roll, (float)ins.pitch, (float)ins.heading };
	//	Eigen::Vector3f ego_Euler = { -(float)ins.roll, -(float)ins.pitch, -(float)ins.heading };
	//Eigen::Matrix3f ego_R;
	//TransRotation::eigenEuler2RotationMatrix(ego_Euler, ego_R);
	//Eigen::Vector3f c = ego_R.eulerAngles(0, 1, 2);
	//c = c * 180.0 / M_PI;

	Eigen::Matrix3f rot = R * front_to_camera.inverse();
	//Eigen::Matrix3f rot2 = front_to_camera * R.inverse();

	//cout << rot1 << endl;
	//cout << rot2 << endl;
	//cout << rot1 * rot2 << endl;
	Eigen::Vector3f rot_euler = rot.eulerAngles(0, 1, 2);
// 	rot_euler = rot_euler * 180.0 / M_PI;

	//rot_euler[0] -= M_PI / 2;
//	rot_euler[2] -= 0.6;
	TransRotation::eigenEuler2RotationMatrix(rot_euler, rot);
	Eigen::Quaternionf qua(rot);
	q = qua.coeffs();
	//Eigen::Vector3d trans = camera_in_world;
	//float yaw, pitch, roll;
	//TransRotation::eigenQuaternion2EulerAngle(qua, yaw, pitch, roll);

	//Eigen::Matrix3f qr;
	//TransRotation::eigenEuler2RotationMatrix(Eigen::Vector3f(roll, pitch, yaw), qr);
	//cout << qr << endl;
#endif
}


void MultiAlign::recoverCurrentPoseVertex(const POSE& pose, const RAW_INS& ins, vector<double>& campose)
{
	Eigen::Vector3d T(ins.point.x, ins.point.y, ins.point.z);
	Eigen::Vector3d Euler(M_PI / 2 + ins.roll, ins.pitch, ins.heading);
	Eigen::Matrix3d R;
	TransRotation::eigenEuler2RotationMatrixd(Euler, R);

	Eigen::Vector3d camera_in_ego = pose.p - T;
	Eigen::Vector3d camera2ego = R * camera_in_ego;

	Eigen::Matrix3d rr(Eigen::Quaterniond(pose.q));
	Eigen::Matrix3d r = R * rr;
	Eigen::Vector3d t = r * camera2ego;

	campose[0] = -t[0];
	campose[1] = -t[1];
	campose[2] = -t[2];
	auto euler = r.eulerAngles(0, 1, 2);
	campose[3] = euler[2];
	campose[4] = euler[1];
	campose[5] = euler[0];

}

bool MultiAlign::calcRelativePoseEdge2d2d(const CamPose& cp1, const CamPose& cp2, Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
	//c1->c2
	/*if (cp1.regist_probability < VALID_SCORE ||
		cp2.regist_probability < VALID_SCORE)
	{
		return false;
	}*/

	string img_file = MY_CONFIG.img_path + cp1.img_na + ".jpg";
	string next_img_file = MY_CONFIG.img_path + cp2.img_na + ".jpg";

	cv::Mat img = cv::imread(img_file);
	cv::Mat img2 = cv::imread(next_img_file);

	vector<cv::KeyPoint> keypoints_1, keypoints_2;
	vector<cv::DMatch> matches;
#ifdef CV_CONTRIB
	findFeatureMatches(img2, img, keypoints_2, keypoints_1, matches);
#endif
	if (matches.size() < 8)
	{
		return false;
	}
//	cout << "一共找到了" << matches.size() << "组匹配点" << endl;

	//--估计两张图像间的运动
	cv::Mat r12;
	cv::Mat t12;
	poseEstimation2d2d(keypoints_2, keypoints_1, matches, r12, t12);

	//--验证 E=t^R*scale
	//--计算反对称矩阵
	//cv::Mat t_x = (cv::Mat_<double>(3, 3) <<
	//	0, -t.at<double>(2, 0), t.at<double>(1, 0),
	//	t.at<double>(2, 0), 0, -t.at<double>(0, 0),
	//	-t.at<double>(1, 0), t.at<double>(0, 0), 0);
	//cout << "t^R=" << endl << t_x*R << endl;
	////-- 验证对极约束
	//cv::Mat K = CalibSpace::intrisicMat;
	//for (cv::DMatch m : matches) {
	//	cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
	//	cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
	//	cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
	//	cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
	//	cv::Mat d = y2.t() * t_x * R * y1;
	//	cout << "epipolar constraint = " << d << endl;
	//}

	//c2->c1->ego->world
	Eigen::Vector3f et12;
	cv::cv2eigen(t12, et12);

	Eigen::Matrix3f er12;
	cv::cv2eigen(r12, er12);

	//cout << t12 << endl;
	//cout << et12 << endl;

	//cout << r12 << endl;
	//cout << er12 << endl;

	t = et12;

//	R = er12;
	Eigen::Vector3f euler = er12.eulerAngles(2, 0, 1);
//	euler = euler / M_PI * 180.0;
	euler[2] = -euler[2];
	TransRotation::eigenEuler2RotationMatrix(euler, R);
	return true;
}

bool MultiAlign::calcRelativePoseEdge3d2d(const CamPose& cp1, const CamPose& cp2, Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
	//c1->c2
	if (cp1.regist_probability < VALID_SCORE ||
		cp2.regist_probability < VALID_SCORE)
	{
//		return false;
	}
	//map<HDMapPointID, cv::Point3f> cam_pts_map1;
	//transformHDMapFromWorld2Camera(m_hdobj_data.hdobj_ld_vec, cp1.ins, cp1.camPose, cam_pts_map1);

	HDObject_VEC local_hdobj_vec;
	//transformHDMapFromWorld2Ego(m_hd_map.ld_vec, cp2.ins, local_hdobj_vec);

	map<int, Point3f_VEC> xyz_vec_map;
	transformEgo2Camera(local_hdobj_vec, cp2.camPose, xyz_vec_map);

	vector<double> campos(6, 0);
//	updateCamPose(campos);
	campos[0] = -1;
	campos[2] = -10.0;
	campos[3] = 0.4;

	const string& folder_path = MY_CONFIG.data_path;
	const string& mid_folder_path = MY_CONFIG.mid_path;

	map<int, LINE_VEC> lines_map;
	m_calib.extractImageDeepLearningMultiLines(mid_folder_path, cp1.img_na, lines_map);

//	findBestPose(xyz_vec_map, lines_map, campos);

	map<int, LINE_VEC> contours_map;
	for (int i = 0; i < local_hdobj_vec.size(); i++)
	{
		const int& type = local_hdobj_vec[i].type;
		const auto& obj = local_hdobj_vec[i].shape;
		vector<cv::Point> contour;
		int oc = convHDMapType2ObjectClassification(type);
		LINE_VEC& contours = contours_map[oc];

		for (int j = 0; j < obj.size(); j++)
		{
			const auto& lp = obj[j];
			double t[3];
			OptimizeCeresWeighted::convertPointByEigen(cp2.camPose, lp.x, lp.y, lp.z, t);
			double xyz[3];
			//xyz[0] = t[0];
			//xyz[1] = t[1];
			//xyz[2] = t[2];
			OptimizeCeresWeighted::convertPointByEigen(campos, t[0], t[1], t[2], xyz);

			double ij[2];
			if (OptimizeCeresWeighted::project2Image(xyz, ij, false))
			{
				contour.push_back(cv::Point(ij[0], ij[1]));
			}
		}
		if (contour.size() > 0)
		{
			LINE_VEC& contours = contours_map[oc];
			contours.push_back(contour);
		}
	}
	
	cv::Mat img = cv::imread(folder_path + SUB_FOLDER_PLY + cp1.img_na + ".jpg");
	saveImage("test\\", cp1.img_na, img, cv::Scalar(255, 255, 255), contours_map);
	//map<HDMapPointID, cv::Point3f> cam_pts_map2;
	//transformHDMapFromWorld2Camera(m_hdobj_data.hdobj_ld_vec, cp2.ins, cp2.camPose, cam_pts_map2);

	//vector<cv::Point3f> cam_3d_vec1;
	//vector<cv::Point3f> cam_3d_vec2;
	///*for_each(cam_pts_map1.begin(), cam_pts_map1.end(), [&](const auto& pt1) {
	//	auto find_ref = cam_pts_map2.find(pt1.first);
	//	if (find_ref != cam_pts_map2.end())
	//	{
	//		cam_3d_vec1.push_back(pt1.second);
	//		cam_3d_vec2.push_back(find_ref->second);
	//	}
	//});*/
	//poseEstimation3d3d(cam_3d_vec1, cam_3d_vec2, R, t);

	return true;
}

bool MultiAlign::calcImage2Image(const cv::Mat& img, const cv::Mat& img2, cv::Mat& homography_matrix)
{
	vector<cv::KeyPoint> keypoints_1, keypoints_2;
	vector<cv::DMatch> matches;
#ifdef CV_CONTRIB
	findFeatureMatches(img, img2, keypoints_1, keypoints_2, matches);
#endif
	if (matches.size() < 8)
	{
		return false;
	}
	//	cout << "一共找到了" << matches.size() << "组匹配点" << endl;

	//--估计两张图像间的运动
	// 相机内参， TUM Freiburg2
	cv::Mat K = CalibSpace::intrisicMat;
	float focal_length = CalibSpace::FX;
	cv::Point2d principal_point(CalibSpace::CX, CalibSpace::CY);  // 光心，ＴＵＭ　ｄａｔａｓｅｔ标定值
	//-- 对齐匹配的点对，并用.pt转化为像素坐标。把匹配点转换为简单的Point2f形式，
	vector<cv::Point> points1;//格式转换，point2f是oepncvpoint的一种数据类型，f就是float浮点的意思
	vector<cv::Point> points2;

	for (int i = 0; i < (int)matches.size(); i++)
	{
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);//queryIdx第一个图像索引,对应查询图像的特征描述子索引
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);//trainIdx第二个图像索引，对应训练图像的特征描述子索引
	}
	vector<int> status;
	homography_matrix = cv::findHomography(points1, points2, status, cv::RANSAC);

	vector<cv::Point> pts1;
	vector<cv::Point> pts2;
	for (int i =0; i < status.size(); i++)
	{
		if (status[i] == 0)
		{
			continue;
		}
		pts1.push_back(points1[i]);
		pts2.push_back(points2[i]);
	}

	homography_matrix = cv::findHomography(pts1, pts2, cv::RANSAC);

	//cv::Mat warp_img;
	//string ply_path_last = MY_CONFIG.data_path + "ply/" + img_na + ".jpg";
	//cv::Mat t_last = cv::imread(ply_path_last);

	//cv::warpPerspective(t_last, warp_img, m, t_last.size());
	//cv::addWeighted(img2, 0.5, warp_img, 0.5, 0, warp_img);
	//cv::imwrite("add_warp.jpg", warp_img);
	return true;
}
void MultiAlign::transformLocalRelative2WorldRelative(Eigen::Matrix3f& er12, Eigen::Vector3f& et12, const CamPose& cp1)
{
	const vector<double>& campose = cp1.camPose;
	const auto& ins = cp1.ins;

	Eigen::Matrix3f er21 = er12.inverse();
	Eigen::Vector3f et21 = -et12;

	Eigen::Vector3f t1 = -Eigen::Vector3f(campose[0], campose[1], campose[2]);
	Eigen::Vector3f euler1(campose[5], campose[4], campose[3]);//xyz
	Eigen::Matrix3f r1;
	TransRotation::eigenEuler2RotationMatrix(euler1, r1);

	Eigen::Vector3f Euler1(M_PI / 2 + ins.roll, ins.pitch, ins.heading);
	Eigen::Matrix3f R1;
	TransRotation::eigenEuler2RotationMatrix(Euler1, R1);


	Eigen::Vector3f cam12ego = r1.inverse() * t1;
	Eigen::Vector3f cam1_in_ego = R1.inverse() * cam12ego;

	Eigen::Vector3f cam22ego = r1.inverse() * (er21 * et21 + t1);
	Eigen::Vector3f cam2_in_ego = R1.inverse() * cam22ego;


	Eigen::Matrix3f cam1_to_cam2 = Eigen::Quaternionf::FromTwoVectors(cam1_in_ego, cam2_in_ego).toRotationMatrix();
	Eigen::Vector3f cam1_to_cam2_euler = cam1_to_cam2.eulerAngles(0, 1, 2);
	cam1_to_cam2_euler = cam1_to_cam2_euler * 180.0 / M_PI;

	Eigen::Matrix3f rot = R1 * cam1_to_cam2.inverse();
	Eigen::Vector3f rot_euler = rot.eulerAngles(0, 1, 2);
	rot_euler = rot_euler * 180.0 / M_PI;

	//for (int i = 0; i < 3; i++)
	//{
	//	et12[i] = cam2_in_ego[i] - cam1_in_ego[i];
	//}
	et12 = cam2_in_ego;
	et12 -=cam1_in_ego;
	er12 = cam1_to_cam2;
	return;
}

#if 1
void MultiAlign::saveG2OVertex(CamPose& cp)
{
	calcCurrentPoseVertex(cp.ins, cp.camPose, cp.pos.p, cp.pos.q);

	const string& folder_path =  MY_CONFIG.data_path;
	ofstream of(folder_path + "g2o", ios::out | ios::app);
	if (!of.is_open())
	{
		return;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);
	string line = "";
	HNString::FormatA(line, "VERTEX_SE3:QUAT %d %.2f %.2f %.2f %.4f %.4f %.4f %.4f",
		cp.idx,
		cp.pos.p[0],
		cp.pos.p[1],
		cp.pos.p[2],
		
		cp.pos.q[0],
		cp.pos.q[1],
		cp.pos.q[2],
		cp.pos.q[3]
		);
	of << line << endl;
	of.close();
}

bool MultiAlign::saveG2OEdge(const CamPose& cp1, const CamPose& cp2)
{
	auto pt = cp1.ins.point - cp2.ins.point;
	double dis = sqrt(pt.ddot(pt));
	if (dis > 30.0)
	{
		return false;
	}
	Eigen::Matrix3f R; 
	Eigen::Vector3f Trans;
	calcRelativePoseEdge2d2d(cp1, cp2, R, Trans);
	Trans[2] = dis;
	//if (!calcRelativePoseEdge3d2d(cp1, cp2, R, Trans))
	//{
	//	if (!calcRelativePoseEdge2d2d(cp1, cp2, R, Trans))
	//	{
	//		return false;
	//	}
	//}

#if 0
	Eigen::Vector3d pa;
	pa << 537456.97,3370050.65,17.09;
	Eigen::Vector4f qa;
//	qa << 0.6815,- 0.1404,0.1507,0.7022;
	qa << -0.6837, 0.2058, - 0.2190, 0.6650;
	Eigen::Vector3d pb;
	pb << 537457.47,3370051.38,17.08;
	Eigen::Vector4f qb;
// 	qb << 0.6825,- 0.1485,0.1569,0.6982;
	qb << -0.6833, 0.2070, - 0.2179, 0.6654;

 	Eigen::Quaternionf qa_inv = Eigen::Quaternionf(qa).conjugate();
 	Eigen::Vector3f a_euler = Eigen::Matrix3f(Eigen::Quaternionf(qa)).eulerAngles(0, 1, 2);
 	a_euler = a_euler * 180.0 / M_PI;
 
 	//Eigen::Vector3f inv_a_euler = qa_inv.eulerAngles(0, 1, 2);
 	//inv_a_euler = inv_a_euler * 180.0 / M_PI;
 
 	Eigen::Quaternionf eqab = qa_inv * Eigen::Quaternionf(qb);
 	cout << "r:..." << endl;
 
 	Eigen::Vector3f ab_euler = Eigen::Matrix3f(eqab).eulerAngles(0, 1, 2);
 	ab_euler = ab_euler * 180.0 / M_PI;
 
 	cout << Eigen::Matrix3f(eqab) << endl;
 	cout << R << endl;
 
 	Eigen::Vector3d epabd(pb - pa);
 	Eigen::Vector3f epab = qa_inv * Eigen::Vector3f((float)epabd[0], (float)epabd[1], (float)epabd[2]);
 	cout << "t:..." << endl;
 	cout << epab << endl;
 	cout << Trans << endl;
#endif

	Eigen::Quaternionf qf(R);
	Eigen::Vector4f Q = qf.coeffs();

	Eigen::Vector3f euler = R.eulerAngles(0, 1, 2);

#if 0
	///////////////////////////////////test  start//////////////////////////////////////////////////

	Eigen::Vector3d p_a(cp1.pos.p);
	Eigen::Quaterniond q_a(cp1.pos.q);

	Eigen::Vector3d p_b(cp2.pos.p);
	Eigen::Quaterniond q_b(cp2.pos.q);

	// Compute the relative transformation between the two frames.
	Eigen::Quaterniond q_a_inverse = q_a.conjugate();
	Eigen::Quaterniond q_ab_estimated = q_a_inverse * q_b;
	Eigen::Vector3d q_ab_estimated_euler = Eigen::Matrix3d(q_ab_estimated).eulerAngles(0, 1, 2);
	q_ab_estimated_euler = q_ab_estimated_euler * 180.0 / M_PI;

	// Represent the displacement between the two frames in the A frame.
	Eigen::Vector3d p_ab_estimated = q_a_inverse * (p_b - p_a);

	// Compute the error between the two orientation estimates.
	Eigen::Quaterniond delta_q =	qf.cast<double>() * q_ab_estimated.conjugate();
	Eigen::Vector3d delta_qeuler = Eigen::Matrix3d(delta_q).eulerAngles(0, 1, 2);
	delta_qeuler = delta_qeuler * 180.0 / M_PI;

	// Compute the residuals.
	// [ position         ]   [ delta_p          ]
	// [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
	Eigen::Matrix<double, 6, 1> residuals;
	residuals.template block<3, 1>(0, 0) = p_ab_estimated - Trans.cast<double>();
	residuals.template block<3, 1>(3, 0) = (2.0) * delta_q.vec();
	///////////////////////////////////test  end//////////////////////////////////////////////////
#endif
	//
	//Eigen::Matrix<double, 6, 6, Eigen::RowMajor> cov_pose = Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero();
	//OptimizeCeresWeighted::estimateCovariance(t, euler, cov_pose);

//	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> infomation = cov_pose.inverse();
	Eigen::Matrix<float, 6, 6, Eigen::RowMajor> infomation = Eigen::MatrixXf::Identity(6,6);
	const string& folder_path =  MY_CONFIG.data_path;
	ofstream of(folder_path + "g2o", ios::out | ios::app);
	if (!of.is_open())
	{
		return false;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);

	of << "EDGE_SE3:QUAT " << cp1.idx << " " << cp2.idx << " ";

	string str = "";
	for (int i = 0; i < 3; i++)
	{
		str += to_string(Trans[i]) + " ";
	}
	for (int i = 0; i < 4; i++)
	{
		str += to_string(Q[i]) + " ";
	}

	for (int i = 0; i < 6; ++i) {
		for (int j = i; j < 6; ++j) {
			str +=  to_string(infomation(i, j)) + " ";
		}
	}
	str.pop_back();
	of << str << endl;
	of.close();

	return true;
}
#endif

bool MultiAlign::readTrajecotry(vector<RAW_INS>& ins_vec)
{
	const auto& data_na = MY_CONFIG.data_name;
	LOG(INFO) <<("read data: [%s]...", data_na.c_str());

	 if (data_na == DATA_ARG)
	{
		m_io = new DataIOArgoverse;
		m_io->initDataPath(MY_CONFIG.data_path);
	}
	
	else
	{
	}

	if (m_io == nullptr)
	{
		return false;
	}

	m_io->getTracePoints(ins_vec);
	/////debug
//120.327,31.451 : 120.356,31.472
//m_trace_box.push_back(cv::Point2f(120.356, 31.472));
//m_trace_box.push_back(cv::Point2f(120.356, 31.451));
//m_trace_box.push_back(cv::Point2f(120.327, 31.451));
//m_trace_box.push_back(cv::Point2f(120.327, 31.472));
//////////
//m_io->getTraceBox(ins_vec, m_trace_box, 10.0);
	m_io->getTraceXYBox(ins_vec, m_trace_box, 10.0);
	if (m_trace_box.size() == 0)
	{
		//		return;
	}

	return ins_vec.size() > 0;
}

void MultiAlign::readHDMap(bool keep_inside_intersection)
{
	if (m_io == nullptr)
	{
		return;
	}

	const auto& data_na = MY_CONFIG.data_name;
	LOG(INFO) <<( "read data: [%s]...", data_na.c_str());
	 if (data_na == DATA_ARG)
	{
		m_io->getHDLaneDividerInBox(m_trace_box, m_hd_map.ld_vec, keep_inside_intersection);
		m_io->getHDObjectsInBox(m_trace_box, m_hd_map.hd_obj_vec);
		m_io->getHDJunctionInBox(m_trace_box, m_hd_map.junction_vec);

	}
	else
	{
		const string& folder_path = MY_CONFIG.mid_path;
		//	string connect_sql = "hostaddr=172.16.100.113 port=9999 user=postgres password=pg$20200922 dbname=hdobj_hn";
		const vector<string>& connect_sql_vec = MY_CONFIG.sql;
		for_each(connect_sql_vec.begin(), connect_sql_vec.end(), [&, this](const auto& connect_sql) {
			LOG(INFO) <<(connect_sql.c_str());
			m_io->initDataPath(connect_sql);
			//lane divider
			m_io->getHDLaneDividerInBox(m_trace_box, m_hd_map.ld_vec);
			//		m_io->saveHDMap(folder_path, m_hdobj_data.hdobj_ld_vec);
			LOG(INFO) <<("hdobj lane divier...%d", m_hd_map.ld_vec.size());

			//ref obejcts
		//	m_io->getHDMapObjectsInBox(connect_sql, m_trace_box, m_hdobj_data.hdobj_obj_vec);
		//	LOG(INFO) <<("hdobj ref object...%d", m_hdobj_data.hdobj_obj_vec.size());
		//	copy(m_hdobj_data.hdobj_obj_vec.begin(), m_hdobj_data.hdobj_obj_vec.end(), back_inserter(m_hdobj_data.hdobj_ld_vec));

			//signboard
		//	m_io->getHDMapSignBoardInBox(connect_sql, m_trace_box, m_hdobj_data.hdobj_isb_vec);
		//	LOG(INFO) <<("hdobj signboard...%d", m_hdobj_data.hdobj_isb_vec.size());
			});
	}
}

//read mask and thin
#if 0
bool MultiAlign::getLocalHDMap(const vector<RAW_INS>& ins_vec, const int& idx,  HDObject_VEC& ego_hdobj_vec)
{
	transformHDMapFromWorld2Ego(m_hd_map.ld_vec, ins_vec[idx], ego_hdobj_vec);
	transformHDMapFromWorld2Ego(m_hd_map.hd_obj_vec, ins_vec[idx], ego_hdobj_vec);

	vector<RAW_INS> local_ins_vec;
	transformInsFromWorld2Ego(ins_vec, idx, local_ins_vec);
//	removeByTraceRange(ego_hdobj_vec, local_ins_vec);

	if (ego_hdobj_vec.size() == 0)
	{
		LOG(ERROR) <<("local hdobj lane divider data empty...");
		return false;
	}
}
#else
bool MultiAlign::getLocalHDMap(const RAW_INS& ins,  map<int, HDObject_VEC>& ego_hdobj_vec)
{
	transformHDMapFromWorld2Ego(m_hd_map.ld_vec, ins, ego_hdobj_vec);
	transformHDMapFromWorld2Ego(m_hd_map.hd_obj_vec, ins, ego_hdobj_vec);

	if (ego_hdobj_vec.size() == 0)
	{
		LOG(ERROR) <<("local hdobj lane divider data empty...");
		return false;
	}
}
#endif


void sortValue(std::vector<double> vec, vector<int>& indices)
{
	// Create a vector of indices
	indices.resize(vec.size());
	std::iota(indices.begin(), indices.end(), 0);

	// Sort the vector and record the order using the indices
	std::sort(indices.begin(), indices.end(), [&](int a, int b) {
		return vec[a] > vec[b];
		});

	for (auto& i : indices)
	{
		if (i == 0)
		{
			i = OC_lane;
		}
		if (i == 1)
		{
			i = OC_pole;
		}
		if (i == 2)
		{
			i = OC_t_sign;
		}
	}

}
bool MultiAlign::calcEORWeight(const string& img_na, OptimizePara& op)
{
	string save_folder = MY_CONFIG.mid_path + "spatial_features\\";
	StatisticData_MAP sd_map;
	m_cc.getFrameSpatialDistribution(save_folder, img_na, sd_map);
	if (sd_map.size() == 0)
	{
		return false;
	}

	set<ObjectClassification> oc_set = { OC_lane ,OC_pole ,OC_t_sign };
	vector<vector<double>> v_vec(5, vector<double>());//occ,gdop,sim

	for (const auto& oc : oc_set)
	{
		const auto& sf = sd_map[oc];
		v_vec[0].push_back(sf.occupancy_ratio_3d);
	}

	for (const auto& oc : oc_set)
	{
		const auto& sf = sd_map[oc];
		v_vec[1].push_back(sf.occupancy_ratio_2d);
	}

	for (const auto& oc : oc_set)
	{
		const auto& sf = sd_map[oc];
		v_vec[2].push_back(1.0 / sf.dop_3d);
	}

	for (const auto& oc : oc_set)
	{
		const auto& sf = sd_map[oc];
		v_vec[3].push_back(1.0 / sf.dop_2d);
	}

	for (const auto& oc : oc_set)
	{
		const auto& sf = sd_map[oc];
		v_vec[4].push_back(sf.local_similarity);
	}

	map<int, vector<int>> oc_orders;
	for (int i = 0; i < 5; i++)
	{
		vector<double> values = v_vec[i];
		vector<int> indices;
		sortValue(values, indices);
		for (int idx = 0; idx < 3; idx++)
		{
			oc_orders[idx].push_back(indices[idx]);
		}

	}

	map<int, int> oc_weight_order;
	for (int idx = 0; idx < 3; idx++) 
	{
		auto values = oc_orders[idx];
		map<int, int> frequencyMap;
		for (const auto& v : values)
		{
			frequencyMap[v]++;
		}
		// Find the mode value(s)
		int maxFrequency = 0;
		std::vector<int> modeValues;
		for (const auto& pair : frequencyMap) {
			if (pair.second > maxFrequency) {
				maxFrequency = pair.second;
				modeValues = { pair.first };
			}
			else if (pair.second == maxFrequency) {
				modeValues.push_back(pair.first);
			}
		}

		oc_weight_order[idx] = modeValues[0];
	}

	auto& weights = op.weights;
	if (oc_weight_order[0] == OC_lane)
	{
		weights[OC_lane] = 1.0;
		weights[OC_pole] = sd_map[OC_pole].occupancy_ratio_3d / sd_map[OC_lane].occupancy_ratio_3d;
		weights[OC_t_sign] = sd_map[OC_t_sign].occupancy_ratio_3d / sd_map[OC_lane].occupancy_ratio_3d;
	}
	//else
	{
		weights[OC_lane] = 1.0;
		weights[OC_pole] = 1.0;
		weights[OC_t_sign] = 1.0;	
	}
	op.occ_major = oc_weight_order[0];
	return true;
}



bool MultiAlign::runSpatialFeatures(const vector<double>& gt_campos,
	const RAW_INS& ins,
	const CamPose& cp)
{
	map<int, HDObject_VEC> ego_hdobj_vec;
	getLocalHDMap(ins, ego_hdobj_vec);

	const string& folder_path = MY_CONFIG.data_path;
	const string& mid_folder_path = MY_CONFIG.mid_path;

	//*********************************2d********************************//
	vector<vector<cv::Point>> line_vec;
	cv::Mat camera_img = cv::imread(MY_CONFIG.img_path + cp.img_na + ".jpg");

	map<int, LINE_VEC> lines_map;
	m_calib.setCameraImage(camera_img);
	m_calib.extractImageDeepLearningMultiLines(mid_folder_path, cp.img_na, lines_map);

	//同一路段spatial distribution是一样的，不用反复跑
	CamPose gt_cp = cp;
	gt_cp.camPose = gt_campos;
	string save_folder = MY_CONFIG.mid_path + "spatial_features\\";
	m_cc.oneIteration(lines_map, ego_hdobj_vec, gt_cp, save_folder);
	return true;
}

bool MultiAlign::registHDImageLaneDividers(const vector<RAW_INS>& ins_vec, const int& idx, CamPose& cp)
{

	//记录当前帧的初始值
	const auto& init_frame_pos = cp.camPose;

	bool ipm_flg = false;
	if (m_hd_map.ld_vec.size() == 0)
	{
		//LOG(ERROR) << "hdobj lane divider data empty...");
		LOG(ERROR) <<("hdobj lane divider data empty...");
		return false;
	}
	const auto& ins = ins_vec[idx];
	map<int, HDObject_VEC> ego_hdobj_vec;
	getLocalHDMap(ins, ego_hdobj_vec);

	const string& folder_path =  MY_CONFIG.data_path;
	const string& mid_folder_path = MY_CONFIG.mid_path;

	//*********************************2d********************************//
	vector<vector<cv::Point>> line_vec;
	cv::Mat camera_img = cv::imread(MY_CONFIG.img_path + cp.img_na + ".jpg");

	map<int, LINE_VEC> lines_map;
	m_calib.setCameraImage(camera_img);
	m_calib.extractImageDeepLearningMultiLines(mid_folder_path, cp.img_na, lines_map);

	string ply_path = mid_folder_path + SUB_FOLDER_PLY + cp.img_na + ".jpg";
	cv::Mat t = camera_img.clone();
	saveImage(SUB_FOLDER_PLY, cp.img_na, t, m_type_seg_color, lines_map);
	//备份
	map<int, LINE_VEC> camera_lines_map = lines_map;
	vector<double> pos_init = cp.camPose;

	StatisticData_MAP sd_map;
	string save_folder = MY_CONFIG.mid_path + "spatial_features\\";
	m_cc.getFrameSpatialDistribution(save_folder, cp.img_na, sd_map);

	if (m_reg_method != RegMethod_NONE &&
		m_reg_method != RegMethod_NONE_GT)
	{

		//3d-3d
		//map<int, LINE_VEC> d3d3_contours_map;
		//getEgoMapContours(ego_hdobj_vec, pos_3d3d, d3d3_contours_map, false);
		if (m_pos_map.size() == 1)
		{
			vector<double> pos_3d3d = cp.camPose;
			iterateClosestPoint3d3d(ego_hdobj_vec, lines_map, pos_3d3d);

			kalmanFilterInitialize(pos_3d3d);
			cp.camPose = pos_3d3d;
		}

		//ekf init
		kalmanFilterEstimate(cp);
		pos_init = cp.camPose;

		//regist
		doRegist(ego_hdobj_vec, lines_map, m_reg_method, sd_map, cp.camPose);
	}
	

	map<int, LINE_VEC> contours_map;
	getEgoMapContours(ego_hdobj_vec, cp.camPose, contours_map,false);
	float eval = evaluteSimilarity(camera_lines_map, contours_map);

	map<int, LINE_VEC> init_contours_map;
	getEgoMapContours(ego_hdobj_vec, pos_init, init_contours_map, false);
	float init_val = evaluteSimilarity(camera_lines_map, init_contours_map);

	if (eval > init_val)
	{
		/*cp.camPose = pos_init;
		contours_map = init_contours_map;
		eval = init_val;*/
	}

	eval = 1 - eval;
	cp.regist_probability = eval;

	string eval_str = to_string(eval);
	eval_str = eval_str.substr(0, eval_str.find(".") + 3);
	eval_str = to_string(cp.idx) + " : " + eval_str;

	// ply image
	t = cv::imread(ply_path);

	cv::putText(t, eval_str, cv::Point(750, 40), 2, 2.0, cv::Scalar(255, 255, 255), 2);


	for_each(init_contours_map.begin(), init_contours_map.end(), [&](const auto& l) {
#if 0
		if (l.first == OC_t_sign)
		{
			cv::drawContours(t, l.second, -1, cv::Scalar(0, 0, 0), 3);
		}
		cv::polylines(t, l.second, false, cv::Scalar(0, 0, 0), 3);
#endif
		cv::polylines(t, contours_map[l.first], false, cv::Scalar(255, 0, 0), 3);
		});

	//saveImage(SUB_FOLDER_PLY, cp.img_na, t, m_type_color, contours_map);
	ply_path = MY_CONFIG.mid_path + SUB_FOLDER_PLY + cp.img_na + ".jpg";
	cv::imwrite(ply_path, t);

//	vector<double> dif_gt;
//	calDiffDistanceAndAngle(gt_campos, cp.camPose, dif_gt);
//	dif_gt.push_back(eval);
	
	return /*cp.regist_flg*/true;
}


float MultiAlign::evaluteMutualInfo(const map<int, LINE_VEC>& lines_map_rgb,
	const map<int, LINE_VEC>& lines_map_hdmap)
{
	cv::Mat blur_mat_rgb;
	m_calib.calcBlurImage(lines_map_rgb, blur_mat_rgb);

	cv::Mat blur_mat_hdmap;
	m_calib.calcBlurImage(lines_map_hdmap, blur_mat_hdmap);

//	MutualInfo mi;
//	float m = mi.calcMutualInformation(blur_mat_rgb, blur_mat_hdmap);

	return 0;
}

float MultiAlign::evaluteSimilarity(const map<int, LINE_VEC>& lines_map_rgb,
	const map<int, LINE_VEC>& lines_map_hdmap)
{
#if 0
	cv::Mat blur_mat_rgb = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
	auto itr = lines_map_rgb.begin();
	for (; itr != lines_map_rgb.end(); itr++)
	{
		const auto& type = itr->first;
		const auto& line_vec = itr->second;
		cv::polylines(blur_mat_rgb, line_vec, false, cv::Scalar(255), 20);
	}
// 	auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(14, 14));
// 	cv::dilate(blur_mat_rgb, blur_mat_rgb, kernel);
	
//	cv::imwrite("0_rgb.jpg", blur_mat_rgb);

	cv::Mat blur_mat_hdmap = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
	itr = lines_map_hdmap.begin();
	for (; itr != lines_map_hdmap.end(); itr++)
	{
		const auto& type = itr->first;
		const auto& line_vec = itr->second;
		cv::polylines(blur_mat_hdmap, line_vec, false, cv::Scalar(255), 20);
	}
// 	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
// 	cv::dilate(blur_mat_hdmap, blur_mat_hdmap, kernel);
	
//	cv::imwrite("0_hd.jpg", blur_mat_hdmap);

	cv::Mat dif;
	cv::bitwise_and(blur_mat_rgb, blur_mat_hdmap, dif);
//	cv::imwrite("0_and.jpg", dif);
	int cnt_and = cv::countNonZero(dif);

	cv::bitwise_or(blur_mat_rgb, blur_mat_hdmap, dif);
//	cv::imwrite("0_or.jpg", dif);
	int cnt_or = cv::countNonZero(dif);

	cv::absdiff(blur_mat_rgb, blur_mat_hdmap, dif);
//	cv::imwrite("0_diff.jpg", dif);
	int cnt_dif = cv::countNonZero(dif);

// 	ImageSimilarity mi;
// 	string hash1 = mi.pHashValue(blur_mat_rgb);
// 	string hash2 = mi.pHashValue(blur_mat_hdmap);
// 
// 	int m = mi.hanmingDist(hash1, hash2);
#endif

	int cnt_and = 0;
	int cnt_or = 0;
	int cnt_dif = 0;

	set<ObjectClassification> reg_oc_set;

	reg_oc_set.insert(OC_lane);
	reg_oc_set.insert(OC_pole);
	reg_oc_set.insert(OC_t_sign);
	//reg_oc_set.insert(OC_crosswalk);

	auto itr_type = reg_oc_set.begin();
	for (; itr_type != reg_oc_set.end(); itr_type++)
	{
		const auto& type = *itr_type;
		const auto& find_ply_rgb = lines_map_rgb.find(type);
		if (find_ply_rgb == lines_map_rgb.end())
		{
			continue;
		}
		const auto& find_ply_hd = lines_map_hdmap.find(type);
		if (find_ply_hd == lines_map_hdmap.end())
		{
			continue;
		}
		const auto& line_vec_rgb = find_ply_rgb->second;
		const auto& line_vec_hd = find_ply_hd->second;

		cv::Mat blur_mat_rgb = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
		cv::Mat blur_mat_hdmap = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));

		cv::polylines(blur_mat_rgb, line_vec_rgb, false, cv::Scalar(255), 30);
		cv::polylines(blur_mat_hdmap, line_vec_hd, false, cv::Scalar(255), 30);

		/*if (type == OC_crosswalk || type == OC_t_sign)
		{
			cv::drawContours(blur_mat_rgb, line_vec_rgb, -1, cv::Scalar(255), cv::FILLED);
			cv::drawContours(blur_mat_hdmap, line_vec_hd, -1, cv::Scalar(255), cv::FILLED);
		}*/
				

		cv::Mat dif;
		cv::bitwise_and(blur_mat_rgb, blur_mat_hdmap, dif);
		//	cv::imwrite("0_and.jpg", dif);
		cnt_and += cv::countNonZero(dif);

		cv::bitwise_or(blur_mat_rgb, blur_mat_hdmap, dif);
		//	cv::imwrite("0_or.jpg", dif);
		cnt_or += cv::countNonZero(dif);

		cv::absdiff(blur_mat_rgb, blur_mat_hdmap, dif);
		//	cv::imwrite("0_diff.jpg", dif);
		cnt_dif += cv::countNonZero(dif);
	}

	float ratio_dif = cnt_dif * 1.0 / cnt_or;
	float ratio_and = cnt_and * 1.0 / cnt_or;
	return ratio_dif;
}


bool MultiAlign::findMatchRect(multimap<string, Mark>::const_iterator low_img_rect, multimap<string, Mark>::const_iterator up_img_rect, const cv::Rect& sb_rect,cv::Rect& rect)
{
	map<float, cv::Rect> rect_map;
	auto low = low_img_rect;
	//
	vector<cv::Rect> detect_rect;
	vector<cv::Rect> temp_rect;
	for (low; low != up_img_rect; low++)
	{
		detect_rect.push_back(low->second.rect);
	}
	for (int i = 0; i < detect_rect.size(); i++)
	{
		for (int j = i+1; j < detect_rect.size(); j++)
		{
			float delta_h = abs(detect_rect[i].height - detect_rect[j].height);
			float delta_w = abs(detect_rect[i].width - detect_rect[j].width);
			auto intersection_rect = detect_rect[i] & detect_rect[j];
			float threshold_h = 0.2*detect_rect[i].height;
			float threshold_w = 0.2*detect_rect[i].width;
			float w = 0;
			float h = 0;
			detect_rect[i].x < detect_rect[j].x ? w = detect_rect[i].width : w = detect_rect[j].width;
			detect_rect[i].y < detect_rect[j].y ? h = detect_rect[i].height : h = detect_rect[j].height;
			float distance = pow((pow(detect_rect[i].x - detect_rect[j].x, 2) + pow(detect_rect[i].y - detect_rect[j].y, 2)), 0.5);
			if ((delta_h < threshold_h && abs(intersection_rect.height - detect_rect[i].height) < threshold_h)
				||(delta_w<threshold_w && abs(intersection_rect.width-detect_rect[i].width) < threshold_w)
				||(delta_h < threshold_h && abs(abs(detect_rect[i].x-detect_rect[j].x)-detect_rect[i].width)<5 && distance<1.2*w)
				|| (delta_h < threshold_h && abs(abs(detect_rect[i].x - detect_rect[j].x)- detect_rect[j].width)<5 &&distance<1.2*w)
				|| (delta_w<threshold_w && abs(abs(detect_rect[i].y - detect_rect[j].y)-detect_rect[i].height)<5 && distance<1.2*h)
				|| (delta_w<threshold_w && abs(abs(detect_rect[i].y - detect_rect[j].y)-detect_rect[i].height)<5)&& distance<1.2*h)
			{
				auto new_rect = detect_rect[i] | detect_rect[j];
				temp_rect.push_back(new_rect);
			}
		}
		
	}
	for (vector<cv::Rect>::iterator it = temp_rect.begin(); it != temp_rect.end(); it++)
	{
		detect_rect.push_back(*it);
	}
	//
	for (vector<cv::Rect>::iterator it = detect_rect.begin(); it != detect_rect.end(); it++)
	{
		cv::Rect intersect_rect = sb_rect & *it;
		if (intersect_rect.width > 0)
		{
			float intersection_area = intersect_rect.area();
			float union_area = sb_rect.area() + it->area() - intersection_area;
			float iou = intersection_area / union_area;
			rect_map.insert(make_pair(iou, *it));
		}
	}
	if (rect_map.size() == 0)
	{
		return false;
	}
	if (rect_map.size() >= 1)
	{
		rect = rect_map.rbegin()->second;
		return true;
	}
}

void MultiAlign::buildImageObject(const string& img_na, HDObject_VEC& isb_vec)
{
	const string& folder_path =  MY_CONFIG.data_path;
	const string& mid_folder_path = MY_CONFIG.mid_path;
	
	cv::Mat camera_img = cv::imread(MY_CONFIG.img_path + img_na + ".jpg");

	auto itr_isb = isb_vec.begin();
	for (; itr_isb != isb_vec.end(); itr_isb++)
	{
		auto& isb = *itr_isb;
		auto& shp = isb.ij_shape;
		if (shp.size() == 0)
		{
			continue;
		}
		/*	vector<vector<cv::Point>> plys(1);
			plys[0] = shp;
			cv::Rect signBoardRect = cv::boundingRect(plys[0]);
			if (signBoardRect.width > 10 * signBoardRect.height)
			{
					splitJumpSignBoard(plys);
			}

			cv::polylines(camera_img, plys, false, cv::Scalar(0, 0, 0), 2);*/

		cv::Rect sb_rect = cv::boundingRect(shp);
		
		isb.rect = sb_rect;
		string text1 = isb.obj_id;

		//string text1 = isb.obj_id + "_" + to_string(isb.majortype) + "_" + to_string(isb.subtype);
		//string text2 = isb.prop_id + "_" + to_string(isb.frame) + "_" + to_string(isb.kind);

		//cv::putText(camera_img, text1, cv::Point2i(isb.rect.x, isb.rect.y + isb.rect.height / 2), 2, 0.7, cv::Scalar(0, 0, 255));
		//cv::putText(camera_img, text2, cv::Point2i(isb.rect.x, isb.rect.y + isb.rect.height / 2 + 30), 2, 0.7, cv::Scalar(0, 0, 255));
		cv::Scalar clr = getHDMapObjectClassifyColor(isb.type);
		//cv::rectangle(camera_img, isb.rect, clr, 2);
		cv::polylines(camera_img, shp, false, clr,2);
	}

	string out_path = mid_folder_path + SUB_FOLDER_PLY + img_na + ".jpg";
	cv::imwrite(out_path, camera_img);

}


cv::Scalar MultiAlign::getSignBoardClassifyColor(const HDSignBoard& sgb)
{
	cv::Scalar clr = cv::Scalar(0, 255, 255);
	switch (sgb.majortype)
	{
	case 1:
		clr = cv::Scalar(0, 255, 255);	//警告标志 黄色
		break;
	case 2:
		clr = cv::Scalar(0, 0, 255);	//禁令标志 红色
		break;
	case 3:
		clr = cv::Scalar(255, 0, 0);	//指示标志 蓝色
		break;
	case 4:
		clr = cv::Scalar(0, 255, 0);	//指路标志 绿色
		break;
	case 5:
		clr = cv::Scalar(0, 125, 255);	//旅游区标志 棕色
		break;
	case 6:
		clr = cv::Scalar(0, 0, 0);	//辅助标志 黑色
		break;
	default:
		break;
	}

	return clr;
}

cv::Scalar MultiAlign::getHDMapObjectClassifyColor(const int& fcode)
{
	cv::Scalar clr = cv::Scalar(0, 0, 0);
	switch (fcode)
	{
	case 13:
		clr = cv::Scalar(255, 255, 255);	//车道线
		break;
	case 66:
		clr = cv::Scalar(0, 0, 255);	//停止线
		break;
	case 71:
		clr = cv::Scalar(0, 255, 0);	//方向箭头
		break;
	case 74:
		clr = cv::Scalar(100, 100, 100);	//人行横道
		break;
	case 81:
		clr = cv::Scalar(255, 0, 255);	//信号灯
		break;
	case 82:
		clr = cv::Scalar(20, 20,255);	//立杆
		break;
	case 85:
		clr = cv::Scalar(255,125 ,0);	//护栏
		break;
	default:
		break;
	}

	return clr;
}


void MultiAlign::getImageVec(const string& folder_path, vector<string>& img_na_vec)
{
	int sub_folder = 2;
	if (CalibSpace::camera_type == CAMERA_MSS_PANO)
	{
		sub_folder = 1;
	}
	string img_file_folder = folder_path + "image\\" + to_string(sub_folder) + "\\";
	
	HN_GENERAL::getAllFilesName(img_file_folder, "JPG", img_na_vec);
	//m_io->getFiles(img_file_folder, "*.jpg", img_na_vec);
	if (img_na_vec.size() == 0)
	{
		//m_io->getFiles(img_file_folder, "*.jpeg", img_na_vec);
		HN_GENERAL::getAllFilesName(img_file_folder, "JPEG", img_na_vec);
	}
	if (img_na_vec.size() == 0)
	{
		return;
	}

	sort(img_na_vec.begin(), img_na_vec.end(), [this](const string& a, const string& b) {
		string img_a = getFileName(a);
		int idx_a = getImageIndex(img_a);

		string img_b = getFileName(b);
		int idx_b = getImageIndex(img_b);

		return idx_a < idx_b;
	});
}



bool MultiAlign::initConfig(const string& para_tbl_na, const string& car_id)
{
	vector<vector<string>> value_vec;
#if 0
	if (m_io->readDBConfig(para_tbl_na, car_id, value_vec))
	{
		//暂时默认取第一条记录
		//后续考虑规则
		auto v = value_vec[0];
		initDBConfig(v);
	}
	else 
#endif
	{
		HN_GENERAL::read_xml_config();
	}

	return initCalibSpace();
}

bool MultiAlign::initDBConfig(const vector<string>& v)
{
	if (v.size() != 22)
	{
		return false;
	}

	int q = 0;
	string para_id = v[q++];
	MY_CONFIG.data_para.camera_type = v[q++];

	int lidartype = stoi(v[q++]);
	MY_CONFIG.data_para.intrinsic_para.resize(4);
	int w = 0;
	MY_CONFIG.data_para.intrinsic_para[w++] = stod(v[q++]);
	MY_CONFIG.data_para.intrinsic_para[w++] = stod(v[q++]);
	MY_CONFIG.data_para.intrinsic_para[w++] = stod(v[q++]);
	MY_CONFIG.data_para.intrinsic_para[w++] = stod(v[q++]);

	MY_CONFIG.data_para.distort.resize(5);
	w = 0;
	MY_CONFIG.data_para.distort[w++] = stod(v[q++]);
	MY_CONFIG.data_para.distort[w++] = stod(v[q++]);
	MY_CONFIG.data_para.distort[w++] = stod(v[q++]);
	MY_CONFIG.data_para.distort[w++] = stod(v[q++]);
	MY_CONFIG.data_para.distort[w++] = stod(v[q++]);

	MY_CONFIG.data_para.trans.x = stod(v[q++]);
	MY_CONFIG.data_para.trans.y = stod(v[q++]);
	MY_CONFIG.data_para.trans.z = stod(v[q++]);

	w = 0;
	MY_CONFIG.data_para.rotate[w++] = stod(v[q++]);
	MY_CONFIG.data_para.rotate[w++] = stod(v[q++]);
	MY_CONFIG.data_para.rotate[w++] = stod(v[q++]);

	//MY_CONFIG.data_para.image_height = stoi(v[q++]);
	//MY_CONFIG.data_para.image_width = stoi(v[q++]);

	//cv::Rect rect;
	//rect.height = stoi(v[q++]);
	//rect.width = stoi(v[q++]);
	//rect.x = stoi(v[q++]);
	//rect.y = stoi(v[q++]);

	//MY_CONFIG.data_para.image_rect = rect;

	//
	vector<cv::Point2f>& corners =  MY_CONFIG.data_para.corners;
	corners.resize(4);

	if (MY_CONFIG.data_para.camera_type == "MSSWide")
	{
		corners[0] = cv::Point2f(930, 625);
		corners[1] = cv::Point2f(989, 625);
		corners[2] = cv::Point2f(817, 790);
		corners[3] = cv::Point2f(1169, 792);
	}
	else
	{
		corners[0] = cv::Point2f(531, 431);
		corners[1] = cv::Point2f(677, 431);
		corners[2] = cv::Point2f(264, 667);
		corners[3] = cv::Point2f(869, 665);
	}


	string card_id = v[q++];
	string sup_str = v[q++];
	string sup_area = v[q++];
	
	LOG(INFO) <<("数据库配准参数初始化成功...");
	LOG(INFO) <<("ParaID:%s...", para_id.c_str());
	LOG(INFO) <<("相机类型:%s...", MY_CONFIG.data_para.camera_type.c_str());
	LOG(INFO) <<("车牌号:%s...", card_id.c_str());
	LOG(INFO) <<("供应商:%s...", sup_str.c_str());
	LOG(INFO) <<("采集区域:%s...", sup_area.c_str());
	return true;
}

bool MultiAlign::initCalibSpace()
{
	if ( MY_CONFIG.data_para.intrinsic_para.size() == 0)
	{
		//LOG(ERROR) << "can't initialize camera...");
		LOG(ERROR) <<("can't initialize camera...");
		return false;
	}
	
	CalibSpace::CX =  MY_CONFIG.data_para.intrinsic_para[0];
	CalibSpace::CY =  MY_CONFIG.data_para.intrinsic_para[1];
	CalibSpace::FX =  MY_CONFIG.data_para.intrinsic_para[2];
	CalibSpace::FY =  MY_CONFIG.data_para.intrinsic_para[3];

	if (MY_CONFIG.data_para.camera_type == "pano")
	{
		CalibSpace::camera_type = CAMERA_MSS_PANO;
//		m_diff_threshold = { 2, 0.5, 0.5, 0.05, 0.05, 0.05 };
	}
	else if (MY_CONFIG.data_para.camera_type == "wide")
	{
		CalibSpace::camera_type = CAMERA_MSS_WIDE;
	}
	else if (MY_CONFIG.data_para.camera_type == "polywide")
	{
		CalibSpace::camera_type = CAMERA_MSS_POLYWIDE;
	}

	if (CalibSpace::camera_type == CAMERA_MSS_WIDE)
	{
		MY_CONFIG.img_path = MY_CONFIG.mid_path + "Image/2/";
	}
	else
	{
		MY_CONFIG.img_path = MY_CONFIG.mid_path + "Image/1/";
	}

	m_calib.initCamera(CalibSpace::intrisicMat, CalibSpace::distCoeffs);

	if (MY_CONFIG.data_para.corners.size() > 0)
	{
		CalibSpace::initInversePerspectiveMappingMat(MY_CONFIG.data_para.corners,
			CalibSpace::warpmat_src2ipm,
			CalibSpace::warpmat_ipm2src);
	}

	CalibSpace::ego_height = MY_CONFIG.data_para.ego_height;
//	updateCamPose();


	if (MY_CONFIG.mid_path.find("arg0c1") != string::npos)
	{
		CalibSpace::data_item = "arg0c1";
	}
	else if (MY_CONFIG.mid_path.find("arg2") != string::npos)
	{
		CalibSpace::data_item = "arg2";
	}
	LOG(INFO) <<("initialize camera pose...");
	LOG(INFO) <<("%f", MY_CONFIG.data_para.trans.x);
	LOG(INFO) <<("%f", MY_CONFIG.data_para.trans.y);
	LOG(INFO) <<("%f", MY_CONFIG.data_para.trans.z);
	LOG(INFO) <<("%f", MY_CONFIG.data_para.rotate[0]);
	LOG(INFO) <<("%f", MY_CONFIG.data_para.rotate[1]);
	LOG(INFO) <<("%f", MY_CONFIG.data_para.rotate[2]);

	return true;
}


void MultiAlign::updateCamPose(vector<double>& campos)
{
	campos.resize(6);

	campos[0] =  MY_CONFIG.data_para.trans.x;
	campos[1] =  MY_CONFIG.data_para.trans.y;
	campos[2] =  MY_CONFIG.data_para.trans.z;

	/*yaw pitch roll*/
	//campos[3] = MY_CONFIG.data_para.rotate[0] / 180.0 * M_PI;  
	//campos[4] =  MY_CONFIG.data_para.rotate[1] / 180.0 * M_PI;
	//campos[5] =  MY_CONFIG.data_para.rotate[2] / 180.0 * M_PI;

	campos[3] = MY_CONFIG.data_para.rotate[0] / 180.0 * M_PI;
	campos[4] = MY_CONFIG.data_para.rotate[1] / 180.0 * M_PI;
	campos[5] = MY_CONFIG.data_para.rotate[2] / 180.0 * M_PI;

}

template<typename T>
void MultiAlign::getEgoMapContours(const map<int, vector<T>>& hdobj_vec, 
	const vector<double>& camPose,
map<int, LINE_VEC>& contours_map, bool ipm)
{
	auto itr = hdobj_vec.begin();
	for (; itr != hdobj_vec.end(); itr++)
	{
		const int& oc = itr->first;
		const auto& objs = itr->second;

		for_each(objs.begin(), objs.end(), [&](const auto& o) {
			const auto& obj = o.shape;

			vector<cv::Point> contour;
			for (int j = 0; j < obj.size(); j++)
			{
				const auto& lp = obj[j];
				cv::Point pp;
				double xyz[3] = { lp.x, lp.y, lp.z };
				double ij[2] = { -1,-1 };
				if (!OptimizeCeresWeighted::convertPoint3dTo2d(camPose, xyz, ij, ipm))
				{
					continue;
				}

				pp.x = ij[0];
				pp.y = ij[1];

				contour.push_back(pp);
			}

			if (contour.size() > 0)
			{
				LINE_VEC& contours = contours_map[oc];
				if (oc == OC_t_sign)
				{
					auto bb = cv::boundingRect(contour);
					cv::Point topLeft(bb.x, bb.y);
					cv::Point topRight(bb.x + bb.width, bb.y);
					cv::Point bottomLeft(bb.x, bb.y + bb.height);
					cv::Point bottomRight(bb.x + bb.width, bb.y + bb.height);
					vector<cv::Point> bb_ct = { topLeft, topRight, bottomRight, bottomLeft,topLeft };
					contour.swap(bb_ct);
				}

				contours.push_back(contour);
			}
			});
		
	}
}


ObjectClassification MultiAlign::convHDMapType2ObjectClassification(int type)
{
	ObjectClassification oc = OC_lane;
	switch (type)
	{
	case HDMapFC_POLE:
		oc = OC_pole;
		break;
	case HDMapFC_PEDSTRAIN:
		oc = OC_crosswalk;
		break;
	case HDMapFC_DIRARR:
		oc = OC_arrows;
		break;

	default:
		break;
	}

	return oc;
}

template<typename T>
void MultiAlign::getValid3dLines(const vector<T>& hdobj_vec, const RAW_INS& ins, const vector<double>& camPose,
	vector<vector<cv::Point3f>>& xyz_vec, vector<vector<cv::Point>>& contours)
{
	//cv::Mat rr;
	//TransRotation::eulerAngles2RotationMatrix(euler, rr);

	for (int i = 0; i < hdobj_vec.size(); i++)
	{
		const auto& obj = hdobj_vec[i].shape;
		vector<cv::Point> contour;
		vector<cv::Point3f> xyz_line;
		for (int j = 0; j < obj.size(); j++)
		{
			const auto& lp = obj[j];

			//cv::Point3d pca = lp;
			//CalibSpace::TranslateAndRot(lp, pca, tt, rr);
			////	CalibSpace::World2Ego(lp, pca, -tt, rr);

			cv::Point pp;
			//CalibSpace::Camera2Image(pca, pp);
			double xyz[3] = { lp.x, lp.y, lp.z };
			double ij[2];
			OptimizeCeresWeighted::convertPoint3dTo2d(camPose, xyz, ij);
			pp.x = ij[0];
			pp.y = ij[1];
			if (pp.x < 0 || pp.x > CalibSpace::IMG_WIDTH ||
				pp.y < 0 || pp.y > CalibSpace::IMG_HEIGHT)
			{
				continue;
			}
			contour.push_back(pp);
			xyz_line.push_back(lp);
		}

		if (contour.size() > 0)
		{
			contours.push_back(contour);
			xyz_vec.push_back(xyz_line);
		}
	}
}

template<typename T>
void MultiAlign::transformHDMapFromWorld2Ego(const vector<T>& hdobj_vec, const RAW_INS& ins, 
	map<int, vector<T>>& ego_hdobj_map)
{

		//for (int i = 0; i < hdobj_vec.size(); i++)
		//{
		//	const auto& obj = hdobj_vec[i];
		//	//add 根据杆底点先过滤
		//	if (obj.type == OC_pole)
		//	{
		//		const auto& ref_z = obj.shape[0].z;
		//		if (abs(ref_z - ins.point.z) > 3.0)
		//		{
		//			continue;
		//		}
		//	}
		//	//end
		//	int64 id = stoull(obj.obj_id);
		//	vector<cv::Point3d> reserv_shp;
		//	for (int j = 0; j < obj.shape.size(); j++)
		//	{
		//		const auto& p = obj.shape[j];
		//		//		cv::Point3d lp;
		//		Eigen::RowVector3d pp;
		//		pp << p.x, p.y, p.z;

		//		
		//		
		//		if (beyongImageRange(obj.type, lp))
		//		{
		//		//	continue;
		//		}
		//		reserv_shp.push_back(cv::Point3d(lp[0], lp[1], lp[2]));
		//	}

		//	if (reserv_shp.size() > 0)
		//	{
		//		T lobj = obj;
		//		lobj.shape.swap(reserv_shp);
		//		ego_hdobj_vec.push_back(lobj);
		//	}
		//}
	Eigen::Vector3f R_euler;
	Eigen::Matrix3f R_matrix;
	Eigen::Vector3d Trans;

	Eigen::Matrix3f R_AXIS;
	R_AXIS << 0, -1, 0, 
		0, 0, -1, 
		1, 0, 0;

	if (MY_CONFIG.data_name == "WHU")
	{
		R_euler << M_PI / 2 + ins.roll, ins.pitch, ins.heading;
		TransRotation::eigenEuler2RotationMatrix(R_euler, R_matrix);
		Trans << ins.point.x, ins.point.y, ins.point.z;
	}
	
	
	for (int i = 0; i < hdobj_vec.size(); i++)
	{
		const auto& obj = hdobj_vec[i];
		auto& ego_hdobj_vec = ego_hdobj_map[obj.type];
		//add 根据杆底点先过滤
		if (obj.type == OC_pole)
		{
			const auto& ref_z = obj.shape[0].z;
			if (abs(ref_z - ins.point.z) > 3.0)
			{
				continue;
			}
		}
		//end
		int64 id = stoull(obj.obj_id);
		vector<cv::Point3d> reserv_shp;
		for (int j = 0; j < obj.shape.size(); j++)
		{
			const auto& p = obj.shape[j];
	
			Eigen::Vector3f lp;
			if (MY_CONFIG.data_name == "ARG")
			{
				Eigen::RowVector3d pp;
				pp << p.x, p.y, p.z;
				pp = pp * ins.R + ins.T.transpose();
				lp << pp[0], pp[1], pp[2];

				//add 20240222
				//直接把坐标轴变换到camera坐标系一致，避免90度左右坐标轴的变换
				lp = R_AXIS * lp;
			}
			else if (MY_CONFIG.data_name == "WHU")
			{
				Eigen::Vector3d pp;
				pp << p.x, p.y, p.z;
				CalibSpace::EigenTranslateAndRot(pp, lp, Trans, R_matrix);
			}
			if (beyongImageRange(obj.type, lp))
			{
				continue;
			}
			
			reserv_shp.push_back(cv::Point3d(lp[0], lp[1], lp[2]));
		}

		if (reserv_shp.size() > 0)
		{
			T lobj = obj;
			lobj.shape.swap(reserv_shp);
			ego_hdobj_vec.push_back(lobj);
		}
	}

}
void MultiAlign::convInsFromWorld2Ego(const vector<RAW_INS>& ins_vec, const RAW_INS& ins,  vector<RAW_INS>& local_ins_vec)
{
	Eigen::Vector3f R_euler;
	R_euler << M_PI / 2 + ins.roll, ins.pitch, ins.heading;

	//if (CalibSpace::camera_type == CAMERA_MSS_WIDE)
	//{
	//	R_euler << M_PI / 2 + ins.roll, ins.pitch, ins.heading;
	//}
	//else if (CalibSpace::camera_type == CAMERA_MSS_PANO)
	//{
	//	R_euler << M_PI / 2 + ins.roll, ins.pitch, ins.heading;
	////	R_euler << ins.roll, ins.pitch, ins.heading - M_PI / 2;
	//}
	Eigen::Matrix3f R_matrix;
	TransRotation::eigenEuler2RotationMatrix(R_euler, R_matrix);

	Eigen::Vector3d Trans;
	Trans << ins.point.x, ins.point.y, ins.point.z;
	for (int i = 0; i < local_ins_vec.size(); i++)
	{
		auto& p = local_ins_vec[i].point;
		Eigen::Vector3d pp;
		pp << p.x, p.y, p.z;
		Eigen::Vector3f lp;
		CalibSpace::EigenTranslateAndRot(pp, lp, Trans, R_matrix);

		p = cv::Point3d(lp[0], lp[1], lp[2]);
	}
}

void MultiAlign::transformInsFromWorld2Ego(const vector<RAW_INS>& ins_vec, const int& idx, vector<RAW_INS>& local_ins_vec)
{
	double len = 100;
	collectLocalIns(ins_vec, idx, len, local_ins_vec);
	convInsFromWorld2Ego(ins_vec, ins_vec[idx], local_ins_vec);
}

double getVerticalValue(const Eigen::Vector3f& lp)
{
	return lp[2];
}

double getHorizonalValue(const Eigen::Vector3f& lp)
{
	return lp[0];
}

double getElevateValue(const Eigen::Vector3f& lp)
{
	double y = lp[1];
	if (y >= 0)
	{
		y = y - CalibSpace::ego_height;
	}
	else
	{
		y = -y + CalibSpace::ego_height;
	}
	return y;
	
}

bool MultiAlign::beyongImageRange(int type, Eigen::Vector3f lp)
{
#if 0
	if (MY_CONFIG.data_name == "ARG")
	{
		//change the axises to camera frame axises
		Eigen::Matrix3f R_AXIS;
		R_AXIS << 0, -1, 0,
			0, 0, -1, 
			1, 0, 0;

		Eigen::Vector3f p = R_AXIS * lp;
		lp = p;
	}
#endif
	double v = getVerticalValue(lp);
	double h = getHorizonalValue(lp);
	double e = getElevateValue(lp);
	if (v < -1)
	{
		return true;
	}

	if (type < OC_pole) // road surface
	{
		if (MY_CONFIG.data_path.find("arg0c1") != string::npos)
		{
			if (abs(h) > 7 ||//横向
			v > 50 ||//纵向
			abs(e) > 5)//高
			{
				return true;
			}
		}
		else
		{
			if (abs(h) > 15 ||//横向
				v > 50 ||//纵向
				abs(e) > 5)//高
			{
				return true;
			}
		}

		
	}
	else if (type == OC_pole) // pole
	{
		if (abs(h) > 20 ||//横向
			v > 70 ||//纵向
			abs(e) > 5)//高
		{
			return true;
		}
	}
	else//sign
	{
		if (abs(h) > 20||
					v > 30||
					(e < 0 || e > 10))//高
		{
			return true;
		}
	}
	
	return false;
}


bool getNearstPoint(const vector<RAW_INS>& ins_vec, const cv::Point3d& point, RAW_INS& ins)
{
	if (ins_vec.size() == 0)
	{
		return false;
	}

	map<double, RAW_INS> dist_map;
	for_each(ins_vec.begin(), ins_vec.end(), [&](const auto& ins) {
		auto  p = point - ins.point;
		double dis = p.ddot(p);
		dist_map[dis] = ins;
	});
	if (dist_map.size() == 0/* ||
		dist_map.begin()->first >*/ )
	{
		return false;
	}
	ins = dist_map.begin()->second;
	return true;
}


bool beyongTraceDiffZlimit(int type, double diff_z)
{
	//地面印刷
	if (type < 13 || type == 255)
	{
		return abs(diff_z)  > 2;
	}
	else //高起
	{
		//return diff_z > 20 || diff_z < -1.5;
		return diff_z > 20 || diff_z < -3.0;
	}

}

template<typename T>
void MultiAlign::removeByTraceRange(vector<T>& ego_hdobj_vec, const vector<RAW_INS>& local_ins_vec)
{
	if (local_ins_vec.size() == 0)
	{
		return;
	}
		
	auto remove_itr = ego_hdobj_vec.begin();
	for (; remove_itr != ego_hdobj_vec.end();)
	{
		auto& obj = *remove_itr;
		const auto& spt = obj.shape.front();
		const auto& ept = obj.shape.back();

		RAW_INS s_ins;
		getNearstPoint(local_ins_vec, spt, s_ins);

		RAW_INS e_ins;
		getNearstPoint(local_ins_vec, ept, e_ins);

		Eigen::Vector3f dif_s;
		dif_s[0] = spt.x - s_ins.point.x;
		dif_s[1] = spt.y - s_ins.point.y;
		dif_s[2] = spt.z - s_ins.point.z;

		Eigen::Vector3f dif_e;
		dif_e[0] = ept.x - e_ins.point.x;
		dif_e[1] = ept.y - e_ins.point.y;
		dif_e[2] = ept.z - e_ins.point.z;

		double diff_s = getElevateValue(dif_s);
		double diff_e = getElevateValue(dif_e);
		if (beyongTraceDiffZlimit(obj.type, diff_s) ||
			beyongTraceDiffZlimit(obj.type, diff_e))
		{
			remove_itr = ego_hdobj_vec.erase(remove_itr);
		}
		else
		{
			remove_itr++;
		}
	}

}

template<typename T>
void MultiAlign::transformEgo2Camera(const vector<T>& hdobj_vec, 
	const vector<double>& camPose,
	map<int, Point3f_VEC>& xyz_vec_map)
{
	for (int i = 0; i < hdobj_vec.size(); i++)
	{
		const int& type = hdobj_vec[i].type;
		const auto& obj = hdobj_vec[i].shape;
		vector<cv::Point> contour;
		int oc = convHDMapType2ObjectClassification(type);

		Point3f_VEC& xyz_vec = xyz_vec_map[oc];
		for (int j = 0; j < obj.size(); j++)
		{
			const auto& lp = obj[j];
			double xyz[3];
			OptimizeCeresWeighted::convertPointByEigen(camPose, lp.x, lp.y, lp.z, xyz);
			xyz_vec.push_back(cv::Point3f(xyz[0], xyz[1], xyz[2]));
		}
	}
}

template<typename T>
void MultiAlign::transformHDMapFromWorld2Camera(const vector<T>& hdobj_vec, 
	const RAW_INS& ins, 
	const vector<double>& camPose, 
	map<HDMapPointID, cv::Point3f>& cam_pts_map)
{
	Eigen::Vector3f R_euler(M_PI / 2 + ins.roll, ins.pitch, ins.heading);
	Eigen::Matrix3f R_matrix;
	TransRotation::eigenEuler2RotationMatrix(R_euler, R_matrix);
	Eigen::Vector3d Trans;
	Trans << ins.point.x, ins.point.y, ins.point.z;
	for (int i = 0; i < hdobj_vec.size(); i++)
	{
		const auto& obj = hdobj_vec[i];
		for (int j = 0; j < obj.shape.size(); j++)
		{
			const auto& p = obj.shape[j];
			Eigen::Vector3d pp;
			pp << p.x, p.y, p.z;
			Eigen::Vector3f lp;
			CalibSpace::EigenTranslateAndRot(pp, lp, Trans, R_matrix);
			if (beyongImageRange(obj.type, lp))
			{
				continue;
			}
			double cp[3];
			OptimizeCeresWeighted::convertPointByEigen(camPose, (double)lp[0], (double)lp[1], (double)lp[2], cp);

			HDMapPointID pid;
			pid.obj_id = obj.obj_id;
			pid.v_id = j;
			cv::Point3f tp(cp[0], cp[1], cp[2]);
			cam_pts_map.insert(make_pair(pid, tp));
		}
	}

}

template<typename T>
void MultiAlign::transformHDMapFromWorld2Image(const vector<T>& hdobj_vec, const POSE& pos, 
	vector<vector<cv::Point>>& line_vec)
{
	Eigen::Vector3d Trans = pos.p;
	Eigen::Matrix3d R_matrixd(Eigen::Quaterniond(pos.q));
	Eigen::Matrix3f R_matrix = R_matrixd.cast<float>();
	for (int i = 0; i < hdobj_vec.size(); i++)
	{
		const auto& obj = hdobj_vec[i];

		vector<cv::Point> line;
		for (int j = 0; j < obj.shape.size(); j++)
		{
			const auto& p = obj.shape[j];
			Eigen::Vector3d pp;
			pp << p.x, p.y, p.z;
			Eigen::Vector3f lp;
			CalibSpace::EigenTranslateAndRot(pp, lp, Trans, R_matrix);
			if (beyongImageRange(obj.type, lp))
			{
				continue;
			}
		
			double xyz[3] = { lp[0], lp[1], lp[2] };
			double ij[2] = { -1, -1 };
			if (!OptimizeCeresWeighted::project2Image(xyz, ij, false))
			{
				continue;
			}

			line.push_back(cv::Point(ij[0], ij[1]));
		}

		if (line.size() > 0)
		{
			line_vec.push_back(line);
		}
	}

}

template<typename T>
void MultiAlign::transformHDMapFromWorld2Image2(const vector<T>& hdobj_vec, const CamPose& cp,
	const Eigen::Matrix3f& RR, const Eigen::Vector3f& TT,
	map<int, LINE_VEC>& contours_map	)
{
	Eigen::Vector3f R_euler(M_PI / 2 + cp.ins.roll, cp.ins.pitch, cp.ins.heading);
	Eigen::Matrix3f R_matrix;
	TransRotation::eigenEuler2RotationMatrix(R_euler, R_matrix);
	Eigen::Vector3d Trans;
	Trans << cp.ins.point.x, cp.ins.point.y, cp.ins.point.z;

	for (int i = 0; i < hdobj_vec.size(); i++)
	{
		const auto& obj = hdobj_vec[i];
		int oc = convHDMapType2ObjectClassification(obj.type);
		vector<vector<cv::Point>>& line_vec = contours_map[oc];
		vector<cv::Point> line;
		for (int j = 0; j < obj.shape.size(); j++)
		{
			const auto& p = obj.shape[j];
			Eigen::Vector3d pp;
			pp << p.x, p.y, p.z;
			Eigen::Vector3f lp;
			CalibSpace::EigenTranslateAndRot(pp, lp, Trans, R_matrix);
			if (beyongImageRange(obj.type, lp))
			{
				continue;
			}

			double cp1[3];
			OptimizeCeresWeighted::convertPointByEigen(cp.camPose, (double)lp[0], (double)lp[1], (double)lp[2], cp1);
			Eigen::Vector3f cpv(cp1[0], cp1[1], cp1[2]);
			Eigen::Vector3f cp2 = RR * cpv + TT;
			double xyz[3] = { cp2[0], cp2[1], cp2[2] };
			double ij[2] = { -1, -1 };
			if (!OptimizeCeresWeighted::project2Image(xyz, ij, false))
			{
				continue;
			}

			line.push_back(cv::Point(ij[0], ij[1]));
		}

		if (line.size() > 0)
		{
			line_vec.push_back(line);
		}
	}

}

void MultiAlign::cutPartImage()
{
	HN_GENERAL::read_xml_config();
	const string& folder_path = MY_CONFIG.data_path + "\\Image\\1\\";
	vector<string> img_na_vec;
	HN_GENERAL::getAllFiles(folder_path, ".JPG", img_na_vec);
	if (img_na_vec.size() == 0)
	{
		HN_GENERAL::getAllFiles(folder_path, ".jpeg", img_na_vec);
	}
	if (img_na_vec.size() == 0)
	{
		return;
	}
	cutPartImage(img_na_vec);
}

void MultiAlign::cutPartImage(const vector<string> &img_na_vec)
{
	if (MY_CONFIG.data_para.image_rect.width == 0 ||
		MY_CONFIG.data_para.image_rect.height == 0)
	{
		return;
	}

	for (auto it = img_na_vec.begin(); it != img_na_vec.end(); it++)
	{
		const auto& itr_img = *it;
		string img_name = getFileName(itr_img);
		cv::Mat &img = cv::imread(itr_img);
		cv::Mat small_img = img(MY_CONFIG.data_para.image_rect);
		string savePath = MY_CONFIG.mid_path + "Image/1/"+ img_name+".jpg";
		cv::imwrite(savePath, small_img);
	}

}

void MultiAlign::splitJumpSignBoard(vector<vector<cv::Point>> &plys)
{
	int index1 = -1;
	int index2 = -1;
	int count = 0;

	auto &ply = plys[0];
	ply.pop_back();

	Point_VEC signBoard1;
	Point_VEC signBoard2;

	for (auto it = ply.begin(); it != ply.end(); it++,count++)
	{
		const auto& p1 = *it;
		const auto& p2 = *(it + 1);

		if (index1 == -1 || index2 != -1)
		{
			signBoard1.push_back(p1);
		}
		else
		{
			signBoard2.push_back(p1);
		}

		int delta = abs(p1.x - p2.x);
		if (delta > 0.7 * MY_CONFIG.data_para.image_width)
		{
			if (index1 == -1 && index2 == -1)
			{
				index1 = count;
			}
			else if(index1 != -1 && index2 == -1)
			{
				index2 = count;
			}
		}

	}

	Point_VEC temp_vec1;
	temp_vec1.push_back(signBoard1[1]);
	temp_vec1.push_back(signBoard1[0]);

	for_each(temp_vec1.begin(), temp_vec1.end(), [](auto& p)
	{
		if (p.x < 0.5 * MY_CONFIG.data_para.image_width)
		{
			p.x = 0;
		}
		else
		{
			p.x = MY_CONFIG.data_para.image_width;
		}
	});


	copy(temp_vec1.begin(), temp_vec1.end(), back_inserter(signBoard1));
	signBoard1.push_back(signBoard1[0]);

	Point_VEC temp_vec2;
	temp_vec2.push_back(signBoard2[1]);
	temp_vec2.push_back(signBoard2[0]);

	for_each(temp_vec2.begin(), temp_vec2.end(), [](auto& p)
	{
		if (p.x < 0.5 * MY_CONFIG.data_para.image_width)
		{
			p.x = 0;
		}
		else
		{
			p.x = MY_CONFIG.data_para.image_width;
		}
	});

	copy(temp_vec2.begin(), temp_vec2.end(), back_inserter(signBoard2));
	signBoard2.push_back(signBoard2[0]);

	plys.clear();
	plys.push_back(signBoard1);
	plys.push_back(signBoard2);

}

void MultiAlign::splitJumpContours(map<int, LINE_VEC>& contours_map)
{
	for (auto it = contours_map.begin(); it != contours_map.end(); it++)
	{
		auto& contours = it->second;
		int contour_idx = 0;
		LINE_VEC new_contours;
		vector<int> remove_indics;
		for (auto itr = contours.begin(); itr != contours.end(); itr++, contour_idx++)
		{
			const auto& contour = *itr;

			Point_VEC contour1;
			Point_VEC contour2;
			int index = -1;
			int count = 0;
			for (auto itr_p = contour.begin(); itr_p != contour.end() - 1; itr_p++, count++)
			{
				const auto& p1 = *itr_p;
				const auto& p2 = *(itr_p + 1);

				if (index == -1)
				{
					contour1.push_back(p1);
				}
				else
				{
					contour2.push_back(p1);
				}

				int delta = abs(p1.x - p2.x);
				if (delta > 0.95*MY_CONFIG.data_para.image_rect.width)
				{
					index = count;
				}

			}
			if (index >= 0)
			{
				new_contours.push_back(contour1);
				new_contours.push_back(contour2);
				remove_indics.push_back(contour_idx);
			}
		}

		int remove_sz = 0;
		for (auto itr_idx = remove_indics.begin(); itr_idx != remove_indics.end(); itr_idx++)
		{
			auto c_idx = *itr_idx - remove_sz;
			auto itr_remove = contours.begin() + c_idx;
			contours.erase(itr_remove);
			remove_sz++;
		}
		
		copy(new_contours.begin(), new_contours.end(), back_inserter(contours));
	}
}

//template<typename T, typename T2>
//void MultiAlign::transformHDMapFromWorld2Image(const vector<T>& hdobj_vec, const RAW_INS& ins, const double* camPose, vector<T2>& image_2d_vec)
//{
//	Eigen::Vector3f R_euler(M_PI / 2 + ins.roll, ins.pitch, ins.heading);
//	Eigen::Matrix3f R_matrix;
//	TransRotation::eigenEuler2RotationMatrix(R_euler, R_matrix);
//	Eigen::Vector3d Trans;
//	Trans << ins.point.x, ins.point.y, ins.point.z;
//	for (int i = 0; i < hdobj_vec.size(); i++)
//	{
//		const auto& obj = hdobj_vec[i];
//		vector<cv::Point3d> reserv_shp;
//		for (int j = 0; j < obj.shape.size(); j++)
//		{
//			const auto& p = obj.shape[j];
//			Eigen::Vector3d pp;
//			pp << p.x, p.y, p.z;
//			Eigen::Vector3f lp;
//			CalibSpace::EigenTranslateAndRot(pp, lp, Trans, R_matrix);
//
//			double cp[3];
//			OptimizeCeresWeighted::convertPointByEigen(camPose, (double)lp[0], (double)lp[1], (double)lp[2], cp);
//
//			T2 tp(cp[0], cp[1], cp[2]);
//			camera_3d_vec.push_back(tp);
//		}
//	}
//
//}



void MultiAlign::transformCoord(vector<RAW_INS>& ins_vec)
{
	CoordinateTransform::LonLatPoints2XY(ins_vec, CalibSpace::band);
}

void MultiAlign::transformCoord(HDObject_VEC& _data)
{
	if (_data.size() < 1 ||
		_data.front().shape.size() == 0)
	{
		return;
	}
	auto spt = _data.front().shape.front();
	bool need_lonlat_2_xy = (abs(spt.x) < 360.0 && abs(spt.y) < 360.0);

	if(need_lonlat_2_xy)
	{
		char GKWGS84_sign[256] = { 0 };
		sprintf_s(GKWGS84_sign, sizeof(GKWGS84_sign), "+proj=tmerc +ellps=WGS84 +lon_0=%d +x_0=500000", CalibSpace::band);
		projLP t_lp;
		projXY t_xy;
		projPJ t_GKWGS84;
		if (!(t_GKWGS84 = pj_init_plus(GKWGS84_sign)))
		{
			return;
		}
		
		//转换
		for_each(_data.begin(), _data.end(), [&](auto& d) {
			for_each(d.shape.begin(), d.shape.end(), [&](auto& p) {
				t_lp.u = p.x * DEG_TO_RAD;
				t_lp.v = p.y * DEG_TO_RAD;
				t_xy = pj_fwd(t_lp, t_GKWGS84);
				p.x = t_xy.u;
				p.y = t_xy.v;
			});
			d.shape_org = d.shape;
		});
	
		pj_free(t_GKWGS84);
	}
	

	for_each(_data.begin(), _data.end(), [&](auto& d) {
		vector<cv::Point> tmp(d.shape.size());
		transform(d.shape.begin(), d.shape.end(), tmp.begin(), [](const auto& pt)->cv::Point {
			return cv::Point(pt.x, pt.y);
			});

		d.rect = cv::boundingRect(tmp);

		if (d.type != OC_pole)
		{
			equalizPolyline(d.shape, 0.5);
		}

		d.shape_org = d.shape;

		});
	return ;
}


void MultiAlign::getImageCenter(cv::Point2f& c)
{
	if (CalibSpace::camera_type == CAMERA_MSS_WIDE)
	{
		c.x = CalibSpace::CX;
		c.y = CalibSpace::CY;
	}
	else
	{
		c.x = CalibSpace::IMG_WIDTH / 2;
		c.y = CalibSpace::IMG_HEIGHT / 2;
	}
}

void MultiAlign::horizonalAssignX(vector<double>& interval_vec, vector<int>& x_vec)
{
	/*cv::Point origin_center;
	getImageCenter(origin_center);
	vector<cv::Point> sf(1);
	sf[0] = origin_center;
	vector<cv::Point> sfpt;
	cv::perspectiveTransform(sf, sfpt, CalibSpace::warpmat_src2ipm);
	if (sfpt.size() == 0)
	{
		return;
	}
	int center_x = sfpt[0].x;
	*/

	x_vec.resize(interval_vec.size());

	int center_x = 400;
	auto find_center_r = find_if(interval_vec.begin(), interval_vec.end(), [&](const auto& it)->bool {
		return it > center_x;
	});
	int dis = distance(interval_vec.begin(), find_center_r);
	int sq = -1;
	for (int i = dis - 1; i >= 0; i--)
	{
		x_vec[i] = sq--;
	}
	sq = 1;
	for (int i = dis; i < interval_vec.size(); i++)
	{
		x_vec[i] = sq++;
	}
}

void MultiAlign::horizonalAssignLocalHDMapX(vector<double>& interval_vec, const RAW_INS& ins, vector<int>& x_vec)
{
	cv::Point3d origin_center = ins.point;
	x_vec.resize(interval_vec.size());

	int center_x = origin_center.x;
	auto find_center_r = find_if(interval_vec.begin(), interval_vec.end(), [&](const auto& it)->bool {
		return it > center_x;
	});
	int dis = distance(interval_vec.begin(), find_center_r);
	int sq = -1;
	for (int i = dis - 1; i >= 0; i--)
	{
		x_vec[i] = sq--;
	}
	sq = 1;
	for (int i = dis; i < interval_vec.size(); i++)
	{
		x_vec[i] = sq++;
	}
}

#include "HNMath/Histogram.h"
void MultiAlign::splitLaneContours(LINE_VEC& contoursf)
{
	LINE_VEC new_contoursf;

	for (auto& line : contoursf)
	{
		if (line.size() == 1)
		{
			new_contoursf.push_back(line);
			continue;
		}
	/*	vector<cv::Point> new_line;
		auto itr_s = line.begin();
		for (; itr_s != line.end() - 1; itr_s++)
		{
			auto itr_e = itr_s + 1;
			if (itr_s->y <= itr_e->y)
			{
				new_line.push_back(*itr_s);
			}
			else
			{
				new_line.push_back(*itr_s);
				new_contoursf.push_back(new_line);
				new_line.clear();
			}
		}
		if (new_line.size() > 0)
		{
		}*/
		while (line.size() > 1)
		{
			vector<cv::Point> new_line;
			int y = line[0].y;
			auto itr_l = find_if(line.begin() + 1, line.end(), [&](const auto& p)->bool {
				if (p.y < y)
				{
					y = p.y;
					return true;
				}
				y = p.y;
				return false;
			});
			copy(line.begin(), itr_l, back_inserter(new_line));
			reverse(new_line.begin(), new_line.end());
			new_contoursf.push_back(new_line);

			line.erase(line.begin(), itr_l);
			if (line.size() <= 1)
			{
				break;
			}
			y = line[0].y;
			itr_l = find_if(line.begin() + 1, line.end(), [&](const auto& p)->bool {
				if (p.y > y)
				{
					y = p.y;
					return true;
				}
				y = p.y;
				return false;
			});
			line.erase(line.begin(), itr_l);
		}
		
	}
	contoursf.swap(new_contoursf);

	int a = 0;
}

void MultiAlign::inversePerspectiveAdjustImageLines(const string& img_na, map<int, LINE_VEC>& lines_map)
{
	auto find_lines = lines_map.find(OC_lane);
	if (find_lines == lines_map.end())
	{
		return;
	}
	auto& lines = find_lines->second;
	if (lines.size() == 0)
	{
		return;
	}

//	splitLaneContours(lines);

//	return;

	if (CalibSpace::warpmat_src2ipm.cols == 0)
	{
		return;
	}
	vector<vector<cv::Point2f>> contoursf;
	for_each(lines.begin(), lines.end(), [&](const auto& line) {
		//auto max_y = max_element(line.rbegin(), line.rend(), [](const auto& x1, const auto& x2) ->bool {
		//	return x1.y < x2.y;
		//});
		//auto max_base = max_y.base();
		vector<cv::Point2f> contour;
		for_each(line.begin(), line.end(), [&](const auto& pt) {
			contour.push_back(cv::Point2f(pt));
		});
		cv::perspectiveTransform(contour, contour, CalibSpace::warpmat_src2ipm);

		//reverse(contour.begin(), contour.end());


		//auto spt = contour.front();
		//auto ept = contour.back();
		//spt.y += 20;
		//ept.y -= 10;
		//contour.insert(contour.begin(), spt);
		//contour.push_back(ept);
 		contoursf.push_back(contour);
	});


	//return;
#if 1
	//区分横向分布
	//auto itr_line = contoursf.begin();
	//for (; itr_line != contoursf.end(); itr_line++)
	//{
	//	auto& line = *itr_line;
	//	auto max_y = max_element(line.begin(), line.end(), [](const auto& x1, const auto& x2) ->bool {
	//		/*return int(x1.y) < int(x2.y);*/
	//		return x1.y < x2.y;
	//	});
	//	line.erase(max_y + 1, line.end());
	//	reverse(line.begin(), line.end());
	//}


//	mergeImageLines(contoursf);

	map<int, LINE_VEC> tmp_map;
	map<int, vector<vector<cv::Point2f>>> h_lines_map;
	for_each(contoursf.begin(), contoursf.end(), [&](const auto& cf) {
		vector<cv::Point> contour(cf.size());
		transform(cf.begin(), cf.end(), contour.begin(), [&](const auto& pf)->cv::Point {
			return cv::Point(pf);
			});

		tmp_map[OC_lane].push_back(contour);
		});
	lines_map.swap(tmp_map);

	return;
	//尝试区分横向车道线
	inversePerspectiveHorizonalAssignImageLines(contoursf, h_lines_map);
	//inversePerspectiveHorizonalAssignImageLines_DBSCAN(contoursf, h_lines_map);
#endif

	lines.clear();
	string ipm_path = MY_CONFIG.mid_path + "ipm/" + img_na + ".jpg";
	cv::Mat ipm_img = cv::imread(ipm_path);
	
	vector<cv::Scalar> sca_vec{ 
		cv::Scalar(255, 0, 0) ,
		cv::Scalar(0, 255, 0) ,
		cv::Scalar(0, 0, 255) ,
		cv::Scalar(0, 0, 0),
		cv::Scalar(255, 0, 255),
	};

	int h = 0;
	
	for_each(h_lines_map.begin(), h_lines_map.end(), [&](auto& hl) {
		auto& tfs = hl.second;
		LINE_VEC ipm_lines;
		for_each(tfs.begin(), tfs.end(), [&](auto& tf) {
			vector<cv::Point> contour(tf.size());
			transform(tf.begin(), tf.end(), contour.begin(), [](const auto& pf)->cv::Point {
				return cv::Point(pf);
				});
			ipm_lines.push_back(contour);

			//暂时装入一个车道线类型中，不区分车道
			tmp_map[13].push_back(contour);
			});
		cv::polylines(ipm_img, ipm_lines, false, sca_vec[h], 3);
		h++;
		//tmp_map[h].swap(ipm_lines);
		
		});
	cv::imwrite(ipm_path, ipm_img);
	lines_map.swap(tmp_map);

#if 0
	//从ipm坐标还原回去
	lines.clear();
	for_each(h_lines_map.begin(), h_lines_map.end(), [&](auto& hl) {
		auto& tfs = hl.second;
		for_each(tfs.begin(), tfs.end(), [&](auto& tf) {
			cv::perspectiveTransform(tf, tf, CalibSpace::warpmat_ipm2src);
			vector<cv::Point> contour(tf.size());
			transform(tf.begin(), tf.end(), contour.begin(), [](const auto& pf)->cv::Point {
				return cv::Point(pf);
			});
			lines.push_back(contour);
		});

	});
#endif

#ifdef H_ASSGIN
	lines_map.swap(h_lines_map);
#endif

}

void MultiAlign::inversePerspectiveHorizonalAssignImageLines(vector<vector<cv::Point2f>>& lines,
	map<int, vector<vector<cv::Point2f>>>& lines_map)
{
	vector<double> x_vec;
	auto itr_line = lines.begin();
	for (; itr_line != lines.end(); itr_line++)
	{
		auto& line = *itr_line;
		for_each(line.begin(), line.end(), [&](const auto& p) {
			x_vec.push_back(p.x);
		});
	}
	if (x_vec.size() == 0)
	{
		return;
	}

	auto itr_max = max_element(x_vec.begin(), x_vec.end(), [](const auto& x1, const auto& x2) ->bool{
		return x1 < x2;
	});

	auto itr_min = min_element(x_vec.begin(), x_vec.end(), [](const auto& x1, const auto& x2) ->bool {
		return x1 < x2;
	});

	map<double, double> value_p_map;
	double unitInterval = 1.0;
	double merge_with = 5;
	HN_GENERAL::calcMergeHistogram(x_vec, *itr_min, *itr_max, value_p_map, unitInterval, merge_with);
	vector<double> interval_vec;
	double merge_width = 50;
	findPeakPoints(value_p_map, interval_vec, merge_width);
	for_each(interval_vec.begin(), interval_vec.end(), [&](auto& v) {
		v = v *unitInterval + *itr_min;
	});

	vector<int> hx_vec;
	horizonalAssignX(interval_vec, hx_vec);

	itr_line = lines.begin();
	for (; itr_line != lines.end(); itr_line++)
	{
		auto& line = *itr_line;
		auto spt = line.front();

		auto find_v = min_element(interval_vec.begin(), interval_vec.end(),
			[&](const auto& v1, const auto& v2)->bool {
			return abs(spt.x - v1) < abs(spt.x - v2);
		});
		
		double dista = distance(interval_vec.begin(), find_v);
		int sq = hx_vec[dista];
		if (sq > 2) { sq = 2; }
		if (sq < -2) { sq = -2; }

		lines_map[sq].push_back(line);
	}

	int i = 0;
	for_each(lines_map.begin(), lines_map.end(), [&, this](auto& tl) {
		if (tl.first == 1 || tl.first == -1)
		{
			auto& lines = tl.second;
			cv::Point max_y;
			int line_idx = 0;
			int max_idx = 0;
			for_each(lines.begin(), lines.end(), [&](auto& line) {
				const auto& p = line.front();
				if (p.y > max_y.y)
				{
					max_y = p;
					max_idx = line_idx;
				}
				line_idx++;
			});

			if (max_y.y > 900)
			{
				auto& max_y_line = lines[max_idx];
				vector<cv::Point2f> l;
				for (int i = 1147; i > max_y.y; i--)
				{
					l.push_back(cv::Point2f(max_y.x, i));
				}
				copy(max_y_line.begin(), max_y_line.end(), back_inserter(l));
				max_y_line.swap(l);
			}
		}	
	});

}



void MultiAlign::mergeImageLines(vector<vector<cv::Point2f>>& line_vec)
{
	auto itr_l = line_vec.begin();
	for (; itr_l != line_vec.end();)
	{
		auto& line = *itr_l;
		const auto& ept = line.back();

		map<double, int> dis_idx_map;
		int idx = 0;
		for_each(line_vec.begin(), line_vec.end(), [&](const auto& line_o){
			const auto& spt = line_o.front();
			auto  p = ept - spt;
			double dis = p.ddot(p);
			//return (abs(p.x) <= 5 &&
			//	        p.y >= 0 && p.y < 150);		
			if (abs(p.x) <= 20 &&
				p.y >= 0 && p.y < 100)
			{
				dis_idx_map.insert(make_pair(dis, idx));
			}
			idx++;
		});

		if (dis_idx_map.size() > 0)
		{
			int idx = dis_idx_map.begin()->second;

			const auto& line_o = line_vec[idx];
			vector<cv::Point2f> insert_line(2);
			insert_line[0] = line.back();
			insert_line[1] = line_o.front();
			insertPoint(insert_line, 1.0);

			//		copy(insert_line.begin() + 1, insert_line.end() - 1, back_inserter(line));

			copy(line_o.begin(), line_o.end(), back_inserter(line));
			auto itr_o = line_vec.begin() + idx;
			line_vec.erase(itr_o);
		}
		else
		{
			itr_l++;
		}
	}
}

template<typename T>
void MultiAlign::horizonalAssignHDMap(vector<T>& ego_hdobj_vec, const vector<RAW_INS>& local_ins_vec)
{
	if (ego_hdobj_vec.size() == 0 ||
		local_ins_vec.size() == 0)
	{
		return;
	}

	vector<double> x_vec;
	auto itr = ego_hdobj_vec.begin();
	for (; itr != ego_hdobj_vec.end(); itr++)
	{
		auto& obj = *itr;
		const auto& spt = obj.shape.front();
		const auto& ept = obj.shape.back();
		x_vec.push_back(spt.x);
	}

	if (x_vec.size() == 0)
	{
		return;
	}

	auto itr_max = max_element(x_vec.begin(), x_vec.end(), [](const auto& x1, const auto& x2) ->bool {
		return x1 < x2;
	});

	auto itr_min = min_element(x_vec.begin(), x_vec.end(), [](const auto& x1, const auto& x2) ->bool {
		return x1 < x2;
	});

	map<double, double> value_p_map;
	double unitInterval = 1.0;
	HN_GENERAL::calcHistogram(x_vec, *itr_min, *itr_max, value_p_map, unitInterval);

	vector<double> interval_vec;
	double merge_width = 3.0;
	findPeakPoints(value_p_map, interval_vec, merge_width);
	for_each(interval_vec.begin(), interval_vec.end(), [&](auto& v) {
		v = v *unitInterval + *itr_min;
	});

	vector<int> hx_vec;
	horizonalAssignLocalHDMapX(interval_vec, local_ins_vec[0], hx_vec);

//	map<int, Point3d_VEC> temp_lines_map;
	itr = ego_hdobj_vec.begin();
	for (; itr != ego_hdobj_vec.end(); itr++)
	{
		auto& obj = *itr;
		const auto& spt = obj.shape.front();
		auto find_v = min_element(interval_vec.begin(), interval_vec.end(),
			[&](const auto& v1, const auto& v2)->bool {
			return abs(spt.x - v1) < abs(spt.x - v2);
		});
	
		double dista = distance(interval_vec.begin(), find_v);
		int sq = hx_vec[dista];
		if (sq > 2) { sq = 2; }
		if (sq < -2) { sq = -2; }
		obj.horizonal_idx = sq;
	}
}

void MultiAlign::saveImage(const string& sub_folder, 
	const string& img_na,
	const cv::Mat& img, 
	const cv::Scalar& sca,
	map<int, LINE_VEC>& lines_map)
{
	//if (!cp.regist_flg)
	//{
	//	cv::putText(t, "RESET", cv::Point(0, 40), 2, 1.0, cv::Scalar(0, 0, 0), 2);
	//}
	//cv::putText(t, to_string(cp.regist_probability), cv::Point(0, 80), 2, 1.0, cv::Scalar(255, 0, 0), 2);
	//cv::putText(t, to_string(cp.res), cv::Point(0, 160), 2, 1.0, cv::Scalar(255, 0, 0), 2);


	cv::Mat mat_ply = img.clone();
	for_each(lines_map.begin(), lines_map.end(), [&](const auto&l) {
		int type = l.first;
		cv::polylines(mat_ply, l.second, false, sca, 5);
	});

	string ply_path = MY_CONFIG.mid_path + sub_folder + img_na + ".jpg";
	cv::imwrite(ply_path, mat_ply);
}

void MultiAlign::saveImage(const string & sub_folder,
	const string & img_na,
	const cv::Mat & img,
	const map<int, cv::Scalar>& type_color,
	map<int, LINE_VEC>&lines_map)
{
	//if (!cp.regist_flg)
	//{
	//	cv::putText(t, "RESET", cv::Point(0, 40), 2, 1.0, cv::Scalar(0, 0, 0), 2);
	//}
	//cv::putText(t, to_string(cp.regist_probability), cv::Point(0, 80), 2, 1.0, cv::Scalar(255, 0, 0), 2);
	//cv::putText(t, to_string(cp.res), cv::Point(0, 160), 2, 1.0, cv::Scalar(255, 0, 0), 2);


	cv::Mat mat_ply = img.clone();
	for_each(lines_map.begin(), lines_map.end(), [&](const auto& l) {
		int type = l.first;

		cv::Scalar clr(0, 0, 0);
		const auto& find_clr = type_color.find(type);
		if (find_clr != type_color.end())
		{
			clr = find_clr->second;
		}

		bool be_closed = false;
		if (l.first == OC_crosswalk || l.first == OC_t_sign)
		{
			be_closed = true;
		}

		cv::polylines(mat_ply, l.second, be_closed, clr, 5);

		});

	string ply_path = MY_CONFIG.mid_path + sub_folder + img_na + ".jpg";
	cv::imwrite(ply_path, mat_ply);


}

void MultiAlign::undistortImage(const cv::Mat& src_img,
	cv::Mat& dest_img)
{
	if (CalibSpace::distCoeffs.at<double>(0, 0) != 0)
	{
		cv::undistort(src_img, dest_img, CalibSpace::intrisicMat, CalibSpace::distCoeffs);
	}
	
}


void MultiAlign::collectLocalIns(const vector<RAW_INS>& ins_vec, int idx, double len, vector<RAW_INS>& local_ins_vec,
	bool add, bool farward)
{
	if (idx >= ins_vec.size())
	{
		return;
	}
	const RAW_INS& cur_ins = ins_vec[idx];
	if (farward)
	{
		local_ins_vec.push_back(cur_ins);
	}

	double dis = 0;
	auto last_ins = cur_ins;

	if (farward)
	{
		for (int i = idx + 1; i < ins_vec.size(); i++)
		{
			auto  p = last_ins.point - ins_vec[i].point;
			dis += sqrt(p.dot(p));
			last_ins = ins_vec[i];
			if (dis > len)
			{
				break;
			}
			local_ins_vec.push_back(last_ins);
		}
	}
	else
	{
		for (int i = idx - 1; i >= 0; i--)
		{
			auto  p = last_ins.point - ins_vec[i].point;
			dis += sqrt(p.dot(p));
			last_ins = ins_vec[i];
			if (dis > len)
			{
				break;
			}
			local_ins_vec.insert(local_ins_vec.begin(), last_ins);
		}
	}
	if (!add)
	{
		return;
	}

	/*float add_dist = farward ? 10.0 : -10.0;
	RAW_INS add_e = farward ? local_ins_vec.back() : cur_ins;
	if (ins_vec.size() - 1 == idx ||
		local_ins_vec.size() == 1)
	{
		add_e.point.x = cur_ins.point.x + add_dist * sin(cur_ins.heading);
		add_e.point.y = cur_ins.point.y + add_dist * cos(cur_ins.heading);
		add_e.point.z = cur_ins.point.z + add_dist * tan(cur_ins.pitch);
	}
	if (farward)
	{
		local_ins_vec.push_back(add_e);
	}
	else
	{
		local_ins_vec.insert(local_ins_vec.begin(), add_e);
	}*/
}


bool MultiAlign::analyzeRobustnessbyRandomSample(const vector<double>& gt_campos,
	const RAW_INS& ins,
	const string& img_na)
{
	SUB_FOLDER_PLY = "prob_distribution\\";

	string directoryPath = MY_CONFIG.mid_path + SUB_FOLDER_PLY;
	createDirectoryIfNotExists(directoryPath);

	m_reg_oc_set = { OC_lane, OC_pole, OC_t_sign };

	map<int, LINE_VEC> lines_map;
	m_calib.extractImageDeepLearningMultiLines(MY_CONFIG.mid_path, 
		img_na, lines_map);

	cv::Mat camera_img = cv::imread(MY_CONFIG.img_path + img_na + ".jpg");
	saveImage(SUB_FOLDER_PLY, img_na, camera_img, m_type_seg_color, lines_map);

	string ply_path = MY_CONFIG.mid_path + SUB_FOLDER_PLY + img_na + ".jpg";
	camera_img = cv::imread(ply_path);

	map<int, HDObject_VEC> ego_hdobj_vec;
	getLocalHDMap(ins, ego_hdobj_vec);

#if 0
	map<int, LINE_VEC> contours_map;
	getEgoMapContours(ego_hdobj_vec, gt_campos, contours_map, false);
	float eval = 1 - evaluteSimilarity(lines_map, contours_map);
	string eval_str = to_string(eval);
	eval_str = eval_str.substr(0, eval_str.find(".") + 3);
	cv::putText(camera_img, eval_str, cv::Point(1000, 40), 2, 2.0, cv::Scalar(0, 255, 0), 2);
	//saveImage(SUB_FOLDER_PLY, img_na, camera_img, m_type_color, contours_map);
#endif
	auto gggt = gt_campos;

//	gggt[0] += 0.09;
//	gggt[1] -= 0.05;
// 
// 
//	gggt[2] += 0.7;

	analyzeSampleRobustness(gggt, lines_map, ego_hdobj_vec, img_na);

	return true;
}

void getSampleHDMap(const map<int, HDObject_VEC>& obj_map,
	const vector<double>& pos,
	map<int, HDObject_VEC>& sample_hdobj_map)
{
	auto itr_type = obj_map.begin();
	for (; itr_type != obj_map.end(); itr_type++)
	{
		const auto& ego_hdobj_vec = itr_type->second;
		auto& sample_hdobj_vec = sample_hdobj_map[itr_type->first];
		sample_hdobj_vec = ego_hdobj_vec;

		Eigen::Vector3d t;
		Eigen::Matrix3d r;
		getPosRT(pos, t, r);
		sample_hdobj_vec = ego_hdobj_vec;

		for_each(sample_hdobj_vec.begin(), sample_hdobj_vec.end(), [&](auto& obj) {

			for_each(obj.shape.begin(), obj.shape.end(), [&](auto& pt) {
				Eigen::Vector3d ep(pt.x, pt.y, pt.z);
				ep = r * ep + t;
				pt.x = ep[0];
				pt.y = ep[1];
				pt.z = ep[2];
				});

			});
	}

}


#include "SampleTest.h"
void MultiAlign::analyzeSampleRobustness(const vector<double>& gt_campos,
	const map<int, LINE_VEC>& lines_map,
	const map<int, HDObject_VEC>& ego_hdobj_vec,
	const string& img_na)
{
	string file_postfix = getPostfix(m_reg_method);
	string folder = MY_CONFIG.mid_path + SUB_FOLDER_PLY + "sample-" + file_postfix + "\\";
	createDirectoryIfNotExists(folder);

	vector<vector<double>> sample_poses;
	SampleTest st;
	st.generateSamplePoses(MY_CONFIG.mid_path, sample_poses);

	vector<vector<double>> error_poses;
	vector<vector<double>> sample_poses_extend;


	int idx = 0;
	//idx = 16;
	StatisticData_MAP sd_map;
	string save_folder = MY_CONFIG.mid_path + "spatial_features\\";
	m_cc.getFrameSpatialDistribution(save_folder, img_na, sd_map);

	auto itr_pos = sample_poses.begin() + idx;
	for (; itr_pos != sample_poses.end(); itr_pos++, idx++)
	{
		LOG(INFO) << ("sample:....%d....", idx);
		const auto& sample = *itr_pos;
		auto pos = gt_campos;

		//抖动
		map<int, HDObject_VEC> sample_hdobj_vec;
		getSampleHDMap(ego_hdobj_vec, sample, sample_hdobj_vec);

		//recoverRegistGTforSample(sample, gt_campos, pos);

		map<int, LINE_VEC> s_contours_map;
		getEgoMapContours(sample_hdobj_vec, pos, s_contours_map, false);
		float s_eval = 1 - evaluteSimilarity(lines_map, s_contours_map);
		//float s_eval = 1 - evaluteSimilarity(gt_hd_contours_map, contours_map);

		auto sample_extend = sample;
		sample_extend.push_back(s_eval);
		sample_poses_extend.push_back(sample_extend);

		//test 3d-3d init start
		//set fixed
		//pos[1] = CalibSpace::camera_height - CalibSpace::ego_height;
		iterateClosestPoint3d3d(sample_hdobj_vec, lines_map, pos);

		map<int, LINE_VEC> d3d3_contours_map;
		getEgoMapContours(sample_hdobj_vec, pos, d3d3_contours_map, false);
		//////test end/////////
		//if(0)
		////if (m_reg_method == RegMethod_EOR) 
		//{
		//	m_reg_oc_set = { OC_lane };
		//	map<int, float> tmp_weights;
		//	tmp_weights.insert(make_pair(OC_lane, 1.0));
		//	op.weights = tmp_weights;
		//	doRegist(sample_hdobj_vec, lines_map, m_reg_method, m_reg_oc_set, op, pos);
		//}


		doRegist(sample_hdobj_vec, lines_map, m_reg_method, sd_map, pos);

		vector<double> dif_gt;
		calSampleDiffPose(sample, pos, gt_campos, dif_gt);

		map<int, LINE_VEC> e_contours_map;
		getEgoMapContours(sample_hdobj_vec, pos, e_contours_map, false);

		float e_eval = 1 - evaluteSimilarity(lines_map, e_contours_map);
		//float e_eval = 1 - evaluteSimilarity(gt_hd_contours_map, contours_map);

		dif_gt.push_back(e_eval);
		error_poses.push_back(dif_gt);
#if 1
		// save image for eye view check
		string eval_str = "";
		for (const auto& t : dif_gt)
		{
			string str_t = to_string(t);
			str_t = str_t.substr(0, str_t.find(".") + 3);
			eval_str = eval_str + str_t + ",";
		}
		//eval_str = eval_str.substr(0, eval_str.find(".") + 3);
		// ply image

		string ply_path = MY_CONFIG.mid_path + SUB_FOLDER_PLY + img_na + ".jpg";
		cv::Mat t = cv::imread(ply_path);
		cv::putText(t, eval_str, cv::Point(0, 80), 2, 2.0, cv::Scalar(255, 255, 255), 2);

		for_each(s_contours_map.begin(), s_contours_map.end(), [&](const auto& l) {
			int type = l.first;
			//init
			cv::polylines(t, l.second, false, cv::Scalar(0, 0, 0), 5);
			//mid
			//cv::polylines(t, d3d3_contours_map[type], false, cv::Scalar(0, 255, 255), 5);
			//end
			cv::polylines(t, e_contours_map[type], false, cv::Scalar(255, 255, 255), 5);
			});

		string save_path = folder + img_na + "_" + to_string(idx) + ".jpg";
		cv::imwrite(save_path, t);
#endif
	}


	const string& file_path = MY_CONFIG.mid_path + SUB_FOLDER_PLY + "0_sample_total_" + file_postfix + ".csv";

	st.saveResults(img_na, file_path, sample_poses_extend, error_poses);
	return;
}




void recoverRegistGTforSample(
	const vector<double>& sample,
	const vector<double>& gt_campos,
	vector<double>& pos)
{
	Eigen::Matrix3d ra, rb, rg; // Your calculated rotation matrices
	Eigen::Vector3d ta, tb, tg; // Your calculated translation vectors
	getPosRT(sample, ta, ra);
	getPosRT(gt_campos, tg, rg);

	Eigen::Matrix4d A, B, E; // Transformation matrices

	// Convert ra and ta to transformation matrix A
	A.setIdentity();
	A.block(0, 0, 3, 3) = ra;
	A.block(0, 3, 3, 1) = ta;

	E.setIdentity();
	E.block(0, 0, 3, 3) = rg;
	E.block(0, 3, 3, 1) = tg;

	B = E * A.inverse();
	Eigen::Vector3d translation = B.block<3, 1>(0, 3);
	Eigen::Matrix3d rotationMatrix = B.block<3, 3>(0, 0);
	Eigen::Vector3d euler = rotationMatrix.eulerAngles(0, 1, 2);

	pos[0] = translation[0];
	pos[1] = translation[1];
	pos[2] = translation[2];

	pos[3] = euler[0];
	pos[4] = euler[1];
	pos[5] = euler[2];
}


string getLabelText(const vector<double>& dif_gt)
{
	string eval_str = "";
	for (const auto& t : dif_gt)
	{
		string str_t = to_string(t);
		str_t = str_t.substr(0, str_t.find(".") + 3);
		eval_str = eval_str + str_t + ",";
	}
	if (eval_str.size() > 0)
	{
		eval_str.pop_back();
	}
	return eval_str;
}

#if 0
void MultiAlign::analyzeSpatialDistribution(const vector<double>& gt_campos,
	const map<int, LINE_VEC>& lines_map,
	const map<int, HDObject_VEC>& ego_hdobj_vec,
	const CamPose& cp,
	const string& save_folder)
{
	modifyFeatureCategories(reg_oc_set);

	CapabilityCalculator cc;
	map<int, LINE_VEC> random_lines_map;
	map<int, HDObject_VEC> random_objs;

	int  map_num = 100;
	vector<double> pos_3d3d;
	pos_3d3d = m_calib.iterateClosestPoint3d3d(ego_hdobj_vec, lines_map);

	map<int, LINE_VEC> d3d3_contours_map;
	getEgoMapContours(ego_hdobj_vec, pos_3d3d, d3d3_contours_map, false);

	vector<double> pos = pos_3d3d;
	for (int i = 0; i < map_num; ++i)
	{
		vector<double> pos = pos_3d3d;

		CamPose s_cp = cp;
		s_cp.idx = i;
		s_cp.camPose = pos;
		cc.generateDistributionSample(lines_map, ego_hdobj_vec, s_cp,
			random_lines_map, random_objs, save_folder);

	//	pos = m_calib.iterateClosestPoint3d3d(random_objs, random_lines_map);
		map<int, LINE_VEC> init_contours_map;
		getEgoMapContours(random_objs, pos, init_contours_map, false);

		doRegist(random_objs, random_lines_map,  pos);

		map<int, LINE_VEC> e_contours_map;
		getEgoMapContours(random_objs, pos, e_contours_map, false);
		float e_eval = 1 - evaluteSimilarity(random_lines_map, e_contours_map);

		vector<double> dif_gt;
		calDiffDistanceAndAngle(gt_campos, pos, dif_gt);
		dif_gt.push_back(e_eval);

		// save image for eye view check
		string eval_str = getLabelText(dif_gt);
		
		// ply image
		string ply_path = MY_CONFIG.img_path + s_cp.img_na + ".jpg";
		cv::Mat t = cv::imread(ply_path);
		cv::putText(t, eval_str, cv::Point(0, 80), 2, 2.0, cv::Scalar(255, 255, 255), 2);
		string folder = save_folder + "regist\\";

		for_each(lines_map.begin(), lines_map.end(), [&](const auto& l) {
			int type = l.first;
			//init
			cv::polylines(t, l.second, false, cv::Scalar(0, 0, 0), 5);
			//3d3d
			cv::polylines(t, d3d3_contours_map[type], false, cv::Scalar(0, 255, 255), 6);

			//mid
			cv::polylines(t, init_contours_map[type], false, cv::Scalar(255, 0, 0), 2);

			//end
			cv::polylines(t, e_contours_map[type], false, cv::Scalar(255, 255, 255), 5);
			});

		string save_path = folder + s_cp.img_na + "_" + to_string(s_cp.idx) + ".jpg";
		cv::imwrite(save_path, t);

		cc.saveRegistPoses(cp, save_folder, dif_gt);
	}

	return;
}
#endif



