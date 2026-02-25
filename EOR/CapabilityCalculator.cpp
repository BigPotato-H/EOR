#include "CapabilityCalculator.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>

#include "HNMath/GeometricAlgorithm2.h"
#include "DataManager/WKTCSV.h"


#include "OptimizeWeighted.h"
#include "MutualInfo.h"
#include <set>
#include <io.h>
#include <random>
#include "HNMath/TransRotation.h"

set<int> oc_set = {13,16,90};
//oc_set.insert(13);
//oc_set.insert(16);
//oc_set.insert(90);

enum MODIFY_TYPE {
	MD_ADDITION = 0,
	MD_DELETION,
	MD_MODIFICATION
};

void calcNormals(const vector<cv::Point3d>& pt_vec,
	vector<cv::Point3d>& normals)
{
	for (int i = 0; i < pt_vec.size() - 1; i++)
	{
		double heading = calcPolylineYaw2(pt_vec, i);
		auto to_left_point = calPointByAltitude(pt_vec[i], true, heading, 1.0);
	//	cv::Point3d normal_point = pt_vec[i] - to_left_point;
		cv::Point3d normal_point = to_left_point;
		normals.push_back(normal_point);
	}
}


void putXZPointsToXYPlane(const vector<cv::Point3d>& ego_shape,
	vector<cv::Point3d>& pt_vec)
{
	pt_vec.resize(ego_shape.size());
	transform(ego_shape.begin(), ego_shape.end(), pt_vec.begin(), [](const auto& pt)->cv::Point3d {
	//return cv::Point3d(pt.x, pt.z, pt.0);
	return cv::Point3d(pt.x, pt.z,pt.y);
	});
}

void putXYPointsToXYPlane(const vector<cv::Point3d>& ego_shape,
	vector<cv::Point3d>& pt_vec)
{
	pt_vec.resize(ego_shape.size());
	transform(ego_shape.begin(), ego_shape.end(), pt_vec.begin(), [](const auto& pt)->cv::Point3d {
		return cv::Point3d(pt.x, pt.y, 0);
		});
	
}


void putYZPointsToXYPlane(const vector<cv::Point3d>& ego_shape,
	vector<cv::Point3d>& pt_vec)
{
	pt_vec.resize(ego_shape.size());
	transform(ego_shape.begin(), ego_shape.end(), pt_vec.begin(), [](const auto& pt)->cv::Point3d {
		return cv::Point3d(pt.y, pt.z, 0);
		});
}

void changeCoordiante(HDObject_VEC& total_objs)
{
	for (auto& obj : total_objs)
	{
		vector<cv::Point3d> xy_pt_vec;
		putXZPointsToXYPlane(obj.shape, xy_pt_vec);
		obj.shape.swap(xy_pt_vec);
	}

}
void calcObjectsNormals(int type, const vector<cv::Point3d>& shape, vector<cv::Point3d>& normals)
{
	vector<cv::Point3d> xy_pt_vec;
	//if (type == OC_lane)
	//{
	//	putXZPointsToXYPlane(shape, xy_pt_vec);
	//}
	//else if (type == OC_pole)
	//{
	//	putYZPointsToXYPlane(shape, xy_pt_vec);
	//}
	//else if (type == OC_t_sign)
	//{
	//	putXYPointsToXYPlane(shape, xy_pt_vec);
	//}
	putXZPointsToXYPlane(shape, xy_pt_vec);
	calcNormals(xy_pt_vec, normals);
}

void saveLocalObjects(string file_na, const HDObject_VEC& obj_vec)
{
	ofstream os(file_na, ios::trunc);
	if (!os.is_open())
	{
		return;
	}
	os.setf(ios::fixed, ios::floatfield);
	os.precision(4);

	for (int i = 0; i < obj_vec.size(); i++)
	{
		const auto& obj = obj_vec[i];
		vector<string> attr_vec;
		attr_vec.push_back(obj.obj_id);
		attr_vec.push_back(to_string(obj.type));
		outAttrCSVFields(os, attr_vec);
		outShpCSVFields(os, obj.shape, 1);
	}
	os.close();
}


void calcNormalDistribution(const vector<vector<cv::Point3d>>& normals,
	map<pair<int,int>, int>& bin_map)
{
	
	float dx = 0.1;
	float dy = 0.1;
	float dz = 0.1;

	int ex = 1.0 / dx;
	int ey = 1.0 / dy;

	for (const auto& obj : normals)
	{
		for (const auto& nor : obj)
		{
			int bin_x = nor.x * ex;
			int bin_y = nor.y * ey;
			bin_map[make_pair(bin_x, bin_y)] ++;
		}
	}
}


float calcNormalEntropy(map<pair<int,int>, int>& bin_map)
{
	float H = 0;
	int total_cnt = 0;
	for_each(bin_map.begin(), bin_map.end(), [&total_cnt](const auto& _FeatureID) {
		total_cnt += _FeatureID.second;
		});

	auto itr = bin_map.begin();
	for (; itr != bin_map.end(); itr++)
	{
		float p = itr->second * 1.0 / total_cnt;
		float h = p * log2f(p);
		H += h;
	}

	H = -H;

	return H;
}

void saveLocalNormals(const HDObject_VEC& obj_vec, 
	vector<vector<cv::Point3d>> normals_vec)
{
	ofstream os("0_local_normal.csv", ios::trunc);
	if (!os.is_open())
	{
		return;
	}
	os.setf(ios::fixed, ios::floatfield);
	os.precision(4);
	//ofstream of("normal.txt", ios::trunc);
	//string line = "";

	for (int o = 0; o < obj_vec.size(); o++)
	{
		//const auto& pt_vec = obj.shape;
		const auto& obj = obj_vec[o];
		const auto& normals = normals_vec[o];
		//calcObjectsNormals(obj.type, obj.shape, normals);

		//pcl::PointCloud<pcl::PointXYZ> pc;
		//pc.resize(pt_vec.size());
		//transform(pt_vec.begin(), pt_vec.end(), pc.begin(), [](const auto& pt)->pcl::PointXYZ {
		//	//return pcl::PointXYZ(pt.x, pt.y, pt.z); 
		//	return pcl::PointXYZ(pt.x, 0, pt.z);
		//});

		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
		//pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
		//tree->setInputCloud(pc.makeShared());
		//normEst.setInputCloud(pc.makeShared());
		//normEst.setSearchMethod(tree);
		//normEst.setKSearch(3);
		////normEst.setRadiusSearch(0.15);
		//normEst.compute(*normals);

		for (int i = 0; i < normals.size() - 1; i++)
		{
			// 			double heading = calcPolylineYaw2(pt_vec, i);
			// 			auto to_left_point = calPointByAltitude(pt_vec[i], true, heading, 1.0);
			// 			cv::Point3d normal_point = pt_vec[i] - to_left_point;
			//			auto normal_point = normals->at(i);
			auto normal_point = normals[i];
			//line = to_string(normal_point.x) + "," + to_string(normal_point.y) + "," + to_string(normal_point.z);
			//of << line << endl;

			vector<string> attr_vec;
			attr_vec.push_back(obj.obj_id);
			attr_vec.push_back(to_string(obj.type));
			attr_vec.push_back(to_string(i));

			vector<cv::Point3d> shape;
			shape.push_back(obj.shape[i]);
			shape.push_back(normals[i]);

			outAttrCSVFields(os, attr_vec);
			outShpCSVFields(os, shape, 1);

		}

	}
	os.close();
	//	of.close();
}


float calcFeatureNormalEntropy(HDObject_VEC obj_vec)
{
	changeCoordiante(obj_vec);

	vector<vector<cv::Point3d>> normals_vec;	
	for (const auto& obj : obj_vec)
	{
		vector<cv::Point3d> normals;
		calcNormals(obj.shape, normals);
		normals_vec.push_back(normals);
	}
	
	//saveLocalNormals(obj_vec, normals_vec);

	//calculate histogram
	//auto t = normals_vec;
	//t.clear();
	//t.push_back(normals_vec[0]);
	//t.push_back(normals_vec.back());
	//normals_vec.swap(t);

	map<pair<int,int>, int> bin_map;
	calcNormalDistribution(normals_vec, bin_map);

	float H = calcNormalEntropy(bin_map);
	return H;
}


CapabilityCalculator::CapabilityCalculator()
{

}

CapabilityCalculator::~CapabilityCalculator()
{
}
//
//void CapabilityCalculator::setObjectTypes(const set<ObjectClassification>& oc_set)
//{
//	m_obj_type_set = oc_set;
//}

//void CapabilityCalculator::setSource(const map<int, LINE_VEC>& lines_map_rgb,
//	const map<int, HDObject_VEC>& ego_objs_map/*,
//	const map<int, LINE_VEC>& lines_map_hdmap*/)
//{
//	m_mat_rgb = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
//	auto itr = lines_map_rgb.begin();
//	for (; itr != lines_map_rgb.end(); itr++)
//	{
//		const auto& type = itr->first;
//		const auto& line_vec = itr->second;
//		cv::polylines(m_mat_rgb, line_vec, false, cv::Scalar(type), 20);
//	}
//	// 	auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(14, 14));
//	// 	cv::dilate(blur_mat_rgb, blur_mat_rgb, kernel);
//
//	//	cv::imwrite("0_rgb.jpg", blur_mat_rgb);
//
//	m_mat_hdmap = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
//	auto itr_objs = ego_objs_map.begin();
//	for (; itr_objs != ego_objs_map.end(); itr_objs++)
//	{
//		const auto& type = itr_objs->first;
//		const auto& obj_vec = itr_objs->second;
//		vector<vector<cv::Point>> line_vec;
//
//		for_each(obj_vec.begin(), obj_vec.end(), [&](const auto& obj) {
//			line_vec.push_back(obj.ij_shape);
//			});
//		cv::polylines(m_mat_hdmap, line_vec, false, cv::Scalar(type), 20);
//
//	}
//
//	// 	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
//	// 	cv::dilate(blur_mat_hdmap, blur_mat_hdmap, kernel);
//
//	//	cv::imwrite("0_hd.jpg", blur_mat_hdmap);
//
//	
//}
//


void getCommonObjects(int type, 
	const cv::Mat& mat_rgb,
	const HDObject_VEC& local_objs,
	HDObject_VEC& and_objs)
{
	int cnt_rgb = cv::countNonZero(mat_rgb);
	if (cnt_rgb < 10)
	{
		return;
	}
	HDObject_VEC objs;
	copy_if(local_objs.begin(), local_objs.end(), back_inserter(objs),  [&](const auto& obj) {
		return obj.type == type;
		});

	auto itr = objs.begin();
	for (; itr != objs.end(); itr++)
	{
		const auto& obj = *itr;
		const auto& obj_ij_shape = obj.ij_shape;
		vector<vector<cv::Point>> line_vec;
		line_vec.push_back(obj_ij_shape);
		auto mat_hdmap = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
		cv::polylines(mat_hdmap, line_vec, false, cv::Scalar(type), 20);

		cv::Mat mat_common;
		cv::bitwise_and(mat_hdmap, mat_rgb, mat_common);
		int cnt_common = cv::countNonZero(mat_common);

		if (cnt_common > 10)
		{
			and_objs.push_back(obj);
		}
	}
	

	//total objects
	HDObject_VEC c_local_objs = local_objs;
	//changeCoordiante(c_local_objs);
	//changeCoordiante(and_objs);

	string file_na = "0_local_total_obj.csv";
	//saveLocalObjects(file_na, c_local_objs);

	//and objects
	string file_na2 = "0_local_obj.csv";
	//saveLocalObjects(file_na2, and_objs);
	//

}


void convLocalObjects2Image(const vector<cv::Point3d>& obj_shape,
	const vector<double>& camPose,
	vector<cv::Point3d>& reserve_shape,
	vector<cv::Point>& contour)
{
	for (int j = 0; j < obj_shape.size(); j++)
	{
		const auto& lp = obj_shape[j];
		cv::Point pp;
		double xyz[3] = { lp.x, lp.y, lp.z };
		double ij[2] = { -1,-1 };
		if (!OptimizeCeresWeighted::convertPoint3dTo2d(camPose, xyz, ij, false))
		{
			continue;
		}
		reserve_shape.push_back(lp);

		pp.x = ij[0];
		pp.y = ij[1];

		contour.push_back(pp);
	}
}


void calcBlurImage(int type, const map<int, LINE_VEC>& line_vec_map, cv::Mat& blur_mat)
{
	blur_mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));

	auto itr = line_vec_map.find(type);
	if (itr == line_vec_map.end())
	{
		return;
	}
	const auto& line_vec = itr->second;
	if (line_vec.size() == 0)
	{
		return;
	}

	if (type == OC_crosswalk || type == OC_t_sign)
	{
		cv::drawContours(blur_mat, line_vec, -1, cv::Scalar(255), cv::FILLED);
	}
	else
	{
		cv::polylines(blur_mat, line_vec, false, cv::Scalar(255), 20);
	}
}

void calLocalMapImage(const map<int, HDObject_VEC>& local_objs,
	const vector<double>& camPose, 
	map<int, HDObject_VEC>& reserve_local_objs,
	map<int, LINE_VEC>& type_line_vec_map)
{
	auto itr = local_objs.begin();
	for (; itr != local_objs.end(); itr++)
	{
		const auto& objs = itr->second;
		LINE_VEC lines;
		for_each(objs.begin(), objs.end(), [&](const auto& obj) {

			vector<cv::Point3d> reserve_shape;
			auto new_o = obj;
			new_o.shape.clear();
			convLocalObjects2Image(obj.shape, camPose, new_o.shape, new_o.ij_shape);
			if (new_o.ij_shape.size() > 1)
			{
				type_line_vec_map[itr->first].emplace_back(new_o.ij_shape);
				reserve_local_objs[itr->first].emplace_back(new_o);
			}
		});
		//move local map objects and project to calculate non-zero-and
	}
}


void moveLocalObjects(const HDObject_VEC& local_objs, 
	const vector<double>& pose,
	cv::Mat& blur_mat_hdmap)
{
	//translate vector
	blur_mat_hdmap = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width,
		CV_8UC1, cv::Scalar(0));

	map<int, vector<vector<cv::Point>>> type_line_vec_map;
	auto itr = local_objs.begin();
	for (; itr != local_objs.end(); itr++)
	{
		const auto& obj = *itr;
		vector<cv::Point> obj_ij_shape;
		vector<cv::Point3d> reserve_shape;
		convLocalObjects2Image(obj.shape, pose, reserve_shape, obj_ij_shape);
		auto& line_vec = type_line_vec_map[obj.type];
		line_vec.push_back(obj_ij_shape);
	}

	for_each(type_line_vec_map.begin(), type_line_vec_map.end(), [&](const auto& _FeatureID){
		int type = _FeatureID.first;
		const auto& line_vec = _FeatureID.second;
		cv::polylines(blur_mat_hdmap, line_vec, false, cv::Scalar(255), 20);
		});
}

float calcAndRatio(const cv::Mat& blur_mat_rgb,
	const cv::Mat& blur_mat_hdmap)
{
	cv::Mat dif;
	cv::bitwise_and(blur_mat_rgb, blur_mat_hdmap, dif);
	//	cv::imwrite("0_common.jpg", dif);
	int cnt_common = cv::countNonZero(dif);

	cv::bitwise_or(blur_mat_rgb, blur_mat_hdmap, dif);
	//	cv::imwrite("0_or.jpg", dif);
	int cnt_or = cv::countNonZero(dif);

	cv::absdiff(blur_mat_rgb, blur_mat_hdmap, dif);
	//	cv::imwrite("0_diff.jpg", dif);
	int cnt_dif = cv::countNonZero(dif);

	if (cnt_or == 0)
	{
		return 0;
	}
	float ratio_dif = cnt_dif * 1.0 / cnt_or;
	float ratio_common = cnt_common * 1.0 / cnt_or;
	return ratio_common;
}


void calcScoreDistribution(const vector<float>& scores,
	map<int, int>& bin_map)
{
	if (scores.size() == 0)
	{
		return;
	}
	float min_s = *min_element(scores.begin(), scores.end());
	float max_s = *max_element(scores.begin(), scores.end());
	int e = 1 / ((max_s - min_s) / 10);

	for (const auto& score : scores)
	{
		int bin_y = score * e;
		bin_map[bin_y] ++;
	}
}


float calcScoreEntropy(map<int, int>& bin_map)
{
	float H = 0;
	int total_cnt = 0;
	for_each(bin_map.begin(), bin_map.end(), [&total_cnt](const auto& _FeatureID) {
		total_cnt += _FeatureID.second;
		});

	auto itr = bin_map.begin();
	for (; itr != bin_map.end(); itr++)
	{
		float p = itr->second * 1.0 / total_cnt;
		float h = p * log2f(p);
		H += h;
	}

	H = -H;

	return H;
}




void getCommonMat(const cv::Mat& mat_rgb, 
	const cv::Mat& mat_hdmap, 
	cv::Mat& mat_common)
{
	auto	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));

	cv::bitwise_and(mat_rgb, mat_hdmap, mat_common);
	cv::morphologyEx(mat_common, mat_common, cv::MORPH_OPEN, kernel);
	int cnt_common = cv::countNonZero(mat_hdmap);
	return;
}



float CapabilityCalculator::getOccupancyRatio()
{

}

float calc2DFeatureDOP(const LINE_VEC& line_vec)
{
	if (line_vec.size() == 0)
	{
		return 1;
	}

	Eigen::MatrixXf A(line_vec.size() * 5, 2);

	int j = 0;
	for (const auto& pt_vec : line_vec)
	{
		cv::RotatedRect rr = cv::minAreaRect(pt_vec);
		cv::Point2f pts[5];
		rr.points(pts);

		pts[4] = rr.center;
		
		for (int i = 0; i < 5; i++)
		{
			cv::Point2f dp = pts[i] - cv::Point2f(CalibSpace::CX, CalibSpace::CY);
			float r = sqrt(dp.ddot(dp));
			float ax = dp.x * 1.0 / r;
			float ay = dp.y * 1.0 / r;
			A(j, 0) = ax;
			A(j, 1) = ay;

			j++;
		}
		
	}

	auto Q = A.transpose() * A;
	//cout << Q << endl;

	float f = sqrt(Q(0,0) + Q(1,1));
	//float f = sqrt(Q(0, 0) );
	f = 1.0 / f;

	//Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	//auto U = svd.matrixU();
	//auto V = svd.matrixV();
	//auto D_t = svd.singularValues();
	//cout << D_t << endl;
	//float x = D_t(0);
	//float y = D_t(1);
	return f;
}


float calc3DFeatureDOP(const HDObject_VEC& obj_vec)
{
	if (obj_vec.size() == 0)
	{
		return 1;
	}

	vector<cv::Point3f> para_vec;
	for (const auto& obj : obj_vec)
	{
		for (const auto& pt : obj.shape)
		{
			cv::Point3f dp(pt.x, pt.y, pt.z);
			float r = sqrt(pt.ddot(pt));
			float ax = dp.x * 1.0 / r;
			float ay = dp.y * 1.0 / r;
			float az = dp.z * 1.0 / r;
			para_vec.push_back(cv::Point3f(ax, ay, az));
		}
	}

	Eigen::MatrixXf A(para_vec.size(), 3);
	int i = 0;
	for_each(para_vec.begin(), para_vec.end(), [&](const auto& p) {
		A(i, 0) = p.x;
		A(i, 1) = p.y;
		A(i, 2) = p.z;
		i++;
		});
	auto Q = A.transpose() * A;
	//cout << Q << endl;

	float f = sqrt(Q(0, 0) + Q(1, 1) + Q(2, 2));
	//float f = sqrt(Q(0, 0) );
	f = 1.0 / f;

	//Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	//auto U = svd.matrixU();
	//auto V = svd.matrixV();
	//auto D_t = svd.singularValues();
	//cout << D_t << endl;
	//float x = D_t(0);
	//float y = D_t(1);
	//float z = D_t(2);
	return f;
}

float get2DFeatureDOP(cv::Mat t_mat)
{
	cv::normalize(t_mat, t_mat, 0, 255, cv::NORM_MINMAX);
	LINE_VEC line_vec;
	cv::findContours(t_mat, line_vec, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	float dop = calc2DFeatureDOP(line_vec);

	return dop;
}

//void CapabilityCalculator::setImageSize(int width, int hight)
//{
////	m_img_size.width = width;
////	m_img_size.height = hight;
//}

void calRatio(StatisticData_MAP& sd_map)
{
	set<int> data_type_set = { Data_COMMON, Data_IMAGE, Data_MAP };
	for (const auto& type : data_type_set)
	{
		int sum_2d = 0;
		int sum_3d = 0;
		for_each(sd_map.begin(), sd_map.end(), [&](const auto& _FeatureID){
			sum_2d += _FeatureID.second.occupancy_count_2d;
			sum_3d += _FeatureID.second.occupancy_count_3d; });

		for_each(sd_map.begin(), sd_map.end(), [&](auto& _FeatureID) {
			auto& sd = _FeatureID.second;
			if (sum_2d == 0)
			{
				sd.occupancy_ratio_2d = 0;
			}
			else
			{
				sd.occupancy_ratio_2d = sd.occupancy_count_2d * 1.0 / sum_2d;
			}
			if (sum_3d == 0)
			{
				sd.occupancy_ratio_3d = 0;
			}
			else
			{
				sd.occupancy_ratio_3d = sd.occupancy_count_3d * 1.0 / sum_3d;
			}
		});
	}

}






void saveStatisticDataWKT(const string& root_path,
	const CamPose& cp, const StatisticData_MAP& sd_map)
{
	ofstream os(root_path + "0_sd_geo.csv", ios::app);
	if (!os.is_open())
	{
		return;
	}
	os.setf(ios::fixed, ios::floatfield);
	os.precision(4);

	vector<string> attr_vec;
	attr_vec.push_back(cp.img_na);

	vector<cv::Point3d> shape;
	shape.push_back(cp.ins.point);

	outAttrCSVFields(os, attr_vec);
	outShpCSVFields(os, shape, 0);

	os.close();
	//	of.close();
}

inline double getRandomDouble(double min, double max)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(min, max);
	return dis(gen);
}

double getRandomInt(double min, double max)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(min, max);
	return dis(gen);
}

void getRandomTransform(Eigen::Vector3d& t, Eigen::Matrix3d& r)
{
	t[0] = getRandomDouble(-1, 1);
	t[1] = getRandomDouble(-1, 1);
	t[2] = getRandomDouble(-1, 1);
	Eigen::Vector3d euler;
	euler[0] = getRandomDouble(-0.1, 0.1);
	euler[1] = getRandomDouble(-0.1, 0.1);
	euler[2] = getRandomDouble(-0.1, 0.1);
	TransRotation::eigenEuler2RotationMatrixd(euler, r);
}

void getRandomIndices(int obj_sz,
	std::vector<int>& random_indices,
	std::vector<int>& difference)
{
	std::vector<int> indices(obj_sz);
	std::iota(indices.begin(), indices.end(), 0);
	int mod_num = getRandomInt(1, obj_sz);
	std::random_shuffle(indices.begin(), indices.end());
	random_indices = std::vector<int>(indices.begin(), indices.begin() + mod_num);

	std::set_difference(indices.begin(), indices.end(),
		random_indices.begin(), random_indices.end(),
		std::back_inserter(difference));
}

void generateSamples(const string& file_path,
	const vector<double>& pose,
	vector<vector<double>>& sample_poses)
{
	const string& file = file_path + "0_sd_similarity_samples.txt";
	if (_access(file.c_str(), 0) == 0)
	{
		ifstream fin(file);

		string line;
		while (getline(fin, line))
		{
			char* pEnd = ",";
			int i = strtol(line.c_str(), &pEnd, 10);
			double x = strtod(pEnd + 1, &pEnd);
			double y = strtod(pEnd + 1, &pEnd);
			double z = strtod(pEnd + 1, &pEnd);
			double ax = strtod(pEnd + 1, &pEnd);
			double ay = strtod(pEnd + 1, &pEnd);
			double az = strtod(pEnd + 1, nullptr);

			vector<double> t_pos = { x,y,z, ax, ay, az };
			for (int i = 0; i < 6; i++)
			{
				t_pos[i] = t_pos[i] + pose[i];
			}
			sample_poses.push_back(t_pos);
		}
		fin.close();
	}
	else
	{

		ofstream of(file, ios::trunc);

		int  numPoints = 100;
		for (int i = 0; i < numPoints; ++i)
		{
#if 0
			double x = getRandomDouble(-1, 1);
			double y = getRandomDouble(-1, 1);
			double z = getRandomDouble(-1, 1);
			
			double ax = getRandomDouble(-0.1, 0.1);
			double ay = getRandomDouble(-0.1, 0.1);
			double az = getRandomDouble(-0.1, 0.1);
#else
			double x = 0, y = 0, ax = 0, ay = 0, az = 0;

			double z = getRandomDouble(-10, 10);
#endif
			string line = "";
			HNString::FormatA(line, "%d,%.3f, %.3f,%.3f,%.3f,%.3f,%.3f",
				i, x, y, z, ax, ay, az);
			of << line << endl;

			vector<double> t_pos = { x,y,z, ax, ay, az };
			for (int i = 0; i < 6; i++)
			{
				t_pos[i] = t_pos[i] + pose[i];
			}
			sample_poses.push_back(t_pos);
		}
		of.close();
	}
}

void modifyObject(HDObject& obj)
{
	//定义随机变换矩阵
	Eigen::Vector3d t;
	Eigen::Matrix3d r;
	getRandomTransform(t, r);

	for_each(obj.shape.begin(), obj.shape.end(), [&](auto& p) {
		Eigen::Vector3d pt(p.x, p.y, p.z);
		//pt = r * pt + t; //不旋转了吧
		pt = pt + t;
		p.x = pt[0];
		p.y = pt[1];
		p.z = pt[2];
		});
}

void modifyImageLines(Point_VEC& contour)
{
	//定义随机变换矩阵
	int tx = getRandomInt(-50, 50); // Random translation along x-axis
	int ty = getRandomInt(-50, 50); // Random translation along y-axis
	cv::Mat translation_matrix = (cv::Mat_<float>(2, 3) << 1, 0, tx, 0, 1, ty);
	std::vector<cv::Point2f> points2f(contour.begin(), contour.end());

	// Rotation matrix
	//cv::Point2f center(0, 0);
	//float angle = getRandomDouble(-10,10);
	//cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
	//std::vector<cv::Point2f> translated_rotated_contour;
	//cv::transform(points2f, translated_rotated_contour, rotation_matrix);
	//cv::transform(translated_rotated_contour, translated_rotated_contour, translation_matrix);
	//contour = vector<cv::Point>(translated_rotated_contour.begin(), translated_rotated_contour.end());

	cv::transform(points2f, points2f, translation_matrix);
	contour = vector<cv::Point>(points2f.begin(), points2f.end());
}

void generateRandomModifiedMap(const map<int, HDObject_VEC>& org_local_objs,
	map<int, HDObject_VEC>& random_objs)
{
	//定义随机修改的要素类型
	vector<ObjectClassification> obj_type_vec{ OC_lane, OC_pole,OC_t_sign,  OC_crosswalk };
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	int subsetSize = 2;
	std::random_shuffle(obj_type_vec.begin(), obj_type_vec.end());
	std::vector<int> randomSubset_obj(obj_type_vec.begin(), obj_type_vec.begin() + subsetSize);

	map<int, HDObject_VEC> type_objs_map;
	for_each(org_local_objs.begin(), org_local_objs.end(), [&](const auto& obj) {
		auto need_to_modify = find_if(randomSubset_obj.begin(), randomSubset_obj.end(), [&](const auto& ootype)->bool {
			return ootype == obj.first;
			});
		if (need_to_modify == randomSubset_obj.end())
		{
			//不需要改的，直接存下来
			random_objs.insert(obj);
		}
		else
		{
			//要改的，下面改
			type_objs_map.insert(obj);
		}

		});

	vector<MODIFY_TYPE> mod_type_vec{ MD_ADDITION, MD_DELETION, MD_MODIFICATION };

	auto itr_type_objs = type_objs_map.begin();
	for (; itr_type_objs != type_objs_map.end(); itr_type_objs++)
	{
		const auto& type = itr_type_objs->first;
		const auto& objs = itr_type_objs->second;

		//定义随机变化的obj数量
		std::vector<int> random_indices;
		vector<int> reserve_indices;
		getRandomIndices(objs.size(), random_indices, reserve_indices);

		//把保留的都存起来
		for_each(reserve_indices.begin(), reserve_indices.end(), [&](const auto& idx) {
			random_objs[type].push_back(objs[idx]);
			});

		//要改的，下面改
		///////////////待修改的对象//////////////////
		for (int i = 0; i < random_indices.size(); i++)
		{
			const auto& obj = objs[i];
			//定义随机的修改类型
			int  mod_type_idx = getRandomInt(0, 3);//随机选取一种修改类型
			int mod_type = mod_type_vec[mod_type_idx];

			if (mod_type == MD_DELETION)
			{
				//删掉
				//不保存就可以了，不做什么
				continue;
			}
			if (mod_type == MD_ADDITION)
			{
				//把自己存起来
				random_objs[type].push_back(obj);
				//再变一个，存起来
			}
			//新增或者修改，都变一个存起来
			auto mod_obj = obj;
			modifyObject(mod_obj);
			random_objs[type].push_back(mod_obj);
		}
	}

	/*
	//定义随机修改的要素类型
	vector<ObjectClassification> obj_type_vec{ OC_lane, OC_pole,OC_t_sign,  OC_crosswalk };
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	int subsetSize = 2;
	std::random_shuffle(obj_type_vec.begin(), obj_type_vec.end());
	std::vector<int> randomSubset_obj(obj_type_vec.begin(), obj_type_vec.begin() + subsetSize);


	map<ObjectClassification, HDObject_VEC> type_objs_map;
	for_each(org_local_objs.begin(), org_local_objs.end(), [&](const auto& obj) {
		auto type = obj.first;

		auto need_to_modify = find_if(randomSubset_obj.begin(), randomSubset_obj.end(), [&](const auto& ootype)->bool {
			return ootype == type;
			});
		if (need_to_modify == randomSubset_obj.end())
		{
			//不需要改的，直接存下来
			random_objs.insert(obj);
		}
		else
		{
			//要改的，下面改
			type_objs_map[ObjectClassification(obj.type)].push_back(obj);
		}
		
		});

	vector<MODIFY_TYPE> mod_type_vec{ MD_ADDITION, MD_DELETION, MD_MODIFICATION };
	//vector<MODIFY_TYPE> mod_type_vec{ MD_ADDITION, MD_DELETION};
	auto itr_type_objs = type_objs_map.begin();
	for (; itr_type_objs != type_objs_map.end(); itr_type_objs++)
	{
		const auto& type = itr_type_objs->first;
		const auto& objs = itr_type_objs->second;

		//定义随机变化的obj数量
		std::vector<int> random_indices;
		vector<int> reserve_indices;
		getRandomIndices(objs.size(), random_indices, reserve_indices);

		//把保留的都存起来
		for_each(reserve_indices.begin(), reserve_indices.end(), [&](const auto& idx) {
			random_objs.push_back(objs[idx]);
			});

		//要改的，下面改
		///////////////待修改的对象//////////////////
		for (int i = 0; i < random_indices.size(); i++)
		{
			const HDObject& obj = objs[i];
			//定义随机的修改类型
			int  mod_type_idx = getRandomInt(0, 2);//随机选取一种修改类型
			int mod_type = mod_type_vec[mod_type_idx];

			if (mod_type == MD_DELETION)
			{
				//删掉
				//不保存就可以了，不做什么
				continue;
			}
			if (mod_type == MD_ADDITION)
			{
				//把自己存起来
				random_objs.push_back(obj);
				//再变一个，存起来
			}
			//新增或者修改，都变一个存起来
			HDObject mod_obj = obj;
			modifyObject(mod_obj);
			random_objs.push_back(mod_obj);
		}
	}*/

}

void generateRandomModifiedImage(const map<int, LINE_VEC>& lines_map_rgb,
	map<int, LINE_VEC>& random_objs)
{
	//定义随机修改的要素类型
	vector<ObjectClassification> obj_type_vec{ OC_lane, OC_pole,OC_t_sign,  OC_crosswalk };
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	int subsetSize = 2;
	std::random_shuffle(obj_type_vec.begin(), obj_type_vec.end());
	std::vector<int> randomSubset_obj(obj_type_vec.begin(), obj_type_vec.begin() + subsetSize);

	map<int, LINE_VEC> type_objs_map;
	for_each(lines_map_rgb.begin(), lines_map_rgb.end(), [&](const auto& obj) {
		auto need_to_modify = find_if(randomSubset_obj.begin(), randomSubset_obj.end(), [&](const auto& ootype)->bool {
			return ootype == obj.first;
			});
		if (need_to_modify == randomSubset_obj.end())
		{
			//不需要改的，直接存下来
			random_objs.insert(obj);
		}
		else
		{
			//要改的，下面改
			type_objs_map.insert(obj);
		}

		});

	vector<MODIFY_TYPE> mod_type_vec{ MD_ADDITION, MD_DELETION, MD_MODIFICATION };

	auto itr_type_objs = type_objs_map.begin();
	for (; itr_type_objs != type_objs_map.end(); itr_type_objs++)
	{
		const auto& type = itr_type_objs->first;
		const auto& objs = itr_type_objs->second;

		//定义随机变化的obj数量
		std::vector<int> random_indices;
		vector<int> reserve_indices;
		getRandomIndices(objs.size(), random_indices, reserve_indices);

		//把保留的都存起来
		for_each(reserve_indices.begin(), reserve_indices.end(), [&](const auto& idx) {
			random_objs[type].push_back(objs[idx]);
			});

		//要改的，下面改
		///////////////待修改的对象//////////////////
		for (int i = 0; i < random_indices.size(); i++)
		{
			const auto& obj = objs[i];
			//定义随机的修改类型
			int  mod_type_idx = getRandomInt(0, 3);//随机选取一种修改类型
			int mod_type = mod_type_vec[mod_type_idx];

			if (mod_type == MD_DELETION)
			{
				//删掉
				//不保存就可以了，不做什么
				continue;
			}
			if (mod_type == MD_ADDITION)
			{
				//把自己存起来
				random_objs[type].push_back(obj);
				//再变一个，存起来
			}
			//新增或者修改，都变一个存起来
			auto mod_obj = obj;
			modifyImageLines(mod_obj);
			random_objs[type].push_back(mod_obj);
		}
	}

}

void saveHDMapInCameraFrame(const string& file_na, const map<int, HDObject_VEC>& obj_map)
{
	ofstream os(file_na, ios::trunc);
	if (!os.is_open())
	{
		return;
	}
	os.setf(ios::fixed, ios::floatfield);
	os.precision(4);
	auto itr_type = obj_map.begin();
	for (; itr_type!= obj_map.end(); itr_type++)
	{
		const auto& obj_vec = itr_type->second;
		for (int i = 0; i < obj_vec.size(); i++)
		{
			const auto& obj = obj_vec[i];
			if (obj.shape.size() < 2)
			{
				continue;
			}
			vector<string> attr_vec;
			attr_vec.push_back(obj.obj_id);
			attr_vec.push_back(to_string(obj.type));
			outAttrCSVFields(os, attr_vec);
			auto shape = obj.shape;
			for_each(shape.begin(), shape.end(), [](auto& pt) {
				double h = -pt.y;
				pt.y = pt.z;
				pt.z = h;
				});

			//outShpCSVFields(os, obj.shape, 1);
			outShpCSVFields(os, shape, 1);
		}
	}
	
	os.close();
}


size_t readHDMapInCameraFrame(const string& file_na, map<int, HDObject_VEC>& obj_vec)
{
	if (_access(file_na.c_str(), 0) != 0)
	{
		return 0;
	}

	ifstream is(file_na, fstream::in);
	is.setf(ios::fixed, ios::floatfield);
	is.precision(4);

	string line = "";
	while (getline(is, line))
	{
		vector<string> attr_vec;
		if (line.size() == 0)
		{
			continue;
		}
		inAttrCSVFields(line, attr_vec);
		if (attr_vec.size() == 0)
		{
			continue;
		}
		
		vector<cv::Point3d> shape;
		inShpCSVFields(attr_vec.back(), shape, 1);

		HDObject info;
		info.obj_id = attr_vec[0];
		info.type = atoi(attr_vec[1].c_str());
		
		for_each(shape.begin(), shape.end(), [](auto& pt) {
			double h = pt.y;
			pt.y = -pt.z;
			pt.z = h;
			});


		info.shape.swap(shape);
		obj_vec[info.type].emplace_back(info);
	}

	is.close();
	return obj_vec.size();
}

void generateMap(const string& file_na, const map<int, HDObject_VEC>& org_local_objs,
	map<int, HDObject_VEC>& random_objs)
{
	
	if (_access(file_na.c_str(), 0) == 0)
	{
		readHDMapInCameraFrame(file_na, random_objs);
	}
	else
	{
		generateRandomModifiedMap(org_local_objs, random_objs);

		saveHDMapInCameraFrame(file_na, random_objs);
	}

}

void setColorMap(map<int, cv::Scalar>& type_color_ref)
{

//	type_color_ref.insert(make_pair(OC_road, cv::Scalar(128, 64, 128)));//road
	type_color_ref.insert(make_pair(OC_lane, cv::Scalar(0, 0, 255)));//lane
//	type_color_ref.insert(make_pair(OC_curbstone, cv::Scalar(255, 0, 255)));//curb

	type_color_ref.insert(make_pair(OC_pole, cv::Scalar(255, 128, 128)));//pole
	type_color_ref.insert(make_pair(OC_t_sign, cv::Scalar(0, 128, 255)));//traffic sign
	type_color_ref.insert(make_pair(OC_crosswalk, cv::Scalar(255, 128, 255)));//crosswalk
}

void saveImageContours(const cv::Mat& img,
	const string& img__na,
	const map<int, LINE_VEC>& lines_map_random,
	const map<int, cv::Scalar>& type_color_ref)
{

	cv::Mat mat_ply = img.clone();
	for_each(lines_map_random.begin(), lines_map_random.end(), [&](const auto& l) {
		int type = l.first;

		cv::Scalar clr(0, 0, 0);
		const auto& find_clr = type_color_ref.find(type);
		if (find_clr != type_color_ref.end())
		{
			clr = find_clr->second;
		}

		cv::polylines(mat_ply, l.second, false, clr, 5);

		});

	cv::imwrite(img__na, mat_ply);
}

void readImageContours(const string& file_na, 
	const map<int, cv::Scalar>& type_color_ref,
	map<int, LINE_VEC>& lines_map_random)
{
	cv::Mat mat_ply = cv::imread(file_na);
	auto itr = type_color_ref.begin();

	for (; itr != type_color_ref.end(); itr++)
	{
		int type = itr->first;
		const auto& color = itr->second;
		cv::Mat img_binary;
		cv::inRange(mat_ply, color, color, img_binary);
		LINE_VEC contours;
		cv::findContours(img_binary, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		if (contours.size() == 0)
		{
			continue;
		}
		lines_map_random.insert(make_pair(type, contours));
	}
}

void generateImages(const string& file_na, const cv::Mat& img,
	const map<int, cv::Scalar>& type_color_ref,
	const map<int, LINE_VEC>& lines_map_rgb,
	map<int, LINE_VEC>& lines_map_random)
{

	if (_access(file_na.c_str(), 0) == 0)
	{
		readImageContours(file_na, type_color_ref, lines_map_random);
	}
	else
	{
		generateRandomModifiedImage(lines_map_rgb, lines_map_random);

		saveImageContours(img, file_na, lines_map_random, type_color_ref);
	}

}

double calculateEntropy(const std::vector<int>& values) 
{
	int n = values.size();

	std::vector<int> uniqueElements(values.begin(), values.end());
	std::sort(uniqueElements.begin(), uniqueElements.end());
	uniqueElements.erase(std::unique(uniqueElements.begin(), uniqueElements.end()), uniqueElements.end());

	double entropy = 0.0;
	for (int element : uniqueElements) 
	{
		int count = std::count(values.begin(), values.end(), element);
		double probability = static_cast<double>(count) / n;
		entropy -= probability * std::log2(probability);
	}

	return entropy;
}

float calcLocalSimilarity(const cv::Mat& blur_mat_rgb,
	const HDObject_VEC& local_objs,
	const vector<vector<double>>& sample_poses)
{
	if (local_objs.size() == 0)
	{
		return 0;
	}

	vector<int> score_vec;

	int idx = 0;
	auto itr_pos = sample_poses.begin();
	for (; itr_pos != sample_poses.end(); itr_pos++, idx++)
	{
		const auto& pose = *itr_pos;

		cv::Mat blur_mat_hdmap;
		moveLocalObjects(local_objs, pose, blur_mat_hdmap);

		float ar = calcAndRatio(blur_mat_rgb, blur_mat_hdmap);
		int a = ar / 0.05 + 1;
		score_vec.push_back(a);
	}

	float H = calculateEntropy(score_vec);
	return H;
}

void CapabilityCalculator::oneIteration(const map<int, LINE_VEC>& lines_map_rgb,
	const map<int, HDObject_VEC>& org_local_objs,
	const CamPose& cp,
	const string& file_path)
{
	StatisticData_MAP sd_map;
	set<ObjectClassification> obj_type_set = {OC_lane, OC_pole, OC_t_sign};

	//copy one piece, some attribute may be changed next
	map<int, HDObject_VEC> reserve_local_objs; 
	/*for_each(org_local_objs.begin(), org_local_objs.end(), [&](const auto& _pair) {
		const auto& objs = _pair.second;
		HDObject_VEC& reserve_objs = reserve_local_objs[_pair.first];
		for_each(objs.begin(), objs.end(), [&](const auto& obj) {
			vector<cv::Point3d> pts;
			copy_if(obj.shape.begin(), obj.shape.end(), back_inserter(pts), [](const auto& pt)->bool {
				return pt.z > 0;
				});
			if (pts.size() > 0)
			{
				HDObject ooo = obj;
				ooo.shape = pts;
				reserve_objs.emplace_back(ooo);
			}
			});

		});*/
	map<int, LINE_VEC> lines_map_hd;
	calLocalMapImage(org_local_objs, cp.camPose, reserve_local_objs, lines_map_hd);

	//sample poses for similarity calculation
	//vector<vector<double>> sample_poses;
	if (m_sample_poses.size() == 0)
	{
		generateSamples(file_path, cp.camPose, m_sample_poses);
	}

	auto itr_type = obj_type_set.begin();
	for (; itr_type != obj_type_set.end(); itr_type++)
	{
		auto type = *itr_type;
		SpatialFeature& sdm = sd_map[type];
		sdm.fid = type;

		auto find_obj = reserve_local_objs.find(type);
		if (find_obj == reserve_local_objs.end())
		{
			continue;
		}
		const HDObject_VEC& local_objs = find_obj->second;
		if (local_objs.size() == 0)
		{
			continue;
		}
		//rgb image
		cv::Mat blur_mat_rgb;
		calcBlurImage(type, lines_map_rgb, blur_mat_rgb);

		cv::Mat blur_mat_hdmap;
		calcBlurImage(type, lines_map_hd, blur_mat_hdmap);

		//(1) image
		
		
		//FeatureID para = FeatureID(type, 0);
		sdm.occupancy_count_2d = cv::countNonZero(blur_mat_rgb);
		sdm.dop_2d = get2DFeatureDOP(blur_mat_rgb);
		
		//(2) map

		sdm.occupancy_count_3d = cv::countNonZero(blur_mat_hdmap);
		sdm.dop_3d = calc3DFeatureDOP(local_objs);
		sdm.normal_entropy = calcFeatureNormalEntropy(local_objs);
		//move map to evaluate the similarity
		sdm.local_similarity = calcLocalSimilarity(blur_mat_rgb, local_objs, m_sample_poses);
		//(3) common
		//first get the common-objects , analyze the distribution of and-objects
		/*SpatialFeature sdc;
		para = FeatureID(type, Data_COMMON);
		cv::Mat common_mat;
		getCommonMat(blur_mat_rgb, blur_mat_hdmap, common_mat);
		sdc.occupancy_count = cv::countNonZero(common_mat);
		sdc.dop_2d = get2DFeatureDOP(common_mat);
		sdc.fid = para;
		HDObject_VEC and_objs;
		getCommonObjects(type, blur_mat_rgb, local_objs, and_objs);
		sdc.normal_entropy = calcFeatureNormalEntropy(and_objs);
		sdc.dop_3d = calc3DFeatureDOP(and_objs);
		sd_map[para] = sdc;*/
	}

	calRatio(sd_map);

	m_img_sd_map.insert(make_pair(cp.img_na, sd_map));

}

size_t readSpatialDistributionDataCSV(const string& root_path,
	map<string, StatisticData_MAP>& img_sd_map)
{
	const string& file_path = root_path + "0_sd_data.csv";
	ifstream of(file_path);
	if (!of.is_open())
	{
		return 0;
	}

	string line = "";
	// skip first line names
	getline(of, line);

	while (getline(of, line))
	{
		vector<string> attr_vec;
		if (line.size() == 0)
		{
			continue;
		}
		SpatialFeature sd;
		vector<string> str_vec;
		HNString::SplitA(line, str_vec, ",");
		int i = 0;
		string img_na = str_vec[i++];
		int idx = atoi(str_vec[i++].c_str());
		int data_type = DataTye(atoi(str_vec[i++].c_str()));
		sd.fid = ObjectClassification(atoi(str_vec[i++].c_str()));
		sd.occupancy_count_2d = atoi(str_vec[i++].c_str());
		sd.occupancy_ratio_2d = atof(str_vec[i++].c_str());
		sd.dop_2d = atof(str_vec[i++].c_str());
		sd.occupancy_count_3d = atof(str_vec[i++].c_str());
		sd.occupancy_ratio_3d = atof(str_vec[i++].c_str());
		sd.dop_3d = atof(str_vec[i++].c_str());
		sd.normal_entropy = atof(str_vec[i++].c_str());
		sd.local_similarity = atof(str_vec[i++].c_str());
		
		StatisticData_MAP& sd_map = img_sd_map[img_na];
		sd_map[sd.fid] = sd;
	}
	of.close();
	return img_sd_map.size();
}

void CapabilityCalculator::saveSpatialDistributionDataCSV(const string& root_path)
{
	if (m_img_sd_map.size() == 0)
	{
		return;
	}
	const string& file_path = root_path + "0_sd_data.csv";
	ofstream of(file_path, ios::out | ios::trunc);
	if (!of.is_open())
	{
		return;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);

	string head_line = "img_na,idx,data_type,feature_type,\
occupancy_count_2d,occupancy_ratio_2d,dop_2d,\
occupancy_count_3d,occupancy_ratio_3d,dop_3d,angular_entropy,local_similarity";
	of << head_line << endl;
	string line = "";

	int idx = 0;
	auto itr_img = m_img_sd_map.begin();
	for (; itr_img != m_img_sd_map.end(); itr_img++, idx++)
	{
		const auto& img_na = itr_img->first;
		const StatisticData_MAP& sd_map = itr_img->second;
		auto itr_sd = sd_map.begin();
		for (; itr_sd != sd_map.end(); itr_sd++)
		{
			const auto& sd = itr_sd->second;
			string line = "";
			HNString::FormatA(line, "%s,%d,%d,%d,%d,%.2f,%.3f,%d,%.2f,%.3f,%.3f,%.3f",
				img_na.c_str(),
				idx,
				//sd.fid.data_type,
				0,
				sd.fid,
				sd.occupancy_count_2d,
				sd.occupancy_ratio_2d,
				sd.dop_2d,
				sd.occupancy_count_3d,
				sd.occupancy_ratio_3d,
				sd.dop_3d,
				sd.normal_entropy,
				sd.local_similarity);

			of << line << endl;
		}
	}


	of.close();
}

bool CapabilityCalculator::getFrameSpatialDistribution(const string& root_path,
	const string& img_na,
	StatisticData_MAP& sf)
{
	if (m_img_sd_map.size() == 0)
	{
		readSpatialDistributionDataCSV(root_path, m_img_sd_map);
	}
	if (m_img_sd_map.size() == 0)
	{
		return false;
	}
	auto find_sd = m_img_sd_map.find(img_na);
	if (find_sd == m_img_sd_map.end())
	{
		return false;
	}
	sf = find_sd->second;
	return true;
}

void CapabilityCalculator::generateDistributionSample(const map<int, LINE_VEC>& lines_map_rgb,
	const map<int, HDObject_VEC>& org_local_objs,
	CamPose cp,
	map<int, LINE_VEC>& random_lines_map,
	map<int, HDObject_VEC>& random_objs,
	const string& file_path)
{
	random_lines_map.clear();
	random_objs.clear();

	map<int, cv::Scalar> type_color_ref;
	setColorMap(type_color_ref);

	string file_na = file_path + "hd\\org_hdobj.csv";
	if (cp.idx == 0)
	{
		saveHDMapInCameraFrame(file_na, org_local_objs);
	}
	
	//random map
	file_na = file_path + "\\hd\\" + to_string(cp.idx) + "_sample_hdobj.csv";
	generateMap(file_na, org_local_objs, random_objs);

	//random image
	//string img_path = "F:\\0hn\\4data\\0image_match\\arg2\\image\\2\\" + cp.img_na + ".jpg";
	//cv::Mat img = cv::imread(img_path);
	cv::Mat img = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC3, cv::Scalar(0, 0, 0));
	string img_na = file_path + "\\image\\" + to_string(cp.idx) + "_" + cp.img_na + ".png";
	generateImages(img_na, img, type_color_ref, lines_map_rgb, random_lines_map);

	//calculate
	
	oneIteration(random_lines_map, random_objs, cp, file_path);
	
	//saveStatisticDataWKT(file_path, cp, sd_map);

	return;
}


void CapabilityCalculator::saveRegistPoses(const CamPose& cp, const string& root_path,
	const vector<double>& err)
{
	const string& file_path = root_path + "\\0_sd_error.csv";
	ofstream of(file_path, ios::out | ios::app);
	if (!of.is_open())
	{
		return;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);

	// write 2 lines
	string line = "";
	HNString::FormatA(line, "%s,%d,\
		%.2f,%.2f,%.2f,\
		%.2f,%.2f,%.2f",
		cp.img_na.c_str(),
		cp.idx,
		err[0],
		err[1],
		err[2],
		err[3],
		err[4],
		err[5]);
	of << line << endl;

	of.close();
}