// Calib.cpp : 定义控制台应用程序的入口点。
// for std
#include <iostream>
#include <io.h>
#include <direct.h>
// for opencv
#include "TestCalib.h"
#include "CenterLine.h"
#include "HNMath/GeometricAlgorithm2.h"
#include "icp/myicp.h"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/ximgproc.hpp"
//#include "opencv2/line_descriptor.hpp"
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>

//#include <boost/concept_check.hpp>

#include <iostream>
#include<fstream>
#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef CV_CONTRIB
#include <opencv2/ximgproc.hpp>
#endif

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/eigen.hpp>

#include "DataIO.h"

//#include "Common.h"
//#include "ManualCalib.h"
#include "DataManager/XmlConfig.h"
#include "DataManager/XmlLabel.h"


///////////for 3d/////////////
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
//////////end

#include "OptimizeWeighted.h"

using namespace std;
using namespace cv;

cv::flann::Index global_kdtree;
map<int, cv::flann::Index> global_kdtree_map;

//cv::flann::Index LANE_TREE;
//cv::flann::Index POLE_TREE;

double calcD(int x, int cycle)
{
	x %= cycle;

	int c = 2;
	double y = 10 * (x - c) * (x - c) + 20;

	//int c = 1;
	//double y = 10 * (x - c) * (x - c) + 10;

	return y;
}

bool searchNeareatPoint(const vector<cv::Point>& d2_pts, cv::flann::Index& kdtree,
	const cv::Point& d2_pt, int& pixel_idx, double& nearest_dist, double D)
{
	unsigned knn = 1;//用于设置返回邻近点的个数
	vector<float> vecQuery(2);//存放 查询点 的容器（本例都是vector类型）
	cv::flann::SearchParams params(-1);//设置knnSearch搜索参数

	/**KD树knn查询**/
	vecQuery[0] = d2_pt.x; //查询点x坐标
	vecQuery[1] = d2_pt.y; //查询点y坐标

	vector<int> vecIndex/*(knn)*/;//存放返回的点索引
	vector<float> vecDist/*(knn)*/;//存放距离

	kdtree.knnSearch(vecQuery, vecIndex, vecDist, knn, params);
	pixel_idx = vecIndex[0];
	if (pixel_idx == 0 && vecDist[0] < 0.01)
	{
//		LOG(INFO) << "knn can't search nearest point....";
		return false;
	}
	//cv::Point search_pt = d2_pts[pixel_idx];

	nearest_dist = sqrt(vecDist[0]);
	if (nearest_dist > D)
	{
		return false;
	}
	return true;
}

bool searchNeareatYPoint(const vector<cv::Point>& d2_pts, cv::flann::Index& kdtree,
	const cv::Point& d2_pt, int& pixel_idx, double& nearest_dist, double D)
{
	unsigned queryNum = 5;//用于设置返回邻近点的个数
	vector<float> vecQuery(2);
	vecQuery[0] = d2_pt.x; //查询点x坐标
	vecQuery[1] = d2_pt.y; //查询点y坐标
	cv::flann::SearchParams params(-1);//设置knnSearch搜索参数

	vector<int> vecIndex(queryNum, -1);//存放返回的点索引
	vector<float> vecDist(queryNum, 0);//存放距离

//	kdtree.knnSearch(vecQuery, vecIndex, vecDist, queryNum, params);
	kdtree.radiusSearch(vecQuery, vecIndex, vecDist, 100, queryNum);
	map<int, int> dy_idx_map;
	for (int i = 0; i < vecIndex.size(); i++)
	{
		pixel_idx = vecIndex[i];
		if (pixel_idx == -1)
		{
			continue;
		}
		cv::Point search_pt = d2_pts[pixel_idx];
		int dy = abs(search_pt.y - d2_pt.y);
		dy_idx_map[dy] = pixel_idx;
	}
	if (dy_idx_map.size() > 0)
	{
		pixel_idx = dy_idx_map.begin()->second;
	}
	return dy_idx_map.size() > 0;
}

cv::Scalar BGR2HSV(const cv::Scalar& bgr)
{
	

	int b = bgr[0], g = bgr[1], r = bgr[2];
	int h, s, v;

	float vmin, diff;

	v = vmin = r;
	if (v < g) v = g;
	if (v < b) v = b;       // v = max(b, g, r)  
	if (vmin > g) vmin = g;
	if (vmin > b) vmin = b;

	diff = v - vmin;
	s = diff / (float)(fabs(v) + FLT_EPSILON);  // s = 1 - min/max  
	diff = (float)(60. / (diff + FLT_EPSILON));
	if (v == r)
		h = (g - b) * diff;
	else if (v == g)
		h = (b - r) * diff + 120.f;
	else
		h = (r - g) * diff + 240.f;

	if (h < 0) h += 360.f;  // h 求值  

	return cv::Scalar(h,s,v);
}


void getTypeInColorMap(map<int, cv::Scalar>& type_color_ref)
{
	/*
	name | id | trainId | category | categoryId | hasInstances | ignoreInEval | color
		--------------------------------------------------------------------------------------------------
		unlabeled | 0 | 255 | void | 0 | 0 | 1 | (0, 0, 0)
		ego vehicle | 1 | 255 | void | 0 | 0 | 1 | (0, 0, 0)
		rectification border | 2 | 255 | void | 0 | 0 | 1 | (0, 0, 0)
		out of roi | 3 | 255 | void | 0 | 0 | 1 | (0, 0, 0)
		static | 4 | 255 | void | 0 | 0 | 1 | (0, 0, 0)
		dynamic | 5 | 255 | void | 0 | 0 | 1 | (111, 74, 0)
		ground | 6 | 255 | void | 0 | 0 | 1 | (81, 0, 81)
		road | 7 | 0 | flat | 1 | 0 | 0 | (128, 64, 128)
		sidewalk | 8 | 1 | flat | 1 | 0 | 0 | (244, 35, 232)
		parking | 9 | 255 | flat | 1 | 0 | 1 | (250, 170, 160)
		rail track | 10 | 255 | flat | 1 | 0 | 1 | (230, 150, 140)
		building | 11 | 2 | construction | 2 | 0 | 0 | (70, 70, 70)
		wall | 12 | 3 | construction | 2 | 0 | 0 | (102, 102, 156)
		fence | 13 | 4 | construction | 2 | 0 | 0 | (190, 153, 153)
		guard rail | 14 | 255 | construction | 2 | 0 | 1 | (180, 165, 180)
		bridge | 15 | 255 | construction | 2 | 0 | 1 | (150, 100, 100)
		tunnel | 16 | 255 | construction | 2 | 0 | 1 | (150, 120, 90)
		pole | 17 | 5 | object | 3 | 0 | 0 | (153, 153, 153)
		polegroup | 18 | 255 | object | 3 | 0 | 1 | (153, 153, 153)
		traffic light | 19 | 6 | object | 3 | 0 | 0 | (250, 170, 30)
		traffic sign | 20 | 7 | object | 3 | 0 | 0 | (220, 220, 0)
		vegetation | 21 | 8 | nature | 4 | 0 | 0 | (107, 142, 35)
		terrain | 22 | 9 | nature | 4 | 0 | 0 | (152, 251, 152)
		sky | 23 | 10 | sky | 5 | 0 | 0 | (70, 130, 180)
		person | 24 | 11 | human | 6 | 1 | 0 | (220, 20, 60)
		rider | 25 | 12 | human | 6 | 1 | 0 | (255, 0, 0)
		car | 26 | 13 | vehicle | 7 | 1 | 0 | (0, 0, 142)
		truck | 27 | 14 | vehicle | 7 | 1 | 0 | (0, 0, 70)
		bus | 28 | 15 | vehicle | 7 | 1 | 0 | (0, 60, 100)
		caravan | 29 | 255 | vehicle | 7 | 1 | 1 | (0, 0, 90)
		trailer | 30 | 255 | vehicle | 7 | 1 | 1 | (0, 0, 110)
		train | 31 | 16 | vehicle | 7 | 1 | 0 | (0, 80, 100)
		motorcycle | 32 | 17 | vehicle | 7 | 1 | 0 | (0, 0, 230)
		bicycle | 33 | 18 | vehicle | 7 | 1 | 0 | (119, 11, 32)
		license plate | -1 | -1 | vehicle | 7 | 0 | 1 | (0, 0, 142)

		————————————————
		版权声明：本文为CSDN博主「一只tobey」的原创文章，遵循CC 4.0 BY - SA版权协议，转载请附上原文出处链接及本声明。
		原文链接：https ://blog.csdn.net/zz2230633069/article/details/84591532*/

//	type_color_ref.insert(make_pair(0, cv::Scalar(0, 0, 0)));//unlabel
//	type_color_ref.insert(make_pair(OC_road, cv::Scalar(128, 64, 128)));//road
//	type_color_ref.insert(make_pair(OC_pole, cv::Scalar(153, 153, 153)));//pole
//	type_color_ref.insert(make_pair(OC_t_sign, cv::Scalar(220, 220, 0)));//traffic sign

//	type_color_ref.insert(make_pair(OC_pole, BGR2HSV(cv::Scalar(153, 153, 153))));//pole
//	type_color_ref.insert(make_pair(OC_t_sign, BGR2HSV(cv::Scalar(220, 220, 0))));//traffic sign

	type_color_ref.insert(make_pair(OC_pole, cv::Scalar(153, 153, 153)));//pole
	type_color_ref.insert(make_pair(OC_t_sign, cv::Scalar(220, 220, 0)));//traffic sign
	type_color_ref.insert(make_pair(OC_crosswalk, cv::Scalar(66,69, 176)));//cross walk
}

Calib::Calib()
{
	m_color_map.clear();
	getTypeInColorMap(m_color_map);
}

Calib::~Calib()
{
}


void Calib::process(vector<double>& camPose, SolveMethod type)
{
	string folder_path = "D:/0huinian/localcode/kitti/2011_09_26_drive_0027/2011_09_26/2011_09_26_drive_0027_sync/";
	string img_path = folder_path + "image_00/data/";

	vector<string> img_files;
//	DataIO::getFiles(img_path, "*.png", img_files);

	CalibSpace::initInversePerspectiveMappingMat(MY_CONFIG.data_para.corners,
		CalibSpace::warpmat_src2ipm,
		CalibSpace::warpmat_ipm2src);

	auto itr_img = img_files.begin();
	for (; itr_img != img_files.end(); itr_img++)
	{
		string cur_img_path = *itr_img;
		string img_na = getFileName(cur_img_path);
		cv::Mat camera_img = cv::imread(*itr_img);
		////******************extract lines in image********************* 
		vector<vector<cv::Point>> line_vec;
		extractImageRoadLines(folder_path, img_na, camera_img, line_vec);

		////******************extract lines in pointcloud********************* 
		vector<vector<cv::Point3f>> xyz_line_vec;
		string xyz_ply_file_path = folder_path + "ply/";
		read3dVectorData(xyz_ply_file_path, xyz_line_vec);

		vector<cv::Point3f> xyz_vec;
		getValid3dPoints(xyz_line_vec, xyz_vec);

		cv::Mat blur_mat(CalibSpace::IMG_HEIGHT, CalibSpace::IMG_WIDTH, CV_8UC1, cv::Scalar(0));
		cv::drawContours(blur_mat, line_vec, -1, cv::Scalar(255));
		cv::GaussianBlur(blur_mat, blur_mat, cv::Size(7, 7), 0, 0);
		cv::threshold(blur_mat, blur_mat, 1, 1, cv::THRESH_BINARY);

		if (type == ICP)
		{
			////******************regist 3d && 2d lines********************* 
		//	regist3D2DLines(xyz_line_vec, line_vec, camPose);
			vector<cv::Point> ij_vec;
			getValid2dPoints(line_vec, ij_vec);

			//OptimizeCeresWeighted::findBestInitialParaByExhaustiveGrid(xyz_vec, blur_mat, camPose);
			iterateClosestPoint2d3d(xyz_vec, ij_vec, camPose);

			//OptimizeCeresWeighted::getPerturbationProbability(xyz_vec, blur_mat, camPose, true);
			//LOG(INFO) << "*************结果****************";
			//LOG(INFO) << "estimated Xs,Ys,Zs,omega,pho,kappa = ";
			//for (int i = 0; i < 6; i++)
			//{
			//	LOG(INFO) << camPose[i];
			//}
		}
		else if (type == JC)
		{
			////******************regist 3d && 2d jc********************* 
	//		regist3D2DJC(xyz_line_vec, line_vec, camPose);
	//		iterateJC(xyz_vec, line_vec, camPose);
		}
		else
		{

		}
		//OptimizeCeresWeighted::getPerturbationProbability(xyz_vec, blur_mat, camPose, true);
		LOG(INFO) << "*************结果****************";
		LOG(INFO) << "estimated Xs,Ys,Zs,omega,pho,kappa = ";
		for (int i = 0; i < 6; i++)
		{
			LOG(INFO) << camPose[i];
		}
		////******************project pointcloud to image********************* 
		string lidar_path = folder_path + "velodyne_points/txt/" + img_na + ".txt";
		vector<CalibSpace::PointXYZI> pc;
	//	DataIO::readPointCloud(lidar_path, pc);

		projectPointCloud2Image(pc, camPose, camera_img);
		string out_path = folder_path + "image_00/proj/" + img_na + ".png";
		cv::imwrite(out_path, camera_img);
	}
}

void Calib::calcBlurImage(const vector<vector<cv::Point>>& line_vec, cv::Mat& blur_mat)
{
	blur_mat = cv::Mat(CalibSpace::IMG_HEIGHT, CalibSpace::IMG_WIDTH, CV_8UC1, cv::Scalar(0));
	cv::drawContours(blur_mat, line_vec, -1, cv::Scalar(255));
	cv::GaussianBlur(blur_mat, blur_mat, cv::Size(7, 7), 0, 0);
	cv::threshold(blur_mat, blur_mat, 1, 1, cv::THRESH_BINARY);
}

void Calib::calcBlurImage(const map<int, LINE_VEC>& line_vec_map, cv::Mat& blur_mat)
{
	blur_mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
	//当前是逆透视投影，固定尺寸800*1280
	//blur_mat = cv::Mat(1280,800, CV_8UC1, cv::Scalar(0));
	auto itr = line_vec_map.begin();
	for (; itr != line_vec_map.end(); itr++)
	{
		const auto& type = itr->first;
		const auto& line_vec = itr->second;
		cv::polylines(blur_mat, line_vec, false, cv::Scalar(255), 20);
	//	cv::drawContours(blur_mat, line_vec, -1, cv::Scalar(type), -1);
	//	cv::drawContours(blur_mat, line_vec, -1, cv::Scalar(1), -1);
	}
//	cv::GaussianBlur(blur_mat, blur_mat, cv::Size(7, 7), 0, 0);
//	cv::threshold(blur_mat, blur_mat, 12, 1, cv::THRESH_BINARY);

// 	auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
// 	cv::dilate(blur_mat, blur_mat, kernel);


	//cv::imshow("dilate", blur_mat);
	//cv::waitKey(0);

	/*itr = line_vec_map.begin();
	for (; itr != line_vec_map.end(); itr++)
	{
		const auto& type = itr->first;
		const auto& line_vec = itr->second;
		for_each(line_vec.begin(), line_vec.end(), [&](const auto& line)
		{
			for_each(line.begin(), line.end(), [&](const auto& p)
			{
				int pp = blur_mat.at<uchar>(p.y, p.x);
				LOG(INFO) << pp;
			});
		});
	}*/
	
}

void Calib::calcDistanceTransformImage(const map<int, LINE_VEC>& line_vec_map, 
cv::Mat& blur_mat)
{
	cv::Mat mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));

	auto itr = line_vec_map.begin();
	for (; itr != line_vec_map.end(); itr++)
	{
		const auto& type = itr->first;
		const auto& line_vec = itr->second;
	//	cv::polylines(mat, line_vec, false, cv::Scalar(255), 100);

		bool be_closed = false;
		if (type == OC_crosswalk || type == OC_t_sign)
		{
			be_closed = true;
			//cv::polylines(mat, line_vec, be_closed, cv::Scalar(255), 30);
			cv::drawContours(mat, line_vec, -1, cv::Scalar(255), 30);
		}
		else
		{
			cv::drawContours(mat, line_vec, -1, cv::Scalar(255), cv::FILLED);
		}
		
	}

	cv::distanceTransform(mat, blur_mat, DIST_L1, 3);
	cv::normalize(blur_mat, blur_mat, 0, 255, cv::NORM_MINMAX);
	double g = mat.at<uchar>(1404, 569);
	g = blur_mat.at<float>(1404, 569);

	//blur_mat = mat;
	// 
	// 

// 	cv::namedWindow("dis", WINDOW_NORMAL);
// 	cv::imshow("dis", blur_mat);
// 	cv::waitKey(0);

	cv::imwrite("dis.jpg", blur_mat);
}

void Calib::extractImageRoadLines(const string& folder_path, const string& img_na,
	const cv::Mat& camera_img, vector<vector<cv::Point>>& line_vec)
{
	cv::Mat ipm_img;
	inversePerspectiveMapping(camera_img, CalibSpace::warpmat_src2ipm, ipm_img);
	string ipm_path = folder_path + "ipm/" + img_na + ".png";
	cv::imwrite(ipm_path, ipm_img);

	//calc2dVectorData(camera_img, line_vec);
	calc2dVectorData(ipm_img, line_vec);

	cv::Mat mat_ply = ipm_img.clone();
	cv::drawContours(mat_ply, line_vec, -1, cv::Scalar(255, 0, 255));
	string ply_path = folder_path + "ply/" + img_na + ".png";
	cv::imwrite(ply_path, mat_ply);

	perspectiveMappingPoints(line_vec, CalibSpace::warpmat_ipm2src);

	//cv::Mat mat_ply = camera_img.clone();
	//cv::drawContours(mat_ply, line_vec, -1, cv::Scalar(255, 0, 255));
	//string ply_path = folder_path + "ply/" + img_na + ".png";
	//cv::imwrite(ply_path, mat_ply);

}

void Calib::extractImageVerticalLines(const string& folder_path, const string& img_na,
	const cv::Mat& camera_img, vector<vector<cv::Point>>& line_vec)
{
#ifdef CV_CONTRIB
	cv::Mat gray;
	cv::cvtColor(camera_img, gray, cv::COLOR_BGR2GRAY);
	int otsu = cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	cv::Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector();
	vector<Vec4f> line_pts;
	fld->detect(gray, line_pts);

	line_vec.clear();
	int i = 0;
	auto itr_line = line_pts.begin();
	for (; itr_line != line_pts.end(); )
	{
		auto& line = *itr_line;
		auto c = camera_img.clone();
		//cv::namedWindow("0", cv::WINDOW_FREERATIO);
		//cv::drawContours(c, line_pts, i, cv::Scalar(255, 0, 255));
		//cv::imshow("0", c);
		//cv::waitKey(0);
		cv::Point spt(line[0], line[1]);
		cv::Point ept(line[2], line[3]);
		cv::Point dpt = ept - spt;
		float theta = atan(dpt.x * 1.0 / dpt.y);
		theta *= 180.0 / CV_PI;

		if (abs(theta) > 5.0)
		{
			itr_line = line_pts.erase(itr_line);
			continue;
		}

		itr_line++;
		i++;
		//内插
		vector<cv::Point> l;
		if (dpt.y > 0)
		{
			for (int y = spt.y; y <= ept.y; y++)
			{
				int x = (dpt.x) * 1.0 / (dpt.y) * (y - spt.y) + spt.x;
				l.push_back(cv::Point(x, y));
			}
		}
		else
		{
			for (int y = spt.y; y >= ept.y; y--)
			{
				int x = (dpt.x) * 1.0 / (dpt.y) * (y - spt.y) + spt.x;
				l.push_back(cv::Point(x, y));
			}
		}
		line_vec.push_back(l);
	}

//	fld->drawSegments(gray, line_pts);
	string ply_path = folder_path + "ply/" + img_na + ".jpg";
	cv::Mat t = cv::imread(ply_path);
	cv::polylines(t, line_vec, false, cv::Scalar(0, 255, 0), 2);
	//string ply_path = folder_path + "ply/" + img_na + ".png";
	cv::imwrite(ply_path, t);
#endif
}

void Calib::extractImageDeepLearningLines(const string& folder_path, const string& img_na, 
	const cv::Mat& camera_img, vector<vector<cv::Point>>& line_vec)
{
	string mask_path = folder_path + "road_mask/" + img_na + ".png";
	if (_access(mask_path.c_str(),0) != 0)
	{
		LOG(INFO) << "path..." << mask_path << " not found!!!";
		return;
	}
	cv::Mat mask = cv::imread(mask_path, 0);
	cv::resize(mask, mask, cv::Size(CalibSpace::IMG_WIDTH, CalibSpace::IMG_HEIGHT));
	cv::Mat img_binary;
	double thresh = cv::threshold(mask, img_binary, 0, 255, /*cv::THRESH_BINARY |*/ cv::THRESH_OTSU);
	cv::threshold(mask, img_binary, thresh * 1.1, 255, cv::THRESH_BINARY);
	mask = img_binary;
	cv::findContours(mask, line_vec, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	//for_each(contours.begin(), contours.end(), [&, this](auto& cont) {
	//	for_each(cont.begin(), cont.end(), [&, this](auto& d2){
	//		d2.x = d2.x * 1.0 / mask.cols * PC_TO_WideAngle_WIDTH - m_rect.tl().x;
	//		d2.y = d2.y * 1.0 / mask.rows * PC_TO_WideAngle_HEIGHT - m_rect.tl().y;
	//	});
	//});
	//cv::imwrite("a.jpg", mask);

	cv::Mat mat_ply = camera_img.clone();
	cv::drawContours(mat_ply, line_vec, -1, cv::Scalar(0, 255, 0), 2);
	string ply_path = folder_path + "ply/" + img_na + ".jpg";
	cv::imwrite(ply_path, mat_ply);

}

void recoverContourSize(cv::Size& img_size,LINE_VEC& line_vec)
{
	cv::Size recover_size = cv::Size(CalibSpace::image_rect.width, CalibSpace::image_rect.height);
	for_each(line_vec.begin(), line_vec.end(), [&](auto& line)
	{
		for_each(line.begin(), line.end(), [&](auto& pt) {
			pt.x = pt.x * 1.0 / img_size.width * recover_size.width;
			pt.y = pt.y * 1.0 / img_size.height * recover_size.height;
		});
	});
}

void recoverImageSize(cv::Mat& img)
{
	if (CalibSpace::camera_type == CAMERA_MSS_PANO)
	{
		int cut_size = 2048;
		cv::resize(img, img, cv::Size(cut_size, cut_size), INTER_NEAREST);

		//		cv::Mat org = cv::Mat::zeros(cv::Size(CalibSpace::IMG_WIDTH, CalibSpace::IMG_HEIGHT), CV_8UC1);
		//		cv::Rect rect(cv::Point(3072, 1024), cv::Point(5120, 3072));
		//		img.copyTo(org(rect));
		//		cv::imwrite("o.jpg", org);
		//		img = org;
	}
	else if (CalibSpace::camera_type == CAMERA_MSS_WIDE)
	{
		cv::resize(img, img, cv::Size(CalibSpace::IMG_WIDTH, CalibSpace::IMG_HEIGHT), INTER_NEAREST);
	}
	else
	{

	}

}


void Calib::extractImageDeepLearningMultiLines(const string& folder_path,
	const string& img_na,
	map<int, LINE_VEC>& lines_map)
{
	//set<ObjectClassification> oc_set = { OC_lane , OC_pole,OC_t_sign,OC_crosswalk };
	set<ObjectClassification> oc_set = { OC_lane , OC_pole,OC_t_sign};
	CenterLine cl;
	LINE_VEC& line_vec = lines_map[OC_lane];

	string ld_mask_path = folder_path + "lane_finding/" + img_na + ".png";
	cv::Mat mask = cv::imread(ld_mask_path, 0);

	//string ld_mask_path = folder_path + "dlink/" + img_na + ".png";
	//cv::Mat mask = cv::imread(ld_mask_path, 0);
	//cv::resize(mask, mask, cv::Size(CalibSpace::IMG_WIDTH, CalibSpace::IMG_HEIGHT));
	cl.getCenterLine(mask, line_vec, 1);

	//string obj_mask_path = folder_path + "deeplabv3/" + img_na + "_pred.png";
	string obj_mask_path = folder_path + "obj_finding/" + img_na + "_pred.png";
	if (_access(obj_mask_path.c_str(), 0) != 0)
	{
		return;
	}
	mask = cv::imread(obj_mask_path);
	for (auto itr = oc_set.begin(); itr != oc_set.end(); itr++)
	{
		if (*itr == OC_lane)
		{
			continue;
		}
		cv::Mat img_binary;
		auto find_color = m_color_map.find(*itr);
		if (find_color == m_color_map.end())
		{
			continue;
		}
		const auto& color = find_color->second;

		cv::inRange(mask, color, color, img_binary);
		LINE_VEC contours;
		cv::findContours(img_binary, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		if (contours.size() == 0)
		{
			continue;
		}
		lines_map.insert(make_pair(*itr, contours));
	}
	return;
}

//void Calib::extractImageDeepLearningMultiLines(const string& folder_path, const string& img_na,
//	const cv::Mat& camera_img, const set<ObjectClassification>& oc_set, map<int, LINE_VEC>& lines_map)
//{
//	//road mask
//	//const string& ld_mask_path = folder_path + "Dlink/" + img_na + ".png";
//	const string& ld_mask_path = folder_path + "BiSeNet/" + img_na + ".png";
//	cv::Mat mat_ply = camera_img.clone();
//	if (oc_set.find(OC_lane) != oc_set.end())
//	{
//		LINE_VEC& line_vec = lines_map[OC_lane];
//		//		extractImageDeepLearningLines(folder_path, img_na, mat_ply, ll_vec);
//
//		CenterLine cl;
//		cl.getCenterLine(ld_mask_path, line_vec, 1);
//		cv::drawContours(mat_ply, line_vec, -1, cv::Scalar(0, 255, 0), 2);
//	}
//
//	//other
//	string mask_path = folder_path + "BiSeNet/" + img_na + ".png";
//	if (_access(mask_path.c_str(), 0) == 0)
//	{
//		cv::Mat mask = cv::imread(mask_path, 0);
//		cv::resize(mask, mask, cv::Size(CalibSpace::IMG_WIDTH, CalibSpace::IMG_HEIGHT));
//		cv::Mat img_binary;
//
//		for (auto itr = oc_set.begin(); itr != oc_set.end(); itr++)
//		{
//			if (*itr == OC_lane)
//			{
//				continue;
//			}
//			cv::compare(mask, *itr, img_binary, cv::CMP_EQ);
//			LINE_VEC& line_vec = lines_map[*itr];
//			cv::findContours(img_binary, line_vec, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//			cv::drawContours(mat_ply, line_vec, -1, cv::Scalar(0, 255, 0), 2);
//		}
//	}
//	
//	string ply_path = folder_path + "ply/" + img_na + ".jpg";
//	cv::imwrite(ply_path, mat_ply);
//
//}

size_t Calib::read3dVectorData(const string& file_path, vector<vector<cv::Point3f>>& line_vec)
{
	//获取该路径下的所有文件
	vector<string> files;
//	DataIO::getFiles(file_path, "*.poly", files);

	int size = files.size();
	for (int i = 0; i < size; i++)
	{
		//定义很多点（单线）的vector
		vector<cv::Point3f> points;

		//定义点，作为push_back的对象
		//这个需要注意，vector的每一项需要作为一个整体来pushback，数组就pushback数组，类就pushback类。
		Point3f temp;

		//定义3个变量，用来存储读取的数据
		float temp1 = 0, temp2 = 0, temp3 = 0;

		ifstream circle;
		//string path = "D:\\vs 项目\\poly\\";
		string path = files[i].c_str();
		circle.open(path);

		//!circle.eof() 用来控制什么时候读取终止
		for (int i = 0; !circle.eof(); i++)
		{
			//读取txt文件的坐标值
			circle >> temp1 >> temp2 >> temp3;
			temp.x = temp1;
			temp.y = temp2;
			temp.z = temp3;

			points.push_back(temp);
		}

		//加密
		equalizPolyline(points, 0.01);
		line_vec.push_back(points);
		circle.close();
	}


	return line_vec.size();
}

size_t Calib::calc2dVectorData(const cv::Mat& img, vector<vector<cv::Point>>& line_vec)
{
	cv::Mat out;
	cv::cvtColor(img, out, cv::COLOR_BGR2GRAY);

	cv::Mat img_binary;
	double thresh = cv::threshold(out, img_binary, 0, 255, cv::THRESH_OTSU);
	thresh = thresh/* * 0.4*/;
//	cv::threshold(out, out, thresh*0.4, 255, cv::ADAPTIVE_THRESH_MEAN_C);

	cv::Mat canny;
	cv::Canny(out, canny, thresh, 255, 3);

	//cv::namedWindow("0", cv::WINDOW_AUTOSIZE);
	////	cv::imshow("0", out);
	//cv::imshow("0", canny);
	//cv::waitKey(0);

	//vector<cv::Vec4i> lines;
	//cv::HoughLines(canny, lines, 1, CV_PI / 180, 10, 10, 5.0 / 180);

	vector<vector<cv::Point>> line_pts;
	cv::findContours(canny, line_pts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	cv::cvtColor(canny, canny, cv::COLOR_GRAY2BGR);
	
	int i = 0;
	auto itr_line = line_pts.begin();
	for (; itr_line != line_pts.end(); )
	{
		auto& line = *itr_line;
		if (line.size() < 2)
		{	
			itr_line = line_pts.erase(itr_line);
			continue;
		}
		auto c = canny.clone();
		//cv::namedWindow("0", cv::WINDOW_FREERATIO);
		//cv::drawContours(c, line_pts, i, cv::Scalar(255, 0, 255));
		//cv::imshow("0", c);
		//cv::waitKey(0);
		
		if (line.size() == 0)
		{
			itr_line = line_pts.erase(itr_line);
			continue;
		}
		cv::Vec4f line_para;
		cv::fitLine(line, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
		float theta = atan(line_para[0] / line_para[1]);
		theta *= 180.0 / CV_PI;

		if (abs(theta) > 5.0)
		{
			itr_line = line_pts.erase(itr_line);
			continue;
		}
		
		////获取点斜式的点和斜率
		//cv::Point point0;
		//point0.x = line_para[2];
		//point0.y = line_para[3];

		//double k = line_para[1] / line_para[0];

		////计算直线的端点(y = k(x - x0) + y0)
		//cv::Point point1, point2;
		//point1.x = point0.x + 10;
		//point1.y = k * (point1.x - point0.x) + point0.y;
		//point2.x = point0.x - 10;
		//point2.y = k * (point2.x - point0.x) + point0.y;

		//line.resize(2);
		//line[0] = point1;
		//line[1] = point2;

		itr_line++;
		i++;
	}
	//cv::polylines(canny, line_pts, 0, cv::Scalar(255, 0, 255));

////	cv::imshow("0", out);
//	cv::imshow("0", canny);
//	cv::waitKey(0);
	line_vec.swap(line_pts);
	return 0;
}

void Calib::perspectiveMappingPoints(vector<vector<cv::Point>>& line_pts,  const cv::Mat& warpmat_ipm2src)
{
	vector<cv::Point2f> contoursf;
	vector<cv::Point2f> t;
	for_each(line_pts.begin(), line_pts.end(), [&](auto& cont) {
		contoursf.resize(cont.size());
		transform(cont.begin(), cont.end(), contoursf.begin(),[](const auto& p)->cv::Point2f{
			cv::Point2f fp = p;
			return fp;
		});
		cv::perspectiveTransform(contoursf, t, warpmat_ipm2src);

		cont.clear();
		cont.push_back(cv::Point(t.front()));
		for (int i = 1; i < t.size(); i++)
		{
			cv::Point p = cv::Point(t[i]);
			if (p == cont.back())
			{
				continue;
			}
			cont.push_back(p);
		}
		
	});

}

void Calib::projectPointCloud2Image(const vector<CalibSpace::PointXYZI>& pc,
	const cv::Mat& intrisicMat, const cv::Mat& distCoeffs, const cv::Mat& rVec, const cv::Mat& tVec,
	cv::Mat&img)
{
	vector<cv::Point2d> imagePoints;
	vector<cv::Point3d> objectPoints(pc.size());
	transform(pc.begin(), pc.end(), objectPoints.begin(), [](const auto& p)->cv::Point3d {
		return cv::Point3d(p.x, p.y, p.z);
	});
	cv::projectPoints(objectPoints, rVec, tVec, intrisicMat, distCoeffs, imagePoints);
	//img = cv::Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
	//	img = cv::Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC1, cv::Scalar(0));
	int i = 0;
	for_each(imagePoints.begin(), imagePoints.end(), [&](const auto& pt)
	{
		if (
			pt.y >= 0 && pt.y < CalibSpace::IMG_HEIGHT &&
			pt.x >= 0 && pt.x < CalibSpace::IMG_WIDTH)
		{
			//		img.at<uchar>(pt.y, pt.x)[2] = pc[i].intensity * 255;
			//		img.at<cv::Vec3b>(pt.y, pt.x)[2] = pc[i].intensity * 255;
			//cv::drawMarker(img, pt, cv::Scalar(0, 0, pc[i].intensity * 255), cv::MARKER_CROSS, 10);
			cv::drawMarker(img, pt, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 10);
		}
		i++;
	});
}

void Calib::projectPointCloud2Image(const vector<CalibSpace::PointXYZI>& pc,
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
		if (OptimizeCeresWeighted::convertPoint3dTo2d(camPose, d3, d2))
		{
			/*cv::drawMarker(img, cv::Point(d2[0], d2[1]), cv::Scalar(0, 0, pc[i].intensity * 255), cv::MARKER_CROSS, 10);*/
			cv::drawMarker(img, cv::Point(d2[0], d2[1]), cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 10);
		}
		i++;
	});

}

void Calib::inversePerspectiveMapping(const cv::Mat& src, const cv::Mat& warpmat_src2ipm, cv::Mat& dest)
{
	if (!src.data)
	{
		return;
	}
	//float roi_height = 30000;
	//float roi_width = 3750;

	//逆透视变换的宽度
	float ipm_width = 800;
	float ipm_height = 1200;
	//float N = 5;
	//保证逆透视变换的宽度大概为5个车头宽
	//float scale = (ipm_width / N) / roi_width;
	//float ipm_height = roi_height * scale;

	dest = cv::Mat::zeros(ipm_height, ipm_width, src.type());

	cv::warpPerspective(src, dest, warpmat_src2ipm, dest.size());


	/*dest = binaryImage(dest);

	vector<vector<cv::Point>> contours;
	cv::findContours(dest, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	cv::cvtColor(dest, dest, cv::COLOR_GRAY2BGR);
	cv::polylines(dest, contours, 1, cv::Scalar(255, 0, 0), 1);
	cv::imwrite("contour_dest.jpg", dest);

	vector<cv::Point2f> contoursf;
	for_each(contours.begin(), contours.end(), [&](const auto& cont) {
	for_each(cont.begin(), cont.end(), [&](const auto& p){
	cv::Point2f pf = p;
	contoursf.push_back(pf);
	});
	});


	vector<cv::Point2f> src_contours;
	cv::Mat warpmat_ipm2src = cv::getPerspectiveTransform(corners_trans, corners);
	cv::perspectiveTransform(contoursf, src_contours, warpmat_ipm2src);

	cv::Mat temp = src.clone();
	cv::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);
	for_each(src_contours.begin(), src_contours.end(), [&](const auto& p) {
	cv::drawMarker(temp, p, cv::Scalar(255, 0, 0));
	});
	cv::imwrite("contour_src.jpg", temp);*/

}

void Calib::getValid3dPoints(const vector<vector<cv::Point3f>>& xyz_line_vec, vector<cv::Point3f>& xyz_vec)
{
	for_each(xyz_line_vec.begin(), xyz_line_vec.end(), [&](const auto&line) {
		for_each(line.begin(), line.end(), [&](const auto& p) {
			if (p.x > 0 && p.x < 25)
			{
				xyz_vec.push_back(p);
			}
		});
	});

	//ofstream of("a.txt");
	//of << xyz_vec;
	//of.close();
}

void Calib::getValid2dPoints(const vector<vector<cv::Point>>& line_vec, vector<cv::Point>& ij_vec)
{
	for_each(line_vec.begin(), line_vec.end(), [&](const auto&line) {
		for_each(line.begin(), line.end(), [&](const auto& p) {
			if (p.x > 0)
			{
				ij_vec.push_back(p);
			}
		});
	});
}

void Calib::regist3D2DLines(const vector<vector<cv::Point3f>>& xyz_line_vec, const vector<vector<cv::Point>>& line_vec, vector<double>& camPose)
{
	vector<cv::Point3f> xyz_vec;
	getValid3dPoints(xyz_line_vec, xyz_vec);
	
	vector<cv::Point> ij_vec;
	getValid2dPoints(line_vec, ij_vec);
	
	iterateClosestPoint2d3d(xyz_vec, ij_vec, camPose);
}

bool Calib::buildKDTree(const map<int, Point_VEC>& ij_linevec, map<int, cv::Mat>& src_map)
{
	bool empty_tree = false;

	auto itr_type = ij_linevec.begin();
	for (; itr_type != ij_linevec.end(); itr_type++)
	{
		const auto& type = itr_type->first;
		const auto& ij_vec = itr_type->second;

		empty_tree |= ij_vec.size() == 0;
		if (empty_tree)
		{
			continue;
		}
		cv::Mat&  source = src_map[type];
		source = cv::Mat(ij_vec).reshape(1);
		source.convertTo(source, CV_32F);

		global_kdtree_map[type].build(source, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);
	}

	return !empty_tree;
}

void Calib::releaseKDTree()
{
//	LANE_TREE.release();
//	POLE_TREE.release();
	auto itr_type = global_kdtree_map.begin();
	for (; itr_type != global_kdtree_map.end(); itr_type++)
	{
		auto& tree = itr_type->second;
		tree.release();
	}
	global_kdtree_map.clear();
}

bool Calib::isValid(const vector<double>&  cam, const vector<double>&  base, const vector<double>& diff)
{
	for (int i = 0; i < 3; i++)
	{
		if (abs(cam[i] - base[i]) > diff[i])
		{
			return false;
		}
	}

	return true;
}

bool isNoChange(const map<int, Point3f_VEC>& xyz_linevec, const vector<double>& v1, const vector<double>& v2)
{
	if (v1.size() != v2.size())
	{
		return true;
	}

	bool no_change = true;
	for (int i = 0; i < v1.size(); i++)
	{
		no_change &= (v1[i] == v2[i]);
	}
	if (no_change)
	{
		return true;
	}

	double diff = 0;
	int diff_sz = 0;
	auto itr_type = xyz_linevec.begin();
	for (; itr_type != xyz_linevec.end(); itr_type++)
	{
		const auto& type = itr_type->first;
		const auto& xyz_vec = itr_type->second;
		vector<cv::Point> lidar_2d_pts;

		auto itr_3d = xyz_vec.begin();
		int i = 0;
		
		for (; itr_3d != xyz_vec.end(); itr_3d++, i++)
		{
			double d3[3] = { itr_3d->x, itr_3d->y, itr_3d->z };
			double d1[2] = {0,0};
			if (!OptimizeCeresWeighted::convertPoint3dTo2d(v1, d3, d1))
			{
				continue;
			}
			cv::Point dd1(d1[0], d1[1]);

			double d2[2] = { 0,0 };
			if (!OptimizeCeresWeighted::convertPoint3dTo2d(v2, d3, d2))
			{
				continue;
			}
			cv::Point dd2(d2[0], d2[1]);

			diff += cv::norm(dd1 - dd2);
			diff_sz++;
		}
	}
	if (diff_sz == 0)
	{
		return false;
	}
	diff /= diff_sz;
	return diff < 1;
}

float isNoChangeSimilarity(const map<int, Point3f_VEC>& xyz_linevec,
	const cv::Mat& blur_mat,
	const vector<double>& v1, const vector<double>& v2)
{
	if (v1.size() != v2.size())
	{
		return true;
	}

	bool no_change = true;
	for (int i = 0; i < v1.size(); i++)
	{
		no_change &= (v1[i] == v2[i]);
	}
	if (no_change)
	{
		return true;
	}

	int count_non_zero1 = 0;
	int count_non_zero2 = 0;
	auto itr_type = xyz_linevec.begin();
	for (; itr_type != xyz_linevec.end(); itr_type++)
	{
		const auto& type = itr_type->first;
		const auto& xyz_vec = itr_type->second;
	
		auto itr_3d = xyz_vec.begin();
		int i = 0;

		for (; itr_3d != xyz_vec.end(); itr_3d++, i++)
		{
			double d3[3] = { itr_3d->x, itr_3d->y, itr_3d->z };
			double d1[2] = { 0,0 };
			if (!OptimizeCeresWeighted::convertPoint3dTo2d(v1, d3, d1, false))
			{
				continue;
			}
			cv::Point dd1(d1[0], d1[1]);

			double d2[2] = { 0,0 };
			if (!OptimizeCeresWeighted::convertPoint3dTo2d(v2, d3, d2, false))
			{
				continue;
			}
			cv::Point dd2(d2[0], d2[1]);

			if (dd1.x < 0 || dd1.y < 0 ||
				dd2.x < 0 || dd2.y < 0)
			{
				continue;
			}
			uchar grey1 = blur_mat.at<uchar>(dd1.y, dd1.x);
			uchar grey2 = blur_mat.at<uchar>(dd2.y, dd2.x);

			if (grey1 > 0)
			{
				count_non_zero1++;
			}
			 if (grey2 > 0)
			 {
				 count_non_zero2++;
			 }
		}
	}

	if (count_non_zero2 > count_non_zero1)
	{
		return true;
	}
}


void convLines2Points(const map<int, LINE_VEC>& lines_map, 
	map<int, Point_VEC>& ij_vec_map)
{
	for_each(lines_map.begin(), lines_map.end(), [&](const auto& _pair) {
		const auto& line_vec = _pair.second;
		auto& ij_vec = ij_vec_map[_pair.first];
		for_each(line_vec.begin(), line_vec.end(), [&](const auto& line) {
			copy(line.begin(), line.end(), back_inserter(ij_vec));
			});
		});
}

double Calib::iterateClosestPoint2d3d(const map<int, Point3f_VEC>& xyz_vec,
	const map<int, LINE_VEC>& lines_map,
	const map<int, LINE_VEC>& camera_lines_map, 
	vector<double>& camPose, const vector<double>&  diff_threshold)
{
	vector<double> init_campose = camPose;

	map<int, Point_VEC>ij_vec_map;
	convLines2Points(lines_map, ij_vec_map);

	map<int, cv::Mat> src_map;
	if (!buildKDTree(ij_vec_map, src_map))
	{
		return 1000;
	}

	cv::Mat blur_mat;
	calcBlurImage(camera_lines_map, blur_mat);

	//ceres::Solver::Summary summary;
	vector<CalibSpace::Point3d2d> p3d2ds;
	//迭代次数
	int total_num = 12;
	int cycle = 4;
	int itr_num = 0;
	double resdual = -1;
	bool no_change = false;

	while (itr_num < total_num)
	{
		p3d2ds.clear();
		double D = calcD(itr_num, cycle);
		//double D = 50;
		int tm = itr_num / cycle;
		findCorrespondPoints(camPose, xyz_vec, ij_vec_map, p3d2ds, D, tm);
		if (p3d2ds.size() < 4)
		{
			break;
		}
		bool  valid = false;
		if (CalibSpace::camera_type == CAMERA_MSS_WIDE)
		{
			vector<double> pnp_camPose = camPose;
			vector<int> inliers;
			if (optimizePnP(p3d2ds, pnp_camPose, inliers, true) &&
				isValid(pnp_camPose, init_campose, diff_threshold))
			{
				camPose.swap(pnp_camPose);
				valid = true;

#if 1
				no_change = isNoChange(xyz_vec, pnp_camPose, camPose);
				//no_change = isNoChangeSimilarity(xyz_vec, blur_mat,pnp_camPose, camPose);
				if (no_change)
				{
					break;
				}
#endif
			}
		}
		///////////////////////
		/*else if (CalibSpace::camera_type == CAMERA_MSS_PANO)
		{
			vector<double> pnp_camPose = camPose;
			vector<int> inliers;
			if (optimizePanoPnP(p3d2ds, pnp_camPose, inliers, true) &&
				isValid(pnp_camPose, init_campose, diff_threshold))
			{
				camPose.swap(pnp_camPose);
				valid = true;
				no_change = isNoChange(xyz_vec, pnp_camPose, camPose);
				if (no_change)
				{
					break;
				}
			}
		}*/
		///////////////////////
		if (!valid)
		{
			vector<double> ceres_camPose = camPose;
			//OptimizeCeresWeighted::optimize(p3d2ds, ceres_camPose);
			if (isValid(ceres_camPose, init_campose, diff_threshold))
			{
				camPose.swap(ceres_camPose);
				valid = true;
				//evaluteSimilarity()
#if 1
				no_change = isNoChange(xyz_vec, ceres_camPose, camPose);
				//no_change = isNoChangeSimilarity(xyz_vec, blur_mat, ceres_camPose, camPose);
				if (no_change)
				{
					break;
				}
#endif
			}
		}

		itr_num++;
		//resdual = OptimizeCeresWeighted::evaluate(p3d2ds, camPose);
		/*if (resdual < 5)
		{
			if (tm == 0)
			{
				if (itr_num > cycle * tm)
				{
					itr_num = cycle * (tm + 1);
				}
			}
			else
			{
				break;
			}
		}
		else
		{
			itr_num++;
		}*/		
	}
	releaseKDTree();

	//	LOG(INFO) << summary.FullReport();
	LOG(INFO) <<("iterate Xs,Ys,Zs,omega,pho,kappa = ");
	for (int i = 0; i < 6; i++)
	{
		LOG(INFO) <<("%.3f",camPose[i]);
	}
	//findCorrespondPoints(camPose, xyz_vec, ij_vec, p3d2ds, 50, 0);
	//resdual = OptimizeCeresWeighted::evaluate(p3d2ds, camPose);
	//LOG(INFO) <<("avg resdual: %.3f", resdual);

	return resdual;
}




Eigen::MatrixXf format(const pcl::PointCloud<pcl::PointXYZ>& pc)
{
	int n = pc.size();
	Eigen::MatrixXf em = Eigen::MatrixXf::Zero(4, n);

	for (int i =0; i < n; i ++)
	{
		const auto& pt = pc[i];
		em(0, i) = pt.x;
		em(1, i) = pt.y;
		em(2, i) = pt.z;
	}

	return em;
}


void Calib::iterateClosestPoint2d3d(const Point3f_VEC& xyz_vec, const Point_VEC& ij_vec, vector<double>& camPose)
{
	cv::Mat  source = cv::Mat(ij_vec).reshape(1);
	source.convertTo(source, CV_32F);
	global_kdtree.build(source, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);

	//ceres::Solver::Summary summary;
	vector<CalibSpace::Point3d2d> p3d2ds;
	//迭代次数
	int total_num = 15;
	int cycle = 5;
	int itr_num = 0;
	while (itr_num < total_num)
	{
		p3d2ds.clear();
		double D = calcD(itr_num, cycle);

		findCorrespondPoints(camPose, xyz_vec, ij_vec, p3d2ds, D);
		//OptimizeCeresWeighted::optimize(p3d2ds, camPose);
		if (p3d2ds.size() < 4)
		{
			break;
		}

		vector<int> inliers;
		if (!optimizePnP(p3d2ds, camPose, inliers, true))
		{
			break;
		}
		itr_num++;
	}

	//	LOG(INFO) << summary.FullReport();
	LOG(INFO) << "*************结果****************";
	LOG(INFO) << "estimated Xs,Ys,Zs,omega,pho,kappa = ";
	for (int i = 0; i < 6; i++)
	{
		LOG(INFO) << camPose[i];
	}

	global_kdtree.release();
}

//void Calib::iterateClosestPoint(const vector<vector<cv::Point3f>>& xyz_line_vec, const vector<vector<cv::Point>>& line_vec, double* camPose)
//{
//	set<int> xyz_s_e_indices;
//	vector<cv::Point3f> xyz_vec;
//	for_each(xyz_line_vec.begin(), xyz_line_vec.end(), [&](const auto& line) {
//		xyz_s_e_indices.insert(xyz_vec.size());
//		copy(line.begin(), line.end(), back_inserter(xyz_vec));
//		xyz_s_e_indices.insert(xyz_vec.size() - 1);
//	});
//
//	vector<cv::Point> ij_vec;
//	set<int> s_e_indices;
//	for_each(line_vec.begin(), line_vec.end(), [&](const auto& line) {
//		s_e_indices.insert(ij_vec.size());
//		copy(line.begin(), line.end(), back_inserter(ij_vec));
//		s_e_indices.insert(ij_vec.size() - 1);
//	});
//
//	cv::Mat  source = cv::Mat(ij_vec).reshape(1);
//	source.convertTo(source, CV_32F);
//	global_kdtree.build(source, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_EUCLIDEAN);
//
//	//ceres::Solver::Summary summary;
//	vector<CalibSpace::Point3d2d> p3d2ds;
//	//迭代次数
//	int total_num = 15;
//	int cycle = 5;
//	int itr_num = 0;
//	while (itr_num < total_num)
//	{
//		p3d2ds.clear();
//		double D = calcD(itr_num, cycle);
//
//		findCorrespondPoints(camPose, xyz_vec, xyz_s_e_indices, ij_vec, s_e_indices,p3d2ds, D);
//		//OptimizeCeresWeighted::optimize(p3d2ds, camPose);
//		if (p3d2ds.size() < 3)
//		{
//			break;
//		}
//		optimizePnP(p3d2ds, camPose, true);
//
//		//LOG(INFO) << "迭代次数:" << itr_num;
//		//for (int i = 0; i < 6; i++)
//		//{
//		//	LOG(INFO) << camPose[i];
//		//}
//		itr_num++;
//	}
//	//itr_num = 0;
//	//total_num = 15;
//	//while (itr_num < total_num)
//	//{
//	//	p3d2ds.clear();
//	//	double D = calcD(itr_num, cycle);
//	//	findCorrespondPoints(camPose, xyz_vec, ij_vec, p3d2ds, D);
//	//	if (p3d2ds.size() < 3)
//	//	{
//	//		break;
//	//	}
//	//	OptimizeCeresWeighted::optimize(p3d2ds, camPose);
//	//	itr_num++;
//	//}
//	//	optimizePnP(p3d2ds, camPose, true);
//
//	//	LOG(INFO) << summary.FullReport();
//	LOG(INFO) << "*************结果****************";
//	LOG(INFO) << "estimated Xs,Ys,Zs,omega,pho,kappa = ";
//	for (int i = 0; i < 6; i++)
//	{
//		LOG(INFO) << camPose[i];
//	}
//	global_kdtree.release();
//
//}

bool Calib::findCorrespondPoints(const vector<double>& camRot, const Point3f_VEC& xyz_vec, const Point_VEC& ij_vec, vector<CalibSpace::Point3d2d>& p3d2ds, float D)
{
	vector<vector<cv::Point>> match_line_vec;
	vector<vector<cv::Point>> reverse_match_line_vec;

	vector<cv::Point> lidar_2d_pts;
	vector<int>  lidar_2d_indices;
	auto itr_3d = xyz_vec.begin();
	int i = 0;
	for (; itr_3d != xyz_vec.end(); itr_3d++, i++)
	{
		double d3[3] = { itr_3d->x, itr_3d->y, itr_3d->z };
		double d2[2];
		if (!OptimizeCeresWeighted::convertPoint3dTo2d(camRot, d3, d2))
		{
			continue;
		}
		cv::Point dd2((int)d2[0], (int)d2[1]);

		lidar_2d_pts.push_back(dd2);
		lidar_2d_indices.push_back(i);

		int idx = -1;

		//	double DD = calcDD(D, dd2);
		double DD = D;
		//LOG(INFO) << D << "," << DD;

		double nearest_dist = 1000;
		if (!searchNeareatPoint(ij_vec, global_kdtree, dd2, idx, nearest_dist, DD))
			//if (!searchNeareatYPoint(ij_vec, global_kdtree, dd2, idx, nearest_dist, DD))
		{
			continue;
		}

		cv::Point search_pt = ij_vec[idx];
		cv::Point delta = search_pt - dd2;
		if (/*abs(search_pt.x - CalibSpace::CX) < 100 &&
			abs(search_pt.y - CalibSpace::CY) < 100*/
			abs(delta.x * 1.0 / delta.y) < 0.4)
		{
			continue;
		}
		vector<cv::Point> line(2);
		line[0] = dd2;
		line[1] = search_pt;
		match_line_vec.push_back(line);

		CalibSpace::Point3d2d p3d2d;
		p3d2d.p3d = *itr_3d;
		p3d2d.p2d = search_pt;
		p3d2ds.push_back(p3d2d);
	}


	//建树
	if (lidar_2d_pts.size() > 0)
	{
		cv::Mat  mat_lidar_2d = cv::Mat(lidar_2d_pts).reshape(1);
		mat_lidar_2d.convertTo(mat_lidar_2d, CV_32F);
		cv::flann::Index tree = cv::flann::Index(mat_lidar_2d, cv::flann::KDTreeIndexParams(1));
		auto itr_2d = ij_vec.begin();
		i = 0;
		for (; itr_2d != ij_vec.end(); itr_2d++, i++)
		{
			const cv::Point& d2 = *itr_2d;
			int idx = -1;
			//	double DD = calcDD(D, dd2);
			double DD = D;

			double nearest_dist = 1000;
			//			if (!searchNeareatPoint(lidar_2d_pts, tree, d2, idx, nearest_dist, DD))
			if (!searchNeareatYPoint(lidar_2d_pts, tree, d2, idx, nearest_dist, DD))
			{
				continue;
			}

			cv::Point search_pt = lidar_2d_pts[idx];

			cv::Point delta = search_pt - d2;
			if (abs(delta.x * 1.0 / delta.y) < 0.4)
			{
				continue;
			}

			vector<cv::Point> line(2);
			line[0] = d2;
			line[1] = search_pt;
			reverse_match_line_vec.push_back(line);

			CalibSpace::Point3d2d p3d2d;
			p3d2d.p3d = xyz_vec[lidar_2d_indices[idx]];
			p3d2d.p2d = d2;
			p3d2ds.push_back(p3d2d);
		}
		tree.release();
	}

	//第一次匹配的时候保存一下
	if (/*D > 100*/1)
	{
		cv::Mat match_mat(CalibSpace::IMG_HEIGHT, CalibSpace::IMG_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));;
		//		cv::cvtColor(match_mat, match_mat, cv::COLOR_GRAY2BGR);
		cv::polylines(match_mat, reverse_match_line_vec, false, cv::Scalar(255, 0, 0), 2);
		cv::polylines(match_mat, match_line_vec, false, cv::Scalar(0, 0, 255), 2);

		//cv::polylines(match_mat, m_lidar_2d_contours, false, cv::Scalar(255, 0, 0), 2);

		string temp_path = "kdtree/";
		cv::imwrite(temp_path + to_string(int(D)) + ".png", match_mat);
	}

	return p3d2ds.size() > 0;
}

bool Calib::findCorrespondPoints(const vector<double>& camRot, const map<int, Point3f_VEC>& xyz_linevec,
	const map<int, Point_VEC>& ij_linevec,
	vector<CalibSpace::Point3d2d>& p3d2ds, 
	float D,
	int tm)
{
	vector<vector<cv::Point>> match_line_vec;
	vector<vector<cv::Point>> reverse_match_line_vec;

	auto itr_type = xyz_linevec.begin();
	for (; itr_type != xyz_linevec.end(); itr_type++)
	{
		const auto& type = itr_type->first;
		//if (H_ASSGIN &&tm == 3 && (type < -1 || type > 1))
		//{
		//	continue;
		//}
		const auto& xyz_vec = itr_type->second;
		auto find_ij = ij_linevec.find(type);
		if (find_ij == ij_linevec.end())
		{
			continue;
		}
		const auto& ij_vec = find_ij->second;
		auto find_tree = global_kdtree_map.find(type);
		if (find_tree == global_kdtree_map.end())
		{
			continue;
		}
		auto& tree = find_tree->second;

		vector<cv::Point> lidar_2d_pts;
		vector<int>  lidar_2d_indices;
		auto itr_3d = xyz_vec.begin();
		int i = 0;
		for (; itr_3d != xyz_vec.end(); itr_3d++, i++)
		{
			double d3[3] = { itr_3d->x, itr_3d->y, itr_3d->z };
			double d2[2];
			if (!OptimizeCeresWeighted::convertPoint3dTo2d(camRot, d3, d2))
			{
				continue;
			}
			cv::Point dd2(d2[0], d2[1]);

			lidar_2d_pts.push_back(dd2);
			lidar_2d_indices.push_back(i);

			int idx = -1;

			//	double DD = calcDD(D, dd2);
			double DD = D;
			//LOG(INFO) << D << "," << DD;

			double nearest_dist = 1000;
			bool search_result = searchNeareatPoint(ij_vec, tree, dd2, idx, nearest_dist, DD);
			
			if (!search_result)
			{
				continue;
			}

			cv::Point search_pt = ij_vec[idx];
			cv::Point delta = search_pt - dd2;
			if (/*abs(search_pt.x - CalibSpace::CX) < 100 &&
				abs(search_pt.y - CalibSpace::CY) < 100*/
				abs(delta.x * 1.0 / delta.y) < 2)
			{
				continue;
			}
			vector<cv::Point> line(2);
			line[0] = dd2;
			line[1] = search_pt;
			match_line_vec.push_back(line);

			CalibSpace::Point3d2d p3d2d;
			p3d2d.p3d = *itr_3d;
			p3d2d.p2d = search_pt;
			p3d2ds.push_back(p3d2d);
		}


		//建树
		if (lidar_2d_pts.size() > 0)
		{
			cv::Mat  mat_lidar_2d = cv::Mat(lidar_2d_pts).reshape(1);
			mat_lidar_2d.convertTo(mat_lidar_2d, CV_32F);
			cv::flann::Index tree = cv::flann::Index(mat_lidar_2d, cv::flann::KDTreeIndexParams(1));
			auto itr_2d = ij_vec.begin();
			i = 0;
			for (; itr_2d != ij_vec.end(); itr_2d++, i++)
			{
				const cv::Point& d2 = *itr_2d;
				int idx = -1;
				//	double DD = calcDD(D, dd2);
				double DD = D;

				double nearest_dist = 1000;
				if (!searchNeareatPoint(lidar_2d_pts, tree, d2, idx, nearest_dist, DD))
				//if (!searchNeareatYPoint(lidar_2d_pts, tree, d2, idx, nearest_dist, DD))
				{
					continue;
				}

				cv::Point search_pt = lidar_2d_pts[idx];

				cv::Point delta = search_pt - d2;
				if (abs(delta.x * 1.0 / delta.y) < 2)
				{
					continue;
				}

				vector<cv::Point> line(2);
				line[0] = d2;
				line[1] = search_pt;
				reverse_match_line_vec.push_back(line);

				CalibSpace::Point3d2d p3d2d;
				p3d2d.p3d = xyz_vec[lidar_2d_indices[idx]];
				p3d2d.p2d = d2;
				p3d2ds.push_back(p3d2d);
			}
			tree.release();
		}
	}

	//int total_match_sz = match_line_vec.size() + reverse_match_line_vec.size();
	//float match_scale = match_line_vec.size()  * 1.0 / total_match_sz;
	//if (match_scale < 0.4 ||
	//	match_scale > 0.6)
	//{
	//	LOG(INFO) << "unbalanced matches..." << match_scale << "&&" << 1 - match_scale;
	////	p3d2ds.clear();
	////	return;
	//}
	

	//第一次匹配的时候保存一下
	if (/*D > 100*/0)
	{
		//cv::Mat match_mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC3, cv::Scalar(0, 0, 0));
		//cv::Mat match_mat(1280, 800, CV_8UC3, cv::Scalar(0, 0, 0));
		cv::Mat match_mat(2560, 800, CV_8UC3, cv::Scalar(0, 0, 0));
		/*	if (m_camera_img.rows > 0)
			{
				match_mat = m_camera_img.clone();
			}*/
//		cv::cvtColor(match_mat, match_mat, cv::COLOR_GRAY2BGR);

		cv::polylines(match_mat, reverse_match_line_vec, false, cv::Scalar(255, 0, 0), 2);
		cv::polylines(match_mat, match_line_vec, false, cv::Scalar(0, 0, 255), 2);

		//cv::polylines(match_mat, m_lidar_2d_contours, false, cv::Scalar(255, 0, 0), 2);

		string temp_path = "kdtree/";
		cv::imwrite(temp_path + to_string(int(D)) + ".png", match_mat);
	}

	return p3d2ds.size() > 0;
}

bool Calib::findCorrespondPoints(const vector<double>& camRot, const vector<cv::Point3f>& xyz_vec, const set<int>& xyz_s_e_indices,
	const vector<cv::Point>& ij_vec, const set<int>& s_e_indices, vector<CalibSpace::Point3d2d>& p3d2ds, float D)
{
	vector<vector<cv::Point>> match_line_vec;
	vector<cv::Point> lidar_2d_pts;
	vector<int>  lidar_2d_indices;
	auto itr_3d = xyz_vec.begin();
	int i = 0;
	for (; itr_3d != xyz_vec.end(); itr_3d++, i++)
	{
		double d3[3] = { itr_3d->x, itr_3d->y, itr_3d->z };
		double d2[2];
		if (!OptimizeCeresWeighted::convertPoint3dTo2d(camRot, d3, d2))
		{
			continue;
		}
		cv::Point dd2(d2[0], d2[1]);

		lidar_2d_pts.push_back(dd2);
		lidar_2d_indices.push_back(i);

		int idx = -1;

		//	double DD = calcDD(D, dd2);
		double DD = D;
		//LOG(INFO) << D << "," << DD;

		double nearest_dist = 1000;
		if (!searchNeareatPoint(ij_vec, global_kdtree, dd2, idx, nearest_dist, DD))
		{
			continue;
		}

		cv::Point search_pt = ij_vec[idx];
		cv::Point delta = search_pt - dd2;
		if (/*abs(delta.x * 1.0 / delta.y) < 0.4*/s_e_indices.find(idx) != s_e_indices.end())
		{
			continue;
		}
		vector<cv::Point> line(2);
		line[0] = dd2;
		line[1] = search_pt;
		match_line_vec.push_back(line);

		CalibSpace::Point3d2d p3d2d;
		p3d2d.p3d = *itr_3d;
		p3d2d.p2d = search_pt;
		p3d2ds.push_back(p3d2d);
	}


	vector<vector<cv::Point>> reverse_match_line_vec;
	//建树
	if (lidar_2d_pts.size() > 0)
	{
		cv::Mat  mat_lidar_2d = cv::Mat(lidar_2d_pts).reshape(1);
		mat_lidar_2d.convertTo(mat_lidar_2d, CV_32F);
		cv::flann::Index tree = cv::flann::Index(mat_lidar_2d, cv::flann::KDTreeIndexParams(1));
		auto itr_2d = ij_vec.begin();
		i = 0;
		for (; itr_2d != ij_vec.end(); itr_2d++, i++)
		{
			const cv::Point& d2 = *itr_2d;
			int idx = -1;
			//	double DD = calcDD(D, dd2);
			double DD = D;

			double nearest_dist = 1000;
			if (!searchNeareatPoint(lidar_2d_pts, tree, d2, idx, nearest_dist, DD))
			{
				continue;
			}

			cv::Point search_pt = lidar_2d_pts[idx];

			cv::Point delta = search_pt - d2;


			vector<cv::Point> line(2);
			line[0] = d2;
			line[1] = search_pt;
			reverse_match_line_vec.push_back(line);

			if (/*abs(delta.x * 1.0 / delta.y) < 0.4*/
				xyz_s_e_indices.find(lidar_2d_indices[idx]) != xyz_s_e_indices.end())
			{
				continue;
			}

			CalibSpace::Point3d2d p3d2d;
			p3d2d.p3d = xyz_vec[lidar_2d_indices[idx]];
			p3d2d.p2d = d2;
			p3d2ds.push_back(p3d2d);
		}
		tree.release();
	}

	//第一次匹配的时候保存一下
	if (/*D > 100*/1)
	{
		cv::Mat match_mat(CalibSpace::IMG_HEIGHT, CalibSpace::IMG_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));;
		//		cv::cvtColor(match_mat, match_mat, cv::COLOR_GRAY2BGR);
		cv::polylines(match_mat, reverse_match_line_vec, false, cv::Scalar(255, 0, 0), 2);
		cv::polylines(match_mat, match_line_vec, false, cv::Scalar(0, 0, 255), 2);

		//cv::polylines(match_mat, m_lidar_2d_contours, false, cv::Scalar(255, 0, 0), 2);

		string temp_path = "kdtree/";
		cv::imwrite(temp_path + to_string(int(D)) + ".png", match_mat);
	}

	return p3d2ds.size() > 0;
}

bool Calib::optimizePnP(const vector<CalibSpace::Point3d2d>& ref_vec, vector<double>& camPose, vector<int>& inliers, bool ransac)
{
	cv::Mat rvec;
	cv::Mat tvec;
	if (!solvePnP(ref_vec, CalibSpace::intrisicMat, CalibSpace::distCoeffs, rvec, tvec, inliers, ransac))
	{
		return false;
	}

	rt2camPose(rvec, tvec, camPose);
	return true;
}



bool Calib::optimizePanoPnP(const vector<CalibSpace::Point3d2d>& ref_vec, vector<double>& camPose, vector<int>& inliers, bool ransac)
{
	
	vector<vector<double>> data(ref_vec.size());
	transform(ref_vec.begin(), ref_vec.end(), data.begin(), [](const auto& ref)->vector<double> {
		vector<double> d(5);
		d[0] = ref.p3d.x;
		d[1] = ref.p3d.y;
		d[2] = ref.p3d.z;
		d[3] = ref.p2d.x + CalibSpace::image_rect.tl().x;
		d[4] = ref.p2d.y + CalibSpace::image_rect.tl().y;
		return d;
	});
	Eigen::Matrix3d r;
	OptimizeCeresWeighted::calcRotation(camPose, r);

	Eigen::Vector3d t(camPose[0], camPose[1], camPose[2]);

	//RansacSolver rs;
	//rs.runRANSAC(data, camPose);

	return true;
}

void Calib::regist3D2DJC(const vector<vector<cv::Point3f>>& xyz_line_vec, const vector<vector<cv::Point>>& line_vec, vector<double>& camPose)
{
	vector<cv::Point3f> xyz_vec;
	getValid3dPoints(xyz_line_vec, xyz_vec);
	//ofstream of("a.txt");
	//of << xyz_vec;
	//of.close();

//	iterateJC(xyz_vec, line_vec, camPose);
}

void Calib::iterateJC(const map<int, Point3f_VEC>& xyz_vec_map, const cv::Mat& blur_mat, vector<double>& camPose)
{
	//*************************************blur***************************
	//OptimizeCeresWeighted::optimizeJC(xyz_vec_map, blur_mat, camPose);

	LOG(INFO) << "*************结果****************";
	LOG(INFO) << "estimated Xs,Ys,Zs,omega,pho,kappa = ";
	for (int i = 0; i < 6; i++)
	{
		LOG(INFO) << camPose[i];
	}
}

double Calib::iterateDistanceTransform(const map<int, Point3f_VEC>& xyz_vec_map, const map<int, LINE_VEC>& lines_map, vector<double>& camPose)
{
	cv::Mat mat;
	calcDistanceTransformImage(lines_map, mat);
	//cout << blur_mat;
	//OptimizeCeresWeighted::optimizeDistanceTransform(xyz_vec_map, mat, camPose);
	//OptimizeCeresWeighted::optimizeDistanceTransform(xyz_vec_map, mat, camPose);
	return 0;
}

void changeDistanceTransform(cv::Mat mat, int h, cv::Mat& blur_mat)
{
#if 1
	int h0 = CalibSpace::image_rect.height / 2.0;
	cv::Rect rect;
	cv::Mat tmp;
	
	int y = h0;
	
	rect = cv::Rect(0, y, CalibSpace::image_rect.width, h);
	mat(rect).setTo(0);
	/*mat(rect).copyTo(tmp);
	auto kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
	cv::dilate(tmp, tmp, kernel);
	tmp.copyTo(mat(rect));

	y = y + h;
	rect = cv::Rect(0, y, CalibSpace::image_rect.width, h0 - h);
	mat(rect).copyTo(tmp);
	kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
	cv::dilate(tmp, tmp, kernel);
	tmp.copyTo(mat(rect));*/

	cv::imwrite("0_mat.jpg", mat);
#else
	//kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
	//cv::dilate(mat, mat, kernel);
#endif
	//cv::distanceTransform(mat, blur_mat, DIST_L1, 3);
	cv::distanceTransform(mat, blur_mat, DIST_L2, 5);

	//cv::normalize(blur_mat, blur_mat, 0, 255, cv::NORM_MINMAX);
	//cv::imwrite("0_dt.jpg", blur_mat);
	//cv::imwrite("0_dilates.jpg", mat);
}

double Calib::iterateCrossEntropy(const map<int, Point3f_VEC>& xyz_vec_map,
	const map<int, LINE_VEC>& lines_map, 
	const float& pitch,
	vector<double>& camPose)
{
	cv::Mat mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));

	auto itr = lines_map.begin();
	for (; itr != lines_map.end(); itr++)
	{
		const auto& type = itr->first;
		const auto& line_vec = itr->second;
		
		//cv::polylines(mat, line_vec, false, cv::Scalar(255), 50);
		
		cv::polylines(mat, line_vec, false, cv::Scalar(255), 10);
	}
	cv::Mat blur_mat;
	//mat.convertTo(blur_mat, CV_32F, 1.0/255);
	
	//cv::distanceTransform(mat, blur_mat, DIST_L1, 3);
	//cv::distanceTransform(mat, blur_mat, DIST_L2, 5);
	int h = 150;
	if (abs(pitch) >= 1.0 / 180 * M_PI)
	{
		h = 50;
	}
	changeDistanceTransform(mat, h, blur_mat);

	//cv::normalize(blur_mat, blur_mat, 0, 255, cv::NORM_MINMAX);

	cv::normalize(blur_mat, blur_mat, 0, 1, cv::NORM_MINMAX);

	//cv::imwrite("0_dt.jpg", blur_mat);

	//cout << blur_mat;
	//OptimizeCeresWeighted::optimizeCrossEntropy(xyz_vec_map, blur_mat, camPose);
	return 0;
}

void Calib::setCameraImage(const cv::Mat& img)
{
	m_camera_img = img;
}


void Calib::initCamera(cv::Mat& intrisicMat, cv::Mat_<double>& distCoeffs)
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
bool Calib::solvePnP(const vector<CalibSpace::Point3d2d>& ref_pt_vec,
	const cv::Mat& intrisicMat, const cv::Mat_<double>& distCoeffs, cv::Mat& rVec, cv::Mat& tVec, vector<int>& inliers, bool ransac)
{
	if (ref_pt_vec.size() < 3)
	{
		return false;
	}
	vector<cv::Point3d> d3_pts(ref_pt_vec.size());
	vector<cv::Point2d> d2_pts(ref_pt_vec.size());

	transform(ref_pt_vec.begin(), ref_pt_vec.end(), d3_pts.begin(), [](const auto& p3d2d)->cv::Point3d {
		return p3d2d.p3d; });

	transform(ref_pt_vec.begin(), ref_pt_vec.end(), d2_pts.begin(), [](const auto& p3d2d)->cv::Point2d {
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

void Calib::rt2camPose(const cv::Mat& rVec, const cv::Mat& tVec, vector<double>& camPose)
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

	camPose[3] = euler[0];
	camPose[4] = euler[1];
	camPose[5] = euler[2];

	//Eigen::Vector3f cr(camPose[5], camPose[4], camPose[3]);
	//Eigen::Matrix3f CR;
	//TransRotation::eigenEuler2RotationMatrix(cr, CR);
	//cout << CR << endl;
}