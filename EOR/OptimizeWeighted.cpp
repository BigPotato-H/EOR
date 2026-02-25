#include "OptimizeWeighted.h"

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <chrono>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>


///////////for 3d/////////////
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/search/kdtree.h> 
#include <pcl/correspondence.h>
#include <pcl/filters/voxel_grid.h>
#include <omp.h>
//////////end

namespace OptimizeCeresWeighted {
// 第一部分：代价函数的计算模型，重载（）符号，仿函数的小技巧 
struct CostFunctor
{
	CostFunctor(cv::Point3f _XYZ, cv::Point _xy)
	{
		XYZ = _XYZ;
		xy = _xy;
	}
	// 残差的计算
	template <typename T>
	bool operator() (const T* const camPose, T* residual) const     // 残差
	{
		vector<T> cc = { camPose[0], camPose[1], camPose[2], camPose[3], camPose[4], camPose[5] };
		T pp_xy[2];
		T ppt_3d[3] = { T(XYZ.x), T(XYZ.y),T(XYZ.z) };

		if (convertPoint3dTo2d(cc, ppt_3d, pp_xy))
		{
			residual[0] = T(xy.x) - T(pp_xy[0]);
			residual[1] = T(xy.y) - T(pp_xy[1]);
		}
		else
		{
			residual[0] = T(CalibSpace::IMG_WIDTH);
			residual[1] = T(CalibSpace::IMG_HEIGHT);
		}

		return true; //千万不要写成return 0,要写成return true
	}
private:
	cv::Point xy;
	cv::Point3f XYZ;
};

#if 0
struct CostFunctorDistanceTransform
{
	CostFunctorDistanceTransform(const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>& _interpolator,
		const cv::Point3f& _XYZ, int _gray):
		interpolator(_interpolator), XYZ(_XYZ), gray(_gray){}

	// 残差的计算
	template <typename T>
	bool operator() (const T*const camPose, 
		T* residual)const      // 残差
	{

		vector<T> cc = { camPose[0], camPose[1], camPose[2], camPose[3], camPose[4], camPose[5] };

		vector<T> cc = { camPose[0], camPose[1], camPose[2], camPose[3], camPose[4], camPose[5], camPose[6] };

		T pp_xy[2];
		T pXYZ[3] = { T(XYZ.x), T(XYZ.y),T(XYZ.z) };

		T gray_img = T(0);
		T weight = T(weights[gray]);
		if (convertPoint3dTo2d(cc, pXYZ, pp_xy, false))
		{
			//pp_xy[0] = T(433);
		//pp_xy[1] = T(1502);

			interpolator.Evaluate(pp_xy[1], pp_xy[0], &gray_img);
		}
		
		residual[0] = T(255) - gray_img;
		residual[0] = residual[0] * weight;

		return true; 
	}
private:
	const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>& interpolator;
	cv::Point3f XYZ;
	int gray;
};
#else
struct CostFunctorWeightIntensity
{
	CostFunctorWeightIntensity(const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>& _interpolator,
		const cv::Point3f& _XYZ, int _gray, const vector<double>& _dimension) :
		interpolator(_interpolator), XYZ(_XYZ), gray(_gray), dimension(_dimension) {}

	// 残差的计算

	template <typename T>
	bool operator() (const T* const camPose,
		T* residual)const      // 残差
	{
		T gray_img = T(0);
		T weight = T(weights[gray]);
		
		T t_cam[7] = { T(0) };
		t_cam[6] = T(1.0);

		for (int i : dimension)
		{
			t_cam[i] = camPose[i];
		}
		T p3[3];
		convertPointByQuternion(t_cam, T(XYZ.x), T(XYZ.y), T(XYZ.z), p3);
		
		T p2[2];
		if (project2Image(p3, p2, false))
		{
			//pp_xy[0] = T(433);
		//pp_xy[1] = T(1502);

			interpolator.Evaluate(p2[1], p2[0], &gray_img);
		}
		if (p3[2] <= T(0))
		{
			gray_img = T(0);
			//p3[2] = T(0);
		}
		//residual[0] = T(255) - gray_img;
		residual[0] = T(1) - gray_img;
		residual[0] = residual[0] * weight;

		T D = T(70.0);
		//T D = T(40.0);
		T weight_distance =/*T(1)+*/ exp(p3[2] / D);
		if (gray == OC_lane)
		{
		//	residual[0] = residual[0] * weight_distance;
		}
		
		return true;
	}
private:
	const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>& interpolator;
	cv::Point3f XYZ;
	int gray;
	vector<double> dimension;
};

struct CostFunctorDistanceWeightIntensity
{
	CostFunctorDistanceWeightIntensity(const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>& _interpolator,
		const cv::Point3f& _XYZ, int _gray, const vector<double>& _dimension) :
		interpolator(_interpolator), XYZ(_XYZ), gray(_gray), dimension(_dimension) {}

	// 残差的计算

	template <typename T>
	bool operator() (const T* const camPose,
		T* residual)const      // 残差
	{
		T gray_img = T(0);
		T weight = T(weights[gray]);

		T t_cam[7] = { T(0) };
		t_cam[6] = T(1.0);

		for (int i : dimension)
		{
			t_cam[i] = camPose[i];
		}
		T p3[3];
		convertPointByQuternion(t_cam, T(XYZ.x), T(XYZ.y), T(XYZ.z), p3);

		T p2[2];
		if (project2Image(p3, p2, false))
		{
			//pp_xy[0] = T(433);
		//pp_xy[1] = T(1502);

			interpolator.Evaluate(p2[1], p2[0], &gray_img);
		}
		if (p3[2] <= T(0))
		{
			gray_img = T(0);
			//p3[2] = T(0);
		}
		//residual[0] = T(255) - gray_img;
		residual[0] = T(1) - gray_img;
		residual[0] = residual[0] * weight;

		T D = T(70.0);
		//T D = T(40.0);
		T weight_distance =/*T(1)+*/ exp(p3[2] / D);
		//residual[0] = residual[0] * weight_distance;

		return true;
	}
private:
	const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>& interpolator;
	cv::Point3f XYZ;
	int gray;
	vector<double> dimension;
};

struct CostFunctorIntensity
{
	CostFunctorIntensity(const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>& _interpolator,
		const cv::Point3f& _XYZ, int _gray) :
		interpolator(_interpolator), XYZ(_XYZ), gray(_gray) {}

	// 残差的计算

	template <typename T>
	bool operator() (const T* const camPose,
		T* residual)const      // 残差
	{
		
		T gray_img = T(0);
		T weight = T(weights[gray]);
		T p3[3];
		convertPointByQuternion(camPose, T(XYZ.x), T(XYZ.y), T(XYZ.z), p3);

		T p2[2];
		if (project2Image(p3, p2, false))
		{
			//pp_xy[0] = T(433);
		//pp_xy[1] = T(1502);

			interpolator.Evaluate(p2[1], p2[0], &gray_img);
		}
		if (p3[2] <= T(0))
		{
			gray_img = T(0);
		}
		//residual[0] = T(255) - gray_img;
		residual[0] = T(1) - gray_img;
		return true;
	}
private:
	const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>& interpolator;
	cv::Point3f XYZ;
	int gray;
	
};


#endif

template<typename T>
T sigmoid(T x)
{
	if (x > T(0))
		return T(1.0) / (T(1.0) + exp(-x));
	else
		return  exp(x) / (T(1.0) + exp(x));
}

template<typename T>
T tanh(T x)
{
	//return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	if (x > T(0))
		return (T(1.0) - exp(-T(2.0) * x)) / (T(1.0) + exp(-T(2.0) * x));
	else
		return (exp(T(2.0) * x) - 1) / (T(1.0) + exp(T(2.0) * x));
}

template <typename T>
T calculateAverage(const std::vector<T>& values) {
	if (values.size() == 0)
	{
		return T(0);
	}
	T sum = std::accumulate(values.begin(), values.end(), T(0.0));
	T average = sum / T(values.size());
	return average;
}

template <typename T>
T calculateStandardDeviation(const std::vector<T>& values) {
	if (values.size() == 0)
	{
		return T(0);
	}
	T average = calculateAverage(values);
	T sumSquaredDiff = T(0.0);

	for (const T& value : values) {
		T diff = value - average;
		sumSquaredDiff += diff * diff;
	}

	T variance = sumSquaredDiff / T(values.size());
//	T standardDeviation = std::sqrt(variance);
	T standardDeviation = variance;
	return standardDeviation;

}


struct CostFunctorLaneRoadSurface
{
	CostFunctorLaneRoadSurface(const vector<cv::Point3d>& _obj) { obj = _obj; }

	// 残差的计算
	template <typename T>
	bool operator() (const T* const camPose,
		T* residual)const      // 残差
	{
		T t_cam[7] = { T(0) };
		t_cam[6] = T(1.0);
		/*for (int k = 3; k < 6; k++)
		{
			t_cam[k] = camPose[k];
		}*/
		t_cam[3] = camPose[3];
		t_cam[5] = camPose[5];

		vector<T> y_vec;		
		for (int i = 0; i < obj.size(); i++)
		{
			const auto& XYZ = obj[i];
			T p[3];//in camera frame

			convertPointByQuternion(t_cam, T(XYZ.x), T(XYZ.y), T(XYZ.z), p);

			if (p[2] > T(-50) && p[2] < T(50) &&
				abs(p[0]) < T(20)) {
				y_vec.push_back(p[1]);
			}
		}
		T std_y = calculateStandardDeviation(y_vec);
		residual[0] = std_y * T(20);

		return true;
	}
private:
	vector<cv::Point3d> obj;

};




struct CostFunctorPole
{
	CostFunctorPole(const vector<cv::Point3d>& _obj) { obj = _obj; }

	// 残差的计算
	template <typename T>
	bool operator() (const T* const camPose, 
		T* residual)const      // 残差
	{
		T t_cam[7] = { T(0) };
		t_cam[6] = T(1.0);
		for (int k = 3; k < 6; k++)
	//	for (int k = 0; k < 7; k++)
		{
			t_cam[k] = camPose[k];
		}

		vector<T> x_vec;
		vector<T> z_vec;
		for (int i = 0; i < obj.size(); i++)
		{
			const auto& XYZ = obj[i];
			T p[3];//in camera frame
			convertPointByQuternion(t_cam, T(XYZ.x), T(XYZ.y), T(XYZ.z), p);

			if (p[2] > T(0) && p[2] < T(150) &&
				abs(p[0]) < T(20))
			{
				x_vec.push_back(p[0]);
				z_vec.push_back(p[2]);
			}
		}
		T std_x = calculateStandardDeviation(x_vec);
		T std_z = calculateStandardDeviation(z_vec);
		residual[0] = (std_x + std_z) * T(20);
		return true;
	}
private:
	vector<cv::Point3d> obj;

};

struct CostFunctorSign
{
	CostFunctorSign(const vector<cv::Point3d>& _obj) { obj = _obj; }

	// 残差的计算
	template <typename T>
	bool operator() (const T* const camPose,
		T* residual)const      // 残差
	{
		T t_cam[7] = { T(0) };
		t_cam[6] = T(1.0);
		for (int k = 3; k < 6; k++)
		//for (int k = 0; k < 7; k++)
		{
			t_cam[k] = camPose[k];
		}

		vector<T> z_vec;

		for (int i = 0; i < obj.size(); i++)
		{
			const auto& XYZ = obj[i];
			T p[3];//in camera frame
			convertPointByQuternion(t_cam, T(XYZ.x), T(XYZ.y), T(XYZ.z), p);
			if (p[2] > T(0) && p[2] < T(150) &&
				abs(p[0]) < T(20))
			{
				//same depth
				z_vec.push_back(p[2]);
			}
		}

		residual[0] = calculateStandardDeviation(z_vec) * T(20);

		return true;
	}
private:
	vector<cv::Point3d> obj;

};

struct CostFunctorCrosswalk
{
	CostFunctorCrosswalk(const vector<cv::Point3d>& _obj) { obj = _obj; }

	// 残差的计算
	template <typename T>
	bool operator() (const T* const camPose,
		T* residual)const      // 残差
	{
		vector<T> y_vec;
		vector<T> z_vec;
		
		for (int i = 0; i < obj.size(); i++)
		{
			const auto& XYZ = obj[i];
			T p[3];//in camera frame
			convertPointByQuternion(camPose, T(XYZ.x), T(XYZ.y), T(XYZ.z), p);

			
			if (p[2] > T(0) && p[2] < T(150) &&
				abs(p[0]) < T(20))
			{
				y_vec.push_back(p[1]);
				z_vec.push_back(p[2]);
				//T xy[2];
				//project2Image(p, xy, true);//ipm
				//if (i == 0)
				//{
				//	xy0[0] = xy[0];
				//	xy0[1] = xy[1];
				//}
				//else
				//{
				//	T y = xy[1] - xy0[1];
				//	diff += abs(y);
				//}
			}
		}
		T std_y = calculateStandardDeviation(y_vec);
		T std_z = calculateStandardDeviation(z_vec);
		residual[0] = std_y + std_z;

		return true;
	}
private:
	vector<cv::Point3d> obj;

};

void drawContours(int line_width, const LINE_VEC& line_vec, int type, cv::Mat& mat)
{
	if (type == OC_crosswalk || type == OC_t_sign)
	{
		bool be_closed = true;
		cv::polylines(mat, line_vec, be_closed, cv::Scalar(255), line_width);
		cv::drawContours(mat, line_vec, -1, cv::Scalar(255), cv::FILLED);
	}
	else
	{
		cv::drawContours(mat, line_vec, -1, cv::Scalar(255), line_width);
	}
}

void calcDistanceTransformImage(int type, const LINE_VEC& line_vec,
	cv::Mat& blur_mat)
{

	int x0 = 100;
	int y0 = 50;
	//int y0 = x0 * 1.0 / CalibSpace::IMG_WIDTH * CalibSpace::IMG_HEIGHT;

	int x1 = 300;
	int y1 = 150;
	
	cv::Rect rect0 = cv::Rect(CalibSpace::CX - x0, 
		CalibSpace::CY - y0, 2.0 * x0, 2.0 * y0);
	cv::Rect rect1 = cv::Rect(CalibSpace::CX - x1,
		CalibSpace::CY - y1, 2.0 * x1, 2.0 * y1);
	
	//最外圈
	
	//vector<int> line_width_vec = { 60,40,20 };
	//vector<int> line_width_vec = { 80,60,40 };
	vector<int> line_width_vec = { 100,80,60 };
	
	if (CalibSpace::data_item == "arg2")
	{
		line_width_vec = { 60,40,20 };
	}

	cv::Mat mat = cv::Mat(blur_mat.rows, blur_mat.cols, CV_8UC1, cv::Scalar(0));
	int line_width = line_width_vec[0];
	drawContours(line_width, line_vec, type, mat);
	//中间
	cv::Mat tmp1= cv::Mat(blur_mat.rows, blur_mat.cols, CV_8UC1, cv::Scalar(0));
	//cv::polylines(tmp1, line_vec, false, cv::Scalar(255), 30);
	line_width = line_width_vec[1];
	drawContours(line_width, line_vec, type, tmp1);
	tmp1(rect1).copyTo(mat(rect1));

	//内圈
	cv::Mat tmp0 = cv::Mat(blur_mat.rows, blur_mat.cols, CV_8UC1, cv::Scalar(0));
	line_width = line_width_vec[2];
	drawContours(line_width, line_vec, type, tmp0);

	tmp0(rect0).copyTo(mat(rect0));

	cv::distanceTransform(mat, blur_mat, cv::DIST_L2, 5);
 	//cv::normalize(blur_mat, blur_mat, 0, 255, cv::NORM_MINMAX);
	cv::normalize(blur_mat, blur_mat, 0, 1, cv::NORM_MINMAX);
	//cv::imwrite("0_layer_mat.jpg", mat);
	//cv::imwrite("0_layer_dt.jpg", blur_mat);
}
void calcBasicDistanceTransformImage(int type, const LINE_VEC& line_vec,
	cv::Mat& blur_mat)
{
	int line_width = 80;
	if (CalibSpace::data_item == "arg2")
	{
		line_width = 60;
	}
	
	cv::Mat mat = cv::Mat(blur_mat.rows, blur_mat.cols, CV_8UC1, cv::Scalar(0));
	drawContours(line_width, line_vec, type, mat);

	cv::distanceTransform(mat, blur_mat, cv::DIST_L2, 5);
	//cv::normalize(blur_mat, blur_mat, 0, 255, cv::NORM_MINMAX);
	cv::normalize(blur_mat, blur_mat, 0, 1, cv::NORM_MINMAX);
	//cv::imwrite("0_layer_mat.jpg", mat);
	//cv::imwrite("0_layer_dt.jpg", blur_mat);
}
map<int, float> weights;

void setQuaternionAngleLimits(Eigen::Quaterniond& quaternion, double minAngle, double maxAngle)
{
	// Convert the quaternion to Euler angles
	Eigen::Vector3d euler = quaternion.toRotationMatrix().eulerAngles(0, 1, 2);

	// Apply limits to the corresponding Euler angles
	euler = euler.cwiseMax(Eigen::Vector3d(minAngle, minAngle, minAngle));
	euler = euler.cwiseMin(Eigen::Vector3d(maxAngle, maxAngle, maxAngle));

	// Convert the Euler angles back to a quaternion
	quaternion = Eigen::AngleAxisd(euler[2], Eigen::Vector3d::UnitZ())
		* Eigen::AngleAxisd(euler[1], Eigen::Vector3d::UnitY())
		* Eigen::AngleAxisd(euler[0], Eigen::Vector3d::UnitX());
}

void setBoundaryLimits(vector<double> limits,
	double* poseQuaternion, ceres::Problem& problem)
{
#if 1
	Eigen::VectorXd qq(7, 1);
	for (int i = 0; i < 7; i++)
	{
		qq[i] = poseQuaternion[i];
	}
	vector<double> cam;
	CalibSpace::QuaternionTopose6DoF(qq, cam);

	vector<double> cam_low(6);
	vector<double> cam_high(6);
	for (int i = 0; i < 6; i++)
	{
		cam_low[i] = - limits[i];
		cam_high[i] =  + limits[i];
	}

	vector<double> poseQuaternion_low(7);
	vector<double> poseQuaternion_high(7);
	CalibSpace::pose6DoFToQuaternion(cam_low, poseQuaternion_low);
	CalibSpace::pose6DoFToQuaternion(cam_high, poseQuaternion_high);
	for (int i = 0; i < 7; i++)
	{
		double minb = min(poseQuaternion_low[i], poseQuaternion_high[i]);
		double maxb = max(poseQuaternion_low[i], poseQuaternion_high[i]);
		problem.SetParameterLowerBound(poseQuaternion, i, minb);
		problem.SetParameterUpperBound(poseQuaternion, i, maxb);
	}

#else
	for (int i = 0; i < 6; i++)
	{
		double min_b = 0;
		double max_b = 0;
		min_b = poseQuaternion[i] - limits[i];
		max_b = poseQuaternion[i] + limits[i];
		problem.SetParameterLowerBound(poseQuaternion, i, min_b);
		problem.SetParameterUpperBound(poseQuaternion, i, max_b);
	}
	problem.SetParameterLowerBound(poseQuaternion, 6, 1 - limits[6]);
	problem.SetParameterUpperBound(poseQuaternion, 6, 1);

#endif
}

bool calcEORWeight(StatisticData_MAP sd_map, OptimizePara& op)
{
	set<ObjectClassification> oc_set = { OC_lane ,OC_pole ,OC_t_sign };

	auto max_occ = max_element(sd_map.begin(), sd_map.end(), [](const auto& ele1, const auto& ele2) ->bool {
		return ele1.second.occupancy_ratio_3d + ele1.second.occupancy_ratio_2d < ele2.second.occupancy_ratio_3d + ele2.second.occupancy_ratio_2d;
		});
	int max_occ_obj = max_occ->first;

	auto min_dop = max_element(sd_map.begin(), sd_map.end(), [](const auto& ele1, const auto& ele2) ->bool {
		return ele1.second.dop_3d + ele1.second.dop_2d < ele2.second.dop_3d + ele2.second.dop_2d;
		});
	int min_dop_obj = min_dop->first;

	auto max_sim = max_element(sd_map.begin(), sd_map.end(), [](const auto& ele1, const auto& ele2) ->bool {
		return ele1.second.local_similarity < ele2.second.local_similarity;
		});
	int max_sim_obj = max_sim->first;


	op.occ_major = max_occ_obj;
	op.dop_major = min_dop_obj;
	op.long_major = max_sim_obj;

	const auto& sd_lane = sd_map[OC_lane];
	const auto& sd_pole = sd_map[OC_pole];
	const auto& sd_sign = sd_map[OC_t_sign];

	/*if (sd_lane.occupancy_count_3d > 0 &&
		sd_lane.occupancy_count_2d > 0)
	{
		double base = sd_lane.occupancy_ratio_3d + sd_lane.occupancy_ratio_2d;

		op.weights[OC_lane] = 1.0;
		op.weights[OC_pole] = (sd_pole.occupancy_ratio_3d + sd_pole.occupancy_ratio_2d) / base;
		op.weights[OC_t_sign] = (sd_sign.occupancy_ratio_3d + sd_sign.occupancy_ratio_2d) / base;
	}
	else
	{
		op.weights[OC_lane] = (sd_lane.occupancy_ratio_3d + sd_lane.occupancy_ratio_2d) / 2.0;
		op.weights[OC_pole] = (sd_pole.occupancy_ratio_3d + sd_pole.occupancy_ratio_2d) / 2.0;
		op.weights[OC_t_sign] = (sd_sign.occupancy_ratio_3d + sd_sign.occupancy_ratio_2d) / 2.0;
	}*/

	double base = sd_lane.occupancy_ratio_3d;

	op.weights[OC_lane] = 1.0;
	op.weights[OC_pole] = 1.0;
	op.weights[OC_t_sign] = 0.5;
//	if (sd_lane.occupancy_count_3d > sd_pole.occupancy_count_3d)

	if (0)	
	{
		double base = sd_lane.occupancy_ratio_3d;

		op.weights[OC_lane] = 1.0;
		op.weights[OC_pole] = (sd_pole.occupancy_ratio_3d) / base;
		op.weights[OC_t_sign] = (sd_sign.occupancy_ratio_3d) / base;
	}

	return true;
}
void addObjectIntensityProblem(
	const map<int, ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>>& type_inter_map,
	const map<int, HDObject_VEC>& local_local_hdobj_vec,
	double* camPose, 
	StatisticData_MAP sd_map,
	ceres::Problem& problem)
{
	OptimizePara op;
	calcEORWeight(sd_map, op);
	weights = op.weights;

	map<int, vector<cv::Point3d>> type_hd_points;
	for_each(local_local_hdobj_vec.begin(), local_local_hdobj_vec.end(), [&](const auto& _pair) {
		int type = _pair.first;
		const auto& objs = _pair.second;
		auto& shp = type_hd_points[type];
		for_each(objs.begin(), objs.end(), [&](const auto& obj) {
			
			copy(obj.shape.begin(), obj.shape.end(), back_inserter(shp));
			});	
		});

	auto sd_lane = sd_map[ObjectClassification(OC_lane)];
	auto sd_pole = sd_map[ObjectClassification(OC_pole)];
	auto sd_sign = sd_map[ObjectClassification(OC_t_sign)];


	auto itr_type = type_hd_points.begin();

	for (; itr_type != type_hd_points.end(); itr_type++)
	{
		int type = itr_type->first;
		const auto& xyz_vec = itr_type->second;
		auto find_inter = type_inter_map.find(type);
		if (find_inter == type_inter_map.end())
		{
			continue;
		}
		auto sd = sd_map[ObjectClassification(type)];
		if (sd.occupancy_count_2d == 0 ||
			sd.occupancy_count_3d == 0)
		{
			continue;
		}

		const auto& base_interpolator = find_inter->second;
		vector<double> dimension = { 0,1,2,3,4,5,6 };


		if (sd_lane.local_similarity < 0.1 &&
			type != OC_lane)
		{
			//dimension = { 0,2,3,4,5,6 };
			//dimension = { 0,3,4,5,6 };
			dimension = { 3,4,5,6 };
			weights[type] = op.weights[type] * 2;
		//	weights[type] = 2;
		}

		if (sd_lane.local_similarity > 0.1 &&
			type != OC_lane)
		{
			if (sd.occupancy_ratio_3d > sd_lane.occupancy_ratio_3d)
			{
				dimension = { 0,1,2,3,4,5,6 };
				weights[type] = op.weights[type] * 2;
			}

// 
// 			if (//(sd.occupancy_ratio_3d > 0.25 && sd.local_similarity > 1.4) ||
// 				sd.occupancy_ratio_3d > sd_lane.occupancy_ratio_3d)
// 			{
// 				dimension = { 0,1,2,3,4,5,6 };
// 				weights[type] = op.weights[type] * 2;
// 			}
		}

		if (sd.local_similarity < 0.1)
		{
			dimension = { 0,1,3,4,5,6 };
		}

		if (sd.dop_2d > 0.4 || sd.dop_3d > 0.4)
		{
			weights[type] = op.weights[type] * 0.5;
		}

 		//weights[OC_pole] = 1;
 		//weights[OC_t_sign] = 0.25;

		//添加边界约束
		vector<double> limits = { 1.5,1.5,1.5,0.2,0.2,0.2,0.02 };

		for (int i = 0; i < xyz_vec.size(); ++i)
		{
			const auto& xyz = xyz_vec[i];	
			ceres::CostFunction* costfunction = new ceres::AutoDiffCostFunction<CostFunctorWeightIntensity, 1, 7>(
					new CostFunctorWeightIntensity(base_interpolator, xyz, type, dimension));
			problem.AddResidualBlock(costfunction,
				new ceres::SoftLOneLoss(1.0),
				//new ceres::TukeyLoss(1.0),
				camPose);
			setBoundaryLimits(limits, camPose, problem);
		}

		
	}
	

}

void addLaneMarkingIntensityProblem(
	const map<int, ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>>& type_inter_map,
	const map<int, HDObject_VEC>& local_local_hdobj_vec,
	double* camPose,
	ceres::Problem& problem)
{
	map<int, vector<cv::Point3d>> type_hd_points;
	for_each(local_local_hdobj_vec.begin(), local_local_hdobj_vec.end(), [&](const auto& _pair) {
		int type = _pair.first;
		const auto& objs = _pair.second;
		auto& shp = type_hd_points[type];
		for_each(objs.begin(), objs.end(), [&](const auto& obj) {

			copy(obj.shape.begin(), obj.shape.end(), back_inserter(shp));
			});
		});

	int type = OC_lane;
	auto find_lane = type_hd_points.find(OC_lane);
	if (find_lane == type_hd_points.end())
	{
		return;
	}
	auto find_inter = type_inter_map.find(type);
	if (find_inter == type_inter_map.end())
	{
		return;
	}

	const auto& xyz_vec = find_lane->second;
	const auto& base_interpolator = find_inter->second;

	vector<double> dimension = { 0,1,2,3,4,5,6 };
	//添加边界约束
	vector<double> limits = { 1.5,1.5,1.5,0.2,0.2,0.2,0.02 };
	for (int i = 0; i < xyz_vec.size(); ++i)
	{
		const auto& xyz = xyz_vec[i];
		ceres::CostFunction* costfunction = new ceres::AutoDiffCostFunction<CostFunctorDistanceWeightIntensity, 1, 7>(
			new CostFunctorDistanceWeightIntensity(base_interpolator, xyz, type, dimension));
		problem.AddResidualBlock(costfunction,
			new ceres::SoftLOneLoss(1.0),
			//new ceres::TukeyLoss(1.0),
			camPose);
		setBoundaryLimits(limits, camPose, problem);
	}


}


void addBasicIntensityProblem(
	const map<int, ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>>& type_inter_map,
	const map<int, HDObject_VEC>& local_local_hdobj_vec,
	double* camPose,
	ceres::Problem& problem)
{
	//cv::Mat blur_mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
	//cv::Mat blur_mat(1280, 800, CV_8UC3, cv::Scalar(0, 0, 0));

	map<int, vector<cv::Point3d>> type_hd_points;
	for_each(local_local_hdobj_vec.begin(), local_local_hdobj_vec.end(), [&](const auto& _pair) {
		int type = _pair.first;
		const auto& objs = _pair.second;
		auto& shp = type_hd_points[type];
		for_each(objs.begin(), objs.end(), [&](const auto& obj) {

			copy(obj.shape.begin(), obj.shape.end(), back_inserter(shp));
			});
		});

	auto itr_type = type_hd_points.begin();
	for (; itr_type != type_hd_points.end(); itr_type++)
	{
		int type = itr_type->first;
		const auto& xyz_vec = itr_type->second;
		auto find_inter = type_inter_map.find(type);
		if (find_inter == type_inter_map.end())
		{
			continue;
		}

		//const auto& image_array = find_inter->second;
		const auto& base_interpolator = find_inter->second;
		//ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> base_interpolator(image_array);

		for (int i = 0; i < xyz_vec.size(); ++i)
		{
			const auto& xyz = xyz_vec[i];
			ceres::CostFunction* costfunction =
				new ceres::AutoDiffCostFunction<CostFunctorIntensity, 1, 7>(
					new CostFunctorIntensity(base_interpolator, xyz, type));

			problem.AddResidualBlock(costfunction,
				new ceres::SoftLOneLoss(1.0),
				//new ceres::TukeyLoss(1.0),
				camPose);
			//添加边界约束
			vector<double> limits = { 1.5,1.5,1.5,0.2,0.2,0.2,0.02 };
			setBoundaryLimits(limits, camPose, problem);
		}
	}


}

// Function to calculate the approximate centerline of a polygon
void calculateCenterline(const std::vector<cv::Point3d>& polygon, std::vector<cv::Point3d>& centerline)
{
	if (polygon.size() == 0)
	{
		return;
	}
	cv::Rect2d rect = cv::boundingRect(polygon);

	size_t numEdges = polygon.size();
	map<double, int> length_idx_map;
	for (size_t i = 0; i < numEdges; ++i) 
	{
		cv::Point3d p1 = polygon[i];
		cv::Point3d p2 = polygon[(i + 1) % numEdges]; // Wrap around to the first point for last edge

		int nextIdx = (i + 1) % numEdges;
		double edgeLength = cv::norm(polygon[i] - polygon[nextIdx]);
		length_idx_map.insert(make_pair(edgeLength, i));
	}
	
	centerline.resize(2);
	auto itr = length_idx_map.rbegin();
	// longest one
	int i1 = itr++ ->second;
	// second longest one
	int i2 = itr->second;
	centerline[0] = (polygon[i1] + polygon[(i2 + 1) % numEdges]) / 2.0;
	centerline[1] = (polygon[i1 +1] + polygon[i2]) / 2.0;

}


void addLaneGeometryProblem(const HDObject_VEC& objs,
	double* camPose, ceres::Problem& problem)
{
	vector<double> limits = { 1.5,1.5,1.5,0.2,0.2,0.2,0.02 };

	std::vector<cv::Point3d> road;
	for_each(objs.begin(), objs.end(), [&](const auto& obj) {
		const auto& shape = obj.shape;
		copy(shape.begin(), shape.end(), back_inserter(road));
		});
	ceres::CostFunction* costfunction = nullptr;
	costfunction = new ceres::AutoDiffCostFunction<CostFunctorLaneRoadSurface, 1, 7>(
		new CostFunctorLaneRoadSurface(road));
	problem.AddResidualBlock(costfunction,
		new ceres::SoftLOneLoss(1.0),
		camPose);
	
	setBoundaryLimits(limits, camPose, problem);
}

void addObjectGeometryProblem(const map<int, HDObject_VEC>& local_local_hdobj_vec,
	double* camPose, ceres::Problem& problem)
{

	auto itr_obj = local_local_hdobj_vec.begin();
	for (; itr_obj != local_local_hdobj_vec.end(); itr_obj++)
	{
		int type = itr_obj->first;
		const auto& objs = itr_obj->second;
		if (type == OC_lane)
		{
			addLaneGeometryProblem(objs, camPose, problem);
			continue;
		}

		for_each(objs.begin(), objs.end(), [&](const auto& obj) {
			const auto& shape = obj.shape;
			std::vector<cv::Point3d> centerline;
			if (type == OC_crosswalk)
			{
				calculateCenterline(shape, centerline);
			}

			ceres::CostFunction* costfunction = nullptr;
			switch (type)
			{
			case OC_pole:
				costfunction = new ceres::AutoDiffCostFunction<CostFunctorPole, 1, 7>(
					new CostFunctorPole(shape));
				break;
			case OC_t_sign:
				costfunction = new ceres::AutoDiffCostFunction<CostFunctorSign, 1, 7>(
					new CostFunctorSign(shape));
				break;
			case OC_crosswalk:
				//costfunction = new ceres::AutoDiffCostFunction<CostFunctorCrosswalk, 1, 7>(
				//	new CostFunctorCrosswalk(centerline));
				break;
			default:
				break;
			}
			if (costfunction != nullptr)
			{
				problem.AddResidualBlock(costfunction,
					new ceres::SoftLOneLoss(1.0),
					camPose);
				//添加边界约束
				vector<double> limits = { 1.5,1.5,1.5,0.2,0.2,0.2,0.02 };
				//vector<double> limits = { 0.5,0.5,2.5,0.2,0.2,0.2};
				setBoundaryLimits(limits, camPose, problem);
			}
		});
	}


}

#if 0
void optimizeGeometryAndIntensity(const map<int, LINE_VEC>& line_vec_map,
	const HDObject_VEC& local_local_hdobj_vec,
	vector<double>& cam)
{
	weights.insert(make_pair(OC_lane, 1.0));
	weights.insert(make_pair(OC_pole, 1.2));
	weights.insert(make_pair(OC_t_sign, 0.5));
	weights.insert(make_pair(OC_crosswalk, 0.5));

	double camPose[6];
	for (int i = 0; i < 6; i++)
	{
		camPose[i] = cam[i];
	}
	//第二部分：构建寻优问题
	ceres::Problem problem;
	///////////////////////////////////////intensity
	map<int, cv::Mat> type_mat_map;
	map<int, ceres::Grid2D<float, 1>> type_grid_map;
	map<int, ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>> type_interpolator_map;
	auto itr_type = line_vec_map.begin();
	for (; itr_type != line_vec_map.end(); itr_type++)
	{
		int type = itr_type->first;

		cv::Mat& blur_mat = type_mat_map[type];
	//	cv::Mat 
		blur_mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
		auto find_lines = line_vec_map.find(type);
		if (find_lines == line_vec_map.end())
		{
			continue;
		}
		const LINE_VEC& lines = find_lines->second;
		calcDistanceTransformImage(lines, blur_mat);
		//ceres::Grid2D<float, 1> image_array((float*)blur_mat.data, 0, blur_mat.size().height, 0, blur_mat.size().width);

		type_grid_map.emplace(type, ceres::Grid2D<float, 1>((float*)blur_mat.data, 0, blur_mat.size().height, 0, blur_mat.size().width));
		ceres::Grid2D<float, 1>& image_array = type_grid_map.find(type)->second;
		type_interpolator_map.emplace(type, image_array);
		
	}
	
	addObjectIntensityProblem(type_interpolator_map, local_local_hdobj_vec, camPose, problem);
		
	//////////////////////////////////////geometry
	addObjectGeometryProblem(local_local_hdobj_vec, camPose, problem);
	//添加边界约束
		//x
	for (int i = 0; i < 6; i++)
	{
		double min_b = 0;
		double max_b = 0;
		if (i < 3)
		{
			min_b = camPose[i] - 1.5;
			max_b = camPose[i] + 1.5;
		}
		else
		{
			/*min_b = camPose[i] - 0.08;
			max_b = camPose[i] + 0.08;*/
			min_b = camPose[i] - 0.2;
			max_b = camPose[i] + 0.2;
		}
		problem.SetParameterLowerBound(camPose, i, min_b);
		problem.SetParameterUpperBound(camPose, i, max_b);
	}

	// 第三部分：配置求解器
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;  // 配置增量方程的解法
	options.minimizer_progress_to_stdout = false;   // 输出到cout

	ceres::Solver::Summary summary;  // 优化信息
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	ceres::Solve(options, &problem, &summary);  // 开始优化
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

	for (int i = 0; i < 6; i++)
	{
		cam[i] = camPose[i];
	}
}
#else




void optimizeEnhancedObjectRegistration(const map<int, LINE_VEC>& line_vec_map,
	const map<int, HDObject_VEC>& local_local_hdobj_vec,
	vector<double>& cam,
	StatisticData_MAP sd_map)
{
	double poseQuaternion[7];
	vector<double> qq_Pose;
	CalibSpace::pose6DoFToQuaternion(cam, qq_Pose);
	for (int i = 0; i < 7; i++)
	{
		poseQuaternion[i] = qq_Pose[i];
	}
 	
	//第二部分：构建寻优问题
	ceres::Problem problem;
	///////////////////////////////////////intensity
	map<int, cv::Mat> type_mat_map;
	map<int, ceres::Grid2D<float, 1>> type_grid_map;
	map<int, ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>> type_interpolator_map;
	auto itr_type = line_vec_map.begin();
	for (; itr_type != line_vec_map.end(); itr_type++)
	{
		int type = itr_type->first;

		cv::Mat& blur_mat = type_mat_map[type];
		//	cv::Mat 
		blur_mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
		auto find_lines = line_vec_map.find(type);
		if (find_lines == line_vec_map.end())
		{
			continue;
		}
		const LINE_VEC& lines = find_lines->second;
		calcDistanceTransformImage(type, lines, blur_mat);
		//ceres::Grid2D<float, 1> image_array((float*)blur_mat.data, 0, blur_mat.size().height, 0, blur_mat.size().width);

		type_grid_map.emplace(type, ceres::Grid2D<float, 1>((float*)blur_mat.data, 0, blur_mat.size().height, 0, blur_mat.size().width));
		ceres::Grid2D<float, 1>& image_array = type_grid_map.find(type)->second;
		type_interpolator_map.emplace(type, image_array);

	}

	addObjectIntensityProblem(type_interpolator_map, local_local_hdobj_vec, poseQuaternion, sd_map,
		problem);
	//////////////////////////////////////geometry
	addObjectGeometryProblem(local_local_hdobj_vec, poseQuaternion, problem);

	// 第三部分：配置求解器
	ceres::Solver::Options options;
//	options.linear_solver_type = ceres::DENSE_QR;  // 配置增量方程的解法
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.dynamic_sparsity = true;
	options.minimizer_progress_to_stdout = false;   // 输出到cout

	ceres::Solver::Summary summary;  // 优化信息
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	ceres::Solve(options, &problem, &summary);  // 开始优化
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	Eigen::VectorXd qq(7,1);
	for (int i = 0; i < 7; i++)
	{
		qq[i] = poseQuaternion[i];
	}
	CalibSpace::QuaternionTopose6DoF(qq, cam);
}

void optimizeLaneMaringOnlyRegistration(const map<int, LINE_VEC>& line_vec_map,
	const map<int, HDObject_VEC>& local_local_hdobj_vec,
	vector<double>& cam)
{
	weights.clear();
	weights.insert(make_pair(OC_lane, 1.0));

	double poseQuaternion[7];
	vector<double> qq_Pose;
	CalibSpace::pose6DoFToQuaternion(cam, qq_Pose);
	for (int i = 0; i < 7; i++)
	{
		poseQuaternion[i] = qq_Pose[i];
	}

	//第二部分：构建寻优问题
	ceres::Problem problem;
	///////////////////////////////////////intensity
	map<int, cv::Mat> type_mat_map;
	map<int, ceres::Grid2D<float, 1>> type_grid_map;
	map<int, ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>> type_interpolator_map;
	int type = OC_lane;

	cv::Mat& blur_mat = type_mat_map[type];
	//	cv::Mat 
	blur_mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
	auto find_lines = line_vec_map.find(type);
	if (find_lines == line_vec_map.end())
	{
		return;
	}
	const LINE_VEC& lines = find_lines->second;
	calcDistanceTransformImage(type, lines, blur_mat);
	//ceres::Grid2D<float, 1> image_array((float*)blur_mat.data, 0, blur_mat.size().height, 0, blur_mat.size().width);

	type_grid_map.emplace(type, ceres::Grid2D<float, 1>((float*)blur_mat.data, 0, blur_mat.size().height, 0, blur_mat.size().width));
	ceres::Grid2D<float, 1>& image_array = type_grid_map.find(type)->second;
	type_interpolator_map.emplace(type, image_array);

	addLaneMarkingIntensityProblem(type_interpolator_map, local_local_hdobj_vec, poseQuaternion, problem);

	//////////////////////////////////////geometry
	addObjectGeometryProblem(local_local_hdobj_vec, poseQuaternion, problem);


	// 第三部分：配置求解器
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // 配置增量方程的解法
	options.dynamic_sparsity = true;
	options.minimizer_progress_to_stdout = false;   // 输出到cout

	ceres::Solver::Summary summary;  // 优化信息
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	ceres::Solve(options, &problem, &summary);  // 开始优化
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	Eigen::VectorXd qq(7, 1);
	for (int i = 0; i < 7; i++)
	{
		qq[i] = poseQuaternion[i];
	}
	CalibSpace::QuaternionTopose6DoF(qq, cam);
}

void optimizeMultiObjectRegistration(const map<int, LINE_VEC>& line_vec_map,
	const map<int, HDObject_VEC>& local_local_hdobj_vec,
	vector<double>& cam)
{
	double poseQuaternion[7];
	vector<double> qq_Pose;
	CalibSpace::pose6DoFToQuaternion(cam, qq_Pose);
	for (int i = 0; i < 7; i++)
	{
		poseQuaternion[i] = qq_Pose[i];
	}

	//第二部分：构建寻优问题
	ceres::Problem problem;
	///////////////////////////////////////intensity
	map<int, cv::Mat> type_mat_map;
	map<int, ceres::Grid2D<float, 1>> type_grid_map;
	map<int, ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>>> type_interpolator_map;
	auto itr_type = line_vec_map.begin();
	for (; itr_type != line_vec_map.end(); itr_type++)
	{
		int type = itr_type->first;

		cv::Mat& blur_mat = type_mat_map[type];
		//	cv::Mat 
		blur_mat = cv::Mat(CalibSpace::image_rect.height, CalibSpace::image_rect.width, CV_8UC1, cv::Scalar(0));
		auto find_lines = line_vec_map.find(type);
		if (find_lines == line_vec_map.end())
		{
			continue;
		}
		const LINE_VEC& lines = find_lines->second;
		calcBasicDistanceTransformImage(type, lines, blur_mat);
		//calcDistanceTransformImage(type, lines, blur_mat);
		//ceres::Grid2D<float, 1> image_array((float*)blur_mat.data, 0, blur_mat.size().height, 0, blur_mat.size().width);

		type_grid_map.emplace(type, ceres::Grid2D<float, 1>((float*)blur_mat.data, 0, blur_mat.size().height, 0, blur_mat.size().width));
		ceres::Grid2D<float, 1>& image_array = type_grid_map.find(type)->second;
		type_interpolator_map.emplace(type, image_array);

	}

	addBasicIntensityProblem(type_interpolator_map, local_local_hdobj_vec, poseQuaternion, problem);

	
	// 第三部分：配置求解器
	ceres::Solver::Options options;
	//options.linear_solver_type = ceres::DENSE_QR;  // 配置增量方程的解法
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;  // 配置增量方程的解法
	options.dynamic_sparsity = true;
	options.minimizer_progress_to_stdout = false;   // 输出到cout

	ceres::Solver::Summary summary;  // 优化信息
	chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
	ceres::Solve(options, &problem, &summary);  // 开始优化
	chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
	chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
	Eigen::VectorXd qq(7, 1);
	for (int i = 0; i < 7; i++)
	{
		qq[i] = poseQuaternion[i];
	}
	CalibSpace::QuaternionTopose6DoF(qq, cam);

}
#endif

}