#pragma once
#ifndef OPTIMIZE_CERES
#define OPTIMIZE_CERES
#endif

#include "DataSet.h"
#include<Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace OptimizeCeresWeighted
{
	template <typename T>
	void calcRotation(const vector<T>& camPose, Eigen::Matrix<T, 3, 3>& rotation_matrix);

	template <typename T>
	void projectPoint(const T* p, T& x, T& y);

	template <typename T>
	void projectPoly(const T* p, T& x, T& y);

	template <typename T>
	void  projectPano(const T* p, T& x, T& y);

	template <typename T>
	bool convertPixel2Camera3d(T x, T y, Eigen::Matrix<T, 3, 1>& p3f);
	template <typename T> T getyInCamera();
	template <typename T> T getzInCamera(const T& yc, const T& row);
	template <typename T> T getxInCamera(const T& zc, const T& col);
	template <typename T>
	void convertPointByEigen(const vector<T>& camPose, T _X, T _Y, T _Z, T* p);

	template <typename T>
	bool project2Image(T* p, T* xy, bool ipm = true);

	template <typename T> 
	void Image2IPM( T* xy);

	template <typename T>
	bool convertPoint3dTo2d(const vector<T>& camPose, const T*const pt_3d, T*pt_2d, bool ipm = true);


	void optimizeEnhancedObjectRegistration(const map<int, LINE_VEC>& line_vec_map,
		const map<int, HDObject_VEC>& local_local_hdobj_vec,
		vector<double>& cam,
		StatisticData_MAP sd_map);

	void optimizeLaneMaringOnlyRegistration(const map<int, LINE_VEC>& line_vec_map,
		const map<int, HDObject_VEC>& local_local_hdobj_vec,
		vector<double>& cam);

	void optimizeMultiObjectRegistration(const map<int, LINE_VEC>& line_vec_map,
		const map<int, HDObject_VEC>& local_local_hdobj_vec,
		vector<double>& cam);

	template <typename T>
	void calcRotation(const vector<T>& camPose, Eigen::Matrix<T, 3, 3>& rotation_matrix)
	{
		//http://www.mamicode.com/info-detail-2730205.html
		//https://blog.csdn.net/u011092188/article/details/77430988
		//旋转矩阵（3X3）:Eigen::Matrix3d
		//	旋转向量（3X1） : Eigen::AngleAxisd
		//	四元数（4X1） : Eigen::Quaterniond
		//	平移向量（3X1） : Eigen::Vector3d
		//	变换矩阵（4X4） : Eigen::Isometry3d
		Eigen::AngleAxis<T> xAngle(camPose[3], Eigen::Matrix<T, 3, 1>::UnitX());
		Eigen::AngleAxis<T> yAngle(camPose[4], Eigen::Matrix<T, 3, 1>::UnitY());
		Eigen::AngleAxis<T> zAngle(camPose[5], Eigen::Matrix<T, 3, 1>::UnitZ());

		rotation_matrix = xAngle * yAngle * zAngle;
	}

	template <typename T>
	void  projectPoint(const T* p, T& x, T& y)
	{
		T fx = T(CalibSpace::FX); //mm
		T fy = T(CalibSpace::FY); //mm

		if (CalibSpace::distCoeffs.at<double>(0, 0) != 0)
		{
#if 0
			//fisheye
			T D[4];
			for (int i = 0; i < 4; i++)
			{
				D[i] = T(CalibSpace::distCoeffs.at<T>(0, i));
			}
			T a = p[0] / p[2];
			T b = p[1] / p[2];
			T r = sqrt(a * a + b * b);
			T theta = atan(r);
			T theta_d = theta * (T(1) + D[0] * pow(theta, 2) + D[1] * pow(theta, 4) + D[2] * pow(theta, 6) + D[3] * pow(theta, 8));
			a = (theta_d / r) * a;
			b = (theta_d / r) * b;
			x = fx * a + T(CalibSpace::CX);
			y = fy * b + T(CalibSpace::CY);
#endif
			T k1 = T(CalibSpace::distCoeffs.at<double>(0, 0));
			T k2 = T(CalibSpace::distCoeffs.at<double>(0, 1));
			T p1 = T(CalibSpace::distCoeffs.at<double>(0, 2));
			T p2 = T(CalibSpace::distCoeffs.at<double>(0, 3));
			T k3 = T(CalibSpace::distCoeffs.at<double>(0, 4));
			x = p[0] / p[2];
			y = p[1] / p[2];
			T r2 = x * x + y * y;
			T x_radial = x * (T(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
			T y_radial = y * (T(1.0) + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
			T x_tangential = T(2.0) * p1 * x * y + p2 * (r2 + T(2.0) * x * x);
			T y_tangential = T(2.0) * p2 * x * y + p1 * (r2 + T(2.0) * y * y);
			T xd = x_radial + x_tangential;
			T yd = y_radial + y_tangential;

			x = fx * xd + T(CalibSpace::CX);
			y = fy * yd + T(CalibSpace::CY);

			T a = fx * p[0] / p[2] + T(CalibSpace::CX);
			T b = fy * p[1] / p[2] + T(CalibSpace::CY);

			int c = 0;
		}
		else
		{
			x = fx * p[0] / p[2] + T(CalibSpace::CX);
			y = fy * p[1] / p[2] + T(CalibSpace::CY);
		}
#if 1						   //无畸变
		//x = fx * p[0] / p[2] + T(CalibSpace::CX);
		//y = fy * p[1] / p[2] + T(CalibSpace::CY);
#else

#endif
	}

	template <typename T>
	void  projectPano(const T* p, T& x, T& y)
	{
#if 0
		const T& _x = p[0];
		const T& _y = -p[1];
		const T& _z = -p[2];

		T rho = sqrt(_x * _x + _y * _y + _z * _z);

		//相机坐标系到像平面，原点到左上角了
		T theta = M_PI - acos(_z / rho);
		//atan2的范围是-π到π
		T phi = atan2(_y, _x) + M_PI;


		x = phi / M_PI * T(CalibSpace::IMG_HEIGHT);
		y = theta / M_PI * T(CalibSpace::IMG_HEIGHT);
#else
		const T& _x = p[0];
		const T& _y = p[1];
		const T& _z = p[2];

		T rho = sqrt(_x * _x + _y * _y + _z * _z);

		//相机坐标系到像平面，原点到左上角了
		T theta = M_PI - acos(_y / rho);
		//atan2的范围是-π到π
		T phi = atan2(_x, _z) + M_PI;


		x = phi / M_PI * T(CalibSpace::IMG_HEIGHT);
		y = theta / M_PI * T(CalibSpace::IMG_HEIGHT);
#endif
	}

	template <typename T>
	void  projectPoly(const T* p, T& x, T& y)
	{
		const T& _x = p[0];
		const T& _y = p[1];
		const T& _z = p[2];

		T rho = sqrt(_x * _x + _y * _y + _z * _z);

		//相机坐标系到像平面，原点到左上角了
		T theta = M_PI - acos(_y / rho);
		//atan2的范围是-π到π
		T phi = atan2(_x, _z) + M_PI / 2;

		x = phi / M_PI * T(CalibSpace::IMG_WIDTH);
		y = theta / M_PI * T(CalibSpace::IMG_HEIGHT);

	}

	template <typename T>
	T getyInCamera()
	{
		return /*CalibSpace::extrinsic_para[1]*/1.90;
		//	return 0;
	}

	template <typename T>
	T getzInCamera(const T& yc, const T& row)
	{
		if (abs(row - CalibSpace::CY) < T(1))
		{
			return T(200);
		}

		//	float zc = yc * CalibSpace::FY / (row - CalibSpace::CY) + 1.0;
		T  zc = yc * CalibSpace::FY / (row - CalibSpace::CY);
		return zc;
	}

	template <typename T>
	T  getxInCamera(const T& zc, const T& col)
	{
		T xc = (col - CalibSpace::CX) * zc / CalibSpace::FX;
		return xc;
	}

	template <typename T>
	bool convertPixel2Camera3d(T x, T y , Eigen::Matrix<T, 3, 1>& p3f)
	{
		x += T(CalibSpace::image_rect.tl().x);
		y += T(CalibSpace::image_rect.tl().y);

		if (CalibSpace::camera_type == CAMERA_MSS_WIDE)
		{
			//	p3f[1] = getyInCamera();
			p3f[2] = getzInCamera(p3f[1], y);
			p3f[0] = getxInCamera(p3f[2], x);
		}
		else if (CalibSpace::camera_type == CAMERA_MSS_PANO)
		{
			T theta = y * T(1.0 / CalibSpace::IMG_HEIGHT * M_PI);
			T phi = x * T(1.0 / CalibSpace::IMG_HEIGHT * M_PI);

			//	p3f[1] = getyInCamera() + 0.65;
			T r = p3f[1] / cos(M_PI - theta);
			if (abs(r) > T(60))
			{
				p3f[2] = T(0);
				p3f[0] = T(0);
				return false;
			}
			p3f[2] = sqrt((r * r - p3f[1] * p3f[1]) / (1.0 + tan(phi - M_PI) * tan(phi - M_PI)));
			p3f[0] = p3f[2] * tan(phi - M_PI);
			if (p3f[2] < T(2))
			{
				return false;
			}
		}

		return true;
	}


	template <typename T>
	void  convertPointByEigen(const vector<T>& camPose, T _X, T _Y, T _Z, T* p)
	{
		//http://www.mamicode.com/info-detail-2730205.html
		//https://blog.csdn.net/u011092188/article/details/77430988
		//旋转矩阵（3X3）:Eigen::Matrix3d
		//	旋转向量（3X1） : Eigen::AngleAxisd
		//	四元数（4X1） : Eigen::Quaterniond
		//	平移向量（3X1） : Eigen::Vector3d
		//	变换矩阵（4X4） : Eigen::Isometry3d
		Eigen::Matrix<T, 3, 3> rotation_matrix;
		calcRotation(camPose, rotation_matrix);
		Eigen::Matrix<T, 3, 1> translate_pt(camPose[0], camPose[1], camPose[2]);
		Eigen::Matrix<T, 3, 1> pt(_X, _Y, _Z);
		Eigen::Matrix<T, 3, 1> rorate_pt = rotation_matrix * pt;
		rorate_pt = rorate_pt + translate_pt;

		// 		Eigen::Matrix<T, 3, 1> translate_pt(camPose[0], camPose[1], camPose[2]);
		// 		Eigen::Matrix<T, 3, 1> pt(_X, _Y, _Z);
		// 		pt = pt - translate_pt;
		// 		Eigen::Matrix<T, 3, 1> rorate_pt = rotation_matrix * pt;

		p[0] = rorate_pt[0];
		p[1] = rorate_pt[1];
		p[2] = rorate_pt[2];
	}

	template<typename T>
	void convertPointByQuternion(const T* camPose, T _X, T _Y, T _Z, T* p)
	{
		Eigen::Matrix<T, 3, 1> translate_pt(camPose[0], camPose[1], camPose[2]);
		//Eigen::Quaterniond constructor, the order of the arguments is(w, x, y, z).
		Eigen::Quaternion<T> q(camPose[6], camPose[3], camPose[4], camPose[5]);

		Eigen::Matrix<T, 3, 3> rotation_matrix = q.toRotationMatrix();

		Eigen::Matrix<T, 3, 1> pt(_X, _Y, _Z);
		Eigen::Matrix<T, 3, 1> rorate_pt = rotation_matrix * pt + translate_pt;

		for (int i = 0; i <3; i++)
		{
			p[i] = rorate_pt[i];
		}
	}

	template <typename T>
	void  Image2IPM(T* xy)
	{
		Eigen::Matrix<T, 3, 3> H;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				H(i, j) = T(CalibSpace::warpmat_src2ipm.at<double>(i, j));
			}

		Eigen::Matrix<T, 3, 1> p(xy[0], xy[1], T(1));
		Eigen::Matrix<T, 3, 1> new_p = H * p;
		xy[0] = new_p(0) / new_p(2);
		xy[1] = new_p(1) / new_p(2);
	}

	template <typename T>
	bool  project2Image(T* p, T* xy, bool ipm)
	{
		if (CalibSpace::camera_type == CAMERA_MSS_WIDE)
		{
			//深度>0
			if (p[2] <= T(0))
			{
				return false;
			}
			projectPoint(p, xy[0], xy[1]);
		}
		else if (CalibSpace::camera_type == CAMERA_MSS_PANO)
		{
			projectPano(p, xy[0], xy[1]);
			xy[0] -= T(CalibSpace::image_rect.tl().x);
			xy[1] -= T(CalibSpace::image_rect.tl().y);
		}
		else if (CalibSpace::camera_type == CAMERA_MSS_POLYWIDE)
		{
			projectPoly(p, xy[0], xy[1]);
			xy[0] -= T(CalibSpace::image_rect.tl().x);
			xy[1] -= T(CalibSpace::image_rect.tl().y);
		}
		else
		{
			return false;
		}
		if (xy[0] < T(0) || xy[0] > T(CalibSpace::image_rect.width - 1) ||
			xy[1] < T(0) || xy[1] > T(CalibSpace::image_rect.height - 1))
		{
			return false;
		}
		if (ipm)
		{
			Image2IPM(xy);
		}

		return true;
	}

	template <typename T>
	bool  convertPoint3dTo2d(const vector<T>& camPose, const T* const XYZ, T* xy, bool ipm)
	{
		T p[3];
		convertPointByEigen(camPose, XYZ[0], XYZ[1], XYZ[2], p);

		return project2Image(p, xy, ipm);
	}

	template <typename T>
	bool  convertCameraPoint2dTo3d(T x, T y, Eigen::Matrix<T, 3, 1>& XYZ)
	{
		
		//p3f[1] = CalibSpace::ego_height + MY_CONFIG.data_para.trans.y;
		//
		//p3f[1] = T(CalibSpace::ego_height) + camPose[1];
		XYZ[1] = T(CalibSpace::camera_height);
		if (!convertPixel2Camera3d(x, y, XYZ))
		{
			return false;
		}
		return true;
	}

	template <typename T>
	bool  convertCamera2dToEgo3d(const vector<T>& camPose, T x, T y, Eigen::Matrix<T, 3, 1>& XYZ)
	{
		//image 2d to camera 3d
		convertCameraPoint2dTo3d(x, y, XYZ);
		//camera 3d to ego 3d
		Eigen::Matrix<T, 3, 1> p3f;
		Eigen::Matrix<T, 3, 3> rotation_matrix;
		calcRotation(camPose, rotation_matrix);
		Eigen::Matrix<T, 3, 1> translate_pt(camPose[0], camPose[1], camPose[2]);
		p3f -= translate_pt;
		p3f = rotation_matrix.inverse() * p3f;
		XYZ = p3f;
		return true;
	}

}