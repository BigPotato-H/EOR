// DataIO.cpp : 定义控制台应用程序的入口点。

// for std
// for opencv
#pragma once
#ifndef DataIO_AGV
#define DataIO_AGV

#include "DataIO.h"

class DataIOArgoverse: public DataIO
{
public:
	DataIOArgoverse();
	~DataIOArgoverse();
	bool readEgo2Cam(Eigen::Matrix3d& r, Eigen::Vector3d& t);
	size_t getTracePoints(vector<RAW_INS>& ins_vec);
	size_t getHDLaneDividerInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec, bool keep_inside_intersection = false);

	size_t getHDJunctionInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec);
	size_t getHDPedCrossingsInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec);
	size_t getHDObjectsInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec);

private:

};

#endif