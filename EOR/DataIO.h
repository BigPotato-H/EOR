// DataIO.cpp : 定义控制台应用程序的入口点。

// for std
// for opencv
#pragma once
#ifndef DataIO_H
#define DataIO_H

#include "DataSet.h"
#include "CapabilityCalculator.h"

class DataIO
{
public:
	DataIO();
	~DataIO();
	void initDataPath(const string& root_path);
	void initSQLPath(const string& sql_path);

	virtual size_t getHDLaneDividerInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec, bool keep_inside_intersection = false);
	virtual size_t getHDJunctionInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec) { return 0; };
	virtual size_t getHDObjectsInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec) { return 0; };
	virtual size_t getTracePoints(vector<RAW_INS>& ins_vec) { return 0; };
	void saveTracePoints(const vector<RAW_INS>& ins_vec);

	void saveJitterCamPose(string file_path, const vector<CamPose>& pose_vec);
	size_t readJitterCamPose(string file_path, map<string, CamPose>& pose_map);

	void saveGTCamPose(string file_path, const vector<CamPose>& pose_vec);
	size_t readGTCamPose(string file_path, map<string, CamPose>& pose_map);

	void importToDB(const string& in_path);
	void saveToCSV(const string& in_path);
	size_t readRefPoints(const string& file_path, vector<CalibSpace::Point3d2d>& ref_pt_vec);
	size_t readPointCloud(const string& file_path, vector<CalibSpace::PointXYZI>& pc);
	size_t readLog(const string& file_path, vector<RAW_INS>& ins_vec);
	size_t readLog0407(const string& file_path, vector<RAW_INS>& ins_vec);

	size_t readHDI(const string& file_path, vector<RAW_INS>& ins_vec);
//	size_t getFiles(string path, string filter, vector<string>& files);
	size_t getHDMapLaneDividerInBox(const string& connect_sql, const vector<cv::Point2f>& box, HDObject_VEC& obj_vec);
	size_t getHDMapSignBoardInBox(const string& connect_sql, const vector<cv::Point2f>& box, HDSignBoard_VEC& obj_vec);
	size_t getHDMapObjectsInBox(const string& connect_sql, const vector<cv::Point2f>& box, HDObject_VEC& obj_vec);


	void saveHDMap(const string& file_path, const HDObject_VEC& obj_vec);


	void buildSignBoard(HDObject_VEC& obj_vec, const HDSignBoard_MAP& sgb_map, HDSignBoard_VEC& sgn_vec);
	void saveParas(const string& folder_path, const CamPose& cp);
	size_t readCamPosePara(const string& folder_path, map<string, CamPose>& pos_map);
	size_t readPosesOptimized(const string& file_path, map<int, POSE>& poses_map);

	//template<typename T>
	// bool get3DCircleCenter(const T&pt1, const T&pt2, const T&pt3, T& normalVector, T& ptCenter, double& dbRadius);

	//

	void writeSignboardCSV(const string& _path, const RAW_INS& ins, const HDSignBoard_VEC& obj_vec);



	void getConstantVertixes(const string& folder_path, vector<int>& indices);


	void getTraceBox(const vector<RAW_INS>& ins_vec, vector<cv::Point2f>& box, float threshold);
	void getTraceXYBox(const vector<RAW_INS>& ins_vec, vector<cv::Point2f>& box, float threshold);
	string formatBoxStr(const vector<cv::Point2f>& box, int srid = 4326);


protected:
	string m_root_path;
	string m_sql;
};



#endif