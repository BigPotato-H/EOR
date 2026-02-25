#pragma once
#ifndef CapabilityCalculator_H
#define CapabilityCalculator_H

#include "DataSet.h"
#include <set>

typedef pair<int, int> PAIR;
typedef pair<float, float> PAIRf;

enum DataTye {
	Data_Unknown = 0,
	Data_MAP = 1,
	Data_IMAGE =2,
	Data_COMMON = 3
};


//struct FeatureID
//{
//	ObjectClassification feature_type; //OC lane, pole, sign
//	DataTye data_type; // map image intersect
//	FeatureID()
//	{
//		feature_type = OC_unkown;
//		data_type = Data_Unknown;
//	}
//	FeatureID(const int& f, const int& d)
//	{
//		feature_type = ObjectClassification(f);
//		data_type = DataTye(d);
//	}
//	bool operator <( const FeatureID a) const 
//	{
//		if (a.feature_type == feature_type)
//		{
//			return a.data_type < data_type;
//		}
//		else
//		{
//			return a.feature_type < feature_type;
//		}
//	}
//};




class CapabilityCalculator
{
public:
	CapabilityCalculator();
	~CapabilityCalculator();

//	void setObjectTypes(const set<ObjectClassification>& oc_set);
// 	void setSource(const map<int, LINE_VEC>& lines_map_rgb,
// 		/*const map<int, LINE_VEC>& lines_map_hdmap*/
// 		const map<int, HDObject_VEC>& ego_objs_map);



	void generateDistributionSample(const map<int, LINE_VEC>& lines_map_rgb, 
		const map<int, HDObject_VEC>& org_local_objs, CamPose cp,
		map<int, LINE_VEC>& random_lines_map, 
		map<int, HDObject_VEC>& random_objs,
		const string& file_path);

	void saveRegistPoses(const CamPose& cp, const string& root_path, 
		const vector<double>& pose_error);

	float getOccupancyRatio();

	//void setImageSize(int width, int hight);

	void oneIteration(const map<int, LINE_VEC>& lines_map_rgb,
		const map<int, HDObject_VEC>& org_local_objs,
		const CamPose& cp,
		const string& file_path);
	void saveSpatialDistributionDataCSV(const string& root_path);
	bool getFrameSpatialDistribution(const string& root_path, const string& img_na, StatisticData_MAP& sf);

private:

	//set<ObjectClassification> m_obj_type_set;
	//cv::Size m_img_size;
	//cv::Mat m_mat_rgb;
	//cv::Mat m_mat_hdmap;
	vector<vector<double>> m_sample_poses;
	map<string, StatisticData_MAP> m_img_sd_map;
};

#endif
