#pragma once
#ifndef SampleTest_H
#define SampleTest_H

#include "DataSet.h"
typedef 

class SampleTest
{
public:
	SampleTest();
	~SampleTest();

	void generateSamplePoses(const string& file_path, 
		vector<vector<double>>& sample_poses);
	void saveResults(const string& img_na, const string& file_path,
		const vector<vector<double>>& sample_poses,
		const vector<vector<double>>& poses_errors);
private:

};

#endif