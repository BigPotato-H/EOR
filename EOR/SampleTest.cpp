#include "SampleTest.h"

#include <fstream>
#include <random>
#include <io.h>
#include "HNMath/TransRotation.h"
#include "HNString/HNString.h"



SampleTest::SampleTest()
{
}

SampleTest::~SampleTest()
{
}

inline double getRandomDouble(double min, double max)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(min, max);
	return dis(gen);
}


void SampleTest::generateSamplePoses(const string&file_path, 
	vector<vector<double>>& sample_poses)
{
	const string& file = file_path + "prob_distribution\\0_sample_points.txt";
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
			double x = getRandomDouble(-1, 1);
			double y = getRandomDouble(-1, 1);
			double z = getRandomDouble(-1, 1);
			double ax = getRandomDouble(-0.1, 0.1);
			double ay = getRandomDouble(-0.1, 0.1);
			double az = getRandomDouble(-0.1, 0.1);
			string line = "";
			HNString::FormatA(line, "%d,%.3f, %.3f,%.3f,%.3f,%.3f,%.3f",
				i, x, y, z, ax, ay, az);
			of << line << endl;

			vector<double> t_pos = { x,y,z, ax, ay, az };
			sample_poses.push_back(t_pos);
		}
		of.close();
	}

}

void SampleTest::saveResults(const string& img_na, const string& file_path,
	const vector<vector<double>>& sample_poses, const vector<vector<double>>& poses_errors)
{
	
	ofstream of(file_path, ios::out | ios::trunc);
	if (!of.is_open())
	{
		return;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);

	string head_line = "img_na,idx,se,x,y,z,ax,ay,az,iou";
	of << head_line << endl;

	for (int i = 0; i < poses_errors.size(); i++)
	{
		auto err = poses_errors[i];
		const auto& spl = sample_poses[i];

		// write 2 lines
		string line = "";
		HNString::FormatA(line, "%s,%d,%d,\
%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f",
			img_na.c_str(),
			i,
			1,
			spl[0],
			spl[1],
			spl[2],
			spl[3],
			spl[4],
			spl[5],
			spl[6]
		);
		of << line << endl;

		HNString::FormatA(line, "%s,%d,%d,\
%.4f,%.4f,%.4f,%.4f,%.4f,%.4f, %.2f",
img_na.c_str(),
i,
2,
err[0],
err[1],
err[2],
err[3],
err[4],
err[5],
err[6]
);
		of << line << endl;
	}

	of.close();
}
