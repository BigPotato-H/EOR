#include "DataIOArgoverse.h"
#include "DataManager/WKTCSV.h"
#include <io.h>
#include <fstream>
#include <iostream>
#include "HNMath/TransRotation.h"
#include <cstdlib>
#include "shapefil.h"
#include "HNMath/GeometricAlgorithm2.h"

DataIOArgoverse::DataIOArgoverse()
{
}

DataIOArgoverse::~DataIOArgoverse()
{
}

bool  DataIOArgoverse::readEgo2Cam(Eigen::Matrix3d& r, Eigen::Vector3d& t)
{
	const string& file_na = m_root_path + "hd_map\\ego_2_cam.csv";
	if (_access(file_na.c_str(), 0) != 0)
	{
		return 0;
	}

	ifstream is(file_na, fstream::in);
	is.setf(ios::fixed, ios::floatfield);
	
	string line = "";
	getline(is, line);
	if (line.size() == 0)
	{
		return false;
	}
	HNString::ReplaceA(line, "[", "");
	HNString::ReplaceA(line, "]", "");
	vector<string> str_vec;
	HNString::SplitA(line, str_vec, ",");
	if (str_vec.size() != 16)
	{
		return false;
	}
	Eigen::Matrix4d rt;
	for (int i =0; i <16; i++)
	{
		int r = i / 4;
		int c = i - r * 4;
		rt(r, c) = stod(str_vec[i]);
	}

	r = rt.topLeftCorner(3, 3);
	Eigen::Matrix3d r_trans = r.transpose();
	r = r_trans;

	t = rt.topRightCorner(3, 1);
	t = -t;
	t = r * t;
	return true;
}


size_t DataIOArgoverse::getHDLaneDividerInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec,
	bool keep_inside_intersection)
{
	const string& file_na = m_root_path + "hd_map\\ld.csv";
	if (_access(file_na.c_str(), 0) != 0)
	{
		return 0;
	}

	vector<string> skip_logic_lines;
	if (m_root_path.find("arg0c1") != string::npos)
	{
		skip_logic_lines = {"",
		};
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
		if (!keep_inside_intersection)
		{
			if (attr_vec[2] == "True")
			{
				continue;
			}
		}
		
		vector<cv::Point3d> pt_vec;
		inShpCSVFields(attr_vec.back(), pt_vec, 2);

		HDObject info;
		info.obj_id = to_string(obj_vec.size());
		info.prop_id = attr_vec[0];
		info.type = 13;
		info.shape.swap(pt_vec);

		obj_vec.emplace_back(info);
	}

	is.close();
	return obj_vec.size();
}



size_t DataIOArgoverse::getHDJunctionInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec)
{
	
	return obj_vec.size();
}

size_t DataIOArgoverse::getHDPedCrossingsInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec)
{
	const string& file_na = m_root_path + "hd_map\\ped_crossings.csv";
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

		vector<cv::Point3d> pt_vec;
		inShpCSVFields(attr_vec.back(), pt_vec, 2);

		HDObject info;
		info.obj_id = attr_vec[0];
		info.type = OC_crosswalk;
		info.shape.swap(pt_vec);

		obj_vec.emplace_back(info);
	}

	is.close();
	return obj_vec.size();
}
size_t DataIOArgoverse::getHDObjectsInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec)
{

	//读取shp
	const string& file_na = m_root_path + "hd_map\\obj.shp";
	SHPHandle hShp = SHPOpen(file_na.c_str(), "r");

	int nShapeType = 0;
	int nEntities = 0;
	double* minB = new double[4];
	double* maxB = new double[4];
	SHPGetInfo(hShp, &nEntities, &nShapeType, minB, maxB);

	obj_vec.resize(nEntities);
	for (int i = 0; i < nEntities; i++)
	{
		int iShape = i;
		SHPObject* obj = SHPReadObject(hShp, iShape);
		int parts = obj->nParts;
		int verts = obj->nVertices;
		if (verts == 0)
		{
			continue;
		}
		auto& hdobj = obj_vec[i];
		for (int s = 0; s < parts; s++)
		{
			int part_start = obj->panPartStart[s];
			int part_end = 0;
			if (s == parts - 1)
			{
				part_end = verts;
			}
			else
			{
				part_end = obj->panPartStart[s + 1];
			}

			int part_size = part_end - part_start;
			if (part_size == 0)
			{
				continue;
			}
			for (size_t i = 0; i < part_size; i++)
			{
				cv::Point3d pt;
				pt.x = obj->padfX[part_start + i];
				pt.y = obj->padfY[part_start + i];
				pt.z = obj->padfZ[part_start + i];
				hdobj.shape.push_back(pt);

			}


		}

	}
	SHPClose(hShp);


	//////////////////dbf///////////////
	const string& dbf_path = m_root_path + "hd_map\\obj.dbf";
	DBFHandle pdbf = DBFOpen(dbf_path.c_str(), "r");
	if (pdbf)
	{
		//> 获取DBF行数和列数目
		int iColCount = DBFGetFieldCount(pdbf);
		int iRowCount = DBFGetRecordCount(pdbf);//行数仅仅是内容的行数，不包含行头

		for (int iRow = 0; iRow < iRowCount; iRow++)
		{
			auto& hdobj = obj_vec[iRow];

			int iCol = 0;
			//DBFReadIntegerAttribute
			hdobj.obj_id = to_string(DBFReadIntegerAttribute(pdbf, iRow, iCol++));
			hdobj.prop_id = hdobj.obj_id;
			hdobj.type = DBFReadIntegerAttribute(pdbf, iRow, iCol++);
			if (hdobj.type == OC_t_sign)
			{
				hdobj.shape.push_back(hdobj.shape[0]);
			}
			if (hdobj.type == OC_pole)
			{
				sort(hdobj.shape.begin(), hdobj.shape.end(), [](const auto& ele1, const auto& ele2)->bool {
					return ele1.z < ele2.z;
					});
				double height = hdobj.shape.back().z - hdobj.shape.front().z;
#if 0
				hdobj.shape.resize(1);
				auto ept = hdobj.shape.front();
				ept.z += height;
				hdobj.shape.emplace_back(ept);
#else
				auto spt = hdobj.shape.back();
				spt.z -= height;
				hdobj.shape[0] = spt;
#endif
				equalizPolyline(hdobj.shape, 0.5);
			}
		}
	}
	//> 使用完关闭DBF文件
	DBFClose(pdbf);

	//////////////////读csv
//	getHDPedCrossingsInBox(box, obj_vec);

	auto remove_obj = remove_if(obj_vec.begin(), obj_vec.end(), [](const auto& t)->bool {
		return t.type == 14;
		});
	obj_vec.erase(remove_obj, obj_vec.end());
	return obj_vec.size();
}

Eigen::Vector3d getEuler(const Eigen::Matrix3d& rr)
{
	Eigen::Vector3d euler = rr.eulerAngles(0, 1, 2);

	// 旋转矩阵 -> 欧拉角(Z-Y-X，即RPY)（确保第一个值的范围在[0, pi]）
	//Eigen::Vector3d euler = rr.eulerAngles(2, 1, 0);


	//旋转矩阵 --> 欧拉角(Z-Y-X，即RPY)（确保pitch的范围[-pi/2, pi/2]）
	Eigen::Vector3d eulerAngle_mine;
	Eigen::Matrix3d rot = rr;
	eulerAngle_mine(2) = std::atan2(rot(2, 1), rot(2, 2));
	eulerAngle_mine(1) = std::atan2(-rot(2, 0), std::sqrt(rot(2, 1) * rot(2, 1) + rot(2, 2) * rot(2, 2)));
	eulerAngle_mine(0) = std::atan2(rot(1, 0), rot(0, 0));

	euler = eulerAngle_mine;
	//cout << eulerAngle_mine;
	return euler;
}

size_t DataIOArgoverse::getTracePoints(vector<RAW_INS>& ins_vec)
{
	const string& file_na = m_root_path + "hd_map\\city_2_ego.csv";
	if (_access(file_na.c_str(), 0) != 0)
	{
		return 0;
	}

	ifstream is(file_na, fstream::in);
//	is.setf(ios::fixed, ios::floatfield);
//	is.precision(4);

	string line = "";
	int k = 0;
	while (getline(is, line))
	{
		k++;
		if (k < 200)
		{
		//	continue;
		}
		vector<string> str_vec;
		if (line.size() == 0)
		{
			continue;
		}
		HNString::SplitA(line, str_vec, ":");
		if (str_vec.size() != 2)
		{
			continue;
		}
		RAW_INS rs;
		rs.name = str_vec[0];

		string aa_str = str_vec[1];
		str_vec.clear();
		HNString::ReplaceA(aa_str, "[", "");
		HNString::ReplaceA(aa_str, "]", "");
		HNString::SplitA(aa_str, str_vec, ",");
		int i = 0;
		Eigen::Matrix4d rt;
		for (const auto& str : str_vec)
		{
			int r = i / 4;
			int c = i - r * 4;
			rt(r, c) = stod(str);

			i++;
		}
		//cout << "---rt:" << endl << rt << endl;

		Eigen::Matrix3d rr = rt.topLeftCorner(3,3);
		Eigen::Vector3d tt = rt.topRightCorner(3,1);
		//cout << "---rr:" << endl << rr << endl;

		//Eigen::Vector3d euler = rr.eulerAngles(0, 1, 2);
		Eigen::Vector3d euler = getEuler(rr);
		//euler = euler *  180 / M_PI;
		//cout << "---euler:" << endl << euler << endl;

		Eigen::Matrix3d R_matrix;
		TransRotation::eigenEuler2RotationMatrixd(euler, R_matrix);

		//cout << "---R:" << endl << R_matrix << endl;
		if (rs.name < "315967989399927216")
		{
	//		continue;
		}
		rs.point = cv::Point3d(tt(0), tt(1), tt(2));
		rs.R = rr;
		rs.T = rs.R.transpose() * (-tt);
		//////////////20230331  argoverse的roll角为正，右边高，反一下，和whu保持一致
		///////////////////////////////whu的roll角为正，左边高
		rs.roll = euler(1);
		rs.pitch = -euler(2);
		rs.heading = -euler(0) + M_PI / 2.0;

		if (rs.heading > M_PI)
		{
			rs.heading -= M_PI * 2.0;
		}
		if (rs.heading < -M_PI)
		{
			rs.heading += M_PI * 2.0;
		}

		//if (rs.name < "315973388249927213")
		
		//if (rs.name < "315973385949927216")
		
		//if (rs.name < "315973405999927221")
		
		if (rs.name < "315973408399927208")
		{
		//	rs.roll = 0;
		//	continue;
		}
		
		/// test
		/*Eigen::AngleAxisd a1(-M_PI / 2.0, Eigen::Vector3d::UnitY());
		Eigen::AngleAxisd a2(M_PI / 2.0, Eigen::Vector3d::UnitZ());
		Eigen::Matrix3d r_yz;
		rr = a2 * a1* rr;
		euler = rr.eulerAngles(0, 1, 2);
		rs.roll = euler(0);
		rs.pitch = euler(1);
		rs.heading = euler(2) - M_PI / 2.0;
		Eigen::Vector3d R_euler;
		R_euler << M_PI / 2 + rs.roll, rs.pitch, rs.heading;

		TransRotation::eigenEuler2RotationMatrixd(R_euler, R_matrix);
		cout << "---R:" << endl << R_matrix << endl;*/

		if (ins_vec.size() > 0)
		{
			const auto& back = ins_vec.back();
			auto dif = rs.point - back.point;

			double dist = sqrt(dif.ddot(dif));
			if (dist < 1.0)
			{
				continue;
			}			
		}
		//随机抖动

		ins_vec.push_back(rs);
		
	}

	//保存出来
	saveTracePoints(ins_vec);
	return ins_vec.size();
}