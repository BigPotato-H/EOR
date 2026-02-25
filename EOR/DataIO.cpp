// DataIO.cpp : 定义控制台应用程序的入口点。
// for std
#include <iostream>
// for opencv
#include "DataIO.h"

#include<fstream>
#include <io.h>
#include "HNMath/CoordinateTransform.h"
#include "HNMath/GeometricAlgorithm2.h"

#include "HNString/HNString.h"
#include "HNString/EncodeStr.h"

#include "DataManager/WKTCSV.h"
#include "Time/LocalTime.h"
#include "HNFile/File.h"

#include "opencv2/imgproc.hpp"
//#include "DBImportor.h"


template<typename T>
bool get3DCircleCenter(const T&pt1, const T&pt2, const T&pt3, T& normalVector, T& ptCenter, double& dbRadius)
{
	double dPx = pt1.x;
	double dQx = pt2.x;
	double dRx = pt3.x;

	double dPy = pt1.y;
	double dQy = pt2.y;
	double dRy = pt3.y;

	double dPz = pt1.z;
	double dQz = pt2.z;
	double dRz = pt3.z;


	//算平面法量;
	double x1 = dQx - dPx;
	double x2 = dRx - dPx;

	double y1 = dQy - dPy;
	double y2 = dRy - dPy;

	double z1 = dQz - dPz;
	double z2 = dRz - dPz;

	double pi = y1*z2 - z1*y2;
	double pj = z1*x2 - x1*z2;
	double pk = x1*y2 - y1*x2;

	if ((pi == 0) && (pj == 0) && (pk == 0))
	{
		return false;
	}
	normalVector.x = pi;
	normalVector.y = pj;
	normalVector.z = pk;

	//求PQ和PR的中垂线
	//1，过PQ的中点(Mx,My,Mz);
	double dMx = (dPx + dQx) / 2;
	double dMy = (dPy + dQy) / 2;
	double dMz = (dPz + dQz) / 2;

	//与（Mi,Mj,Mk）＝（pi，pj，pk）×（x1,y1,z1）垂直;
	double dMi = pj*z1 - pk*y1;
	double dMj = pk*x1 - pi*z1;
	double dMk = pi*y1 - pj*x1;

	//2，过PR的中点(Nx,Ny,Nz);
	double dNx = (dPx + dRx) / 2;
	double dNy = (dPy + dRy) / 2;
	double dNz = (dPz + dRz) / 2;

	//与（Ni,Nj,Nk）＝（pi，pj，pk）×（x2,y2,z2）垂直;
	double dNi = pj*z2 - pk*y2;
	double dNj = pk*x2 - pi*z2;
	double dNk = pi*y2 - pj*x2;

	double tn;

	//<Modified by huix2890 2019/01/29 是否共线的判断允许误差>
	if (abs(dNj*dMi - dMj*dNi) >= 1e-50)
	{
		tn = ((dMy - dNy)*dMi + dMj*(dNx - dMx)) / (dNj*dMi - dMj*dNi);
	}
	else if (abs(dNk*dMi - dMk*dNi) >= 1e-50)
	{
		tn = ((dMz - dNz)*dMi + dMk*(dNx - dMx)) / (dNk*dMi - dMk*dNi);
	}
	else if (abs(dNk*dMj - dMk*dNj) >= 1e-50)
	{
		tn = ((dMz - dNz)*dMj + dMk*(dNy - dMy)) / (dNk*dMj - dMk*dNj);
	}
	else
	{
		return false;
	}
	//< /Modified by huix2890 2019/01/29 是否共线的判断允许误差>

	double dX0 = dNx + dNi*tn;
	double dY0 = dNy + dNj*tn;
	double dZ0 = dNz + dNk*tn;

	ptCenter.x = dX0;
	ptCenter.y = dY0;
	ptCenter.z = dZ0;
	//得半径;
	dbRadius = (dX0 - dPx)*(dX0 - dPx) + (dY0 - dPy)*(dY0 - dPy) + (dZ0 - dPz)*(dZ0 - dPz);
	dbRadius = sqrt(dbRadius);
	return true;
}


DataIO::DataIO()
{
}

DataIO::~DataIO()
{
}

void DataIO::initDataPath(const string& root_path)
{
	m_root_path = root_path;
}

void DataIO::initSQLPath(const string& sql_path)
{
	m_sql = sql_path;
}

string DataIO::formatBoxStr(const vector<cv::Point2f>& box, int srid)
{
	if (box.size() != 4)
	{
		return "";
	}

	string box_str = "";
	HNString::FormatA(box_str, "st_geometryfromtext('polygon((%.8f %.8f,%.8f %.8f,%.8f %.8f,%.8f %.8f,%.8f %.8f))', %d)",
		box[0].x, box[0].y,
		box[1].x, box[1].y,
		box[2].x, box[2].y,
		box[3].x, box[3].y,
		box[0].x, box[0].y,
		srid);

	return box_str;
}

size_t DataIO::readRefPoints(const string& file_path, vector<CalibSpace::Point3d2d>& ref_pt_vec)
{
	ref_pt_vec.clear();

	ifstream fin(file_path.c_str());
	string line;
	while (getline(fin, line))
	{
		char* pEnd;
		CalibSpace::Point3d2d ref;
		ref.p2d.x = strtod(line.c_str(), &pEnd);
		ref.p2d.y = strtod(pEnd, &pEnd);
		ref.p3d.x = strtod(pEnd, &pEnd);
		ref.p3d.y = strtod(pEnd, &pEnd);
		ref.p3d.z = strtod(pEnd, nullptr);

		if (abs(ref.p3d.x) < 180 &&
			abs(ref.p3d.y) < 180)
		{
			CoordinateTransform::LonLat2XY(ref.p3d.x, ref.p3d.y, CalibSpace::band, ref.p3d.x, ref.p3d.y);
		}
		ref_pt_vec.push_back(ref);
	}
	fin.close();

	return ref_pt_vec.size();
}

size_t DataIO::readPointCloud(const string& file_path, vector<CalibSpace::PointXYZI>& pc)
{
	pc.clear();

	ifstream fin(file_path.c_str());
	string line;
	while (getline(fin, line))
	{
		char* pEnd = ",";
		CalibSpace::PointXYZI pt;
		pt.x = strtod(line.c_str(), &pEnd);
		pt.y = strtod(pEnd + 1, &pEnd);
		pt.z = strtod(pEnd + 1, &pEnd);
		pt.intensity = strtod(pEnd + 1, nullptr);

		pc.push_back(pt);
	}
	fin.close();

	return pc.size();
}

size_t DataIO::readLog(const string& file_path, vector<RAW_INS>& ins_vec)
{
	// 定义输入文件流类对象infile
	ifstream infile(file_path, ios::in);

	if (!infile)
	{  // 判断文件是否存在
		cerr << "open error." << endl;
		return 0;
	}

	char str[255]; // 定义字符数组用来接受读取一行的数据
	RAW_INS raw;
	infile.getline(str, 255);
	while (infile)
	{
		infile.getline(str, 255);  // getline函数可以读取整行并保存在str数组里
		vector<string> str_vec;
		HNString::SplitA(str, str_vec, ",");
		if (str_vec.size() < 16)
		{
			continue;
		}
		raw.time = atof(str_vec[0].c_str());
		raw.lonlat.y = atof(str_vec[1].c_str());
		raw.lonlat.x = atof(str_vec[2].c_str());
		raw.lonlat.z = atof(str_vec[3].c_str());

		raw.roll = atof(str_vec[7].c_str());
		raw.pitch = atof(str_vec[8].c_str());
		raw.heading = atof(str_vec[9].c_str());

		raw.name = to_string((int)(raw.time));
		raw.point = raw.lonlat;
		if (raw.heading  < M_PI)
		{
			raw.heading = -raw.heading;
		}
		else
		{
			raw.heading = M_PI * 2 - raw.heading;
		}

		if (int(raw.time + 0.01) <= int(raw.time))
		{
			continue;
		}
		
		if (ins_vec.size() > 0)
		{
			const auto& back = ins_vec.back();
			double dist = abs(raw.lonlat.x - back.lonlat.x) + abs(raw.lonlat.y - back.lonlat.y);
			if (dist < 1e-5)
			{
				continue;
			}
		}

		ins_vec.push_back(raw);
	}
	infile.close();

	CoordinateTransform::LonLatPoints2XY(ins_vec, CalibSpace::band);

	return ins_vec.size();
}

void relate2ImageTimestamp(long long tsp)
{
	tm tm_ = HN_GENERAL::stamp_to_standard(tsp);
	char s[100] = { 0 };
	strftime(s, sizeof(s), "%Y-%m-%d %H:%M:%S", &tm_);
}

void renameImage(const string& folder_path)
{
	char image_start_time[100] = "2022-04-07 13:24:32";
	long long s_tsp = HN_GENERAL::standard_to_stamp(image_start_time);

	vector<string> files;
//	HN_GENERAL::getAllFilesName(folder_path, ".JPEG", files);
	HN_GENERAL::getAllFilesName(folder_path, ".PNG", files);
	for (const auto& file : files)
	{
		string old_name = folder_path + file;
#if 0
		vector<string> str_vec;
		HNString::SplitA(file, str_vec, "-");
		if (str_vec.size() < 2)
		{
			continue;
		}
		//什么单位??? 300->12s
		string time_str = str_vec[1];
		HNString::ReplaceA(time_str, ".JPEG", "");
		int second = (stoi(time_str) / 300 - 1) * 12;
		int millisecond = 0;
		int abs_time = s_tsp + second + millisecond;
//		string new_name = folder_path + to_string(abs_time) + ".jpg";
#endif
		string new_name = old_name;
		HNString::ReplaceA(new_name, ".PNG", ".jpg");
		rename(old_name.c_str(), new_name.c_str());
	}
}

size_t DataIO::readLog0407(const string& file_path, vector<RAW_INS>& ins_vec)
{
	// 定义输入文件流类对象infile
	ifstream infile(file_path, ios::in);

	if (!infile)
	{  // 判断文件是否存在
		cerr << "open error." << endl;
		return 0;
	}

	char str[255]; // 定义字符数组用来接受读取一行的数据
	RAW_INS raw;
	infile.getline(str, 255);
	while (infile)
	{
		infile.getline(str, 255);  // getline函数可以读取整行并保存在str数组里
		vector<string> str_vec;
		HNString::SplitA(str, str_vec, ",");
		if (str_vec.size() < 22)
		{
			continue;
		}
		raw.time = atof(str_vec[0].c_str());
		float sec_in_week = atof(str_vec[2].c_str());

		raw.lonlat.y = atof(str_vec[3].c_str());
		raw.lonlat.x = atof(str_vec[4].c_str());
		raw.lonlat.z = atof(str_vec[5].c_str());

		raw.roll = atof(str_vec[12].c_str());
		raw.pitch = atof(str_vec[13].c_str());
		raw.heading = atof(str_vec[14].c_str());

	//	raw.name = to_string((long long)(raw.time / 1000));
	//	raw.name = to_string((long long)(raw.time));
		raw.name = "";
		raw.point = raw.lonlat;
		if (raw.heading < M_PI)
		{
			raw.heading = -raw.heading;
		}
		else
		{
			raw.heading = M_PI * 2 - raw.heading;
		}

		/*if (int(sec_in_week + 0.1) <= int(sec_in_week))
		{
			continue;
		}*/
		if (int(raw.time / 1000) <= int(raw.time))//ms
		{
//			continue;
		}

		/*if (ins_vec.size() > 0)
		{
			const auto& back = ins_vec.back();
			double dist = abs(raw.lonlat.x - back.lonlat.x) + abs(raw.lonlat.y - back.lonlat.y);
			if (dist < 1e-5)
			{
				continue;
			}
		}*/

		//long long t = raw.time / 1000;
		//relate2ImageTimestamp(t);

		ins_vec.push_back(raw);


	}
	infile.close();

	CoordinateTransform::LonLatPoints2XY(ins_vec, CalibSpace::band);

	return ins_vec.size();
}

size_t DataIO::readHDI(const string& file_path, vector<RAW_INS>& ins_vec)
{
	// 定义输入文件流类对象infile
	ifstream infile(file_path, ios::in);

	if (!infile)
	{  // 判断文件是否存在
		cerr << "open error." << endl;
		return 0;
	}

	char str[255]; // 定义字符数组用来接受读取一行的数据
	RAW_INS raw;
	while (infile)
	{
		infile.getline(str, 255);  // getline函数可以读取整行并保存在str数组里
		vector<string> str_vec;
//		HNString::SplitA(str, str_vec, "	");
//		HNString::SplitA(str, str_vec, " ");
		HNString::SplitA(str, str_vec, "\t");
		if (str_vec.size() < 16)
		{
			continue;
		}
		raw.name = str_vec[0].c_str();
		raw.point.x = atof(str_vec[9].c_str());

		if (int(raw.point.x) == 0 &&
			int(raw.point.y) == 0)
		{
			continue;
		}
		raw.point.y = atof(str_vec[10].c_str());
		raw.point.z = atof(str_vec[11].c_str());

		raw.lonlat.x = atof(str_vec[12].c_str());
		raw.lonlat.y = atof(str_vec[13].c_str());
		raw.lonlat.z = atof(str_vec[11].c_str());

		raw.heading = atof(str_vec[14].c_str()) / 180.0 * M_PI;
		raw.pitch = atof(str_vec[15].c_str()) / 180.0 * M_PI;
		raw.roll = atof(str_vec[16].c_str()) / 180.0 * M_PI;
		if (raw.heading < 0)
		{
			raw.heading += 2.0 * M_PI;
		}
		ins_vec.push_back(raw);
	}
	infile.close();

	if (ins_vec.size() == 0)
	{
		return 0;
	}
	raw = ins_vec.front();
	if (abs(raw.point.x) < 180 &&
		abs(raw.point.y) < 180)
	{
		CoordinateTransform::LonLatPoints2XY(ins_vec, CalibSpace::band);
	}
	return ins_vec.size();
}

size_t getFiles(string path, string filter, vector<string>& files)
{
	//文件句柄
	intptr_t   hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p = path + filter;
	if ((hFile = _findfirst(p.c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(path + fileinfo.name, "", files);
			}
			else
			{
				files.push_back(path + fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

	
	return files.size();
}

size_t DataIO::getHDMapLaneDividerInBox(const string& connect_sql, const vector<cv::Point2f>& box , HDObject_VEC& obj_vec)
{
	string box_str = formatBoxStr(box);	
	string sql_cmd = "";
	string tbl_name = "";
	string join_tbl_name = "";
	//lane divider
	
	tbl_name = "hdobj_lanelevel_geometry";
	join_tbl_name = "hdobj_lane_divider";

		/*HNString::FormatA(sql_cmd, "select g.shapeid, g.shapeid,a.logical, st_astext(st_intersection(shape, %s)) from\
 %s a inner join %s g on a.shapeid = g.shapeid and a.logical=1 where st_intersects(shape, %s)",
		box_str.c_str(),
		join_tbl_name.c_str(),
		tbl_name.c_str(),
		box_str.c_str());*/
		HNString::FormatA(sql_cmd, "select g.shapeid, g.shapeid,13, st_astext(st_intersection(shape, %s)) from\
 %s a inner join %s g on a.shapeid = g.shapeid where st_intersects(shape, %s)",
		box_str.c_str(),
		join_tbl_name.c_str(),
		tbl_name.c_str(),
		box_str.c_str());

	//getHDMap(connect_sql, sql_cmd, obj_vec);
	return obj_vec.size();
}

size_t DataIO::getHDMapSignBoardInBox(const string& connect_sql, const vector<cv::Point2f>& box, HDSignBoard_VEC& sgn_vec)
{
	string box_str = formatBoxStr(box);
	string sql_cmd = "";
	string tbl_name = "";
	string join_tbl_name = "";
	//object
	HDObject_VEC obj_vec;
	sql_cmd = "";
	tbl_name = "hdobj_objlevel_geometry";
	join_tbl_name = "hdobj_object_base";
	//	string sql_cmd = "select shapeid, st_astext(geom) from %s where st_intersects(geom, %s);";
	HNString::FormatA(sql_cmd, "select g.shapeid, a.propid, a.fcode, st_astext(shape) from\
 %s a inner join %s g on a.shapeid = g.shapeid and g.shapetype=3 and a.fcode in (80) where st_contains(%s, shape)",
		join_tbl_name.c_str(),
		tbl_name.c_str(),
		box_str.c_str());

	//getHDMap(connect_sql, sql_cmd, obj_vec, false);

	//signboard
	HDSignBoard_MAP sgb_map;
	//getHDMapSignBoardAttr(connect_sql, box_str, sgb_map);

	buildSignBoard(obj_vec, sgb_map, sgn_vec);
	return sgn_vec.size();
}

size_t DataIO::getHDMapObjectsInBox(const string& connect_sql, const vector<cv::Point2f>& box, HDObject_VEC& obj_vec)
{
	string box_str = formatBoxStr(box);
	string sql_cmd = "";
	string tbl_name = "";
	string join_tbl_name = "";
	//object
	/////////////////////////////////////////////////////////////////////////////////////////
	sql_cmd = "";
	tbl_name = "hdobj_objlevel_geometry";
	join_tbl_name = "hdobj_object_base";
	HNString::FormatA(sql_cmd, "select g.shapeid, a.propid, a.fcode, st_astext(shape) from\
 %s a inner join %s g on a.shapeid = g.shapeid and g.shapetype in (2,3) and a.fcode in (74,71,66,80)\
 where st_contains(%s, shape)",
		join_tbl_name.c_str(),
		tbl_name.c_str(),
		box_str.c_str());

	//getHDMap(connect_sql, sql_cmd, obj_vec, false);

	//////////////////////////////////////////pole//////////////////////////////////////////
	sql_cmd = "";
	HNString::FormatA(sql_cmd, "select g.shapeid, a.propid, a.fcode, st_astext(shape) from\
 %s a inner join %s g on a.shapeid = g.shapeid and g.shapetype=5 and a.fcode in (82) where st_contains(%s, st_collectionextract(shape,1))",
		join_tbl_name.c_str(),
		tbl_name.c_str(),
		box_str.c_str());

	HDObject_VEC org_vec;
	//getHDMap(connect_sql, sql_cmd, org_vec, false);

	//pole
	auto itr_obj = org_vec.begin();
	for (;itr_obj != org_vec.end(); itr_obj++)
	{
		if (itr_obj->type != 82 || itr_obj->shape.size() < 5)
		{		
			continue;
		}
		auto& shp = itr_obj->shape;
		auto ph = shp.back();
		
		for (int i = 0; i < 3; i++)
		{
			HDObject pole_obj = *itr_obj;
			vector<cv::Point3d>& new_shp = pole_obj.shape;
			auto p = shp[i];
			for (double h = p.z; h < ph.z; h+=0.1)
			{
				auto new_p = p;
				new_p.z = h;
				new_shp.push_back(new_p);
			}
			auto new_p = p;
			new_p.z = ph.z;
			new_shp.push_back(new_p);

			obj_vec.push_back(pole_obj);
		}		
	}
	return obj_vec.size();
}



void DataIO::saveHDMap(const string& file_path, const HDObject_VEC& obj_vec)
{
	string file_na = file_path + "hdobj.csv";
	ofstream os(file_na, ios::app);
	if (!os.is_open())
	{
		return;
	}
	os.setf(ios::fixed, ios::floatfield);
	os.precision(4);

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
		outShpCSVFields(os, obj.shape, 1);
	}
	os.close();
}


void DataIO::buildSignBoard(HDObject_VEC& obj_vec, const HDSignBoard_MAP& sgb_map, HDSignBoard_VEC& sgn_vec)
{
	int CIRCLE_POINT_NUM = 10;

	auto itr_obj = obj_vec.begin();
	for (; itr_obj != obj_vec.end(); itr_obj++)
	{
		auto& obj = *itr_obj;
		if (obj.type != 80)
		{
			continue;
		}
		auto& shp = obj.shape;		
		auto find_prop = sgb_map.find(obj.prop_id);
		if (find_prop == sgb_map.end())
		{
			continue;
		}
		const auto& prop = find_prop->second;
		if (prop.frame == 2)
		{
			shp.pop_back();
			shp.push_back(shp[0] + shp[2] - shp[1]);
			shp.push_back(shp[0]);
		}
		//else if (prop.frame == 3)
		//{
		//	//计算法向量，圆心，半径
		//	cv::Point3d normalVec, ptCenter;
		//	double dbRadius;
		//	if (get3DCircleCenter(shp.at(0), shp.at(1), shp.at(2), normalVec, ptCenter, dbRadius))
		//	{
		//		//计算法向量旋转矩阵
		//		normalVec.normalize();
		//		ccGLMatrixd roatMat = ccGLMatrixd::FromToRotation(CCVector3d(0, 0, 1), normalVec);
		//		vector<cv::Point3d> circles;
		//		double radStep = (double)(2 * M_PI / CIRCLE_POINT_NUM);
		//		for (int iSeq = 0; iSeq < CIRCLE_POINT_NUM; iSeq++)
		//		{
		//			//计算z=0平面上，以原点为圆心的圆
		//			double rad = radStep*iSeq;
		//			double xSeq = sin(rad)*dbRadius;
		//			double ySeq = cos(rad)*dbRadius;
		//			cv::Point3d ptXOY(xSeq, ySeq, 0);
		//			//将z=0平面上的点旋转到P0,P1,P2面上
		//			ptXOY = roatMat*ptXOY;
		//			//将圆移动为以ptCenter为圆心。
		//			circles.push_back(ptXOY + ptCenter);
		//		}
		//	}
		//}

		HDSignBoard sb = prop;
		sb.obj_id = obj.obj_id;
		sb.shape = obj.shape;
		sgn_vec.push_back(sb);
	}
}

void DataIO::saveParas(const string& folder_path, const CamPose& cp)
{
	if (cp.camPose.size() != 6)
	{
		return;
	}

	ofstream of(folder_path, ios::out | ios::app);
	if (!of.is_open())
	{
		return;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);
	string line = "";
	HNString::FormatA(line, "%s,%d,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
		cp.img_na.c_str(),
		cp.idx,
		cp.regist_probability,
		cp.camPose[0],
		cp.camPose[1],
		cp.camPose[2],
		cp.camPose[3],
		cp.camPose[4],
		cp.camPose[5]);
	of << line << endl;
	of.close();
}



size_t DataIO::readCamPosePara(const string& folder_path, map<string, CamPose>& pos_map)
{
	ifstream fin(folder_path + "para.csv");
	string line = "";
	while (getline(fin, line))
	{
		vector<string> str_vec;
		HNString::SplitA(line, str_vec, ",");
		if (str_vec.size() == 0)
		{
			continue;
		}
		CamPose cp;
		int seq = 0;
		cp.img_na = str_vec[seq++];

		cp.regist_flg = stoi(str_vec[seq++]);
		cp.regist_probability = stoi(str_vec[seq++]);
		cp.ins.lonlat.x = stod(str_vec[seq++]);
		cp.ins.lonlat.y = stod(str_vec[seq++]);
//		cp.ins.lonlat.z = stod(str_vec[seq++]);
		cp.ins.point.x = stod(str_vec[seq++]);
		cp.ins.point.y = stod(str_vec[seq++]);
		cp.ins.point.z = stod(str_vec[seq++]);
		cp.ins.heading = stod(str_vec[seq++]) / 180.0 * M_PI;
		cp.ins.pitch = stod(str_vec[seq++]) / 180.0 * M_PI;
		cp.ins.roll = stod(str_vec[seq++]) / 180.0 * M_PI;
		cp.camPose.resize(6);
		cp.camPose[0] = stod(str_vec[seq++]);
		cp.camPose[1] = stod(str_vec[seq++]);
		cp.camPose[2] = stod(str_vec[seq++]);
		cp.camPose[3] = stod(str_vec[seq++]) / 180.0 * M_PI;
		cp.camPose[4] = stod(str_vec[seq++]) / 180.0 * M_PI;
		cp.camPose[5] = stod(str_vec[seq++]) / 180.0 * M_PI;
		
		pos_map[cp.img_na] = cp;
	}

	return pos_map.size();
}

size_t DataIO::readPosesOptimized(const string& file_path, map<int, POSE>& poses_map)
{

	ifstream infile(file_path, ios::in);

	if (!infile)
	{  // 判断文件是否存在
		cerr << "open error." << endl;
		return 0;
	}

	string str = ""; 
	while (getline(infile, str))
	{
		vector<string> str_vec;
		HNString::SplitA(str, str_vec, " ");
		
		int id = atoi(str_vec[0].c_str());
		POSE& pose = poses_map[id];
		pose.id = id;

		pose.p[0] = stod(str_vec[1]);
		pose.p[1] = stod(str_vec[2]); 
		pose.p[2] = stod(str_vec[3]);

		pose.q[0] = stof(str_vec[4]);
		pose.q[1] = stof(str_vec[5]);
		pose.q[2] = stof(str_vec[6]);
		pose.q[3] = stof(str_vec[7]);
	}
	infile.close();

	return poses_map.size();
}

void DataIO::writeSignboardCSV(const string& _path, const RAW_INS& ins, const HDSignBoard_VEC& obj_vec)
{
	string file_na(_path + "signboard.csv");
	ofstream os(file_na, ios::app);
	if (!os.is_open())
	{
		return;
	}
	os.setf(ios::fixed, ios::floatfield);
	os.precision(4);

	for (int i = 0; i < obj_vec.size(); i++)
	{
		const auto& obj = obj_vec[i];
		if (obj.shape.size() < 2)
		{
			continue;
		}
		vector<string> attr_vec;
		attr_vec.push_back(obj.obj_id);
		attr_vec.push_back(ins.name);
		outAttrCSVFields(os, attr_vec);

		vector<cv::Point3d> shape;
		shape.push_back(ins.point);
		shape.push_back(obj.shape_org.front());
		outShpCSVFields(os, shape, 1);
	}
	os.close();
}

void DataIO::getConstantVertixes(const string& folder_path, vector<int>& indices)
{
	ifstream of(folder_path + "idc");
	string line = "";
	while (getline(of, line))
	{
		if (line.size() == 0)
		{
			continue;
		}
		int idx = stoi(line);
		indices.push_back(idx);
	}
	of.close();
}



void DataIO::getTraceBox(const vector<RAW_INS>& ins_vec, vector<cv::Point2f>& box, float threshold)
{
	cv::Point2f right_up(-180.0, -90.0);
	cv::Point2f right_down(-180.0, 90.0);
	cv::Point2f left_down(180.0, 90.0);
	cv::Point2f left_up(180.0, -90.0);

	int count = ins_vec.size();
	for (int i = 0; i < count; i++)
	{
		//right_up
		if (right_up.x < ins_vec[i].lonlat.x)
		{
			right_up.x = ins_vec[i].lonlat.x;
		}
		if (right_up.y < ins_vec[i].lonlat.y)
		{
			right_up.y = ins_vec[i].lonlat.y;
		}


		//right_down
		if (right_down.x < ins_vec[i].lonlat.x)
		{
			right_down.x = ins_vec[i].lonlat.x;
		}
		if (right_down.y > ins_vec[i].lonlat.y)
		{
			right_down.y = ins_vec[i].lonlat.y;
		}

		//left_down
		if (left_down.x > ins_vec[i].lonlat.x)
		{
			left_down.x = ins_vec[i].lonlat.x;
		}
		if (left_down.y > ins_vec[i].lonlat.y)
		{
			left_down.y = ins_vec[i].lonlat.y;
		}


		//left_up
		if (left_up.x > ins_vec[i].lonlat.x)
		{
			left_up.x = ins_vec[i].lonlat.x;
		}
		if (left_up.y < ins_vec[i].lonlat.y)
		{
			left_up.y = ins_vec[i].lonlat.y;
		}

	}

	threshold /= 100000;
	right_up.x += threshold;
	right_up.y += threshold;
	right_down.x += threshold;
	right_down.y -= threshold;
	left_down.x -= threshold;
	left_down.y -= threshold;
	left_up.x -= threshold;
	left_up.y += threshold;

	box.push_back(right_up);
	box.push_back(right_down);
	box.push_back(left_down);
	box.push_back(left_up);

}

void DataIO::getTraceXYBox(const vector<RAW_INS>& ins_vec, vector<cv::Point2f>& box, float threshold)
{
	int count = ins_vec.size();
	if (count == 0)
	{
		return;
	}
	auto init_x = ins_vec[0].point.x;
	auto init_y = ins_vec[0].point.y;
	cv::Point2f right_up(init_x, init_y);
	cv::Point2f right_down(right_up);
	cv::Point2f left_down(right_up);
	cv::Point2f left_up(right_up);

	
	for (int i = 0; i < count; i++)
	{
		//right_up
		if (right_up.x < ins_vec[i].point.x)
		{
			right_up.x = ins_vec[i].point.x;
		}
		if (right_up.y < ins_vec[i].point.y)
		{
			right_up.y = ins_vec[i].point.y;
		}


		//right_down
		if (right_down.x < ins_vec[i].point.x)
		{
			right_down.x = ins_vec[i].point.x;
		}
		if (right_down.y > ins_vec[i].point.y)
		{
			right_down.y = ins_vec[i].point.y;
		}

		//left_down
		if (left_down.x > ins_vec[i].point.x)
		{
			left_down.x = ins_vec[i].point.x;
		}
		if (left_down.y > ins_vec[i].point.y)
		{
			left_down.y = ins_vec[i].point.y;
		}


		//left_up
		if (left_up.x > ins_vec[i].point.x)
		{
			left_up.x = ins_vec[i].point.x;
		}
		if (left_up.y < ins_vec[i].point.y)
		{
			left_up.y = ins_vec[i].point.y;
		}

	}

	right_up.x += threshold;
	right_up.y += threshold;
	right_down.x += threshold;
	right_down.y -= threshold;
	left_down.x -= threshold;
	left_down.y -= threshold;
	left_up.x -= threshold;
	left_up.y += threshold;

	box.push_back(right_up);
	box.push_back(right_down);
	box.push_back(left_down);
	box.push_back(left_up);

}


size_t DataIO::getHDLaneDividerInBox(const vector<cv::Point2f>& box, HDObject_VEC& obj_vec, bool keep_inside_intersection)
{
	const string& connect_sql = m_root_path;
	string box_str = formatBoxStr(box, 4547);
	string sql_cmd = "";
	string tbl_name = "";
	string join_tbl_name = "";
	//lane divider
	//select UpdateGeometrySRID('whu_z', 'geom', 4547);

	tbl_name = "whu_z";

	HNString::FormatA(sql_cmd, "select id, st_astext(st_intersection(geom, %s)) from\
 %s where st_intersects(geom, %s)",
		box_str.c_str(),
		tbl_name.c_str(),
		box_str.c_str());

	//getHDMapObjects(connect_sql, sql_cmd, obj_vec);
	return obj_vec.size();
}


void DataIO::saveTracePoints(const vector<RAW_INS>& ins_vec)
{
	ofstream of(m_root_path + "trace_points.csv", ios::out | ios::trunc);
	if (!of.is_open())
	{
		return;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);

	for (const auto& ins : ins_vec)
	{
		string line = "";
		HNString::FormatA(line, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
			ins.name.c_str(),			
			ins.point.x,
			ins.point.y,
			ins.point.z,
			ins.heading / M_PI * 180.0,
			ins.pitch / M_PI * 180.0,
			ins.roll / M_PI * 180.0
			);
		of << line << endl;
	}
	
	of.close();
}


void DataIO::saveJitterCamPose(string file_path, const vector<CamPose>& pose_vec)
{
	ofstream of(file_path, ios::out | ios::trunc);
	if (!of.is_open())
	{
		return;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);

	for (const auto& cp : pose_vec)
	{
		string line = "";
		HNString::FormatA(line, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
			cp.img_na.c_str(),
			cp.camPose[0],
			cp.camPose[1],
			cp.camPose[2],
			cp.camPose[3],
			cp.camPose[4],
			cp.camPose[5]
		);
		of << line << endl;
	}

	of.close();
}

size_t DataIO::readJitterCamPose(string file_path, map<string, CamPose>& pose_map)
{
	ifstream infile(file_path, ios::in);

	if (!infile)
	{  // 判断文件是否存在
		cerr << "open error." << endl;
		return 0;
	}

	string str = "";
	while (getline(infile, str))
	{
		vector<string> str_vec;
		HNString::SplitA(str, str_vec, ",");

		CamPose cp;
		cp.img_na = str_vec[0];
		auto& pose = cp.camPose;
		pose.resize(6);
		for (int i = 0; i < 6; i++)
		{
			pose[i] = stod(str_vec[i +1]);
		}
		pose_map[cp.img_na] = cp;
	}
	infile.close();

	return pose_map.size();
}

void DataIO::saveGTCamPose(string file_path, const vector<CamPose>& pose_vec)
{
	ofstream of(file_path, ios::out | ios::trunc);
	if (!of.is_open())
	{
		return;
	}
	of.setf(ios::fixed, ios::floatfield);
	of.precision(4);

	for (const auto& cp : pose_vec)
	{
		string line = "";
		HNString::FormatA(line, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
			cp.img_na.c_str(),
			cp.camPose[0],
			cp.camPose[1],
			cp.camPose[2],
			cp.camPose[3],
			cp.camPose[4],
			cp.camPose[5]
		);
		of << line << endl;
	}

	of.close();
}

size_t DataIO::readGTCamPose(string file_path, map<string, CamPose>& pose_map)
{
	ifstream infile(file_path, ios::in);

	if (!infile)
	{  // 判断文件是否存在
		cerr << "open error." << endl;
		return 0;
	}

	string str = "";
	while (getline(infile, str))
	{
		vector<string> str_vec;
		HNString::SplitA(str, str_vec, ",");

		CamPose cp;
		cp.img_na = str_vec[0];
		auto& pose = cp.camPose;
		pose.resize(6);
		for (int i = 0; i < 6; i++)
		{
			pose[i] = stod(str_vec[i + 1]);
		}
		pose_map[cp.img_na] = cp;
	}
	infile.close();

	return pose_map.size();
}

void DataIO::importToDB(const string& in_path)
{
//	DBImportor im;
//	im.importToDB(in_path, m_sql);
}

void DataIO::saveToCSV(const string& in_path)
{
//	DBImportor im;
//	im.saveToCSV(in_path, in_path);
}
