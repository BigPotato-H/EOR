
#include "MutualInfo.h"

//using namespace cv;

//

MutualInfo::MutualInfo()
{
}

MutualInfo::~MutualInfo()
{
}

float MutualInfo::calcImageEntropy(const cv::Mat &img)
{
    assert(img.channels() == 1);

    float temp[256] = { 0.0 };

    // 计算每个像素的累积值
    for (int m = 0; m < img.rows; m++)
    {
        // 有效访问行列的方式
        const uchar* t = img.ptr<uchar>(m);
        for (int n = 0; n < img.cols; n++)
        {
            temp[t[n]]++;
        }
    }

    // 计算每个像素的概率
    long imgSize = img.rows*img.cols;
    for (int i = /*0*/1; i < 256; i++)
    {
        temp[i] /= imgSize;
    }

    float result = 0;
    // 计算图像信息熵
    for (int i = /*0*/1; i < 256; i++)
    {
        if (temp[i] > 0)
        {
            result = result - temp[i] * (log(temp[i]) / log(2.0));
        }
    }

    return result;
}

// 两幅图像联合信息熵计算
float MutualInfo::calcCombineEntropy(const cv::Mat &img1, const cv::Mat &img2)
{
    assert(img1.channels() == 1 && img2.channels() == 1);
    assert(img1.cols == img2.cols && img1.rows == img2.rows);


    float temp[256][256] = { 0.0 };

    // 计算联合图像像素的累积值
    for (int m1 = 0; m1 < img1.rows; m1++)
    {
        // 有效访问行列的方式
        const uchar* t1 = img1.ptr<uchar>(m1);
        const uchar* t2 = img2.ptr<uchar>(m1);
        for (int n1 = 0; n1 < img1.cols; n1++)
        {
            temp[t1[n1]][t2[n1]]++;
        }
    }

    // 计算每个联合像素的概率
    long imgSzie = img1.rows*img1.cols;
    for (int i = /*0*/1; i < 256; i++)
    {
        for (int j = /*0*/1; j < 256; j++)
        {
            temp[i][j] /= imgSzie;
        }
    }

    float result = 0.0;
    //计算图像联合信息熵
    for (int i = /*0*/1; i < 256; i++)
    {
        for (int j = /*0*/1; j < 256; j++)
        {
            if (temp[i][j] > 0.0)
                result = result - temp[i][j] * (log(temp[i][j]) / log(2.0));
        }
    }

    return result;
}

float MutualInfo::calcMutualInformation(const cv::Mat &img1, const cv::Mat &img2)
{
    //得到两幅图像的互信息熵
    float img1_entropy = calcImageEntropy(img1);
    float img2_entropy = calcImageEntropy(img2);
    float combineEntropy = calcCombineEntropy(img1, img2);
    float result = img1_entropy + img2_entropy - combineEntropy;
	result = (img1_entropy + img2_entropy) / combineEntropy;
    return result;
}

float MutualInfo::calcMutualInformation(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &mask)
{
    assert(img1.channels() == 1 && img2.channels() == 1 && mask.channels() == 1);
    assert(img1.cols == img2.cols && img1.rows == img2.rows && img1.cols == mask.cols && img1.rows == mask.rows);

    float I1[256] = { 0 };
    float I2[256] = { 0 };
    float combin[256][256] = { 0.0 };

    int sum = 0;
    // 计算联合图像像素的累积值
    for (int i = 0; i < img1.rows; i++)
    {
        // 有效访问行列的方式
        const uchar* t1 = img1.ptr<uchar>(i);
        const uchar* t2 = img2.ptr<uchar>(i);
        const uchar* m = mask.ptr<uchar>(i);
        for (int j = 0; j < img1.cols; j++)
        {
            if (m[j] != 0)
            {
                ++sum;
                combin[t1[j]][t2[j]]++;
                I1[t1[j]]++;
                I2[t2[j]]++;
            }
        }
    }

    // 计算每个联合像素的概率
    long imgSzie = img1.rows*img1.cols;
    for (int i = /*0*/1; i < 256; i++)
    {
        I1[i] /= imgSzie;
        I2[i] /= imgSzie;
        for (int j = /*0*/1; j < 256; j++)
        {
            combin[i][j] /= imgSzie;
        }
    }

    float entropy1(.0), entropy2(.0);
    float result = 0.0;
    //计算图像联合信息熵
    for (int i = /*0*/1; i < 256; i++)
    {
        if (I1[i] > 0)
        {
            entropy1 -= I1[i] * (log(I1[i]) / log(2.0));
        }
        if (I1[i] > 0)
        {
            entropy2 -= I2[i] * (log(I2[i]) / log(2.0));
        }
        for (int j = /*0*/1; j < 256; j++)
        {
            if (combin[i][j] > 0.0)
                result = result - combin[i][j] * (log(combin[i][j]) / log(2.0));
        }
    }

    return entropy1 + entropy2 - result;
}


//https://blog.csdn.net/u013972657/article/details/106826924
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/features2d.hpp>
//#include<opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <algorithm>
#include <functional>
#include <memory>
#include <iterator>
#include <fstream>
#include <numeric>
#include <utility>

using namespace std;


//mutual_information函数
float MutualInfo::mutual_information(cv::Mat ref, cv::Mat flt)
{
	cv::Mat joint_histogram(256, 256, CV_64FC1, cv::Scalar(0));

	for (int i = 0; i < ref.cols; ++i) {
		for (int j = 0; j < ref.rows; ++j) {
			int ref_intensity = ref.at<uchar>(j, i);
			int flt_intensity = flt.at<uchar>(j, i);
			//if (flt_intensity == 0 || ref_intensity == 0)
			//{
			//	continue;
			//}
			joint_histogram.at<double>(ref_intensity, flt_intensity) = joint_histogram.at<double>(ref_intensity, flt_intensity) + 1;
			double v = joint_histogram.at<double>(ref_intensity, flt_intensity);
		}
	}



	for (int i = 0; i < 256; ++i) {
		for (int j = 0; j < 256; ++j) {
			joint_histogram.at<double>(j, i) = joint_histogram.at<double>(j, i) / (1.0*ref.rows*ref.cols);
			double v = joint_histogram.at<double>(j, i);
		}
	}

	cv::Size ksize(7, 7);
	cv::GaussianBlur(joint_histogram, joint_histogram, ksize, 7, 7);


	double entropy = 0.0;
	for (int i = 0; i < 256; ++i) {
		for (int j = 0; j < 256; ++j) {
			double v = joint_histogram.at<double>(j, i);
			if (v > 0.000000000000001) {
				entropy += v*log(v) / log(2);
			}
		}
	}
	entropy *= -1;

	//    std::cout << entropy << "###";

	std::vector<double> hist_ref(256, 0.0);
	for (int i = 0; i < joint_histogram.rows; ++i) {
		for (int j = 0; j < joint_histogram.cols; ++j) {
			hist_ref[i] += joint_histogram.at<double>(i, j);
		}
	}

	cv::Size ksize2(5, 0);
	//  cv::GaussianBlur(hist_ref, hist_ref, ksize2, 5);


	std::vector<double> hist_flt(256, 0.0);
	for (int i = 0; i < joint_histogram.cols; ++i) {
		for (int j = 0; j < joint_histogram.rows; ++j) {
			hist_flt[i] += joint_histogram.at<double>(j, i);
		}
	}

	//   cv::GaussianBlur(hist_flt, hist_flt, ksize2, 5);



	double entropy_ref = 0.0;
	for (int i = 0; i < 256; ++i) {
		if (hist_ref[i] > 0.000000000001) {
			entropy_ref += hist_ref[i] * log(hist_ref[i]) / log(2);
		}
	}
	entropy_ref *= -1;
	//std::cout << entropy_ref << "~~ ";

	double entropy_flt = 0.0;
	for (int i = 0; i < 256; ++i) {
		if (hist_flt[i] > 0.000000000001) {
			entropy_flt += hist_flt[i] * log(hist_flt[i]) / log(2);
		}
	}
	entropy_flt *= -1;
	// std::cout << entropy_flt << "++ ";

	double mutual_information = entropy_ref + entropy_flt - entropy;

//	double mutual_information = (entropy_ref + entropy_flt - entropy) / (entropy_ref + entropy_flt);
	return mutual_information;
}


//int main()
//{
//
//	string refpath = "1.png";
//	string fltpath = "2.png";
//	//通过配准融合
//	Mat fused = perform_fusion_from_files(refpath, fltpath, "mutualinformation");
//	//不配准直接融合
//	Mat fused_unregistered = perform_fusion_from_files(refpath, fltpath, "identity");
//
//	imwrite("配准融合图像.png", fused);
//	imwrite("不配准直接融合.png", fused_unregistered);
//	return 0;
//}