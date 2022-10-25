#pragma once
#include "stdafx.h"

#include <iostream>

#include<opencv2\core.hpp>

#include<opencv2\highgui.hpp>

#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/highgui/highgui.hpp"

#include <opencv2/core/core.hpp>  


#include <opencv2\imgproc\types_c.h> 

#include<opencv2/imgproc/imgproc.hpp>

#include"highgui.h"

#include <stdio.h>

using namespace cv;

using namespace std;

class PlateRecognition{

public:
	PlateRecognition(string image_path) {
		
		this -> image = imread(image_path);
	}

	string getPlate(PlateRecognition& plate);

	
private:
	string image_path;
	Mat image;

	struct pattern
	{
		double feature[33]; //样本的特征向量
		int number;        //待识别字符在样本库中的序列号
	};

	const int g_width = 20;   //设置归一化的宽度 
	const int g_height = 40; //设置归一化的高度 

	
	IplImage * change_dst_image[7];   //存放归一化后的字符的图像数组

	Mat ImageStretchByHistogram(Mat src); //增强对比度

	int Otsu(Mat img);  //二值化

	Mat delRivet(Mat image); //去除铆钉

	Mat reSize(Mat image, int w, int h); //重置大小

	Mat toushi(Mat image, Point2f pt1, Point2f pt2, Point2f pt3, Point2f pt4); //透视变换

	cv::Point jiaodian(cv::Point 点1, cv::Point 点2, cv::Point 点3, cv::Point 点4); //计算上下左右边框的交点

	Mat setright(Mat Image); //矫正

	Mat delEdge(Mat image); //去边

	Mat reverse(Mat src); //颜色反转

	vector<Mat> verticalProjectionMat(Mat srcImg); //垂直投影

	void GetFeature(IplImage* src, pattern &pat); //特征提取

	string on_actionCharacterRecognition_C_triggered(); //字符识别并返回

	void saveImage(Mat srcImg); //保存字符分割后所得的7张图片


};