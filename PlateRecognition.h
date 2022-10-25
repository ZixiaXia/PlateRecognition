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
		double feature[33]; //��������������
		int number;        //��ʶ���ַ����������е����к�
	};

	const int g_width = 20;   //���ù�һ���Ŀ�� 
	const int g_height = 40; //���ù�һ���ĸ߶� 

	
	IplImage * change_dst_image[7];   //��Ź�һ������ַ���ͼ������

	Mat ImageStretchByHistogram(Mat src); //��ǿ�Աȶ�

	int Otsu(Mat img);  //��ֵ��

	Mat delRivet(Mat image); //ȥ��í��

	Mat reSize(Mat image, int w, int h); //���ô�С

	Mat toushi(Mat image, Point2f pt1, Point2f pt2, Point2f pt3, Point2f pt4); //͸�ӱ任

	cv::Point jiaodian(cv::Point ��1, cv::Point ��2, cv::Point ��3, cv::Point ��4); //�����������ұ߿�Ľ���

	Mat setright(Mat Image); //����

	Mat delEdge(Mat image); //ȥ��

	Mat reverse(Mat src); //��ɫ��ת

	vector<Mat> verticalProjectionMat(Mat srcImg); //��ֱͶӰ

	void GetFeature(IplImage* src, pattern &pat); //������ȡ

	string on_actionCharacterRecognition_C_triggered(); //�ַ�ʶ�𲢷���

	void saveImage(Mat srcImg); //�����ַ��ָ�����õ�7��ͼƬ


};