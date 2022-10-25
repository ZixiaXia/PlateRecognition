#include "stdafx.h"
#include "PlateRecognition.h"


int expandCount = 7;
int expandRang[2] = { 9,46 }, pointRang[2] = { 85,100 };
int wFilter = 6, hFilter = 15;
int denoiseNum = 5;
#define PI 3.1415926


string PlateRecognition::getPlate(PlateRecognition& plate)
{
	imwrite("picture.bmp", plate.image);

	plate.image = imread("picture.bmp", 0);

	int delimage = remove("picture.bmp");

	//创建无初始化矩阵
	Mat result;

	//增强对比度
	result = plate.ImageStretchByHistogram(plate.image);

	//高斯模糊
	GaussianBlur(result, result, Size(3, 3), 0, 0, BORDER_DEFAULT);

	////灰度化
	//cvtColor(result, result, CV_RGB2GRAY);

	//矫正
	result = plate.setright(result);

	//二值化
	int thres = plate.Otsu(result);
	threshold(result, result, thres, 255, THRESH_BINARY);

	//去边
	result = plate.delEdge(result);

	//去除铆钉	
	result = plate.delRivet(result);

	// 重置图片大小为300 * 50，便于字符分割
	result = plate.reSize(result, 300, 50);

	//保存预处理所得的图片
	imwrite("result.png", result);  

	//imshow("结果", result);
	//imshow("原始图片", image);

	// 等待6000 ms后窗口自动关闭
	//waitKey(6000000);

	result = reverse(result);
	this->saveImage(result);     //字符分割,并保存所得的图片
	
	
	//将字符分割后所得的图片载入到图像数组中
	change_dst_image[0] = cvLoadImage("0.png", 0);
	change_dst_image[1] = cvLoadImage("1.png", 0);
	change_dst_image[2] = cvLoadImage("2.png", 0);
	change_dst_image[3] = cvLoadImage("3.png", 0);
	change_dst_image[4] = cvLoadImage("4.png", 0);
	change_dst_image[5] = cvLoadImage("5.png", 0);
	change_dst_image[6] = cvLoadImage("6.png", 0);


	return plate.on_actionCharacterRecognition_C_triggered();
	
	
}

//增强对比度
Mat PlateRecognition::ImageStretchByHistogram(Mat src)
{
	Mat dest;
	dest = Mat::zeros(src.size(), src.type());
	int width = src.cols;
	int height = src.rows;
	int channels = src.channels();

	int alphe = 1.8; //(alphe > 1)
	int beta = -30;// 负数对比度越高
	Mat m1;
	src.convertTo(m1, CV_32F); //将原始图片数据（CV_8U类型）转换成CV_32类型，以提高操作的精度
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (channels == 3) { //对于3通道
				float b = m1.at<Vec3f>(row, col)[0];
				float g = m1.at<Vec3f>(row, col)[1];
				float r = m1.at<Vec3f>(row, col)[2];

				dest.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(alphe * b + beta);
				dest.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(alphe * g + beta);
				dest.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(alphe * r + beta);
			}
			else if (channels == 1) { //对于单通道
				int pix = src.at<uchar>(row, col);
				dest.at<uchar>(row, col) = saturate_cast<uchar>(alphe * pix + beta);
			}
		}
	}

	return dest;
}

//二值化
int PlateRecognition::Otsu(Mat img)
{
	int height = img.rows; // Mat 中读数据的方式类型与Ipilmage中的区别
	int width = img.cols;
	float histogram[256] = { 0 };
	for (int i = 0; i < height; i++)
	{
		unsigned char *p = (unsigned char *)img.data + img.step*i;
		for (int j = 0; j < width; j++)
		{
			histogram[*p++]++;
		}
	}
	int size = height * width;
	for (int i = 0; i < 256; i++)
	{
		histogram[i] = histogram[i] / size;
	}
	float avgValue = 0;
	for (int i = 0; i < 256; i++)
	{
		avgValue += i * histogram[i];
	}
	int threshold;
	float maxVariance = 0;
	float w = 0, u = 0;
	for (int i = 0; i < 256; i++)
	{
		w += histogram[i];
		u += i * histogram[i];
		float t = avgValue * w - u;
		float variance = t * t / (w*(1 - w));
		if (variance > maxVariance)
		{
			maxVariance = variance;
			threshold = i;
		}
	}
	return threshold;
}


//去除铆钉
Mat PlateRecognition::delRivet(Mat image)
{
	int  thisPoint, lastPoint, hop;
	int upperbound = image.rows / 2;
	int lowerbound = image.rows / 2;

	Mat result;
	//寻找字的下边界
	for (int i = image.rows / 2; i <= image.rows - 1; ++i)
	{
		hop = 0;
		for (int j = 0; j <= image.cols - 1; ++j)
		{
			thisPoint = image.at<uchar>(i, j);
			if (j == 0)
			{
				lastPoint = image.at<uchar>(i, j);
			}
			else
			{
				lastPoint = image.at<uchar>(i, j - 1);
			}

			if (thisPoint != lastPoint)
			{
				hop++;
			}
		}

		if (hop < 7)
		{
			lowerbound = i;
			goto findlowerbound;
		}
	}
    findlowerbound:

	//寻找字的上边界
	for (int i = image.rows / 2; i >= 0; --i)
	{
		hop = 0;
		for (int j = 0; j <= image.cols - 1; ++j)
		{
			thisPoint = image.at<uchar>(i, j);
			if (j == 0)
			{
				lastPoint = image.at<uchar>(i, j);
			}
			else
			{
				lastPoint = image.at<uchar>(i, j - 1);
			}

			if (thisPoint != lastPoint)
			{
				hop++;
			}
		}

		if (hop < 7)
		{
			upperbound = i;
			goto findupperbound;
		}
	}
    findupperbound:

	for (int i = upperbound; i <= lowerbound; ++i)
	{
		Mat temp = image.row(i).clone();
		result.push_back(temp);
	}
	return result;
}

//重置大小
Mat PlateRecognition::reSize(Mat image, int w, int h)
{
	CvSize czSize;
	IplImage *pSrcImage = &IplImage(image);
	IplImage *pDstImage = NULL;

	//图像大小
	czSize.width = w;
	czSize.height = h;

	//创建图像并缩放
	pDstImage = cvCreateImage(czSize, pSrcImage->depth, pSrcImage->nChannels);
	cvResize(pSrcImage, pDstImage, CV_INTER_AREA);

	cv::Mat result = cv::cvarrToMat(pDstImage);
	return result;
}

//透视变换
Mat PlateRecognition::toushi(Mat image, Point2f pt1, Point2f pt2, Point2f pt3, Point2f pt4)
{
	Mat src = image;

	//取车牌四边形四个顶点
	cv::Point2f srcPts[4];
	srcPts[0] = pt1;	//左上
	srcPts[1] = pt2;	//左下
	srcPts[2] = pt3;	//右上
	srcPts[3] = pt4;	//右下

	//计算原图中四个点的横纵坐标最大值小值,考虑位置特点,无需一一比较
	int MinX = min(srcPts[0].x, srcPts[1].x);
	int MaxX = max(srcPts[2].x, srcPts[3].x);
	int MinY = min(srcPts[0].y, srcPts[2].y);
	int MaxY = max(srcPts[1].y, srcPts[3].y);

	//根据最大最小坐标值设定目标图像中的矩形四个顶点,注意对应关系
	cv::Point2f dstPts[4];
	dstPts[0] = cv::Point2f(MinX, MinY);
	dstPts[1] = cv::Point2f(MinX, MaxY);
	dstPts[2] = cv::Point2f(MaxX, MinY);
	dstPts[3] = cv::Point2f(MaxX, MaxY);

	//计算透视变换矩阵
	cv::Mat perspectiveMat = getPerspectiveTransform(srcPts, dstPts);

	//对原图进行透视变换，完成车牌校正
	cv::Mat dst;
	cv::warpPerspective(src, dst, perspectiveMat, src.size());

	return dst;
}

//计算上下左右边框的交点
cv::Point PlateRecognition::jiaodian(cv::Point 点1, cv::Point 点2, cv::Point 点3, cv::Point 点4)
{
	int x, y;

	int X1 = 点1.x - 点2.x, Y1 = 点1.y - 点2.y, X2 = 点3.x - 点4.x, Y2 = 点3.y - 点4.y;

	if (X1*Y2 == X2 * Y1)return cv::Point((点2.x + 点3.x) / 2, (点2.y + 点3.y) / 2);

	int A = X1 * 点1.y - Y1 * 点1.x, B = X2 * 点3.y - Y2 * 点3.x;

	y = (A*Y2 - B * Y1) / (X1*Y2 - X2 * Y1);

	x = (B*X1 - A * X2) / (Y1*X2 - Y2 * X1);

	return cv::Point(x, y);
}

//矫正
Mat PlateRecognition::setright(Mat Image)
{
	//边缘检测
	Mat CannyImg;
	Canny(Image, CannyImg, 140, 250, 3);

	//灰度化
	Mat DstImg;
	cvtColor(Image, DstImg, CV_GRAY2BGR);

	//霍夫变换
	vector<Vec4i> Lines;
	HoughLinesP(CannyImg, Lines, 1, CV_PI / 360, 1, 30, 15);

	//初始化
	int umaxlength = 0, lmaxlength = 0, lmaxwidth = 0, rmaxwidth = 0;
	size_t ulevel = -1, llevel = Lines.size(), lvertical = -1, rvertical = Lines.size();

	//确定上下四个边界
	for (size_t i = 0; i < Lines.size(); i++)
	{
		line(DstImg, Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]), Scalar(0, 0, 255), 2, 8);		int distance = (Lines[i][0] - Lines[i][2])*(Lines[i][0] - Lines[i][2]) + (Lines[i][1] - Lines[i][3])*(Lines[i][1] - Lines[i][3]);
		if (Lines[i][2] != Lines[i][0])
		{
			int angle = (Lines[i][3] - Lines[i][1]) / (Lines[i][2] - Lines[i][0]);
			if (angle > -1 && angle < 1)//水平边框
			{
				if (max(Lines[i][1], Lines[i][3]) < Image.rows / 5)
				{
					umaxlength = max(distance, umaxlength);
					ulevel = umaxlength == distance ? i : ulevel;
				}
				else if (min(Lines[i][1], Lines[i][3]) > 4 * (Image.rows / 5))
				{
					lmaxlength = max(distance, lmaxlength);
					llevel = lmaxlength == distance ? i : llevel;
				}
			}
			else//近似垂直边框
			{
				if (max(Lines[i][0], Lines[i][2]) < Image.cols / 20)
				{
					lmaxwidth = max(distance, lmaxwidth);
					lvertical = lmaxwidth == distance ? i : lvertical;
				}
				else if (min(Lines[i][0], Lines[i][2]) > 19 * (Image.cols / 20))
				{
					rmaxwidth = max(distance, rmaxwidth);
					rvertical = rmaxwidth == distance ? i : rvertical;
				}
			}
		}
		else//垂直边框
		{
			if (max(Lines[i][0], Lines[i][2]) < Image.cols / 20)
			{
				lmaxwidth = max(distance, lmaxwidth);
				lvertical = lmaxwidth == distance ? i : lvertical;
			}
			else if (min(Lines[i][0], Lines[i][2]) > 19 * (Image.cols / 20))
			{
				rmaxwidth = max(distance, rmaxwidth);
				rvertical = rmaxwidth == distance ? i : rvertical;
			}
		}
	}

	int ulevel0 = ulevel == -1 ? 1 : Lines[ulevel][0];
	int ulevel1 = ulevel == -1 ? 1 : Lines[ulevel][1];
	int ulevel2 = ulevel == -1 ? Image.cols - 1 : Lines[ulevel][2];
	int ulevel3 = ulevel == -1 ? 1 : Lines[ulevel][3];

	int llevel0 = llevel == Lines.size() ? 1 : Lines[llevel][0];
	int llevel1 = llevel == Lines.size() ? Image.rows - 1 : Lines[llevel][1];
	int llevel2 = llevel == Lines.size() ? Image.cols - 1 : Lines[llevel][2];
	int llevel3 = llevel == Lines.size() ? Image.rows - 1 : Lines[llevel][3];

	int lvertical0 = lvertical == -1 ? 1 : Lines[lvertical][0];
	int lvertical1 = lvertical == -1 ? 1 : Lines[lvertical][1];
	int lvertical2 = lvertical == -1 ? 1 : Lines[lvertical][2];
	int lvertical3 = lvertical == -1 ? Image.rows - 1 : Lines[lvertical][3];

	int rvertical0 = rvertical == Lines.size() ? Image.cols - 1 : Lines[rvertical][0];
	int rvertical1 = rvertical == Lines.size() ? 1 : Lines[rvertical][1];
	int rvertical2 = rvertical == Lines.size() ? Image.cols - 1 : Lines[rvertical][2];
	int rvertical3 = rvertical == Lines.size() ? Image.rows - 1 : Lines[rvertical][3];

	//左上角
	Point2f pt1 = jiaodian(Point(ulevel0, ulevel1), Point(ulevel2, ulevel3),
		Point(lvertical0, lvertical1), Point(lvertical2, lvertical3));
	//左下角
	Point2f pt2 = jiaodian(Point(llevel0, llevel1), Point(llevel2, llevel3),
		Point(lvertical0, lvertical1), Point(lvertical2, lvertical3));
	//右上角
	Point2f pt3 = jiaodian(Point(ulevel0, ulevel1), Point(ulevel2, ulevel3),
		Point(rvertical0, rvertical1), Point(rvertical2, rvertical3));
	//右下角
	Point2f pt4 = jiaodian(Point(llevel0, llevel1), Point(llevel2, llevel3),
		Point(rvertical0, rvertical1), Point(rvertical2, rvertical3));
	cv::circle(DstImg, pt1, 4, cv::Scalar(0, 255, 0));
	cv::circle(DstImg, pt2, 4, cv::Scalar(0, 255, 0));
	cv::circle(DstImg, pt3, 4, cv::Scalar(0, 255, 0));
	cv::circle(DstImg, pt4, 4, cv::Scalar(0, 255, 0));

	imshow("划线", DstImg);

	//透视变换
	Mat result = toushi(Image, pt1, pt2, pt3, pt4);

	return result;
}

//去边
Mat PlateRecognition::delEdge(Mat image)
{
	Mat result;

	for (int i = 2; i <= image.rows - 3; ++i)
	{
		Mat temp = image.row(i).clone();

		result.push_back(temp);
	}

	return result;
}



//颜色反转
Mat PlateRecognition::reverse(Mat src)
{
	Mat dst = src < 100;
	return dst;
}


//垂直投影
vector<Mat> PlateRecognition::verticalProjectionMat(Mat srcImg)
{
	Mat binImg;
	blur(srcImg, binImg, Size(3, 3));
	threshold(binImg, binImg, 0, 255, CV_THRESH_OTSU);
	int perPixelValue;//每个像素的值
	int width = srcImg.cols;
	int height = srcImg.rows;
	int* projectValArry = new int[width];//创建用于储存每列白色像素个数的数组
	memset(projectValArry, 0, width * 4);//初始化数组
	for (int col = 0; col < width; col++)
	{
		for (int row = 0; row < height; row++)
		{
			perPixelValue = binImg.at<uchar>(row, col);
			if (perPixelValue == 0)//如果是白底黑字
			{
				projectValArry[col]++;
			}
		}
	}
	Mat verticalProjectionMat(height, width, CV_8UC1);//垂直投影的画布
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			perPixelValue = 255;  //背景设置为白色
			verticalProjectionMat.at<uchar>(i, j) = perPixelValue;
		}
	}
	for (int i = 0; i < width; i++)//垂直投影直方图
	{
		for (int j = 0; j < projectValArry[i]; j++)
		{
			perPixelValue = 0;  //直方图设置为黑色  
			verticalProjectionMat.at<uchar>(height - 1 - j, i) = perPixelValue;
		}
	}
	//imshow("垂直投影", verticalProjectionMat);
	//cvWaitKey(0);
	vector<Mat> roiList;//用于储存分割出来的每个字符
	int startIndex = 0;//记录进入字符区的索引
	int endIndex = 0;//记录进入空白区域的索引
	bool inBlock = false;//是否遍历到了字符区内
	for (int i = 0; i < srcImg.cols; i++)//cols=width
	{
		cout << "projectValArry[" << i << "]= " << projectValArry[i] << endl;
		if (!inBlock && projectValArry[i] != 0)//进入字符区
		{
			if (pointRang[0] < i && i < pointRang[1]) {//忽略圆点
				inBlock = false;
			}
			else {
				inBlock = true;
				startIndex = i;
			}
		}
		else if (projectValArry[i] == 0 && inBlock)//进入空白区
		{
			int sign = 0;
			if (expandRang[0] < i && i < expandRang[1]) {//防止膨胀边框
				for (int j = i; j < i + expandCount; j++) {
					sign += projectValArry[j];
				}
			}
			if (sign == 0) {
				endIndex = i;
				inBlock = false;
				Mat roiImg = srcImg(Range(0, srcImg.rows), Range(startIndex, endIndex + 1));
				//imshow("t",roiImg);
				roiList.push_back(roiImg);
			}
		}
	}
	//cout << "projectValArry= " << projectValArry << endl;
	delete[] projectValArry;
	return roiList;
}




//保存字符分割所得的7张图片
void PlateRecognition::saveImage(Mat srcImg)
{
	vector<Mat> b = verticalProjectionMat(srcImg);//先进行垂直投影	

    //cout << "b_size= " << b.size() << endl;
	char szName[30] = { 0 };
	//Mat resize_image = Mat::zeros(Size(g_width, g_height), CV_8UC3); //设置图片大小为20 * 40
	for (int i = 0, j = 0; i < b.size(); i++)
	{

		//imshow("b", b[i]);
		//保存切分的结果
		Mat Img = b[i];
		Img = reverse(Img);
		//Img = cvarrToMat(&img);
		int w = Img.cols;
		int h = Img.rows;
		if (w > wFilter && h > hFilter) {
			sprintf_s(szName, "%d.png", j);
			//imshow(szName, b[i]);
			//resize(Img, resize_image,resize_image.size()); //重置图片大小再保存
			Img = reSize(Img, 20, 40);
			imwrite(szName, Img);
			j++;
		}
	}
}




//特征提取
void PlateRecognition::GetFeature(IplImage * src, pattern & pat)
{
	CvScalar s;
	int i, j;
	for (i = 0; i < 33; i++)
		pat.feature[i] = 0.0;
	//图像大小是20*40大小的，分成25块

	//********第一行***********	
	//第一块
	for (j = 0; j < 8; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[0] += 1.0;
		}
	}

	//第二块
	for (j = 0; j < 8; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[1] += 1.0;
		}
	}
	//第三块
	for (j = 0; j < 8; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[2] += 1.0;
		}
	}
	//第四块
	for (j = 0; j < 8; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[3] += 1.0;
		}
	}
	//第五块
	for (j = 0; j < 8; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[4] += 1.0;
		}
	}
	//********第二行***********	
	//第六块
	for (j = 8; j < 16; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[5] += 1.0;
		}
	}
	//第七块
	for (j = 8; j < 16; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[6] += 1.0;
		}
	}
	//第八块
	for (j = 8; j < 16; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[7] += 1.0;
		}
	}
	//第九块
	for (j = 8; j < 16; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[8] += 1.0;
		}
	}
	//第十块
	for (j = 8; j < 16; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[9] += 1.0;
		}
	}
	//********第三行***********
	//第十一块
	for (j = 16; j < 24; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[10] += 1.0;
		}
	}
	//第十二块
	for (j = 16; j < 24; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[11] += 1.0;
		}
	}
	//第十三块
	for (j = 16; j < 24; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[12] += 1.0;
		}
	}
	//第十四块
	for (j = 16; j < 24; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[13] += 1.0;
		}
	}
	//第十五块
	for (j = 16; j < 24; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[14] += 1.0;
		}
	}
	//********第四行***********
	//第十六块
	for (j = 24; j < 32; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[15] += 1.0;
		}
	}
	//第十七块
	for (j = 24; j < 32; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[16] += 1.0;
		}
	}
	//第十八块
	for (j = 24; j < 32; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[17] += 1.0;
		}
	}
	//第十九块
	for (j = 24; j < 32; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[18] += 1.0;
		}
	}
	//第二十块
	for (j = 24; j < 32; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[19] += 1.0;
		}
	}
	//********第五行***********
	//第二十一块
	for (j = 32; j < 40; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[20] += 1.0;
		}
	}
	//第二十二块
	for (j = 32; j < 40; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[21] += 1.0;
		}
	}
	//第二十三块
	for (j = 32; j < 40; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[22] += 1.0;
		}
	}
	//第二十四块
	for (j = 32; j < 40; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[23] += 1.0;
		}
	}
	//第二十五块
	for (j = 32; j < 40; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[24] += 1.0;
		}
	}

	//下面统计方向交点特征
	for (i = 0; i < 20; i++)
	{
		s = cvGet2D(src, 8, i);
		if (s.val[0] == 255)
			pat.feature[25] += 1.0;
	}
	for (i = 0; i < 20; i++)
	{
		s = cvGet2D(src, 16, i);
		if (s.val[0] == 255)
			pat.feature[26] += 1.0;
	}
	for (i = 0; i < 20; i++)
	{
		s = cvGet2D(src, 24, i);
		if (s.val[0] == 255)
			pat.feature[27] += 1.0;
	}
	for (i = 0; i < 20; i++)
	{
		s = cvGet2D(src, 32, i);
		if (s.val[0] == 255)
			pat.feature[28] += 1.0;
	}
	for (j = 0; j < 40; j++)
	{
		s = cvGet2D(src, j, 4);
		if (s.val[0] == 255)
			pat.feature[29] += 1.0;
	}
	for (j = 0; j < 40; j++)
	{
		s = cvGet2D(src, j, 8);
		if (s.val[0] == 255)
			pat.feature[30] += 1.0;
	}
	for (j = 0; j < 40; j++)
	{
		s = cvGet2D(src, j, 12);
		if (s.val[0] == 255)
			pat.feature[31] += 1.0;
	}
	for (j = 0; j < 40; j++)
	{
		s = cvGet2D(src, j, 16);
		if (s.val[0] == 255)
			pat.feature[32] += 1.0;
	}
}

//字符识别
string PlateRecognition::on_actionCharacterRecognition_C_triggered()
{
	IplImage * char_sample[34];//字符样本图像数组
	IplImage * hanzi_sample[9];//汉字样本图像数组
	pattern char_pattern[34];//定义字符样品库结构数组
	pattern hanzi_pattern[9];//定义汉字样品库结构数组
	pattern TestSample[7];//定义待识别字符结构数组

	IplImage * charO_sample[24];//仅含字母，用于地级市识别
	pattern charO_pattern[24];


	//载入字符模板
	char_sample[0] = cvLoadImage("template\\0.bmp", 0);
	char_sample[1] = cvLoadImage("template\\1.bmp", 0);
	char_sample[2] = cvLoadImage("template\\2.bmp", 0);
	char_sample[3] = cvLoadImage("template\\3.bmp", 0);
	char_sample[4] = cvLoadImage("template\\4.bmp", 0);
	char_sample[5] = cvLoadImage("template\\5.bmp", 0);
	char_sample[6] = cvLoadImage("template\\6.bmp", 0);
	char_sample[7] = cvLoadImage("template\\7.bmp", 0);
	char_sample[8] = cvLoadImage("template\\8.bmp", 0);
	char_sample[9] = cvLoadImage("template\\9.bmp", 0);
	char_sample[10] = cvLoadImage("template\\A.bmp", 0);
	char_sample[11] = cvLoadImage("template\\B.bmp", 0);
	char_sample[12] = cvLoadImage("template\\C.bmp", 0);
	char_sample[13] = cvLoadImage("template\\D.bmp", 0);
	char_sample[14] = cvLoadImage("template\\E.bmp", 0);
	char_sample[15] = cvLoadImage("template\\F.bmp", 0);
	char_sample[16] = cvLoadImage("template\\G.bmp", 0);
	char_sample[17] = cvLoadImage("template\\H.bmp", 0);
	char_sample[18] = cvLoadImage("template\\J.bmp", 0);
	char_sample[19] = cvLoadImage("template\\K.bmp", 0);
	char_sample[20] = cvLoadImage("template\\L.bmp", 0);
	char_sample[21] = cvLoadImage("template\\M.bmp", 0);
	char_sample[22] = cvLoadImage("template\\N.bmp", 0);
	char_sample[23] = cvLoadImage("template\\P.bmp", 0);
	char_sample[24] = cvLoadImage("template\\Q.bmp", 0);
	char_sample[25] = cvLoadImage("template\\R.bmp", 0);
	char_sample[26] = cvLoadImage("template\\S.bmp", 0);
	char_sample[27] = cvLoadImage("template\\T.bmp", 0);
	char_sample[28] = cvLoadImage("template\\U.bmp", 0);
	char_sample[29] = cvLoadImage("template\\V.bmp", 0);
	char_sample[30] = cvLoadImage("template\\W.bmp", 0);
	char_sample[31] = cvLoadImage("template\\X.bmp", 0);
	char_sample[32] = cvLoadImage("template\\Y.bmp", 0);
	char_sample[33] = cvLoadImage("template\\Z.bmp", 0);

	charO_sample[0] = cvLoadImage("template\\A.bmp", 0);
	charO_sample[1] = cvLoadImage("template\\B.bmp", 0);
	charO_sample[2] = cvLoadImage("template\\C.bmp", 0);
	charO_sample[3] = cvLoadImage("template\\D.bmp", 0);
	charO_sample[4] = cvLoadImage("template\\E.bmp", 0);
	charO_sample[5] = cvLoadImage("template\\F.bmp", 0);
	charO_sample[6] = cvLoadImage("template\\G.bmp", 0);
	charO_sample[7] = cvLoadImage("template\\H.bmp", 0);
	charO_sample[8] = cvLoadImage("template\\J.bmp", 0);
	charO_sample[9] = cvLoadImage("template\\K.bmp", 0);
	charO_sample[10] = cvLoadImage("template\\L.bmp", 0);
	charO_sample[11] = cvLoadImage("template\\M.bmp", 0);
	charO_sample[12] = cvLoadImage("template\\N.bmp", 0);
	charO_sample[13] = cvLoadImage("template\\P.bmp", 0);
	charO_sample[14] = cvLoadImage("template\\Q.bmp", 0);
	charO_sample[15] = cvLoadImage("template\\R.bmp", 0);
	charO_sample[16] = cvLoadImage("template\\S.bmp", 0);
	charO_sample[17] = cvLoadImage("template\\T.bmp", 0);
	charO_sample[18] = cvLoadImage("template\\U.bmp", 0);
	charO_sample[19] = cvLoadImage("template\\V.bmp", 0);
	charO_sample[20] = cvLoadImage("template\\W.bmp", 0);
	charO_sample[21] = cvLoadImage("template\\X.bmp", 0);
	charO_sample[22] = cvLoadImage("template\\Y.bmp", 0);
	charO_sample[23] = cvLoadImage("template\\Z.bmp", 0);

	//载入汉字模板
	/*hanzi_sample[0]=cvLoadImage("template\\川.bmp",0);
	hanzi_sample[1]=cvLoadImage("template\\黑.bmp",0);
	hanzi_sample[2]=cvLoadImage("template\\京.bmp",0);
	hanzi_sample[3]=cvLoadImage("template\\辽.bmp",0);
	hanzi_sample[4]=cvLoadImage("template\\琼.bmp",0);
	hanzi_sample[5]=cvLoadImage("template\\粤.bmp",0);
	hanzi_sample[6]=cvLoadImage("template\\浙.bmp",0);*/

	//载入汉字模板
	hanzi_sample[0] = cvLoadImage("template\\川.bmp", 0);
	hanzi_sample[1] = cvLoadImage("template\\鄂.bmp", 0);
	hanzi_sample[2] = cvLoadImage("template\\黑.bmp", 0);
	hanzi_sample[3] = cvLoadImage("template\\京.bmp", 0);
	hanzi_sample[4] = cvLoadImage("template\\辽.bmp", 0);
	hanzi_sample[5] = cvLoadImage("template\\琼.bmp", 0);
	hanzi_sample[6] = cvLoadImage("template\\湘.bmp", 0);
	hanzi_sample[7] = cvLoadImage("template\\粤.bmp", 0);
	hanzi_sample[8] = cvLoadImage("template\\浙.bmp", 0);

	//提取字符样本特征
	for (int i = 0; i < 34; i++)
	{
		GetFeature(char_sample[i], char_pattern[i]);
	}
	//提取汉字样本特征
	for (int i = 0; i < 9; i++)
	{
		GetFeature(hanzi_sample[i], hanzi_pattern[i]);
	}

	//提取字母样本特征
	for (int i = 0; i < 24; i++)
	{
		GetFeature(charO_sample[i], charO_pattern[i]);
	}

	//提取待识别字符特征
	for (int i = 0; i < 7; i++)
	{
		GetFeature(change_dst_image[i], TestSample[i]);
	}

	//进行汉字模板匹配	
	double min = 100000.0;
	for (int num = 0; num < 1; num++)
	{
		for (int i = 0; i < 9; i++)
		{
			double diff = 0.0;
			for (int j = 0; j < 25; j++)
			{
				diff += fabs(TestSample[num].feature[j] - hanzi_pattern[i].feature[j]);
			}
			for (int j = 25; j < 33; j++)
			{
				diff += fabs(TestSample[num].feature[j] - hanzi_pattern[i].feature[j]) * 9;
			}
			if (diff < min)
			{
				min = diff;
				TestSample[num].number = i;
			}
		}
	}
	//进行字母模板匹配	
	for (int num = 1; num < 2; num++)
	{
		double min_min = 1000000.0;
		for (int i = 0; i < 24; i++)
		{
			double diff_diff = 0.0;
			for (int j = 0; j < 25; j++)
			{
				diff_diff += fabs(TestSample[num].feature[j] - charO_pattern[i].feature[j]);
			}
			for (int j = 25; j < 33; j++)
			{
				diff_diff += fabs(TestSample[num].feature[j] - charO_pattern[i].feature[j]);
			}
			if (diff_diff < min_min)
			{
				min_min = diff_diff;
				TestSample[num].number = i;
			}
		}
	}
	//进行字符模板匹配
	for (int num = 2; num < 7; num++)
	{
		double min_min = 1000000.0;
		for (int i = 0; i < 34; i++)
		{
			double diff_diff = 0.0;
			for (int j = 0; j < 25; j++)
			{
				diff_diff += fabs(TestSample[num].feature[j] - char_pattern[i].feature[j]);
			}
			for (int j = 25; j < 33; j++)
			{
				diff_diff += fabs(TestSample[num].feature[j] - char_pattern[i].feature[j]);
			}
			if (diff_diff < min_min)
			{
				min_min = diff_diff;
				TestSample[num].number = i;
			}
		}
	}



	string result;//存放识别出的字符

	for (int i = 0; i < 1; i++)
	{
		switch (TestSample[i].number)
		{
		case 0:
			result += "川";
			break;
		case 1:
			result += "鄂";
			break;
		case 2:
			result += "黑";
			break;
		case 3:
			result += "京";
			break;
		case 4:
			result += "辽";
			break;
		case 5:
			result += "琼";
			break;
		case 6:
			result += "湘";
			break;
		case 7:
			result += "粤";
			break;
		case 8:
			result += "浙";
			break;
		default:

			break;
		}
	}

	for (int i = 1; i < 2; i++)
	{
		switch (TestSample[i].number)
		{
		case 0:
			result += "A";
			break;
		case 1:
			result += "B";
			break;
		case 2:
			result += "C";
			break;
		case 3:
			result += "D";
			break;
		case 4:
			result += "E";;
			break;
		case 5:
			result += "F";
			break;
		case 6:
			result += "G";
			break;
		case 7:
			result += "H";
			break;
		case 8:
			result += "J";
			break;
		case 9:
			result += "K";
			break;
		case 10:
			result += "L";
			break;
		case 11:
			result += "M";
			break;
		case 12:
			result += "N";
			break;
		case 13:
			result += "P";
			break;
		case 14:
			result += "Q";
			break;
		case 15:
			result += "R";
			break;
		case 16:
			result += "S";
			break;
		case 17:
			result += "T";
			break;
		case 18:
			result += "U";
			break;
		case 19:
			result += "U";
			break;
		case 20:
			result += "W";
			break;
		case 21:
			result += "X";
			break;
		case 22:
			result += "Y";
			break;
		case 23:
			result += "Z";
			break;
		default:

			break;
		}
	}

	for (int i = 2; i < 7; i++)
	{
		switch (TestSample[i].number)
		{
		case 0:
			result += "0";
			break;
		case 1:
			result += "1";
			break;
		case 2:
			result += "2";
			break;
		case 3:
			result += "3";
			break;
		case 4:
			result += "4";
			break;
		case 5:
			result += "5";
			break;
		case 6:
			result += "6";
			break;
		case 7:
			result += "7";
			break;
		case 8:
			result += "8";
			break;
		case 9:
			result += "9";
			break;
		case 10:
			result += "A";
			break;
		case 11:
			result += "B";
			break;
		case 12:
			result += "C";
			break;
		case 13:
			result += "D";
			break;
		case 14:
			result += "E";;
			break;
		case 15:
			result += "F";
			break;
		case 16:
			result += "G";
			break;
		case 17:
			result += "H";
			break;
		case 18:
			result += "J";
			break;
		case 19:
			result += "K";
			break;
		case 20:
			result += "L";
			break;
		case 21:
			result += "M";
			break;
		case 22:
			result += "N";
			break;
		case 23:
			result += "P";
			break;
		case 24:
			result += "Q";
			break;
		case 25:
			result += "R";
			break;
		case 26:
			result += "S";
			break;
		case 27:
			result += "T";
			break;
		case 28:
			result += "U";
			break;
		case 29:
			result += "U";
			break;
		case 30:
			result += "W";
			break;
		case 31:
			result += "X";
			break;
		case 32:
			result += "Y";
			break;
		case 33:
			result += "Z";
			break;
		default:

			break;
		}
	}
	//cout << result << endl;  //控制台输出车牌
	return result;
}




