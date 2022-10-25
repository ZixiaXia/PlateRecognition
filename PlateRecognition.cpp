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

	//�����޳�ʼ������
	Mat result;

	//��ǿ�Աȶ�
	result = plate.ImageStretchByHistogram(plate.image);

	//��˹ģ��
	GaussianBlur(result, result, Size(3, 3), 0, 0, BORDER_DEFAULT);

	////�ҶȻ�
	//cvtColor(result, result, CV_RGB2GRAY);

	//����
	result = plate.setright(result);

	//��ֵ��
	int thres = plate.Otsu(result);
	threshold(result, result, thres, 255, THRESH_BINARY);

	//ȥ��
	result = plate.delEdge(result);

	//ȥ��í��	
	result = plate.delRivet(result);

	// ����ͼƬ��СΪ300 * 50�������ַ��ָ�
	result = plate.reSize(result, 300, 50);

	//����Ԥ�������õ�ͼƬ
	imwrite("result.png", result);  

	//imshow("���", result);
	//imshow("ԭʼͼƬ", image);

	// �ȴ�6000 ms�󴰿��Զ��ر�
	//waitKey(6000000);

	result = reverse(result);
	this->saveImage(result);     //�ַ��ָ�,���������õ�ͼƬ
	
	
	//���ַ��ָ�����õ�ͼƬ���뵽ͼ��������
	change_dst_image[0] = cvLoadImage("0.png", 0);
	change_dst_image[1] = cvLoadImage("1.png", 0);
	change_dst_image[2] = cvLoadImage("2.png", 0);
	change_dst_image[3] = cvLoadImage("3.png", 0);
	change_dst_image[4] = cvLoadImage("4.png", 0);
	change_dst_image[5] = cvLoadImage("5.png", 0);
	change_dst_image[6] = cvLoadImage("6.png", 0);


	return plate.on_actionCharacterRecognition_C_triggered();
	
	
}

//��ǿ�Աȶ�
Mat PlateRecognition::ImageStretchByHistogram(Mat src)
{
	Mat dest;
	dest = Mat::zeros(src.size(), src.type());
	int width = src.cols;
	int height = src.rows;
	int channels = src.channels();

	int alphe = 1.8; //(alphe > 1)
	int beta = -30;// �����Աȶ�Խ��
	Mat m1;
	src.convertTo(m1, CV_32F); //��ԭʼͼƬ���ݣ�CV_8U���ͣ�ת����CV_32���ͣ�����߲����ľ���
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			if (channels == 3) { //����3ͨ��
				float b = m1.at<Vec3f>(row, col)[0];
				float g = m1.at<Vec3f>(row, col)[1];
				float r = m1.at<Vec3f>(row, col)[2];

				dest.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(alphe * b + beta);
				dest.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(alphe * g + beta);
				dest.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(alphe * r + beta);
			}
			else if (channels == 1) { //���ڵ�ͨ��
				int pix = src.at<uchar>(row, col);
				dest.at<uchar>(row, col) = saturate_cast<uchar>(alphe * pix + beta);
			}
		}
	}

	return dest;
}

//��ֵ��
int PlateRecognition::Otsu(Mat img)
{
	int height = img.rows; // Mat �ж����ݵķ�ʽ������Ipilmage�е�����
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


//ȥ��í��
Mat PlateRecognition::delRivet(Mat image)
{
	int  thisPoint, lastPoint, hop;
	int upperbound = image.rows / 2;
	int lowerbound = image.rows / 2;

	Mat result;
	//Ѱ���ֵ��±߽�
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

	//Ѱ���ֵ��ϱ߽�
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

//���ô�С
Mat PlateRecognition::reSize(Mat image, int w, int h)
{
	CvSize czSize;
	IplImage *pSrcImage = &IplImage(image);
	IplImage *pDstImage = NULL;

	//ͼ���С
	czSize.width = w;
	czSize.height = h;

	//����ͼ������
	pDstImage = cvCreateImage(czSize, pSrcImage->depth, pSrcImage->nChannels);
	cvResize(pSrcImage, pDstImage, CV_INTER_AREA);

	cv::Mat result = cv::cvarrToMat(pDstImage);
	return result;
}

//͸�ӱ任
Mat PlateRecognition::toushi(Mat image, Point2f pt1, Point2f pt2, Point2f pt3, Point2f pt4)
{
	Mat src = image;

	//ȡ�����ı����ĸ�����
	cv::Point2f srcPts[4];
	srcPts[0] = pt1;	//����
	srcPts[1] = pt2;	//����
	srcPts[2] = pt3;	//����
	srcPts[3] = pt4;	//����

	//����ԭͼ���ĸ���ĺ����������ֵСֵ,����λ���ص�,����һһ�Ƚ�
	int MinX = min(srcPts[0].x, srcPts[1].x);
	int MaxX = max(srcPts[2].x, srcPts[3].x);
	int MinY = min(srcPts[0].y, srcPts[2].y);
	int MaxY = max(srcPts[1].y, srcPts[3].y);

	//���������С����ֵ�趨Ŀ��ͼ���еľ����ĸ�����,ע���Ӧ��ϵ
	cv::Point2f dstPts[4];
	dstPts[0] = cv::Point2f(MinX, MinY);
	dstPts[1] = cv::Point2f(MinX, MaxY);
	dstPts[2] = cv::Point2f(MaxX, MinY);
	dstPts[3] = cv::Point2f(MaxX, MaxY);

	//����͸�ӱ任����
	cv::Mat perspectiveMat = getPerspectiveTransform(srcPts, dstPts);

	//��ԭͼ����͸�ӱ任����ɳ���У��
	cv::Mat dst;
	cv::warpPerspective(src, dst, perspectiveMat, src.size());

	return dst;
}

//�����������ұ߿�Ľ���
cv::Point PlateRecognition::jiaodian(cv::Point ��1, cv::Point ��2, cv::Point ��3, cv::Point ��4)
{
	int x, y;

	int X1 = ��1.x - ��2.x, Y1 = ��1.y - ��2.y, X2 = ��3.x - ��4.x, Y2 = ��3.y - ��4.y;

	if (X1*Y2 == X2 * Y1)return cv::Point((��2.x + ��3.x) / 2, (��2.y + ��3.y) / 2);

	int A = X1 * ��1.y - Y1 * ��1.x, B = X2 * ��3.y - Y2 * ��3.x;

	y = (A*Y2 - B * Y1) / (X1*Y2 - X2 * Y1);

	x = (B*X1 - A * X2) / (Y1*X2 - Y2 * X1);

	return cv::Point(x, y);
}

//����
Mat PlateRecognition::setright(Mat Image)
{
	//��Ե���
	Mat CannyImg;
	Canny(Image, CannyImg, 140, 250, 3);

	//�ҶȻ�
	Mat DstImg;
	cvtColor(Image, DstImg, CV_GRAY2BGR);

	//����任
	vector<Vec4i> Lines;
	HoughLinesP(CannyImg, Lines, 1, CV_PI / 360, 1, 30, 15);

	//��ʼ��
	int umaxlength = 0, lmaxlength = 0, lmaxwidth = 0, rmaxwidth = 0;
	size_t ulevel = -1, llevel = Lines.size(), lvertical = -1, rvertical = Lines.size();

	//ȷ�������ĸ��߽�
	for (size_t i = 0; i < Lines.size(); i++)
	{
		line(DstImg, Point(Lines[i][0], Lines[i][1]), Point(Lines[i][2], Lines[i][3]), Scalar(0, 0, 255), 2, 8);		int distance = (Lines[i][0] - Lines[i][2])*(Lines[i][0] - Lines[i][2]) + (Lines[i][1] - Lines[i][3])*(Lines[i][1] - Lines[i][3]);
		if (Lines[i][2] != Lines[i][0])
		{
			int angle = (Lines[i][3] - Lines[i][1]) / (Lines[i][2] - Lines[i][0]);
			if (angle > -1 && angle < 1)//ˮƽ�߿�
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
			else//���ƴ�ֱ�߿�
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
		else//��ֱ�߿�
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

	//���Ͻ�
	Point2f pt1 = jiaodian(Point(ulevel0, ulevel1), Point(ulevel2, ulevel3),
		Point(lvertical0, lvertical1), Point(lvertical2, lvertical3));
	//���½�
	Point2f pt2 = jiaodian(Point(llevel0, llevel1), Point(llevel2, llevel3),
		Point(lvertical0, lvertical1), Point(lvertical2, lvertical3));
	//���Ͻ�
	Point2f pt3 = jiaodian(Point(ulevel0, ulevel1), Point(ulevel2, ulevel3),
		Point(rvertical0, rvertical1), Point(rvertical2, rvertical3));
	//���½�
	Point2f pt4 = jiaodian(Point(llevel0, llevel1), Point(llevel2, llevel3),
		Point(rvertical0, rvertical1), Point(rvertical2, rvertical3));
	cv::circle(DstImg, pt1, 4, cv::Scalar(0, 255, 0));
	cv::circle(DstImg, pt2, 4, cv::Scalar(0, 255, 0));
	cv::circle(DstImg, pt3, 4, cv::Scalar(0, 255, 0));
	cv::circle(DstImg, pt4, 4, cv::Scalar(0, 255, 0));

	imshow("����", DstImg);

	//͸�ӱ任
	Mat result = toushi(Image, pt1, pt2, pt3, pt4);

	return result;
}

//ȥ��
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



//��ɫ��ת
Mat PlateRecognition::reverse(Mat src)
{
	Mat dst = src < 100;
	return dst;
}


//��ֱͶӰ
vector<Mat> PlateRecognition::verticalProjectionMat(Mat srcImg)
{
	Mat binImg;
	blur(srcImg, binImg, Size(3, 3));
	threshold(binImg, binImg, 0, 255, CV_THRESH_OTSU);
	int perPixelValue;//ÿ�����ص�ֵ
	int width = srcImg.cols;
	int height = srcImg.rows;
	int* projectValArry = new int[width];//�������ڴ���ÿ�а�ɫ���ظ���������
	memset(projectValArry, 0, width * 4);//��ʼ������
	for (int col = 0; col < width; col++)
	{
		for (int row = 0; row < height; row++)
		{
			perPixelValue = binImg.at<uchar>(row, col);
			if (perPixelValue == 0)//����ǰ׵׺���
			{
				projectValArry[col]++;
			}
		}
	}
	Mat verticalProjectionMat(height, width, CV_8UC1);//��ֱͶӰ�Ļ���
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			perPixelValue = 255;  //��������Ϊ��ɫ
			verticalProjectionMat.at<uchar>(i, j) = perPixelValue;
		}
	}
	for (int i = 0; i < width; i++)//��ֱͶӰֱ��ͼ
	{
		for (int j = 0; j < projectValArry[i]; j++)
		{
			perPixelValue = 0;  //ֱ��ͼ����Ϊ��ɫ  
			verticalProjectionMat.at<uchar>(height - 1 - j, i) = perPixelValue;
		}
	}
	//imshow("��ֱͶӰ", verticalProjectionMat);
	//cvWaitKey(0);
	vector<Mat> roiList;//���ڴ���ָ������ÿ���ַ�
	int startIndex = 0;//��¼�����ַ���������
	int endIndex = 0;//��¼����հ����������
	bool inBlock = false;//�Ƿ���������ַ�����
	for (int i = 0; i < srcImg.cols; i++)//cols=width
	{
		cout << "projectValArry[" << i << "]= " << projectValArry[i] << endl;
		if (!inBlock && projectValArry[i] != 0)//�����ַ���
		{
			if (pointRang[0] < i && i < pointRang[1]) {//����Բ��
				inBlock = false;
			}
			else {
				inBlock = true;
				startIndex = i;
			}
		}
		else if (projectValArry[i] == 0 && inBlock)//����հ���
		{
			int sign = 0;
			if (expandRang[0] < i && i < expandRang[1]) {//��ֹ���ͱ߿�
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




//�����ַ��ָ����õ�7��ͼƬ
void PlateRecognition::saveImage(Mat srcImg)
{
	vector<Mat> b = verticalProjectionMat(srcImg);//�Ƚ��д�ֱͶӰ	

    //cout << "b_size= " << b.size() << endl;
	char szName[30] = { 0 };
	//Mat resize_image = Mat::zeros(Size(g_width, g_height), CV_8UC3); //����ͼƬ��СΪ20 * 40
	for (int i = 0, j = 0; i < b.size(); i++)
	{

		//imshow("b", b[i]);
		//�����зֵĽ��
		Mat Img = b[i];
		Img = reverse(Img);
		//Img = cvarrToMat(&img);
		int w = Img.cols;
		int h = Img.rows;
		if (w > wFilter && h > hFilter) {
			sprintf_s(szName, "%d.png", j);
			//imshow(szName, b[i]);
			//resize(Img, resize_image,resize_image.size()); //����ͼƬ��С�ٱ���
			Img = reSize(Img, 20, 40);
			imwrite(szName, Img);
			j++;
		}
	}
}




//������ȡ
void PlateRecognition::GetFeature(IplImage * src, pattern & pat)
{
	CvScalar s;
	int i, j;
	for (i = 0; i < 33; i++)
		pat.feature[i] = 0.0;
	//ͼ���С��20*40��С�ģ��ֳ�25��

	//********��һ��***********	
	//��һ��
	for (j = 0; j < 8; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[0] += 1.0;
		}
	}

	//�ڶ���
	for (j = 0; j < 8; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[1] += 1.0;
		}
	}
	//������
	for (j = 0; j < 8; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[2] += 1.0;
		}
	}
	//���Ŀ�
	for (j = 0; j < 8; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[3] += 1.0;
		}
	}
	//�����
	for (j = 0; j < 8; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[4] += 1.0;
		}
	}
	//********�ڶ���***********	
	//������
	for (j = 8; j < 16; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[5] += 1.0;
		}
	}
	//���߿�
	for (j = 8; j < 16; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[6] += 1.0;
		}
	}
	//�ڰ˿�
	for (j = 8; j < 16; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[7] += 1.0;
		}
	}
	//�ھſ�
	for (j = 8; j < 16; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[8] += 1.0;
		}
	}
	//��ʮ��
	for (j = 8; j < 16; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[9] += 1.0;
		}
	}
	//********������***********
	//��ʮһ��
	for (j = 16; j < 24; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[10] += 1.0;
		}
	}
	//��ʮ����
	for (j = 16; j < 24; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[11] += 1.0;
		}
	}
	//��ʮ����
	for (j = 16; j < 24; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[12] += 1.0;
		}
	}
	//��ʮ�Ŀ�
	for (j = 16; j < 24; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[13] += 1.0;
		}
	}
	//��ʮ���
	for (j = 16; j < 24; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[14] += 1.0;
		}
	}
	//********������***********
	//��ʮ����
	for (j = 24; j < 32; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[15] += 1.0;
		}
	}
	//��ʮ�߿�
	for (j = 24; j < 32; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[16] += 1.0;
		}
	}
	//��ʮ�˿�
	for (j = 24; j < 32; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[17] += 1.0;
		}
	}
	//��ʮ�ſ�
	for (j = 24; j < 32; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[18] += 1.0;
		}
	}
	//�ڶ�ʮ��
	for (j = 24; j < 32; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[19] += 1.0;
		}
	}
	//********������***********
	//�ڶ�ʮһ��
	for (j = 32; j < 40; j++)
	{
		for (i = 0; i < 4; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[20] += 1.0;
		}
	}
	//�ڶ�ʮ����
	for (j = 32; j < 40; j++)
	{
		for (i = 4; i < 8; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[21] += 1.0;
		}
	}
	//�ڶ�ʮ����
	for (j = 32; j < 40; j++)
	{
		for (i = 8; i < 12; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[22] += 1.0;
		}
	}
	//�ڶ�ʮ�Ŀ�
	for (j = 32; j < 40; j++)
	{
		for (i = 12; i < 16; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[23] += 1.0;
		}
	}
	//�ڶ�ʮ���
	for (j = 32; j < 40; j++)
	{
		for (i = 16; i < 20; i++)
		{
			s = cvGet2D(src, j, i);
			if (s.val[0] == 255)
				pat.feature[24] += 1.0;
		}
	}

	//����ͳ�Ʒ��򽻵�����
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

//�ַ�ʶ��
string PlateRecognition::on_actionCharacterRecognition_C_triggered()
{
	IplImage * char_sample[34];//�ַ�����ͼ������
	IplImage * hanzi_sample[9];//��������ͼ������
	pattern char_pattern[34];//�����ַ���Ʒ��ṹ����
	pattern hanzi_pattern[9];//���庺����Ʒ��ṹ����
	pattern TestSample[7];//�����ʶ���ַ��ṹ����

	IplImage * charO_sample[24];//������ĸ�����ڵؼ���ʶ��
	pattern charO_pattern[24];


	//�����ַ�ģ��
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

	//���뺺��ģ��
	/*hanzi_sample[0]=cvLoadImage("template\\��.bmp",0);
	hanzi_sample[1]=cvLoadImage("template\\��.bmp",0);
	hanzi_sample[2]=cvLoadImage("template\\��.bmp",0);
	hanzi_sample[3]=cvLoadImage("template\\��.bmp",0);
	hanzi_sample[4]=cvLoadImage("template\\��.bmp",0);
	hanzi_sample[5]=cvLoadImage("template\\��.bmp",0);
	hanzi_sample[6]=cvLoadImage("template\\��.bmp",0);*/

	//���뺺��ģ��
	hanzi_sample[0] = cvLoadImage("template\\��.bmp", 0);
	hanzi_sample[1] = cvLoadImage("template\\��.bmp", 0);
	hanzi_sample[2] = cvLoadImage("template\\��.bmp", 0);
	hanzi_sample[3] = cvLoadImage("template\\��.bmp", 0);
	hanzi_sample[4] = cvLoadImage("template\\��.bmp", 0);
	hanzi_sample[5] = cvLoadImage("template\\��.bmp", 0);
	hanzi_sample[6] = cvLoadImage("template\\��.bmp", 0);
	hanzi_sample[7] = cvLoadImage("template\\��.bmp", 0);
	hanzi_sample[8] = cvLoadImage("template\\��.bmp", 0);

	//��ȡ�ַ���������
	for (int i = 0; i < 34; i++)
	{
		GetFeature(char_sample[i], char_pattern[i]);
	}
	//��ȡ������������
	for (int i = 0; i < 9; i++)
	{
		GetFeature(hanzi_sample[i], hanzi_pattern[i]);
	}

	//��ȡ��ĸ��������
	for (int i = 0; i < 24; i++)
	{
		GetFeature(charO_sample[i], charO_pattern[i]);
	}

	//��ȡ��ʶ���ַ�����
	for (int i = 0; i < 7; i++)
	{
		GetFeature(change_dst_image[i], TestSample[i]);
	}

	//���к���ģ��ƥ��	
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
	//������ĸģ��ƥ��	
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
	//�����ַ�ģ��ƥ��
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



	string result;//���ʶ������ַ�

	for (int i = 0; i < 1; i++)
	{
		switch (TestSample[i].number)
		{
		case 0:
			result += "��";
			break;
		case 1:
			result += "��";
			break;
		case 2:
			result += "��";
			break;
		case 3:
			result += "��";
			break;
		case 4:
			result += "��";
			break;
		case 5:
			result += "��";
			break;
		case 6:
			result += "��";
			break;
		case 7:
			result += "��";
			break;
		case 8:
			result += "��";
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
	//cout << result << endl;  //����̨�������
	return result;
}




