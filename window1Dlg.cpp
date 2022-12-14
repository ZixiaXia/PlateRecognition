
// window1Dlg.cpp: 实现文件
//

#include "stdafx.h"
#include "window1.h"
#include "window1Dlg.h"
#include "afxdialogex.h"
#include "PlateRecognition.h"
#include <atlstr.h>
#include <iostream>  
#include <cstring>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2\imgproc\types_c.h> 
#include <opencv2/imgproc/imgproc.hpp>
#include"highgui.h"


using namespace cv;
using namespace std;

void CImageToMat(CImage& cimage, Mat& mat);
Mat reverse(Mat src);
string replace(string path);

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

string filename = "sample1.png";
//int expandCount = 7;
//int wFilter = 6, hFilter = 15;

// 用于应用程序“关于”菜单项的 CAboutDlg 对话框
CString filePath;

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// Cwindow1Dlg 对话框



Cwindow1Dlg::Cwindow1Dlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_WINDOW1_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void Cwindow1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(Cwindow1Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &Cwindow1Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &Cwindow1Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDOK, &Cwindow1Dlg::OnBnClickedOk)
END_MESSAGE_MAP()


// Cwindow1Dlg 消息处理程序

BOOL Cwindow1Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void Cwindow1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void Cwindow1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR Cwindow1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void Cwindow1Dlg::OnBnClickedButton1()
{
	//打开文件 定义四种格式的文件bmp gif jpg tiff
	CString filter;
	filter = "所有文件(*.bmp,*.jpg,*.gif,*tiff)|*.bmp;*.jpg;*.gif;*.tiff| BMP(*.bmp)|*.bmp| JPG(*.jpg)|*.jpg| GIF(*.gif)|*.gif| TIFF(*.tiff)|*.tiff||";
	CFileDialog dlg(TRUE, NULL, NULL, OFN_HIDEREADONLY, filter, NULL);

	//按下确定按钮 dlg.DoModal() 函数显示对话框
	if (dlg.DoModal() == IDOK)
	{
		//打开对话框获取图像信息
		CString BmpName = dlg.GetPathName();     //获取文件路径名   如D:\pic\abc.bmp
		CString EntName = dlg.GetFileExt();      //获取文件扩展名
		EntName.MakeLower();                     //将文件扩展名转换为一个小写字符

		if (EntName.Compare(_T("bmp")) == 0)
		{
			//定义变量存储图片信息
			BITMAPINFO *pBmpInfo;       //记录图像细节
			BYTE *pBmpData;             //图像数据
			BITMAPFILEHEADER bmpHeader; //文件头
			BITMAPINFOHEADER bmpInfo;   //信息头
			CFile bmpFile;              //记录打开文件

										//以只读的方式打开文件 读取bmp图片各部分 bmp文件头 信息 数据
			if (!bmpFile.Open(BmpName, CFile::modeRead | CFile::typeBinary))
				return;
			if (bmpFile.Read(&bmpHeader, sizeof(BITMAPFILEHEADER)) != sizeof(BITMAPFILEHEADER))
				return;
			if (bmpFile.Read(&bmpInfo, sizeof(BITMAPINFOHEADER)) != sizeof(BITMAPINFOHEADER))
				return;
			pBmpInfo = (BITMAPINFO *)new char[sizeof(BITMAPINFOHEADER)];
			//为图像数据申请空间
			memcpy(pBmpInfo, &bmpInfo, sizeof(BITMAPINFOHEADER));
			DWORD dataBytes = bmpHeader.bfSize - bmpHeader.bfOffBits;
			pBmpData = (BYTE*)new char[dataBytes];
			bmpFile.Read(pBmpData, dataBytes);
			filePath = bmpFile.GetFilePath();
			SetDlgItemText(IDC_EDIT3, filePath);
			bmpFile.Close();

			//显示图像
			CWnd *pWnd = GetDlgItem(IDC_STATIC_PIC); //获得pictrue控件窗口的句柄
			CRect rect;
			pWnd->GetClientRect(&rect); //获得pictrue控件所在的矩形区域
			CDC *pDC = pWnd->GetDC(); //获得pictrue控件的DC
			pDC->SetStretchBltMode(COLORONCOLOR);
			StretchDIBits(pDC->GetSafeHdc(), 0, 0, rect.Width(), rect.Height(), 0, 0,
				bmpInfo.biWidth, bmpInfo.biHeight, pBmpData, pBmpInfo, DIB_RGB_COLORS, SRCCOPY);
		}
	}
}


void Cwindow1Dlg::OnBnClickedButton2()
{
	std::string strPath = CT2A(filePath.GetBuffer());
	strPath = replace(strPath);
	CString cstr1(strPath.c_str());//
	MessageBox(cstr1);//

	PlateRecognition image = PlateRecognition(strPath);
	string result = image.getPlate(image);
	int temp = result.length(); //
	CString STemp;//
	STemp.Format(_T("%d"), temp);//
	MessageBox(STemp);//
	CString cstr2(result.c_str());//
	CString str;//
	MessageBox(str+"!");//
	SetDlgItemText(IDC_EDIT2, cstr2);
}


void CImageToMat(CImage& cimage, Mat& mat)
{
	if (true == cimage.IsNull())
	{
		return;
	}


	int nChannels = cimage.GetBPP() / 8;
	if ((1 != nChannels) && (3 != nChannels))
	{
		return;
	}
	int nWidth = cimage.GetWidth();
	int nHeight = cimage.GetHeight();


	//重建mat
	if (1 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC1);
	}
	else if (3 == nChannels)
	{
		mat.create(nHeight, nWidth, CV_8UC3);
	}


	//拷贝数据


	uchar* pucRow;									//指向数据区的行指针
	uchar* pucImage = (uchar*)cimage.GetBits();		//指向数据区的指针
	int nStep = cimage.GetPitch();					//每行的字节数,注意这个返回值有正有负


	for (int nRow = 0; nRow < nHeight; nRow++)
	{
		pucRow = (mat.ptr<uchar>(nRow));
		for (int nCol = 0; nCol < nWidth; nCol++)
		{
			if (1 == nChannels)
			{
				pucRow[nCol] = *(pucImage + nRow * nStep + nCol);
			}
			else if (3 == nChannels)
			{
				for (int nCha = 0; nCha < 3; nCha++)
				{
					pucRow[nCol * 3 + nCha] = *(pucImage + nRow * nStep + nCol * 3 + nCha);
				}
			}
		}
	}
}


void Cwindow1Dlg::OnBnClickedOk()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialogEx::OnOK();
}

Mat reverse(Mat src)
{
	Mat dst = src<100;
	return dst;
}

string replace(string path) {
	for (int i = 0; i < path.length(); i++) {
		if (path.at(i) == '\\') {
			path.at(i) = '/';
		}
	}
	return path;
}