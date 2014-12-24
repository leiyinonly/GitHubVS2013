/*************************************************************************
* @文件名：
*	assessment.cpp
* @说明:
*   该文件包括模糊图像鉴别、图像质量评估相关的实现函数
*************************************************************************/
#include "basicOperation.h"
#include "assessment.h"


/*************************************************************************
* @函数名称：
*	blurIdentify()
* @输入:
*   IplImage* input            - 输入灰度图像
* @返回值:
*   uchar                      - 返回图像标志，清晰返回1，模糊返回0
* @说明:
*   该函数通过计算梯度图像，并进行直方图分析，计算相关鉴别指标GMG和NGN，
*	实现模糊鉴别功能
*************************************************************************/
uchar blurIdentify(const IplImage* input)
{
	int rows=input->height;
	int cols=input->width;
	int i=0,j=0;
	uchar flag=0;
	CvSize size1,size2;
	size1.width=cols;
	size1.height=rows;
	size2.width=2*cols;
	size2.height=rows;


	IplImage* tempr=cvCreateImage(size1,IPL_DEPTH_8U,1);
	IplImage* tempc=cvCreateImage(size1,IPL_DEPTH_8U,1);
	cvZero(tempr);
	cvZero(tempc);

	/*计算水平梯度图像*/
	for (i=0;i<rows;i++)
	{
		uchar* input_data=(uchar*)(input->imageData+i*input->widthStep);
		uchar* tempr_data=(uchar*)(tempr->imageData+i*tempr->widthStep);
		for (j=0;j<cols-1;j++)
		{
			tempr_data[j]=abs(input_data[j+1]-input_data[j]);
		}
	}
	/*计算垂直梯度图像*/
	for (i = 0; i<rows - 1; i++)
	{
		uchar* input_data1 = (uchar*)(input->imageData + i*input->widthStep);
		uchar* input_data2 = (uchar*)(input->imageData + (i + 1)*input->widthStep);
		uchar* tempc_data = (uchar*)(tempc->imageData + i*tempc->widthStep);
		for (j = 0; j<cols; j++)
		{

			tempc_data[j]=abs(input_data2[j]-input_data1[j]);
		}
	}
	/*将两个梯度图像合并为一个，以便统计计算*/
	IplImage* gradient=cvCreateImage(size2,IPL_DEPTH_8U,1);
	cvZero(gradient);
	cvSetImageROI(gradient,cvRect(0,0,cols,rows));
	cvCopy(tempr,gradient);							//将水平梯度图存入
	cvResetImageROI(gradient);

	cvSetImageROI(gradient,cvRect(cols,0,cols,rows));
	cvCopy(tempc,gradient);							//将垂直梯度图存入
	cvResetImageROI(gradient);

	int nHistSize=256;
	float fRange[]={0,255};   //灰度级范围
	float* pfRanges[]={fRange};
	//CvHistogram* hist=cvCreateHist(1,&nHistSize,CV_HIST_ARRAY,pfRanges);	//CV_HIST_ARRAY多维密集数组
	//cvCalcHist(&gradient,hist);
	CvHistogram* hist1=cvCreateHist(1,&nHistSize,CV_HIST_ARRAY,pfRanges);	//CV_HIST_ARRAY多维密集数组
	CvHistogram* hist2=cvCreateHist(1,&nHistSize,CV_HIST_ARRAY,pfRanges);	//CV_HIST_ARRAY多维密集数组

	cvCalcHist(&tempr,hist1);
	cvCalcHist(&tempc, hist2);

	//int NGN=cvCountNonZero(hist->bins);
	int NX=cvCountNonZero(hist1->bins);
	int NY = cvCountNonZero(hist2->bins);

	double GMG=calGMG(input);
	double s = (NX + NY) / (2 * 256.0);
	double BIM=s*GMG;
	
	if(BIM>700)
	{
		flag=1;
	}
	//测试代码，显示梯度图像
	/*cvNamedWindow("gx",1);
	cvShowImage("gx",tempr);
	cvNamedWindow("gy",1);
	cvShowImage("gy",tempc);
	cvNamedWindow("g",1);
	cvShowImage("g",gradient);*/
	printf("GMG=%f,NX=%d,NY=%d,s=%f,BIM=%f\n", GMG, NX, NY,s, BIM);

	//for(i=0;i<256;i++)
	//{
	//	printf("%.f ",((CvMatND*)hist->bins)->data.fl[i]);
	//}

	cvReleaseImage(&tempr);
	cvReleaseImage(&tempc);
	cvReleaseImage(&gradient);
	cvReleaseHist(&hist1);
	cvReleaseHist(&hist2);


	return 1;
}


/*************************************************************************
* @函数名称：
*	calGMG()
* @输入:
*   IplImage* input            - 输入灰度图像
* @返回值:
*   double                     - 灰度平均梯度值GMG
* @说明:
*   该函数计算图像灰度平均梯度值，其值越大表示图像越清晰
*************************************************************************/
double  calGMG(const IplImage* input)
{
	int rows=input->height;
	int cols=input->width;
	int i=0,j=0;
	int num=(rows-1)*(cols-1);
	double sum=0;
	double GMG=0;

	for(i=0;i<rows-1;i++)
	{
		uchar* input_data1=(uchar*)(input->imageData+i*input->widthStep);
		uchar* input_data2 = (uchar*)(input->imageData + (i + 1)*input->widthStep);

		for(j=0;j<cols-1;j++)
		{
			sum+=sqrt(((input_data1[j+1]-input_data1[j])*(input_data1[j+1]-input_data1[j])
				+((input_data2[j]-input_data1[j])*(input_data2[j]-input_data1[j])))/2);
		}
	}
	GMG=sum/num;
	
	return GMG;
}

/*************************************************************************
* @函数名称：
*	calLuminanceSim()
* @输入:
*   IplImage* input1            - 输入图像1
*   IplImage* input2            - 输入图像2
* @返回值: 
*   double                      - 返回图像亮度相似性
* @说明:
*   计算图像的亮度相似度，符合亮度掩盖模型
*	计算公式为：l(x,y)=(2*u_x*u_y+c1)/(u_x*u_x+u_y*u_y+c1)
*************************************************************************/
double calLuminanceSim(const IplImage* input1, const IplImage* input2)
{
	double lum=0,c1=0;
	double k1=0.01;
	CvScalar mean1,mean2;

	mean1=cvAvg(input1);
	mean2=cvAvg(input2);
	c1=(k1*255)*(k1*255);
	lum=(2*mean1.val[0]*mean2.val[0]+c1)/(mean1.val[0]*mean1.val[0]+mean2.val[0]*mean2.val[0]+c1);
	return lum;
}

/*************************************************************************
* @函数名称：
*	calContrastSim()
* @输入:
*   IplImage* input1            - 输入图像1
*   IplImage* input2            - 输入图像2
* @返回值: 
*   double                      - 返回图像对比度相似性
* @说明:
*   计算图像对比度相似性，考虑了图像的对比度掩盖效应
*	计算公式为：l(x,y)=(2*std_x*std_y+c2)/(std_x*std_x+std_y*std_y+c2)
*************************************************************************/
double calContrastSim(const IplImage* input1, const IplImage* input2)
{
	double con=0, c2=0;
	double k2=0.03;
	CvScalar stdev1, stdev2;

	cvAvgSdv(input1,NULL,&stdev1);
	cvAvgSdv(input2,NULL,&stdev2);
	c2=(k2*255 )*(k2*255);
	con=(2*stdev1.val[0]*stdev2.val[0]+c2)/(stdev1.val[0]*stdev1.val[0]+stdev2.val[0]*stdev2.val[0]+c2);
	return con;
}

/*************************************************************************
* @函数名称：
*	calStructSim()
* @输入:
*   const IplImage* input1            - 输入图像1
*   const IplImage* input2            - 输入图像2
* @返回值: 
*   double							  - 结构度相似性
* @说明:
*   计算图像结构度相似性，用图像像素间的相关来刻画结构关系
*	计算公式为：l(x,y)=(cov_xy+c2)/(std_x+std_y+c2)
*************************************************************************/
double calStructSim(const IplImage* input1, const IplImage* input2)
{
	double stc=0, c3=0;
	double k2=0.03;
	CvScalar stdev1, stdev2;
	CvScalar mean1,mean2;

	cvAvgSdv(input1,&mean1,&stdev1);
	cvAvgSdv(input2,&mean2,&stdev2);

	double cov = 0;
	double sum = 0;
	int i, j;
	int rows=input1->height;
	int cols=input1->width;

	for (i = 0; i < rows; i++)
	{
		uchar* input1_data=(uchar*)(input1->imageData+i*input1->widthStep);
		uchar* input2_data=(uchar*)(input2->imageData+i*input2->widthStep);
		for (j = 0; j < cols; j++)
		{
			sum += ((input1_data[j] - mean1.val[0]) *(input2_data[j] - mean2.val[0]));
		}
	}
	cov = sum / (rows*cols);
	c3 = (k2 * 255) * (k2 * 255) / 2;
	stc=(cov+c3)/(stdev1.val[0]*stdev2.val[0]+c3);
	return stc;
}



/*************************************************************************
* @函数名称：
*	estSNR()
* @输入:
*   IplImage* input1            - 输入图像
* @返回值:
*   double                      - 估计的信噪比
* @说明:
*	通过计算局部方差估计信噪比
*	局部方差的最大值为信号方差，最小值为噪声方差，再用经验公式修正
*************************************************************************/
//double estSNR(IplImage* input)
//{
//	double snr=0, max, min, k;
//	CvSize size = cvGetSize(input);
//
//	Mat f = Mat_<float>(input), av, v;
//	Mat g = f;// Mat(f, Rect(2, 2, size.width - 4, size.height - 4));
//	Mat mask = Mat::ones(5, 5, CV_32F);
//	mask = mask / 25;
//
//	cvFilter2D(g, av, -1, mask);
//	pow((g - av), 2, v);
//	filter2D(v, v, -1, mask);
//	minMaxLoc(v, &min, &max);
//	snr = 10 * log(max / min);
//	/*snr = 1.04*snr - 7;//k为维纳滤波参数
//	k=pow(10, (-snr / 10));
//	k = 5 * k;
//	return k;*/
//
//	return snr;
//}

/*************************************************************************
* @函数名称：
*	calMISSIM()
* @输入:
*   const IplImage* image1           - 输入图像1
*   const IplImage* image2           - 输入图像2
*	int n							 - 每个方块的大小
* @返回值:
*   double MISSIM                    - 返回图像的平均改进结构相似度
* @说明:
*   计算图像的平均改进结构相似度
**************************************************************************/
double calMISSIM(const IplImage* image1, const IplImage* image2, int n)
{
	double MISSIM = 0;
	int i, j, k;
	int row1 = image1->height;
	int col1 = image1->width;
	int row2 = image2->height;
	int col2 = image2->width;
	if (row1 != row2 || col1 != col2)
	{
		printf("Size can't match in calMISSIM()!!");
	}

	int nr = cvFloor(row1 / n);
	int nc = cvFloor(col1 / n);
	int N = nr*nc;
	double ISSIM=0;
	double sum = 0;

	CvMat tmp1;
	CvMat tmp2;
	IplImage* temp1 = cvCreateImage(cvSize(n, n), image1->depth, image1->nChannels);
	IplImage* temp2 = cvCreateImage(cvSize(n, n), image1->depth, image1->nChannels);

	for (i = 0, k = 0; i < nr; i++)
	{
		for (j = 0; j < nc; j++, k++)
		{
			cvGetSubRect(image1, &tmp1, cvRect(j*n, i*n, n, n));
			cvGetSubRect(image2, &tmp2, cvRect(j*n, i*n, n, n));

			cvScale(&tmp1, temp1, 1, 0);
			cvScale(&tmp2, temp2, 1, 0);

			ISSIM = calISSIM(temp1, temp2);
			sum += ISSIM;
		}
	}

	MISSIM = sum / N;
	cvReleaseImage(&temp1);
	cvReleaseImage(&temp2);

	return MISSIM;
}

/*************************************************************************
* @函数名称：
*	calISSIM()
* @输入:
*   const IplImage* image1           - 输入图像1
*   const IplImage* image2           - 输入图像2
* @返回值:
*   double ISSIM                     - 返回图像的改进的结构相似度
* @说明:
*   计算图像的改进的结构相似度
**************************************************************************/
double calISSIM(const IplImage* image1, const IplImage* image2)
{
	double ISSIM = 0;
	double l = 0, c = 0, g = 0, s = 0;

	l = calLuminanceSim(image1, image2);
	c = calContrastSim(image1, image2);
	g = calGradSim(image1, image2);
	s = calStructSim(image1, image2);
	
	//printf("l=%f\n", l);
	//printf("c=%f\n", c);
	//printf("g=%f\n", g);
	//printf("s=%f\n", s);

	ISSIM = pow(l, 1)*pow(c, 1)*pow(g,1)*pow(s, 1);
	return ISSIM;
}

/*************************************************************************
* @函数名称：
*	calGradSim()
* @输入:
*   const IplImage* image1           - 输入图像1
*   const IplImage* image2           - 输入图像2
* @返回值:
*   double g			             - 梯度相似性
* @说明:
*   计算图像梯度相似性
*************************************************************************/
double calGradSim(const IplImage* image1, const IplImage* image2)
{
	double c4 = 0;   
	double g = 0;

	IplImage* g1;
	IplImage* g2;
	IplImage* tmp;

	g1=gradientImage(image1);
	g2=gradientImage(image2);
	tmp = cvCloneImage(g1);

	cvMul(g1, g2, tmp);
	cvMul(g1, g1, g1);
	cvMul(g2, g2, g2);

	CvScalar s1 = cvSum(tmp);
	CvScalar s2 = cvSum(g1);
	CvScalar s3 = cvSum(g2);

	c4 = (0.03 * 255) * (0.03 * 255);
	g = (2 * s1.val[0] + c4) / (s2.val[0] +s3.val[0] + c4);

	cvReleaseImage(&g1);
	cvReleaseImage(&g2);
	cvReleaseImage(&tmp);

	return g;
}

/*************************************************************************
* @函数名称：
*	gradientImage()
* @输入:
*   const IplImage* input           - 输入8U图像
* @输出:
*   IplImage* gradient			    - 输出8U梯度图像
* @说明:
*   计算图像的梯度幅值矩阵
**************************************************************************/
IplImage* gradientImage(const IplImage* input)
{
	int i, j;
	int row = input->height;
	int col = input->width;
	IplImage* gradient=cvCreateImage(cvSize(col, row), input->depth,input->nChannels);
	IplImage* gx = cvCreateImage(cvSize(col, row), input->depth, input->nChannels);
	IplImage* gy = cvCreateImage(cvSize(col, row), input->depth, input->nChannels);
	cvZero(gradient);
	cvZero(gx);
	cvZero(gy);

	/*计算水平梯度图像*/
	for (i = 0; i < row; i++)
	{
		uchar* current = (uchar*)(input->imageData+i*input->widthStep);
		uchar* gxcurrent = (uchar*)(gx->imageData + i*gx->widthStep);

		for (j = 0; j < col - 1; j++)
		{
			gxcurrent[j] = abs(current[j + 1] - current[j]);
		}
	}
	/*计算垂直梯度图像*/
	for (i = 0; i < row - 1; i++)
	{
		uchar* current = (uchar*)(input->imageData + i*input->widthStep);
		uchar* next = (uchar*)(input->imageData + (i+1)*input->widthStep);
		uchar* gycurrent = (uchar*)(gx->imageData + i*gx->widthStep);

		for (j = 0; j < col; j++)
		{
			gycurrent[j] = abs(next[j] - current[j]);
		}
	}
	
	cvAdd(gx,gy,gradient);
	cvReleaseImage(&gx);
	cvReleaseImage(&gy);

	return gradient;
}


/*************************************************************************
* @函数名称：
*	calINRSS()
* @输入:
*   const IplImage* input           - 输入8U图像
* @输出:
*   double INRSS				    - 输出INRSS值
* @说明:
*   计算图像无参考结构相似度
**************************************************************************/
double calINRSS(const IplImage* input)
{
	double INRSS = 0;
	double missim = 0;

	IplImage* lp_image = cvCloneImage(input);

	cvSmooth(input, lp_image, CV_GAUSSIAN, 7, 7, 6);
	missim = calMISSIM(input, lp_image, 8);

	INRSS = 1 - missim;

	cvReleaseImage(&lp_image);

	return INRSS;
}

/*************************************************************************
* @函数名称：
*	calRingMetric()
* @输入:
*   const IplImage* input           - 输入8U图像
*   int d							- 使用的模糊核边长的一半
* @输出:
*   double INRSS				    - 输出INRSS值
* @说明:
*   计算图像的平行边缘值用以评价振铃效应
**************************************************************************/
double calRingMetric(const IplImage* input, int d)
{
	int i = 0, j = 0, p = 0, q = 0;
	int id = 0, jd = 0, is = 0, js = 0;
	double rm = 0;
	double cos45 = cos(45 / 180 * PI);
	double sin45 = sin(45 / 180 * PI);
	double cos135 = cos(135 / 180 * PI);
	double sin135 = sin(135 / 180 * PI);

	IplImage* edge = cvCloneImage(input);
	CvMat* rm1 = cvCreateMat(input->height, input->width, CV_8UC1);
	CvMat* rm2 = cvCreateMat(input->height, input->width, CV_8UC1);
	CvMat* rm3 = cvCreateMat(input->height, input->width, CV_8UC1);
	CvMat* rm4 = cvCreateMat(input->height, input->width, CV_8UC1);

	cvZero(rm1);
	cvZero(rm2);
	cvZero(rm3);
	cvZero(rm4);

	//边缘检测
	cvCanny(input, edge, 0.04*255, 0.1*255, 3);
	cvScale(edge, edge, 1.0 / 255, 0);
	//cvNamedWindow("psf",1);
	//cvShowImage("psf", edge);
	int lambda = 3;
	for (i = d + lambda; i < input->height - (d + lambda); i++)
	{
		uchar* pe = (uchar*)(edge->imageData + i*edge->widthStep);
		uchar* prm1 = (uchar*)(rm1->data.ptr + i*rm1->step);
		uchar* prm2 = (uchar*)(rm2->data.ptr + i*rm2->step);
		uchar* prm3 = (uchar*)(rm3->data.ptr + i*rm3->step);
		uchar* prm4 = (uchar*)(rm4->data.ptr + i*rm4->step);

		for (j = d + lambda; j < input->width - (d + lambda); j++)
		{
			if (pe[j] == 1)
			{
				//0度检测
				for (p = d - lambda; p < d + lambda; p++)
				{
					id = i + d * 1;
					jd = j;

				}
			}
		}
	}
	

	return rm;
}