#include "basicOperation.h"

/************************************************************************
* @函数名称：
*	genaratePsf()
* @输入参数:
*	double len                 - 指定PSF的模糊长度
*	double angle               - 指定PSF的模糊角度
* @输出：
*   IplImage* psf              - 输出生成的模糊核
* @说明:
*   该函数用来生成指定参数的运动模糊核
*	模拟Matlab中的fspecial的‘motion’功能
************************************************************************/
void genaratePsf(IplImage* psf, double len, double angle)
{
	if (len < 1.0)
		len = 1.0;
	double half = (len-1) / 2;
	double phi = angle/ 180.0* PI;
	double cosphi = cos(phi);
	double sinphi = sin(phi);
	double xsign;
	int i=0,j=0;

	if (cosphi < 0){
		xsign = -1;
	}
	else{
		if (cosphi == 90){
			xsign = 0;
		}
		else{
			xsign = 1;
		}
	}
	double linewdt = 1.0;
	//计算0到90度PSF，实际角度-90到0度
	double sx = half*cosphi + linewdt*xsign - len*EPS;
	double sy = half*sinphi + linewdt - len*EPS;

	sx = cvFloor(sx);
	sy = cvFloor(sy);
	 
	//创建存储数据的各种矩阵
	CvMat* X = cvCreateMat(int(sy) + 1, int(sx) + 1, CV_32FC1);
	CvMat* XX = cvCreateMat(int(sy) + 1, int(sx) + 1, CV_32FC1);
	CvMat* Y = cvCreateMat(int(sy) + 1, int(sx) + 1, CV_32FC1);
	CvMat* YY = cvCreateMat(int(sy) + 1, int(sx) + 1, CV_32FC1);
	CvMat* dist2line = cvCreateMat(int(sy) + 1, int(sx) + 1, CV_32FC1);
	CvMat* rad = cvCreateMat(int(sy) + 1, int(sx) + 1, CV_32FC1);
	CvMat* h = cvCreateMat(int(sy) + 1, int(sx) + 1, CV_32FC1);
	cvZero(dist2line);
	cvZero(rad);
	cvZero(h);

	meshgrid(0, int(sx), 0, int(sy), X, Y);
	int rows = X->rows;
	int cols = X->cols;

	cvAddWeighted(Y, cosphi, X, -sinphi,0,dist2line);
	cvMul(X,X,XX);
	cvMul(Y,Y,YY);
	cvAdd(XX,YY, rad);
	cvPow(rad,rad,0.5);

	/*为减小运算量，先计算一半大小的PSF*/
	for (i = 0; i < rows; i++)
	{
		float* prvalue = (float*)(rad->data.ptr+i*rad->step);
		float* pdvalue = (float*)(dist2line->data.ptr+i*dist2line->step);
		float* pxvalue = (float*)(X->data.ptr+i*X->step);
		float* pyvalue = (float*)(Y->data.ptr+i*Y->step);

		for (j = 0; j < cols; j++)
		{
			if (prvalue[j] >= half && fabs(pdvalue[j]) <= linewdt)    //端点情况另外处理
			{
				float x = pxvalue[j];
				float temp = float(half - fabs((x + pdvalue[j] * sinphi) / cosphi));
				pdvalue[j] = sqrt(pdvalue[j] * pdvalue[j] + temp*temp);		//计算到端点的实际距离
			}
			pdvalue[j] = float(linewdt + EPS - fabs(pdvalue[j]));
			if (pdvalue[j] < 0)
			{
				pdvalue[j] = 0;
			}
		}
	}

	/*将模糊核矩阵扩展至实际大小*/
	cvFlip(dist2line, h, -1);		//flip_mode=0沿X-轴翻转,flip_mode>0沿Y-轴翻转,flip_mode<0沿X-轴和Y-轴翻转	
	//测试代码
	/*for (i = 0; i < rows; i++)
	{
		float* ppvalue = (float*)(h->data.ptr + i*h->step);
		for (j = 0; j < cols; j++)
		{
			printf("%f ", ppvalue[j]);
		}
		printf("\n");
	}*/

	int nrows = rows + rows - 1;
	int ncols = cols + cols - 1;
	CvSize size2;
	size2.width = ncols;
	size2.height = nrows;
	
	IplImage* p = cvCreateImage(size2, IPL_DEPTH_32F, 1);
	cvZero(p);

	for (i = 0; i < rows; i++)
	{
		float* phvalue = (float*)(h->data.ptr+i*h->step);
		float* ppvalue = (float*)(p->imageData+i*p->widthStep);
		for (int j = 0; j < cols; j++)
		{
			ppvalue[j] = phvalue[j];
		}
	}

	for (i = 0; i < rows; i++)
	{
		float* pdvalue = (float*)(dist2line->data.ptr+i*dist2line->step);
		float* ppvalue = (float*)(p->imageData+(rows-1+i)*p->widthStep);
		for (j = 0; j < cols;j++)
		{
			ppvalue[cols-1+j] = pdvalue[j];
		}
	}

	/*for (i = 0; i < nrows; i++)
	{
		float* ppvalue = (float*)(p->imageData + i*p->widthStep);
		for (j = 0; j < ncols; j++)
		{
			printf("%f ", ppvalue[j]);
		}
		printf("\n");
	}*/


	/*保持图像总能量不变，归一化矩阵*/
	CvScalar s1 = cvSum(p);
	for (i = 0; i < nrows; i++)
	{
		float* ppvalue = (float*)(p->imageData+i*p->widthStep);
		for (j = 0; j < ncols; j++)
		{
			ppvalue[j]=(float)(ppvalue[j]/s1.val[0]); 
		}
	}

	if (cosphi > 0)
		cvFlip(p, p, 0);


	/*将p填充至指定尺寸的psf*/
	CvSize psfsize = cvGetSize(psf);
	for (i = 0; i < nrows; i++)
	{
		float* ppsfvalue = (float*)(psf->imageData + ((psfsize.height-nrows)/2+i)*psf->widthStep);
		float* ppvalue = (float*)(p->imageData + i*p->widthStep);
		for (j = 0; j < ncols; j++)
		{
			ppsfvalue[(psfsize.width - ncols) / 2+j] = ppvalue[j];
		}
	}
	//测试代码
	/*printf("psf=");
	for (i = 0; i < psfsize.height; i++)
	{
		float* ppsfvalue = (float*)(psf->imageData + i*psf->widthStep);
		for (j = 0; j < psfsize.width; j++)
		{
			printf("%f ", ppsfvalue[j]);
		}
		printf("\n");
	}*/


	cvReleaseMat(&X);
	cvReleaseMat(&XX);
	cvReleaseMat(&Y);
	cvReleaseMat(&YY);
	cvReleaseMat(&dist2line);
	cvReleaseMat(&rad);
	cvReleaseMat(&h);

	cvReleaseImage(&p);
}

/************************************************************************
* @函数名称：
*	meshgrid()
* @输入:
*   int x0           - 输入X轴的起始值
*	int xt           - 输入X轴的结束值
*	int y0           - 输入y轴的起始值
*	int yt           - 输入y轴的起始值
* @输出：
*	CvMat* X           - 生成的X矩阵指针
*	CvMat* Y           - 生成的Y矩阵指针
* @说明:
*   该函数用来模拟Matlab中的meshgrid函数
************************************************************************/
void meshgrid(int x0, int xt, int y0, int yt, CvMat* X, CvMat* Y)
{
	int p, q;
	int pd = yt + 1 - y0;
	int qd = xt + 1 - x0;
	
	for (p = 0; p < pd; p++)
	{
//		uchar* pxvalue = (uchar*)(X->data.ptr+p*X->step);	//用data.ptr指向数据
//		uchar* pyvalue = (uchar*)(Y->data.ptr+p*Y->step);
	//	float* pxvalue = X->data.fl + p*X->step/sizeof(float);	//用data.fl指向数据,不用强制类型转化则需要调整偏移量
	//	float* pxvalue = Y->data.fl + p*X->step/sizeof(float);	

		float* pxvalue = (float*)(X->data.ptr + p*X->step);	//用data.fl指向数据
		float* pyvalue = (float*)(Y->data.ptr + p*Y->step);
		float temp=(float)(y0+p);
		for (q = 0; q < qd; q++)
		{
			pxvalue[q] = (float)(x0 + q);
			pyvalue[q] = temp;
		}
	}
}

/************************************************************************
* @函数名称：
*	mat2Image(()
* @输入参数:
*	CvMat* input                  - 指定PSF的模糊长度
* @输出：
*   IplImage* output              - 输出生成的模糊核
* @说明:
*   该函数用来将矩阵结构转换为图像，以便显示，数据均为单精度浮点类型
************************************************************************/
void mat2Image(CvMat* input, IplImage* output)
{
	int i = 0, j = 0;
	int rows = input->rows;
	int cols = input->cols;

	for (i = 0; i < rows; i++)
	{
		float* pivalue = (float*)(input->data.ptr + i*input->step);
		float* povalue = (float*)(output->imageData + i*output->widthStep);

		for (j = 0; j < cols; j++)
		{
			povalue[j] = pivalue[j];
		}
	}
}

/************************************************************************
* @函数名称：
*	image2Mat()
* @输入参数:
*	IplImage* input            - 指定PSF的模糊长度
* @输出：
*   CvMat* output              - 输出生成的模糊核
* @说明:
*   该函数用来将图像结构转换为矩阵结构，数据均为单精度浮点类型
************************************************************************/
void image2Mat(IplImage* input, CvMat* output)
{
	int i = 0, j = 0;
	int rows = input->height;
	int cols = input->width;

	for (i = 0; i < rows; i++)
	{
		float* pivalue = (float*)(input->imageData + i*input->widthStep);
		float* povalue = (float*)(output->data.ptr + i*output->step);
		
		for (j = 0; j < cols; j++)
		{
			povalue[j] = pivalue[j];
		}
	}
}


/************************************************************************
* @函数名称：
*	shiftDFT()
* @输入参数:
*	CvArr * input                - 输入图像或矩阵
* @输出：
*   CvArr * output               - 输出图像或矩阵
* @说明:
*   该函数在图像或矩阵进行DFT过后将图像中心化，便与显示
************************************************************************/
void shiftDFT(CvArr * input, CvArr * output)
{
	CvMat q1stub, q2stub;
	CvMat q3stub, q4stub;
	CvMat d1stub, d2stub;
	CvMat d3stub, d4stub;
	CvMat * q1, *q2, *q3, *q4;
	CvMat * d1, *d2, *d3, *d4;

	CvSize size = cvGetSize(input);
	CvSize dst_size = cvGetSize(output);
	CvMat * tmp = cvCreateMat(size.height / 2, size.width / 2, cvGetElemType(input));
	int cx, cy;
	//尺寸匹配检查
	if (dst_size.width != size.width || dst_size.height != size.height){
		cvError(CV_StsUnmatchedSizes, "shiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__);
	}

	cx = size.width / 2;
	cy = size.height / 2; //图像中心

	q1 = cvGetSubRect(input, &q1stub, cvRect(0, 0, cx, cy));	//左上角
	q2 = cvGetSubRect(input, &q2stub, cvRect(cx, 0, cx, cy));	//右上角
	q3 = cvGetSubRect(input, &q3stub, cvRect(cx, cy, cx, cy));//右下角
	q4 = cvGetSubRect(input, &q4stub, cvRect(0, cy, cx, cy)); //左下角
	d1 = cvGetSubRect(output, &d1stub, cvRect(0, 0, cx, cy));
	d2 = cvGetSubRect(output, &d2stub, cvRect(cx, 0, cx, cy));
	d3 = cvGetSubRect(output, &d3stub, cvRect(cx, cy, cx, cy));
	d4 = cvGetSubRect(output, &d4stub, cvRect(0, cy, cx, cy));

	if (input != output){
		//类型匹配检查
		if (!CV_ARE_TYPES_EQ(q1, d1)){
			cvError(CV_StsUnmatchedFormats, "shiftDFT", "Source and Destination arrays must have the same format", __FILE__, __LINE__);
		}
		cvCopy(q3, d1, 0);
		cvCopy(q4, d2, 0);
		cvCopy(q1, d3, 0);
		cvCopy(q2, d4, 0);
	}
	else{
		cvCopy(q3, tmp, 0);
		cvCopy(q1, q3, 0);
		cvCopy(tmp, q1, 0);
		cvCopy(q4, tmp, 0);
		cvCopy(q2, q4, 0);
		cvCopy(tmp, q2, 0);
	}

	cvReleaseMat(&tmp);
}

/************************************************************************
* @函数名称：
*	showDFT2()
* @输入参数:
*	IplImage* input                - 输入图像
* @输出：
*   无
* @说明:
*   该函数实现图像的二维傅里叶变换，并显示幅值谱
************************************************************************/
void showDFT2(IplImage* input)
{
	int dft_M = cvGetOptimalDFTSize(input->height);
	int dft_N = cvGetOptimalDFTSize(input->width);	//获得最优DFT变换尺寸
	CvMat* dft_A = cvCreateMat(dft_M, dft_N, CV_64FC2);	//创建DFT变换后的图像
	DFT2(input, dft_A, 1);
	cvNamedWindow("magnitude", 1);
	IplImage* image_Re = cvCreateImage(cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
	IplImage* image_Im = cvCreateImage(cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
	double m, M;
	//分离结果的实部和虚部
	cvSplit(dft_A, image_Re, image_Im, 0, 0);

	//计算频谱的幅值 Mag = sqrt(Re^2 + Im^2)
	cvPow(image_Re, image_Re, 2.0);
	cvPow(image_Im, image_Im, 2.0);
	cvAdd(image_Re, image_Im, image_Re, NULL);
	cvPow(image_Re, image_Re, 0.5);

	//计算对数谱log(1 + Mag)，便于显示
	cvAddS(image_Re, cvScalarAll(1.0), image_Re, NULL); // 1 + Mag
	cvLog(image_Re, image_Re); // log(1 + Mag)

	//中心化
	shiftDFT(image_Re, image_Re);

	cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
	cvScale(image_Re, image_Re, 1.0 / (M - m), 1.0*(-m) / (M - m));//归一化
	cvShowImage("magnitude", image_Re);

	cvReleaseMat(&dft_A);
	
	cvReleaseImage(&image_Re);
	cvReleaseImage(&image_Im);
}

/************************************************************************
* @函数名称：
*	DFT2()
* @输入参数:
*	IplImage* input            - 输入图像，只具有实部单通道
*	int flag			       - 标志位为1进行正变换，标志位为-1进行逆变换
* @输出：
*   CvMat* output              - 输出矩阵，具有实部和虚部两个通道
* @说明:
*   该函数实现图像的二维傅里叶变换
************************************************************************/
void DFT2(IplImage* input, CvMat* output, int flag)
{
	IplImage* realInput;
	IplImage* imaginaryInput;
	IplImage* complexInput;
	int dft_M, dft_N;
	CvMat tmp;
	CvSize size = cvGetSize(input);

	//创建输入图像的实部、虚部和双通道图像
	realInput = cvCreateImage(size, IPL_DEPTH_64F, 1);
	imaginaryInput = cvCreateImage(size, IPL_DEPTH_64F, 1);
	complexInput = cvCreateImage(size, IPL_DEPTH_64F, 2);

	cvScale(input, realInput, 1.0, 0.0);	//将输入图像转换为双精度浮点型
	cvZero(imaginaryInput);
	cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);

	dft_M = output->height;
	dft_N = output->width;	//获得最优DFT变换尺寸

	//将图像填充至最优尺寸，零填充
	cvGetSubRect(output, &tmp, cvRect(0, 0, input->width, input->height));
	cvCopy(complexInput, &tmp, NULL);
	if (dft_N > input->width){
		cvGetSubRect(output, &tmp, cvRect(input->width, 0, dft_N - input->width, input->height));
		cvZero(&tmp);
	}
	if (flag == 1){
		cvDFT(output, output, CV_DXT_FORWARD, complexInput->height);	//正向DFT变换，因为使用了CV_DXT_ROWS参数，不需要进行行填充 
	}
	if (flag == -1){
		cvDFT(output, output, CV_DXT_INV_SCALE, complexInput->height);	//逆向DFT变换，因为使用了CV_DXT_ROWS参数，不需要进行行填充 
	}

	cvReleaseImage(&realInput);
	cvReleaseImage(&imaginaryInput);
	cvReleaseImage(&complexInput);

}

/************************************************************************
* @函数名称：
*	psf2otf()
* @输入参数:
*   IplImage* psf         - 待处理点扩散函数
* @输出：
*   CvMat* otf            - 转换为指定大小的光学传递函数，为一个两通道矩阵
* @说明:
*   该函数用来把一个空间点扩散函数转换为指定大小的频域的光学传递函数
************************************************************************/
void psf2otf(IplImage* psf, CvMat* otf)
{
	int cx, cy;
	int dft_M, dft_N;
	IplImage* image_Re;
	IplImage* image_Im;
	IplImage* temp;
	IplImage* dpsf;
	CvMat* dft_A;

	CvMat q1stub, q2stub;
	CvMat q3stub, q4stub;
	CvMat d1stub, d2stub;
	CvMat d3stub, d4stub;

	CvMat * q1, *q2, *q3, *q4;
	CvMat * d1, *d2, *d3, *d4;

	cx = (psf->width - 1) / 2;
	cy = (psf->height - 1) / 2;

	dft_M = otf->rows;
	dft_N = otf->cols;	//获得最优DFT变换尺寸

	temp = cvCreateImage(cvGetSize(otf), IPL_DEPTH_64F, 1);
	dpsf = cvCreateImage(cvGetSize(psf), IPL_DEPTH_64F, 1);

	cvScale(psf, dpsf, 1.0, 0.0);	//将输入图像转换为双精度浮点型
	cvZero(temp);
	q1 = cvGetSubRect(dpsf, &q1stub, cvRect(0, 0, cx, cy));	//左上角
	q2 = cvGetSubRect(dpsf, &q2stub, cvRect(cx, 0, cx, cy));	//右上角
	q3 = cvGetSubRect(dpsf, &q3stub, cvRect(cx, cy, cx, cy));//右下角
	q4 = cvGetSubRect(dpsf, &q4stub, cvRect(0, cy, cx, cy)); //左下角
	//扩展后的新的位置
	d1 = cvGetSubRect(temp, &d1stub, cvRect(dft_N - cx, dft_M - cy, cx, cy));
	d2 = cvGetSubRect(temp, &d2stub, cvRect(0, dft_M - cy, cx, cy));
	d3 = cvGetSubRect(temp, &d3stub, cvRect(0, 0, cx, cy));
	d4 = cvGetSubRect(temp, &d4stub, cvRect(dft_N - cx, 0, cx, cy));
	//扩展psf至otf尺寸
	cvCopy(q1, d1);
	cvCopy(q2, d2);
	cvCopy(q3, d3);
	cvCopy(q4, d4);

	dft_A = cvCreateMat(dft_M, dft_N, CV_64FC2);	//创建DFT变换后的图像
	image_Re = cvCreateImage(cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
	image_Im = cvCreateImage(cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
	
	DFT2(temp, dft_A, 1);
	cvCopy(dft_A,otf);

	cvReleaseMat(&dft_A);

	cvReleaseImage(&temp);
	cvReleaseImage(&dpsf);
	cvReleaseImage(&image_Re);
	cvReleaseImage(&image_Im);

}




/************************************************************************
* @函数名称：
*	circularShift()
* @输入参数:
*	const CvMat* input			    	- 输入的待处理矩阵
*	const int rows						- 水平方向位移
*	const int cols						- 垂直方向位移
* @输出：
*	CvMat* output						- 循环位移后的矩阵
* @说明:
*	为保证傅里叶变换的中心对齐，将填充后的模糊核循环位移，使之位于填充矩阵的四角上
************************************************************************/
CvMat* circularShift(const CvMat* input,int yshift,int xshift)
{
	int i = 0;
	int j = 0;
	CvMat itmp1, itmp2;
	CvMat otmp1,otmp2;
	CvMat* tmp = cvCloneMat(input);
	CvMat* output = cvCloneMat(input);
	xshift = xshift%input->cols;
	yshift = yshift%input->rows;

	//水平方向循环移位,xshift为正右移，否则左移
	if (xshift > 0)
	{
		cvGetCols(input, &itmp1, 0, input->cols - xshift);
		cvGetCols(input, &itmp2, input->cols - xshift, input->cols);
		cvGetCols(tmp, &otmp1, xshift, tmp->cols);
		cvGetCols(tmp, &otmp2, 0, xshift);
		cvCopy(&itmp1, &otmp1);
		cvCopy(&itmp2, &otmp2);
	}
	if (xshift < 0)
	{
		xshift = -xshift;
		cvGetCols(input, &itmp1, 0, xshift);
		cvGetCols(input, &itmp2,  xshift, input->cols);
		cvGetCols(tmp, &otmp1, input->cols - xshift, tmp->cols);
		cvGetCols(tmp, &otmp2, 0, input->cols - xshift);
		cvCopy(&itmp1, &otmp1);
		cvCopy(&itmp2, &otmp2);
	}
	cvCopy(tmp, output);

	//垂直方向循环移位,yshift为正下移，否则上移
	if (yshift > 0)
	{
		cvGetRows(tmp, &itmp1, 0, tmp->rows - yshift);
		cvGetRows(tmp, &itmp2, tmp->rows - yshift, tmp->rows);
		cvGetRows(output, &otmp1, yshift, output->rows);
		cvGetRows(output, &otmp2, 0, yshift);
		cvCopy(&itmp1, &otmp1);
		cvCopy(&itmp2, &otmp2);
	}
	if (yshift < 0)
	{
		yshift = -yshift;
		cvGetRows(tmp, &itmp1, 0, yshift);
		cvGetRows(tmp, &itmp2, yshift, tmp->rows);
		cvGetRows(output, &otmp1, tmp->rows-yshift, output->rows);
		cvGetRows(output, &otmp2, 0, tmp->rows-yshift);
		cvCopy(&itmp1, &otmp1);
		cvCopy(&itmp2, &otmp2);
	}

	cvReleaseMat(&tmp);

	return output;
}

/************************************************************************
* @函数名称：
*	calConv()
* @输入参数:
*   IplImage* input         - 输入图像
*	IplImage* kernal	    - 卷积核
* @输出：
*   IplImage* output        - 输出图像
* @说明:
*   用FFT计算卷积,时域卷积即频域相乘
************************************************************************/
void calConv(IplImage* input, IplImage* kernal, CvMat* output)
{
	int rows = output->height;
	int cols = output->width;
	CvMat* temp1 = cvCreateMat(rows, cols, CV_64FC2);
	CvMat* temp2 = cvCreateMat(rows, cols, CV_64FC2);
	CvMat* temp3 = cvCreateMat(rows, cols, CV_64FC2);

	DFT2(input,temp1,1);
	psf2otf(kernal,temp2);
	cvMulSpectrums(temp1, temp2, temp3, CV_DXT_ROWS);
	cvDFT(temp3, temp3, CV_DXT_INV_SCALE,temp3->height);
	cvCopy(temp3, output);

	cvReleaseMat(&temp1);
	cvReleaseMat(&temp2);
	cvReleaseMat(&temp3);
} 

/************************************************************************
* @函数名称：
*	kcvRectIntersection()，kcvRectFromCenterAndSiz()
*	void sort(uchar a[], int n),clip(),	percentileValue()
*	percentileFilter(),percentileValue2(),percentileFilter2()
*	上述函数簇是为了实现统计排序滤波
************************************************************************/
CvRect kcvRectIntersection(CvRect rect1, CvRect rect2)
{
	CvRect rect;
	rect.x = max(rect1.x, rect2.x);
	rect.y = max(rect1.y, rect2.y);
	rect.width = min(rect1.x + rect1.width, rect2.x + rect2.width);
	rect.width = rect.width - rect.x;
	rect.height = min(rect1.y + rect1.height, rect2.y + rect2.height);
	rect.height = rect.height - rect.y;
	return rect;
}

CvRect kcvRectFromCenterAndSize(int cx, int cy, int w, int h)
{
	CvRect rect;
	h = (h == 0 ? w : h);
	rect.x = cx - (w >> 1);
	rect.y = cy - (h >> 1);
	rect.width = w;
	rect.height = h;
	return rect;
}

// 冒泡排序  
void sort(uchar a[], int n)
{
	uchar t = 0;
	for (int i = 0; i<n - 1; ++i)
	{
		for (int j = 0; j<n - 1 - i; ++j)
		{
			if (a[j]>a[j + 1])
			{
				t = a[j];
				a[j] = a[j + 1];
				a[j + 1] = t;
			}
		}
	}
}

// 将p限定在[min(a,b),max(a,b)]  
double clip(double p, double a, double b)
{
	return min(max(p, min(a, b)), max(a, b));
}

uchar percentileValue(IplImage* img, CvRect rect, double p)
{
	int n = rect.width*rect.height;
	//uchar* data = new uchar[n];
	uchar* data = (uchar*)malloc(sizeof(uchar)*n);

	for (int j = 0; j<rect.width; ++j)
	{
		for (int i = 0; i<rect.height; ++i)
		{
			data[j + i*rect.width] = CV_IMAGE_ELEM(img, uchar, rect.y + i, rect.x + j);
		}
	}
	sort(data, n);
	p = clip(p, 0, 1);
	return data[int(p*(n - 1))];
}

// p取值范围为[0,1]。0表示最小值滤波，1表示最大值滤波，0.5表示中值滤波  
void percentileFilter(IplImage* src, IplImage* dst, double p, int kwidth, int kheight)
{
	CvRect rect1 = cvRect(0, 0, src->width, src->height);
	for (int j = 0; j<src->width; ++j)
	{
		for (int i = 0; i<src->height; ++i)
		{
			CvRect rect2 = kcvRectFromCenterAndSize(j, i, kwidth, kheight);
			CvRect rect = kcvRectIntersection(rect1, rect2);
			CV_IMAGE_ELEM(dst, uchar, i, j) = percentileValue(src, rect, p);
		}
	}
}

uchar percentileValue2(IplImage* img, CvRect rect, double p)
{
	int n = rect.width*rect.height;
	//uchar* data = new uchar[n];
	uchar* data = (uchar*)malloc(sizeof(uchar)*n);
	for (int j = 0; j<rect.width; ++j)
	{
		for (int i = 0; i<rect.height; ++i)
		{
			data[j + i*rect.width] = CV_IMAGE_ELEM(img, uchar, rect.y + i, rect.x + j);
		}
	}
	sort(data, n);
	p = clip(p, 0, 1);

	double index = (n - 1)*p;
	int indexc = (int)ceil(index);
	int indexf = (int)floor(index);
	if (indexc == indexf)  // 即index是整数（不是整型）  
	{
		return data[(int)index];
	}
	else
	{
		// 注意到indexc-indexf==1  
		return (uchar)(data[indexc] * (index - indexf) + data[indexf] * (indexc - index));
	}
}

// p取值范围为[0,1]。0表示最小值滤波，1表示最大值滤波，0.5表示中值滤波  
void percentileFilter2(IplImage* src, IplImage* dst, double p, int kwidth, int kheight)
{
	CvRect rect1 = cvRect(0, 0, src->width, src->height);
	for (int j = 0; j<src->width; ++j)
	{
		for (int i = 0; i<src->height; ++i)
		{
			CvRect rect2 = kcvRectFromCenterAndSize(j, i, kwidth, kheight);
			CvRect rect = kcvRectIntersection(rect1, rect2);
			CV_IMAGE_ELEM(dst, uchar, i, j) = percentileValue2(src, rect, p);
		}
	}
}

void adpMedian(IplImage* src, IplImage* dst, int Smax)
{
	CvRect rect1 = cvRect(0, 0, src->width, src->height);
	uchar min, max, med,adpmed,tmp;
	for (int j = 0; j<src->width; ++j)
	{
		for (int i = 0; i<src->height; ++i)
		{
			for (int s = 3; s < Smax + 1; s = s + 2)
			{
				CvRect rect2 = kcvRectFromCenterAndSize(j, i, s, s);
				CvRect rect = kcvRectIntersection(rect1, rect2);
				min = percentileValue2(src, rect, 0);
				max = percentileValue2(src, rect, 1);
				med = percentileValue2(src, rect, 0.5);
				adpmed = med;
				if ((med>min) && (med < max))
				{
					tmp = CV_IMAGE_ELEM(src, uchar, i, j);
					if ((tmp>min) && (tmp < max))
					{
						adpmed = tmp;
					}
					adpmed = med;
				}
				CV_IMAGE_ELEM(dst, uchar, i, j) = adpmed;
			}
		}
	}
}

/*************************************************************************
* @函数名称：
*	complexMatrixDivide()
* @输入:
*   CvMat* m            - 被除数，为一个两通道复数矩阵
*	CvMat* n			- 除数，为一个两通道复数矩阵
* @返回值:
*   CvMat* result       - 返回复数除法结果
* @说明:
*   该函数用来对复数进行点除运算，输入和输出均为两通道矩阵，
*	两个通道分别存放实部和虚部
*************************************************************************/
CvMat* complexMatrixDivide(const CvMat* m, const CvMat* n)
{
	if ((m->rows != n->rows) || (m->cols != n->cols))
	{
		printf("Wrong size in complexMatrixDivide!");
	}

	CvMat* a = cvCreateMat(m->rows, m->cols, CV_64FC1);
	CvMat* b = cvCreateMat(m->rows, m->cols, CV_64FC1);
	CvMat* c = cvCreateMat(m->rows, m->cols, CV_64FC1);
	CvMat* d = cvCreateMat(m->rows, m->cols, CV_64FC1);
	CvMat* ac = cvCreateMat(m->rows, m->cols, CV_64FC1);
	CvMat* bd = cvCreateMat(m->rows, m->cols, CV_64FC1);
	CvMat* bc = cvCreateMat(m->rows, m->cols, CV_64FC1);
	CvMat* ad = cvCreateMat(m->rows, m->cols, CV_64FC1);
	CvMat* result = cvCloneMat(m);

	cvSplit(m, a, b, 0, 0);
	cvSplit(n, c, d, 0, 0);

	cvMul(a, c, ac);
	cvMul(b, d, bd);
	cvMul(b, c, bc);
	cvMul(a, d, ad);

	cvMul(c, c, c);
	cvMul(d, d, d);

	cvAdd(c, d, c);
	cvMaxS(c, EPS, c);
	cvAdd(ac, bd, ac);
	cvSub(bc, ad, bc);

	cvDiv(ac, c, ac);
	cvDiv(bc, c, bc);

	cvMerge(ac, bc, 0, 0, result);

	cvReleaseMat(&a);
	cvReleaseMat(&b);
	cvReleaseMat(&c);
	cvReleaseMat(&d);
	cvReleaseMat(&ac);
	cvReleaseMat(&bd);
	cvReleaseMat(&bc);
	cvReleaseMat(&ad);

	return result;
}