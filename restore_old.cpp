#include "basicOperation.h"
#include "restore.h"

/************************************************************************
* @函数名称：
*	doubFreqAngle()
* @输入参数:
*	IplImage* input               - 输入图像
* @输出：
*   double angle    			  - 输出估计出的模糊角度
* @说明:
*	采用二次频谱对图像模糊方向进行提取，利用自适应中值滤波、形态学滤波等手段
*	提取方向特征
************************************************************************/
double doubFreqAngle(IplImage* input)
{
	double angle = 0;
	int i, j;
	int rows = input->height;
	int cols = input->width;	//获取图像尺寸
	CvSize size_input;
	size_input.height = input->height;
	size_input.width = input->width;

	CvMat* dft = cvCreateMat(rows,cols,CV_64FC2);
	IplImage* dft_Re = cvCreateImage(size_input, IPL_DEPTH_64F,1);
	IplImage* dft_Im = cvCreateImage(size_input, IPL_DEPTH_64F, 1);

	DFT2(input, dft, 1);//第一次傅里叶变换
	cvSplit(dft, dft_Re, dft_Im, 0, 0);	//分离结果的实部和虚部

	//计算频谱的幅值 Mag = sqrt(Re^2 + Im^2)
	cvPow(dft_Re, dft_Re, 2.0);
	cvPow(dft_Im, dft_Im, 2.0);
	cvAdd(dft_Re, dft_Im, dft_Re, NULL);
	cvPow(dft_Re, dft_Re, 0.5);

	//计算对数谱log(1 + Mag)，便于显示
	cvAddS(dft_Re, cvScalarAll(1.0), dft_Re, NULL); // 1 + Mag
	cvLog(dft_Re, dft_Re); // log(1 + Mag)

	DFT2(dft_Re, dft, 1);//第二次傅里叶变换
	cvSplit(dft, dft_Re, dft_Im, 0, 0);	//分离结果的实部和虚部

	//计算频谱的幅值 Mag = sqrt(Re^2 + Im^2)
	cvPow(dft_Re, dft_Re, 2.0);
	cvPow(dft_Im, dft_Im, 2.0);
	cvAdd(dft_Re, dft_Im, dft_Re, NULL);
	cvPow(dft_Re, dft_Re, 0.5);

	//计算对数谱log(1 + Mag)，便于显示
	cvAddS(dft_Re, cvScalarAll(1.0), dft_Re, NULL); // 1 + Mag
	cvLog(dft_Re, dft_Re); // log(1 + Mag)

	//中心化
	shiftDFT(dft_Re, dft_Re);

	//cvNormalize(dft_Re, dft_Re, 1, 0, CV_MINMAX);//归一化
	//cvNamedWindow("double frequency analysis", 1);
	//cvShowImage("double frequency analysis", dft_Re);

	cvNormalize(dft_Re, dft_Re, 255, 0, CV_MINMAX);//归一化
	IplImage* input_char = cvCreateImage(size_input, IPL_DEPTH_8U, 1);
	cvScale(dft_Re, input_char, 1.0, 0.0);	//将输入图像转换为双精度浮点型

	//double M, m;
	//cvMinMaxLoc(dft_Re, &m, &M, NULL, NULL, NULL);
	//printf("min=%f max=%f\n", m, M);
	//printf("cvEqualizeHist=");
	//for (i = 0; i < rows; i++)
	//{
	//	uchar* prvalue = (uchar *)(input_char->imageData + i*input_char->widthStep);
	//	for (j = 0; j < cols; j++)
	//	{
	//		printf("%d ", prvalue[j]);
	//	}
	//	printf("\n");
	//}
	cvEqualizeHist(input_char, input_char);//直方图均衡化，增强对比度
	cvThreshold(input_char, input_char, 0.85*255, 1, CV_THRESH_BINARY);//阈值处理
	//中值滤波不支持原地操作
	IplImage* input_median = cvCreateImage(size_input, IPL_DEPTH_8U, 1);
	//cvSmooth(input_char, input_median, CV_MEDIAN, 5, 0, 0, 0);//中值滤波
	//percentileFilter2(input_char, input_median,0.5,5,5);
	adpMedian(input_char, input_median,7);
	//重写自适应中值滤波
	//cvScale(input_median, input_median, 255, 0.0);
	//cvNamedWindow("median", 1);
	//cvShowImage("median", input_median);
	//cvNormalize(input_median, input_median, 1, 0, CV_MINMAX);//归一化
	//形态学滤波
	
	cvMorphologyEx(input_median, input_median, 0, 0, CV_MOP_CLOSE,2);	
	cvErode(input_median, input_median, 0, 1);//腐蚀
	adpMedian(input_median, input_char, 5);

	//cvScale(input_median, input_median, 255, 0.0);
	//cvNamedWindow("median", 1);
	//cvShowImage("median", input_median);
	//cvNormalize(input_median, input_median, 1, 0, CV_MINMAX);//归一化

	//cvScale(input_char, input_median, 255, 0.0);
	//cvNamedWindow("Morphol", 1);
	//cvShowImage("Morphol", input_median);
	//cvNormalize(input_median, input_median, 1, 0, CV_MINMAX);//归一化

	//外轮廓检测
	IplImage* pContourImg = NULL;
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	CvSeq* contoursTemp = 0;
	int mode = CV_RETR_EXTERNAL;
	int num = 0,imax=0;

	pContourImg = cvCreateImage(cvGetSize(input_median),IPL_DEPTH_8U,1);
	num=cvFindContours(input_median, storage, &contour, sizeof(CvContour),mode, CV_CHAIN_APPROX_SIMPLE);
	//printf("contour number=%d\n", num);
	//cvDrawContours(pContourImg, contour,CV_RGB(255, 255, 255), CV_RGB(0, 0, 0),2, 1, 8);//显示图像
	//cvShowImage("contour", pContourImg);

	int* pnum = (int*)malloc(sizeof(int)*num);
	contoursTemp = contour;
	//for (; contoursTemp != 0; contoursTemp = contoursTemp->h_next)  // 这样可以访问每一个轮廓  ====横向轮廓  
	for (i = 0; i < num; i++)
	{
		pnum[i] = contoursTemp->total;
	//	printf("The %d contour has %d points!\n", i, pnum[i]);
		contoursTemp = contoursTemp->h_next;
	}

	for (j = 0; j < num; j++)
	{
		if (pnum[imax]< pnum[j])
		{
			imax = j;
		}
	}

	CvPoint* ppoint = (CvPoint*)malloc(sizeof(CvPoint)*pnum[imax]);
	int* px = (int*)malloc(sizeof(int)*pnum[imax]);
	int* py = (int*)malloc(sizeof(int)*pnum[imax]);


	contoursTemp = contour;
	for (i = 0; i < num; i++)
	{
		if (i == imax)
		{
			for (j = 0; j < contoursTemp->total; j++)    // 提取一个轮廓的所有坐标点  
			{
				ppoint[j] = *((CvPoint*)cvGetSeqElem(contoursTemp, j));   // 得到一个轮廓中一个点的函数cvGetSeqElem  
				px[j] = ppoint[j].x;
				py[j] = ppoint[j].y;
			//	printf("[%d %d] ", px[j], py[j]);
			}
		}
		contoursTemp = contoursTemp->h_next;
	}
	//查找最大轮廓对应的x,y方向的最大和最小值
	int xmax = 0, ymax = 0;
	int xmin = 0, ymin = 0;
	for (i = 0; i < pnum[imax]; i++)
	{
		if ((px[xmax] < px[i]) || (px[xmax] == px[i]))
		{
			xmax = i;
		}
		if ((py[ymax] < py[i]) || (py[ymax] == py[i]))
		{
			ymax = i;
		}
		if ((px[xmin] > px[i]) || (px[xmin] == px[i]))
		{
			xmin = i;
		}
		if ((px[ymin] > py[i]) || (px[ymin] == py[i]))
		{
			ymin = i;
		}
	}
	//判断斜率符号
	int sign = 1;
	if (py[xmax] < py[xmin])
	{
		sign = -1;
	}
	angle = -atan(((double)(py[xmax] - py[xmin])) / (px[xmax] - px[xmin]));
	angle = angle / PI * 180;
	if (angle<0)
	{
		angle = angle + 180;
	}

	if (angle>50)
	{
		angle = -atan(((double)(py[ymax] - py[ymin])) / (px[ymax] - px[ymin]));
		angle = angle / PI * 180;
	}

	free(pnum);
	free(ppoint);
	free(px);
	free(py);

	cvReleaseMat(&dft);

	cvReleaseImage(&dft_Re);
	cvReleaseImage(&input_char);
	cvReleaseImage(&input_median);
	cvReleaseImage(&pContourImg);

	cvReleaseMemStorage(&storage);

	return angle;
}



/************************************************************************
* @函数名称：
*	edgetaper()
* @输入参数:
*	IplImage* input               - 输入图像
*	IplImage* psf				  - 模糊核
* @输出：
*   IplImage* output			  - 输出图像
* @说明:
*	对待处理图像边缘进行预处理，使之具有相关性
*	J = alpha.*I + (1-alpha).*blurredI;
*	alpha = (1-beta[0]) *(1- beta[1]);
************************************************************************/
void edgetaper(IplImage* input, IplImage* psf,IplImage* output)
{
	int irow = input->height;
	int icol = input->width;
	int prow = psf->height;
	int pcol = psf->width;
	int i, j;

	IplImage* psfRowProj = cvCreateImage(cvSize(prow,1), IPL_DEPTH_64F, 1);
	IplImage* psfColProj = cvCreateImage(cvSize(pcol, 1), IPL_DEPTH_64F, 1);
	CvMat* blur = cvCreateMat(irow, icol, CV_64FC2);
	IplImage* blur_Re = cvCreateImage(cvGetSize(input),IPL_DEPTH_64F,1);
	IplImage* blur_Im = cvCreateImage(cvGetSize(input), IPL_DEPTH_64F, 1);
	cvZero(psfRowProj);
	cvZero(psfColProj);

	calConv(input, psf, blur);
	cvSplit(blur, blur_Re, blur_Im, 0, 0);
	//cvNormalize(blur_Re, blur_Re, 255, 0, CV_MINMAX);
	//测试代码
	//printf("blur=");
	//for (i = 0; i < 4; i++)
	//{
	//	double* prvalue = (double *)(blur_Re->imageData + i*blur_Re->widthStep);
	//	for (j = 0; j < icol; j++)
	//	{
	//		printf("%f ", prvalue[j]);
	//	}
	//	printf("\n");
	//}

	//计算psf的行投影
	for (i = 0; i < prow; i++)
	{
		float temp = 0;
		float* ppvalue = (float*)(psf->imageData+i*psf->widthStep);
		double* prvalue = (double*)(psfRowProj->imageData);
		for (j = 0; j < pcol; j++)
		{
			temp += ppvalue[j];	//计算行和
		}
		prvalue[i] = temp;
	}

	CvSize tsize1, tsize2;
	//自相关，用FFT实现
	tsize1.width = irow;
	tsize1.height = 1;

	IplImage* psfRowProjPadded = cvCreateImage(tsize1, IPL_DEPTH_64F, 1);
	IplImage* temp1_Re = cvCreateImage(tsize1, IPL_DEPTH_64F, 1);
	IplImage* temp1_Im = cvCreateImage(tsize1, IPL_DEPTH_64F, 1);
	CvMat* psfRowProjDFT = cvCreateMat(tsize1.height, tsize1.width, CV_64FC2);
	CvMat q1stub, q2stub;
	CvMat d1stub, d2stub;
	CvMat * q1, *q2;
	CvMat * d1, *d2;
	int cr = (prow - 1) / 2;

	cvZero(psfRowProjPadded);
	q1 = cvGetSubRect(psfRowProj, &q1stub, cvRect(0, 0, cr, 1));	//左
	q2 = cvGetSubRect(psfRowProj, &q2stub, cvRect(cr, 0, cr, 1));	//右
	//扩展后的新的位置
	d1 = cvGetSubRect(psfRowProjPadded, &d1stub, cvRect(irow - cr, 0, cr, 1));
	d2 = cvGetSubRect(psfRowProjPadded, &d2stub, cvRect(0, 0, cr, 1));
	cvCopy(q1, d1);
	cvCopy(q2, d2);

	DFT2(psfRowProjPadded, psfRowProjDFT, 1);
	cvSplit(psfRowProjDFT, temp1_Re, temp1_Im, 0, 0);

	//计算频谱的幅值 Mag = sqrt(Re^2 + Im^2)
	cvPow(temp1_Re, temp1_Re, 2.0);
	cvPow(temp1_Im, temp1_Im, 2.0);
	cvAdd(temp1_Re, temp1_Im, temp1_Re, NULL);
	DFT2(temp1_Re, psfRowProjDFT,-1);
	cvSplit(psfRowProjDFT, temp1_Re, temp1_Im, 0, 0);
	cvNormalize(temp1_Re, temp1_Re, 1, 0, CV_MINMAX);

	//计算psf的列投影
	for (i = 0; i < prow; i++)
	{
		float* ppvalue = (float*)(psf->imageData + i*psf->widthStep);
		double* pcvalue = (double*)(psfColProj->imageData);
		for (j = 0; j < pcol; j++)
		{
			pcvalue[j] += ppvalue[j];	//计算列和
		}
	}

	tsize2.width = icol ;
	tsize2.height = 1;

	IplImage* psfColProjPadded = cvCreateImage(tsize2, IPL_DEPTH_64F, 1);
	IplImage* temp2_Re = cvCreateImage(tsize2, IPL_DEPTH_64F, 1);
	IplImage* temp2_Im = cvCreateImage(tsize2, IPL_DEPTH_64F, 1);
	CvMat* psfColProjDFT = cvCreateMat(tsize2.height, tsize2.width, CV_64FC2);
	int cc = (prow - 1) / 2;

	cvZero(psfColProjPadded);
	q1 = cvGetSubRect(psfColProj, &q1stub, cvRect(0, 0, cc, 1));	//左
	q2 = cvGetSubRect(psfColProj, &q2stub, cvRect(cc, 0, cc, 1));	//右
	//扩展后的新的位置
	d1 = cvGetSubRect(psfColProjPadded, &d1stub, cvRect(icol - cc, 0, cc, 1));
	d2 = cvGetSubRect(psfColProjPadded, &d2stub, cvRect(0, 0, cc,1));
	cvCopy(q1, d1);
	cvCopy(q2, d2);

	DFT2(psfColProjPadded, psfColProjDFT, 1);
	cvSplit(psfColProjDFT, temp2_Re, temp2_Im, 0, 0);

	//计算频谱的幅值 Mag = sqrt(Re^2 + Im^2)
	cvPow(temp2_Re, temp2_Re, 2.0);
	cvPow(temp2_Im, temp2_Im, 2.0);
	cvAdd(temp2_Re, temp2_Im, temp2_Re, NULL);
	DFT2(temp2_Re, psfColProjDFT, -1);
	cvSplit(psfColProjDFT, temp2_Re, temp2_Im, 0, 0);
	cvNormalize(temp2_Re, temp2_Re, 1, 0, CV_MINMAX);

	IplImage* alpha = cvCreateImage(cvGetSize(input), IPL_DEPTH_64F, 1);
	IplImage* alphaTmp = cvCreateImage(cvGetSize(input), IPL_DEPTH_64F, 1);

	cvConvertScale(temp1_Re, temp1_Re, -1, 1);
	cvConvertScale(temp2_Re, temp2_Re, -1, 1);

	//printf("tmp1= ");
	//for (i = 0; i <irow; i++)
	//{
	//	double* ppvalue = (double*)(temp1_Re->imageData);
	//	printf("%f ", ppvalue[i]);
	//}
	//printf("\n ");
	//printf("tmp2= ");
	//for (i = 0; i <icol; i++)
	//{
	//	double* ppvalue = (double*)(temp2_Re->imageData);
	//	printf("%f ", ppvalue[i]);
	//}
	//printf("\n ");

	cvGEMM(temp1_Re, temp2_Re, 1, 0, 0, alpha, CV_GEMM_A_T);
	cvConvertScale(alpha, alphaTmp, -1, 1);
	IplImage* inputScale = cvCreateImage(cvGetSize(input), IPL_DEPTH_64F, 1);
	IplImage* blurScale = cvCreateImage(cvGetSize(input), IPL_DEPTH_64F, 1);
	IplImage* doubleInput = cvCreateImage(cvGetSize(input), IPL_DEPTH_64F, 1);

	cvConvertScale(input, doubleInput, 1.0, 0.0);	//将输入图像转换为双精度浮点型
	cvMul(doubleInput, alpha, inputScale);	//原始图像加权
	cvMul(blur_Re, alphaTmp, blurScale);	//模糊图像加权
	cvAdd(inputScale, blurScale, output);	//得到加权图像
	//cvNormalize(output, output, 1, 0, CV_MINMAX);//归一化
	cvMaxS(output, 0, output);
	cvMinS(output, 255, output);

	cvReleaseMat(&blur);
	cvReleaseMat(&psfRowProjDFT);
	cvReleaseMat(&psfColProjDFT);
	cvReleaseImage(&psfRowProj);
	cvReleaseImage(&psfColProj);
	cvReleaseImage(&blur_Re);
	cvReleaseImage(&blur_Im);
	cvReleaseImage(&psfRowProjPadded);
	cvReleaseImage(&psfColProjPadded);
	cvReleaseImage(&temp1_Re);
	cvReleaseImage(&temp1_Im);
	cvReleaseImage(&temp2_Re);
	cvReleaseImage(&temp2_Im);
	cvReleaseImage(&alpha);
	cvReleaseImage(&alphaTmp);
	cvReleaseImage(&inputScale);
	cvReleaseImage(&blurScale);
	cvReleaseImage(&doubleInput);
}


/************************************************************************
* @函数名称：
*	doubFreqAngle()
* @输入参数:
*	IplImage* input               - 输入图像
*	double angle				  - 模糊角度
* @输出：
*   double len   				  - 输出估计出的模糊长度
* @说明:
*	采用倒谱法对图像模糊长度进行提取，利用自适应中值滤波、形态学滤波等手段
*	提取模糊长度信息
************************************************************************/
double cepstrumLen(IplImage* input,double angle)
{
	double len = 0;
	int i, j;
	int rows = input->height;
	int cols = input->width;	//获取图像尺寸
	CvSize size_input;
	size_input.height = input->height;
	size_input.width = input->width;

	CvMat* dft = cvCreateMat(rows, cols, CV_64FC2);
	IplImage* dft_Re = cvCreateImage(size_input, IPL_DEPTH_64F, 1);
	IplImage* dft_Im = cvCreateImage(size_input, IPL_DEPTH_64F, 1);

	//将图像转换到倒谱域
	//Cep(g(x, y)) = invFT{ log(FT(g(x, y))) }
	DFT2(input, dft, 1);//第一次傅里叶变换
	cvSplit(dft, dft_Re, dft_Im, 0, 0);	//分离结果的实部和虚部

	//计算频谱的幅值 Mag = sqrt(Re^2 + Im^2)
	cvPow(dft_Re, dft_Re, 2.0);
	cvPow(dft_Im, dft_Im, 2.0);
	cvAdd(dft_Re, dft_Im, dft_Re, NULL);
	cvPow(dft_Re, dft_Re, 0.5);

	//计算对数谱log(1 + Mag)
	cvAddS(dft_Re, cvScalarAll(1.0), dft_Re, NULL); // 1 + Mag
	cvLog(dft_Re, dft_Re); // log(1 + Mag)

	DFT2(dft_Re, dft, -1);//傅里叶逆变换
	cvSplit(dft, dft_Re, dft_Im, 0, 0);	//分离结果的实部和虚部

	//计算频谱的幅值 Mag = sqrt(Re^2 + Im^2)
	cvPow(dft_Re, dft_Re, 2.0);
	cvPow(dft_Im, dft_Im, 2.0);
	cvAdd(dft_Re, dft_Im, dft_Re, NULL);
	cvPow(dft_Re, dft_Re, 0.5);

	//计算对数谱log(1 + Mag)，便于显示
	cvAddS(dft_Re, cvScalarAll(1.0), dft_Re, NULL); // 1 + Mag
	cvLog(dft_Re, dft_Re); // log(1 + Mag)

	//中心化
	shiftDFT(dft_Re, dft_Re);

	//cvNormalize(dft_Re, dft_Re, 1, 0, CV_MINMAX);//归一化
	//cvNamedWindow("cepstrum analysis", 0);
	//cvShowImage("cepstrum analysis", dft_Re);

	//旋转倒谱，便于检测模糊长度
	IplImage* input_rotate = cvCreateImage(cvGetSize(dft_Re), IPL_DEPTH_8U, 1);
	IplImage* output_rotate = cvCreateImage(cvGetSize(dft_Re), IPL_DEPTH_8U, 1);
	cvScale(dft_Re, input_rotate, 255, 0);
	imRotate(input_rotate, output_rotate, -angle);
	cvNamedWindow("rotate",0);
	cvShowImage("rotate", output_rotate);
	//计算列和
	CvMat* col_avg = cvCreateMat(1, output_rotate->width, CV_64FC1);
	cvZero(col_avg);
	//CvMat col_center;
	//cvGetRow(output_rotate, &col_center, output_rotate->height / 2-1 );
	//cvScale(&col_center, col_avg, 1, 0);
	for (i = output_rotate->height / 2 -8; i < output_rotate->height/2+8; i++)
	{
		uchar* pvalue = (uchar*)(output_rotate->imageData + i*output_rotate->widthStep);
		double* pcvalue = (double*)(col_avg->data.ptr);
		for (j = 0; j < output_rotate->width; j++)
		{
			pcvalue[j] += (double)(pvalue[j]);	//计算列和
		}
	}
//	uchar* pcvalue = (uchar*)(col_avg->data.ptr);

	double* pcvalue = (double*)(col_avg->data.ptr);
	for (j = 0; j < output_rotate->width; j++)
	{
		printf("%f ", pcvalue[j]);
	}

	double m, M;
	CvPoint pM,pm;
	int minloc=0;

	cvMinMaxLoc(col_avg, &m, &M, &pm, &pM, 0);
	//得到最大值坐标后，向一侧遍历，求第一个极小值位置
	for (i = pM.x-1; i>0; i--)
	{
		double* pcvalue = (double*)(col_avg->data.ptr);
		if ((pcvalue[i] < pcvalue[i + 1]) && (pcvalue[i] < pcvalue[i - 1]))
		{
			minloc = i;
			break;
		}
	}

	//for (i = 1; i<col_avg ->cols/ 2 - 1; i++)
	//{
	//	double* pcvalue = (double*)(col_avg->data.ptr);
	//	if ((pcvalue[i] < pcvalue[i + 1]) && (pcvalue[i] < pcvalue[i - 1]))
	//	{
	//		minloc = i;
	//		break;
	//	}
	//}
	printf("min=%f,max=%f\n", m, M);
	//printf("minloc=%d, maxloc=%d \n", pm.x,pM.x);
	printf("minloc=%d, maxloc=%d \n", minloc, pM.x);

	len = pM.x - minloc;
	//len = minloc;

	cvReleaseMat(&dft);
	cvReleaseMat(&col_avg);
	cvReleaseImage(&input_rotate);
	cvReleaseImage(&output_rotate);
	cvReleaseImage(&dft_Re);
	cvReleaseImage(&dft_Im);
	
	return len;
}


/************************************************************************
* @函数名称：
*	imRotate()
* @输入参数:
*	IplImage* input		- 输入图像
*	double angle		- 旋转角度
* @输出：
*   IplImage*output		- 旋转后的图像
* @说明:
*	输入输出均需保证图像为8位无符号整形数据
************************************************************************/
void imRotate(IplImage* input, IplImage* output, double angle)
{
	float m[6];
	double factor = 1;
	int w = input->width;
	int h = input->height;

	m[0] = (float)(factor*cos(-angle * CV_PI / 180.));
	m[1] = (float)(factor*sin(-angle * CV_PI / 180.));
	m[2] = w*0.5f;
	m[3] = -m[1];
	m[4] = m[0];
	m[5] = h*0.5f;

	CvMat M = cvMat(2, 3, CV_32F, m);
	cvGetQuadrangleSubPix(input, output, &M);
}

/************************************************************************
* @函数名称：
*	deconvRL()
* @输入参数:
*	IplImage* input		- 输入图像
*	IplImage* kernal    - 输入模糊核
* @输出：
*   IplImage*output		- 输出图像
* @说明:
*	 实现RL迭代复原，使用FFT计算卷积，使用矢量外推法进行优化加速
*		 边缘预处理抑制振铃
*	f(k+1)=f(k)(conv((g(k)/(conv(h,f(k)))),hInv))
*	     =beta(f(k))
*	y(k)=x(k)+lambda(k)*h(k)
*	h(k)=x(k)-x(k-1)
*	x(k+1)=y(k)+g(k)
*	g(k)=beta(y(k))-y(k)
*	lambda(k)=sum(g(k-1).*g(k-2))/sum(g(k-2).*g(k-2)),0<lambda(k)<1
************************************************************************/
void deconvRL(IplImage* input, IplImage* kernal, IplImage* output, int num)
{
	CvSize size = cvGetSize(input);
	CvSize psize = cvGetSize(kernal);

	IplImage* input1 = cvCreateImage(size, IPL_DEPTH_64F, 1);
	IplImage* psfInv = cvCreateImage(psize, IPL_DEPTH_32F, 1);
	IplImage* g=cvCreateImage(size, IPL_DEPTH_64F, 1);
	IplImage* image_char = cvCreateImage(size, IPL_DEPTH_8U, 1);
	
	//edgetaper(input, kernal, input1);

	//cvScale(input1, image_char, 1);
	//cvNamedWindow("input1", 1);
	//cvShowImage("input1", image_char);
	cvCopy(input, input1);

	cvZero(g);
	cvScale(g, g, 1, 0.5);

	cvFlip(kernal, psfInv, -1);


	IplImage* J1 = cvCreateImage(size, IPL_DEPTH_64F, 1);//存放模糊图像
	IplImage* J2 = cvCreateImage(size, IPL_DEPTH_64F, 1);//存放清晰图像当前估计值
	IplImage* J3 = cvCreateImage(size, IPL_DEPTH_64F, 1);//存放上一次估计结果
	IplImage* L1 = cvCreateImage(size, IPL_DEPTH_64F, 1);//存放当前估计值和预测值的差值
	IplImage* L2 = cvCreateImage(size, IPL_DEPTH_64F, 1);//存放上次估计值和预测值的差值
	IplImage* Y = cvCreateImage(size, IPL_DEPTH_64F, 1);//存放当前预测的清晰图像
	cvCopy(g, J2);
	cvCopy(input1, J1);

	cvZero(J3);
	cvZero(L1);
	cvZero(L2);

	CvMat* H = cvCreateMat(input1->width, input1->height, CV_64FC2);
	CvMat* H1 = cvCreateMat(input1->width, input1->height, CV_64FC2);
	CvMat* CC = cvCreateMat(input1->width, input1->height, CV_64FC2);


	double lambda = 0;
	psf2otf(kernal, H);
	psf2otf(psfInv, H1);

	CvScalar s1, s2;
	IplImage* g1 = cvCreateImage(size, IPL_DEPTH_64F, 1);
	IplImage* g2 = cvCreateImage(size, IPL_DEPTH_64F, 1);
	IplImage* temp = cvCreateImage(size, IPL_DEPTH_64F, 1);
	IplImage* CC_Re = cvCreateImage(size, IPL_DEPTH_64F, 1);
	IplImage* CC_Im = cvCreateImage(size, IPL_DEPTH_64F, 1);


	for (int i = 0; i < num; i++)
	{
		if (i>1)
		{
			cvMul(L1, L2, g1);
			cvMul(L2, L2, g2);
			s1=cvSum(g1);
			s2=cvSum(g2);
			lambda = (double)(s1.val[0] / max(s2.val[0], EPS));
			lambda = max(min(lambda, (double)(1.0)), (double)(0));
			//printf("lambda=%f\n", lambda);
		}
		//Y = J2 + lambda*(J2 - J3);
		cvSub(J2, J3, temp);
		cvScale(temp, temp, lambda, 0);
		cvAdd(J2,temp,Y);
		cvMaxS(Y, 0, Y);

		corelucy(Y, H, J1,CC);

		cvMulSpectrums(CC, H1, CC, CV_DXT_ROWS);
		cvDFT(CC, CC, CV_DXT_INV_SCALE, CC->height);
		cvSplit(CC, CC_Re, CC_Im,0,0);

		cvCopy(J2,J3,0);
		cvMul(CC_Re, Y,J2);

		cvCopy(L1, L2);
		cvSub(J2,Y,L1);
	}
	cvCopy(J2, output);
	cvMaxS(output, 0, output);
	cvMinS(output, 255, output);//越界处理

	//释放内存空间
	cvReleaseMat(&H);
	cvReleaseMat(&H1);
	cvReleaseMat(&CC);

	cvReleaseImage(&input1);
	cvReleaseImage(&psfInv);
	cvReleaseImage(&g);
	cvReleaseImage(&J1);
	cvReleaseImage(&J2);
	cvReleaseImage(&J3);
	cvReleaseImage(&L1);
	cvReleaseImage(&L2);
	cvReleaseImage(&Y);
	cvReleaseImage(&g1);
	cvReleaseImage(&g2);
	cvReleaseImage(&temp);
	cvReleaseImage(&CC_Re);
	cvReleaseImage(&CC_Im);
	cvReleaseImage(&image_char);
}

/************************************************************************
* @函数名称：
*	corelucy()
* @输入参数:
*	IplImage* input_Y		- 输入图像
*	CvMat* input_H          - 输入经过FFT的2通道模糊核矩阵
*	IplImage* input_g		- 输入模糊图像
* @输出：
*   CvMat* output			- 输出2通道矩阵
* @说明:
*	 RL算法核心迭代步骤
*	CC=FFT(g(k)/conv(h,f(k)))
************************************************************************/
void corelucy(IplImage* input_Y, CvMat* input_H, IplImage* input_g,CvMat* output)
{
	CvMat* YF = cvCreateMat(input_Y->width, input_Y->height, CV_64FC2);
	IplImage* image_Re = cvCreateImage(cvGetSize(input_Y), IPL_DEPTH_64F, 1);
	IplImage* image_Im = cvCreateImage(cvGetSize(input_Y), IPL_DEPTH_64F, 1);

	DFT2(input_Y, YF, 1);
	cvMulSpectrums(YF, input_H, YF, CV_DXT_ROWS);
	cvDFT(YF, YF, CV_DXT_INV_SCALE, YF->height);
	cvSplit(YF, image_Re, image_Im,0,0);
	cvMaxS(image_Re, EPS, image_Re);
	cvDiv(input_g, image_Re, image_Re, 1);
	DFT2(image_Re, output, 1);
	
	cvReleaseMat(&YF);
	cvReleaseImage(&image_Re);
	cvReleaseImage(&image_Im);
}


/************************************************************************
* @函数名称：
*	deconvTV()
* @输入参数:
*	IplImage* input_Y		- 输入图像
*	CvMat* input_H          - 输入经过FFT的2通道模糊核矩阵
*	IplImage* input_g		- 输入模糊图像
* @输出：
*   CvMat* output			- 输出2通道矩阵
* @说明:
*	 RL算法核心迭代步骤
*	CC=FFT(g(k)/conv(h,f(k)))
************************************************************************/
double autocorrLen(IplImage* input, double angle)
{
	int i, j;
	double len = 0;
	CvSize size = cvGetSize(input);

	//旋转图像，便于检测模糊长度
	IplImage* input_rotate = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage* output_rotate = cvCreateImage(size, IPL_DEPTH_8U, 1);
	IplImage* corr_image = cvCreateImage(size, IPL_DEPTH_64F, 1);
	IplImage* row_image = cvCreateImage(cvSize(size.width, 1), IPL_DEPTH_64F, 1);

	CvMat* input_real = cvCreateMat(1, size.width, CV_64FC1);
	CvMat* input_imaginary = cvCreateMat(1, size.width, CV_64FC1);

	CvMat* dft = cvCreateMat(1, size.width, CV_64FC2);
	CvMat tmp;
	CvMat* ptmp;

	cvScale(input, input_rotate, 1, 0);
	imRotate(input_rotate, output_rotate, -angle);

	cvZero(corr_image);
	cvZero(row_image);

	
	//计算行差分
	for (i = 0; i<size.height; i++)
	{
		uchar* prvalue = (uchar*)(output_rotate->imageData + i*output_rotate->widthStep);
		double* pcvalue = (double*)(corr_image->imageData + i*corr_image->widthStep);
		for (j = 0; j<size.width - 1; j++)
		{
			pcvalue[j] = abs(prvalue[j + 1] - prvalue[j]);
		}
	}

	//计算各行自相关,通过傅里叶变换实现
	for (i = 0; i<size.height; i++)
	{	
		cvGetRow(corr_image, &tmp,i);
		cvCopy(&tmp, input_real);
		cvZero(input_imaginary);
		cvMerge(input_real, input_imaginary, 0, 0, dft);//构造2通道存储空间
		cvDFT(dft, dft, CV_DXT_FORWARD);//傅里叶变换
		cvMulSpectrums(dft, dft, dft, CV_DXT_MUL_CONJ);//频域共轭相乘
		cvDFT(dft, dft, CV_DXT_INV_SCALE);//傅里叶逆变换，完成自相关
		cvSplit(dft, input_real, input_imaginary, 0, 0);//取实部
		cvNormalize(input_real, input_real, 1, 0, CV_MINMAX);
		cvCopy(input_real, &tmp);
	}

	//计算列和
	for (i = 0; i<size.height; i++)
	{
		double* pcvalue = (double*)(corr_image->imageData + i*corr_image->widthStep);
		double* prvalue = (double*)(row_image->imageData);
		for (j = 0; j < size.width; j++)
		{
			prvalue[j] += pcvalue[j];	//计算列和
	//		printf("%f ", pcvalue[j]);
		}
//		printf("\n\n ");
	}


	//得到最大值坐标后，向一侧遍历，求第一个极小值位置
	for (i = 1; i<(size.width/2-1); i++)
	{
		double* prvalue = (double*)(row_image->imageData);
		if ((prvalue[i] < prvalue[i - 1]) && (prvalue[i] < prvalue[i + 1]))
		{
			len = (double)i;
			break;
		}
	}
	////测试代码
	//double* prvalue = (double*)(row_image->imageData);
	//for (j = 0; j <size.width; j++)
	//{
	//	printf("%f ", prvalue[j]);
	//}

	cvReleaseMat(&input_real);
	cvReleaseMat(&input_imaginary);
	cvReleaseMat(&dft);

	cvReleaseImage(&input_rotate);
	cvReleaseImage(&output_rotate);
	cvReleaseImage(&corr_image);
	cvReleaseImage(&row_image);

	return len;
}

/************************************************************************
* @函数名称：
*	deconvTV()
* @输入参数:
*	IplImage* input_Y		- 输入图像
*	CvMat* input_H          - 输入经过FFT的2通道模糊核矩阵
*	IplImage* input_g		- 输入模糊图像
* @输出：
*   CvMat* output			- 输出2通道矩阵
* @说明:
*	 RL算法核心迭代步骤
*	CC=FFT(g(k)/conv(h,f(k)))
************************************************************************/
//void deconvTV()
//{
//
//}



/************************************************************************
* @函数名称：
*	blindEstKernel()
* @输入参数:
*	IplImage* blur				- 输入模糊图像
* @输出：
*   IplImage* psf    			- 输出模糊核,为单通道浮点类型
* @说明:
*	 归一化L1正则化盲复原估计模糊核,优化函数为：
*	argmin_{x,k}lambda/2 |y-x\oplusk|^2+|x|_1/|x|_2+k_reg_wt*|k|_1
*	f为图像高频分量的估计
************************************************************************/
//IplImage* blindEstKernel(const IplImage* blur,const int psf_max_size)
//{
//	CvSize isize_max = cvGetSize(blur);//获取图像尺寸
//	
//	int psize_max = psf_max_size;//获取最大模糊核尺寸
//	int i = 0, j = 0;
//	int x_in_iter = 2;//估计f时内/外迭代次数
//	int x_out_iter = 2;
//	int xk_iter = 21;//每层交替求解f/k的次数，这个参数需在性能和效率之间折中
//
//	double lambda_min = 100;//似然项权重，噪声较大时用较小的值，减少这个参数会使模糊核估计更粗，增大则更细
//	double delta = 0.001;//ISTA算法更新步长，增加会导致离散，减少太多会导致收敛变慢
//	double k_reg_wt = 0.1;//k的正则化权重
//
//	IplImage* psf = cvCreateImage(cvSize(psf_max_size, psf_max_size), IPL_DEPTH_64F, 1);
//	CvMat* g = cvCreateMat(isize_max.height, isize_max.width, CV_64FC1);
//	cvScale(blur, g, 1, 0);//将图像转换为浮点矩阵，方便计算
//
//	int psize_min = max(3, 2 * cvFloor((psize_max - 1) / 16) + 1);//最粗层模糊核尺寸
//	double resize_step = sqrt(2);//尺度缩放步长
//
//	//确定多尺度框架的层数
//	int num_scales = 1;
//	int NUM = 0;
//	int tmp = psize_min;
//
//	while (tmp < psize_max)
//	{	
//		num_scales++;
//		tmp = cvCeil(tmp*resize_step);
//		if (tmp % 2 == 0)
//		{
//			tmp++;
//		}
//	}
//
//	NUM = num_scales;
//	int* ksize = (int*)malloc(NUM *sizeof(int));//分配不同尺度模糊核尺寸存放空间
//	CvSize* ysize = (CvSize*)malloc(NUM *sizeof(CvSize));//分配不同尺度图像尺寸存放空间
//	double* lambda = (double*)malloc(NUM *sizeof(double));//分配不同尺度似然项参数存放空间
//
//	lambda[0] = lambda_min;
//	num_scales = 0;
//	tmp = psize_min;
//	//将每个尺度的模糊核尺寸存放到ksize数组
//	while (tmp < psize_max)
//	{
//		ksize[num_scales] = tmp;
//		ysize[num_scales].height = cvFloor(((double)tmp / psize_max)*isize_max.height);
//		ysize[num_scales].width = cvFloor(((double)tmp / psize_max)*isize_max.width);
//
//		num_scales++;
//		tmp = cvCeil(tmp*resize_step);
//		if (tmp % 2 == 0)
//		{
//			tmp++;
//		}
//	}
//	ksize[num_scales] = psize_max;
//	ysize[num_scales].height = isize_max.height;
//	ysize[num_scales].width = isize_max.width;
//
//	//分配各层次的模糊核矩阵空间
//	CvMat* k0 = cvCreateMat(3, 3, CV_64FC1);
//	CvMat* k1 = cvCreateMat(5, 5, CV_64FC1);
//	CvMat* k2 = cvCreateMat(9, 9, CV_64FC1);
//	CvMat* k3 = cvCreateMat(13, 13, CV_64FC1);
//	CvMat* k4 = cvCreateMat(19, 19, CV_64FC1);
//	CvMat* k5 = cvCreateMat(25, 25, CV_64FC1);
//	//分配各层次的模糊图像矩阵空间
//	CvMat* y0 = cvCreateMat(36, 45, CV_64FC1);
//	CvMat* y1 = cvCreateMat(60, 76, CV_64FC1);
//	CvMat* y2 = cvCreateMat(109, 137, CV_64FC1);
//	CvMat* y3 = cvCreateMat(158, 198, CV_64FC1);
//	CvMat* y4 = cvCreateMat(231, 289, CV_64FC1);
//	CvMat* y5 = cvCreateMat(256, 320, CV_64FC1);
//	//分配各层次的估计图像矩阵空间
//	CvMat* xx0 = cvCreateMat(36, 45, CV_64FC1);
//	CvMat* xx1 = cvCreateMat(60, 76, CV_64FC1);
//	CvMat* xx2 = cvCreateMat(109, 137, CV_64FC1);
//	CvMat* xx3 = cvCreateMat(158, 198, CV_64FC1);
//	CvMat* xx4 = cvCreateMat(231, 289, CV_64FC1);
//	CvMat* xx5 = cvCreateMat(256, 320, CV_64FC1);
//	CvMat* xy0 = cvCreateMat(36, 45, CV_64FC1);
//	CvMat* xy1 = cvCreateMat(60, 76, CV_64FC1);
//	CvMat* xy2 = cvCreateMat(109, 137, CV_64FC1);
//	CvMat* xy3 = cvCreateMat(158, 198, CV_64FC1);
//	CvMat* xy4 = cvCreateMat(231, 289, CV_64FC1);
//	CvMat* xy5 = cvCreateMat(256, 320, CV_64FC1);
//
//	cvZero(k0);
//	cvZero(k1);
//	cvZero(k2);
//	cvZero(k3);
//	cvZero(k4);
//	cvZero(k5);
//
//	cvZero(y0);
//	cvZero(y1);
//	cvZero(y2);
//	cvZero(y3);
//	cvZero(y4);
//	cvZero(y5);
//
//	cvZero(xx0);
//	cvZero(xx1);
//	cvZero(xx2);
//	cvZero(xx3);
//	cvZero(xx4);
//	cvZero(xx5);
//
//	cvZero(xy0);
//	cvZero(xy1);
//	cvZero(xy2);
//	cvZero(xy3);
//	cvZero(xy4);
//	cvZero(xy5);
//
//	CvMat* k[6] = { k0, k1, k2, k3, k4, k5 };
//	CvMat* y[6] = { y0, y1, y2, y3, y4, y5 };
//	CvMat* xx[6] = { xx0, xx1, xx2, xx3, xx4, xx5 };
//	CvMat* xy[6] = { xy0, xy1, xy2, xy3, xy4, xy5 };
//
//
//
//	//对模糊图像进行多尺度处理
//	for (int s = 0; s < NUM; s++)
//	{
//		//分配梯度图像的存储空间
//		CvMat* yx = cvCreateMat(ysize[s].height, ysize[s].width,CV_64FC1);
//		CvMat* yy = cvCreateMat(ysize[s].height, ysize[s].width, CV_64FC1);
//		cvZero(yx);
//		cvZero(yy);
//
//		//初始化模糊核
//		if (s == 0)
//		{
//			double* pvalue = (double *)(k[0]->data.ptr + (ksize[0]-1)/2*k0->step);
//			for (j = (ksize[0] - 1) / 2 ; j < (ksize[0] - 1) / 2 + 1; j++)
//			{
//				pvalue[j] = 0.5;
//			}
//		}
//		//从前一层次模糊核得到当前层
//		else
//		{
//			cvResize(k[s-1], k[s], CV_INTER_LINEAR);//模糊核双线性插值上采样
//			cvMaxS(k[s], 0, k[s]);//保证非负
//			CvScalar sum;
//			sum = cvSum(k[s]);
//			cvScale(k[s], k[s], 1 / sum.val[0], 0);//归一化
//		}
//		printf("scales:%d", s);
//		printf("[%d %d %d]\n", ksize[s], ysize[s].height, ysize[s].width);
//
//		cvResize(g, y[s], CV_INTER_LINEAR);//模糊图像双线性插值下采样
//		
//		/*计算水平梯度图像*/
//		for (i = 0; i<ysize[s].height; i++)
//		{
//			double* ys_data = (double*)(y[s]->data.ptr + i*y[s]->step);
//			double* yxs_data = (double*)(yx->data.ptr + i*yx->step);
//			for (j = 0; j<ysize[s].width - 1; j++)
//			{
//				yxs_data[j] = abs(ys_data[j + 1] - ys_data[j]);
//			}
//		}
//		/*计算垂直梯度图像*/
//		for (i = 0; i<ysize[s].width; i++)
//		{
//			for (j = 0; j<ysize[s].height - 1; j++)
//			{
//				double* ys_data = (double*)(y[s]->data.ptr + j*y[s]->step);
//				double* ysp_data = (double*)(y[s]->data.ptr + (j+1)*y[s]->step);
//				double* yys_data = (double*)(yy->data.ptr + j*yy->step);
//
//				yys_data[i] = abs(ysp_data[i] - ys_data[i]);
//			}
//		}
//
//		double l2norm = 6;
//		cvScale(yx, yx, l2norm / cvNorm(yx, 0, CV_L2), 0);
//		cvScale(yy, yy, l2norm / cvNorm(yy, 0, CV_L2), 0);
//
//		//初始化估计图像
//		if (s == 0)
//		{
//			cvCopy(yx, xx0);
//			cvCopy(yy, xy0);
//		}
//		//从上次估计图像中上采样
//		else
//		{
//			cvResize(xx[s - 1], xx[s], CV_INTER_LINEAR);//估计图像的双线性插值上采样
//			cvResize(xy[s - 1], xy[s], CV_INTER_LINEAR);//估计图像的双线性插值上采样
//		}
//
//		cvScale(xx[s], xx[s], l2norm / cvNorm(xx[s], 0, CV_L2), 0);
//		cvScale(xy[s], xy[s], l2norm / cvNorm(xy[s], 0, CV_L2), 0);
//
//		coreBlindEstKernel(k[s], xx[s], xy[s], yx, yy, 100, delta, k_reg_wt, x_in_iter, x_out_iter, xk_iter);
//		cvReleaseMat(&yx);
//		cvReleaseMat(&yy);
//	}
//	cvScale(k[5], kernel, 1, 0);
//	//释放动态数组
//	free(ksize);
//	free(ysize);
//	free(lambda);
//	//释放矩阵
//	cvReleaseMat(&k0);
//	cvReleaseMat(&k1);
//	cvReleaseMat(&k2);
//	cvReleaseMat(&k3);
//	cvReleaseMat(&k4);
//	cvReleaseMat(&k5);
//	cvReleaseMat(&y0);
//	cvReleaseMat(&y1);
//	cvReleaseMat(&y2);
//	cvReleaseMat(&y3);
//	cvReleaseMat(&y4);
//	cvReleaseMat(&y5);
//	cvReleaseMat(&xx0);
//	cvReleaseMat(&xx1);
//	cvReleaseMat(&xx2);
//	cvReleaseMat(&xx3);
//	cvReleaseMat(&xx4);
//	cvReleaseMat(&xx5);
//	cvReleaseMat(&xy0);
//	cvReleaseMat(&xy1);
//	cvReleaseMat(&xy2);
//	cvReleaseMat(&xy3);
//	cvReleaseMat(&xy4);
//	cvReleaseMat(&xy5);
//
//	return psf;
//}


IplImage* blindEstKernel(const IplImage* blur, const int psf_max_size)
{
	CvSize isize_max = cvGetSize(blur);//获取图像尺寸

	int psize_max = psf_max_size;//获取最大模糊核尺寸
	int i = 0, j = 0;
	int x_in_iter = 2;//估计f时内/外迭代次数
	int x_out_iter = 2;
	int xk_iter = 21;//每层交替求解f/k的次数，这个参数需在性能和效率之间折中

	double lambda_min = 100;//似然项权重，噪声较大时用较小的值，减少这个参数会使模糊核估计更粗，增大则更细
	double delta = 0.001;//ISTA算法更新步长，增加会导致离散，减少太多会导致收敛变慢
	double k_reg_wt = 0.1;//k的正则化权重

	IplImage* psf = cvCreateImage(cvSize(psf_max_size, psf_max_size), IPL_DEPTH_64F, 1);
	CvMat* g = cvCreateMat(isize_max.height, isize_max.width, CV_64FC1);
	cvScale(blur, g, 1.0/255, 0);//将图像转换为浮点矩阵并缩放至[0 1]，方便计算
	
	int psize_min = max(3, 2 * cvFloor((psize_max - 1) / 16) + 1);//最粗层模糊核尺寸
	double resize_step = sqrt(2);//尺度缩放步长

	//确定多尺度框架的层数
	int num_scales = 1;
	int NUM = 0;
	int tmp = psize_min;

	while (tmp < psize_max)
	{
		num_scales++;
		tmp = cvCeil(tmp*resize_step);
		if (tmp % 2 == 0)
		{
			tmp++;
		}
	}

	NUM = num_scales;
	int* ksize = (int*)malloc(NUM *sizeof(int));//分配不同尺度模糊核尺寸存放空间
	double* lambda = (double*)malloc(NUM *sizeof(double));//分配不同尺度似然项参数存放空间

	CvSize* ysize = (CvSize*)malloc(NUM *sizeof(CvSize));//分配不同尺度图像尺寸存放空间
	CvMat** k = (CvMat**)malloc(NUM *sizeof(CvMat*));//分配不同尺度模糊核存放空间
	CvMat** y = (CvMat**)malloc(NUM *sizeof(CvMat*));//分配不同尺度模糊图像存放空间
	CvMat** xx = (CvMat**)malloc(NUM *sizeof(CvMat*));//分配不同尺度清晰图像的水平梯度图像存放空间
	CvMat** xy = (CvMat**)malloc(NUM *sizeof(CvMat*));//分配不同尺度清晰图像的垂直梯度图像存放空间



	lambda[0] = lambda_min;
	num_scales = 0;
	tmp = psize_min;
	//将每个尺度的模糊核尺寸存放到ksize数组
	while (tmp < psize_max)
	{
		ksize[num_scales] = tmp;
		ysize[num_scales].height = cvFloor(((double)tmp / psize_max)*isize_max.height);
		ysize[num_scales].width = cvFloor(((double)tmp / psize_max)*isize_max.width);

		num_scales++;
		tmp = cvCeil(tmp*resize_step);
		if (tmp % 2 == 0)
		{
			tmp++;
		}
	}
	ksize[num_scales] = psize_max;
	ysize[num_scales].height = isize_max.height;
	ysize[num_scales].width = isize_max.width;

	for (i = 0; i < NUM; i++)
	{
		k[i] = cvCreateMat(ksize[i], ksize[i], CV_64FC1);//分配各层次的模糊核矩阵空间
		y[i] = cvCreateMat(ysize[i].height,ysize[i].width, CV_64FC1);//分配各层次的模糊图像矩阵空间
		xx[i] = cvCreateMat(ysize[i].height-1, ysize[i].width-1, CV_64FC1);//分配各层次的清晰图像的水平梯度矩阵空间
		xy[i] = cvCreateMat(ysize[i].height-1, ysize[i].width-1, CV_64FC1);//分配各层次的清晰图像的垂直梯度矩阵空间
		/*初始化各种存储空间*/
		cvZero(k[i]);
		cvZero(y[i]);
		cvZero(xx[i]);
		cvZero(xy[i]);
	}

	//printf("g=");
	//for (int i = 0; i < 2; i++)
	//{
	//	double* pvalue = (double*)(g->data.ptr + i*g->step);
	//	for (int j = 0; j < g->cols; j++)
	//	{
	//		printf("%f ", pvalue[j]);
	//	}
	//	printf("\n");
	//}
	//对模糊图像进行多尺度处理
	for (int s = 0; s < NUM; s++)
	{
		//分配梯度图像的存储空间
		CvMat* yx = cvCreateMat(ysize[s].height-1, ysize[s].width-1, CV_64FC1);
		CvMat* yy = cvCreateMat(ysize[s].height-1, ysize[s].width-1, CV_64FC1);
		cvZero(yx);
		cvZero(yy);

		//初始化模糊核
		if (s == 0)
		{
			double* pvalue = (double *)(k[0]->data.ptr + ((ksize[0] - 1) / 2-1) * k[0]->step);
			for (j = (ksize[0] - 1) / 2 - 1; j < (ksize[0] - 1) / 2+1 ; j++)
			{
				pvalue[j] = 0.5;
			}
		}

		//从前一层次模糊核得到当前层
		else
		{
			cvResize(k[s - 1], k[s], CV_INTER_LINEAR);//模糊核双线性插值上采样
			cvMaxS(k[s], 0, k[s]);//保证非负
			CvScalar sum;
			sum = cvSum(k[s]);
			cvScale(k[s], k[s], 1 / sum.val[0], 0);//归一化
		}
		printf("scales:%d", s);
		printf("[%d %d %d]\n", ksize[s], ysize[s].height, ysize[s].width);

		cvResize(g, y[s], CV_INTER_LINEAR);//模糊图像双线性插值下采样

		/*计算梯度图像*/
		double dx[4] = { -1, 1, 0, 0 };
		double dy[4] = { -1, 0, 1, 0 };

		CvMat dxx = cvMat(2, 2, CV_64FC1, dx);
		CvMat dyy = cvMat(2, 2, CV_64FC1, dy);
		
		yx = cvMatDftConv2(y[s], &dxx, CONVOLUTION_VALID);
		yy = cvMatDftConv2(y[s], &dyy, CONVOLUTION_VALID);
		//归一化
		double l2norm = 6;
		printf("yxn=%f\n", cvNorm(yx, 0, CV_L2));
		printf("yyn=%f\n", cvNorm(yy, 0, CV_L2));

		cvScale(yx, yx, l2norm / cvNorm(yx, 0, CV_L2), 0);
		cvScale(yy, yy, l2norm / cvNorm(yy, 0, CV_L2), 0);

		//初始化估计图像
		if (s == 0)
		{
			cvCopy(yx, xx[0]);
			cvCopy(yy, xy[0]);
		}
		//从上次估计图像中上采样
		else
		{
			cvResize(xx[s - 1], xx[s], CV_INTER_LINEAR);//估计图像的双线性插值上采样
			cvResize(xy[s - 1], xy[s], CV_INTER_LINEAR);//估计图像的双线性插值上采样
		}

		cvScale(xx[s], xx[s], l2norm / cvNorm(xx[s], 0, CV_L2), 0);
		cvScale(xy[s], xy[s], l2norm / cvNorm(xy[s], 0, CV_L2), 0);


		//printf("xx=");
		//for (int i = 0; i < k[0]->rows; i++)
		//{
		//	double* pvalue = (double*)(xx[0]->data.ptr + i*xx[0]->step);
		//	for (int j = 0; j < xx[0]->cols; j++)
		//	{
		//		printf("%f ", pvalue[j]);
		//	}
		//	printf("\n");
		//}

		coreBlindEstKernel(k[s], xx[s], xy[s], yx, yy, 100, delta, k_reg_wt, x_in_iter, x_out_iter, xk_iter);
		cvReleaseMat(&yx);
		cvReleaseMat(&yy);
	}
	cvScale(k[NUM-1], psf, 1, 0);

	//释放动态数组
	free(ksize);
	free(ysize);
	free(lambda);
	free(k);
	free(y);
	free(xx);
	free(xy);

	return psf;
}

/************************************************************************
* @函数名称：
*	coreBlindEstKernel()
* @输入参数:
*	CvMat* ks				- 上次估计的模糊核k经过上采样得到的矩阵
*	CvMat* xxs				- 当前估计的x方向梯度图像矩阵
*	CvMat* xys				- 当前估计的y方向梯度图像矩阵
*	CvMat* yxs				- 当前尺度模糊图像的水平梯度矩阵
*	CvMat* yys				- 当前尺度模糊图像的垂直梯度矩阵
*	double lambdas			- 似然项的正则化参数
*	double delta			- 迭代停止条件
*	double k_reg_wt			- 对模糊核的正则化参数，一般取0.01-1之间
*	double x_in_iter		- 优化x时的内部迭代次数
*	double x_out_iter		- 优化x时的外部迭代次数
*	double xk_iter			- x/k交替优化次数
* @输出：
*	CvMat* ks				- 当前层估计出的模糊核
*	CvMat* xxs				- 当前层估计出的x方向梯度图像矩阵
*	CvMat* xys				- 当前层估计出的y方向梯度图像矩阵
* @说明:
*	 但尺度模糊核估计核心优化算法
************************************************************************/
void coreBlindEstKernel(CvMat* ks, CvMat* xxs, CvMat*xys, CvMat* yxs, CvMat* yys, double lambdas,
						double delta, double k_reg_wt, int x_in_iter, int x_out_iter, int xk_iter)
{
	int i = 0, j = 0,p=0;
	int skip_rest = 1;//标志位，如果该变量为零，则停止循环
	int totiter = 0;//总迭代次数
	int totiter_before_x = 0;
	double lambda = lambdas;
	double delta_iter = delta;
	double lcost[10000] = { 0 };//似然项代价变量
	double pcost[10000] = { 0 };//正则项代价变量
	double norm2xx = 0;
	double norm2xy = 0;
	double betax = 0;
	double betay = 0;
	double cost_before_x = 0;
	double cost_after_x = 0;
	double pcg_tol = 0.0001;
	double pcg_its = 1;

	CvSize isize = cvGetSize(yxs);

	CvMat* tmp1 = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* tmp2 = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* temp1 = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* temp2 = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* xx2= cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* xy2 = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* kt = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* kprev = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* vx = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* vy = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* xxprev = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* xyprev = cvCreateMat(isize.height, isize.width, CV_64FC1);

	//计算y - x \oplus k
	cvFilter2D(xxs, tmp1, ks);
	cvFilter2D(xys, tmp2, ks);
	cvSub(tmp1, yxs, tmp1);
	cvSub(tmp2, yys, tmp2);
	//测试
	double a = cvNorm(tmp1, 0, CV_L2);
	double b = cvNorm(tmp2, 0, CV_L2);

	lcost[totiter] = (lambdas / 2)*(cvNorm(tmp1, 0, CV_L2)*cvNorm(tmp1, 0, CV_L2) + cvNorm(tmp2, 0, CV_L2)*cvNorm(tmp2, 0, CV_L2));
	pcost[totiter] = cvNorm(xxs, 0, CV_L1) / cvNorm(xxs, 0, CV_L2) + cvNorm(xys, 0, CV_L1) / cvNorm(xys, 0, CV_L2);

	//x、k交替迭代
	for (i = 0; i < xk_iter; i++)
	{
		lambda = lambdas; // / (1.15 ^ (xk_iter - iter)); 
		totiter_before_x = totiter;
		cost_before_x = lcost[totiter] + pcost[totiter];//在更新x前的迭代次数和函数代价

		cvCopy(xxs, xx2);
		cvCopy(xys, xy2);//备份

		while (delta_iter>0.0001)
		{
			for (j = 0; j < x_out_iter; j++)
			{
				if (skip_rest == 0)
				{
					break;
				}

				norm2xx = cvNorm(xxs, 0, CV_L2);
				norm2xy = cvNorm(xys, 0, CV_L2);
				betax = lambda*norm2xx;
				betay = lambda*norm2xy;//将正则项的L2范数通过参数beta体现

				for (p = 0; p < x_in_iter; p++)
				{
					if (skip_rest == 0)
					{
						break;
					}
					totiter++;
					xxprev = xxs;
					xyprev = xys;
					//采用ISTA算法求解该层次的估计图像xxs和xys
					cvFilter2D(xxprev, tmp1, ks);
					cvFilter2D(xyprev, tmp2, ks);
					cvSub(yxs, tmp1, tmp1);
					cvSub(yys, tmp2, tmp2);
			
					cvFlip(ks, kt, -1);
					cvFilter2D(tmp1, tmp1, kt);
					cvFilter2D(tmp2, tmp2, kt);

					cvScale(tmp1, vx, betax*delta_iter, 0);
					cvScale(tmp2, vy, betay*delta_iter, 0);

					cvAdd(xxprev, vx, vx);
					cvAdd(xyprev, vy, vy);

					//收缩操作，求解下一个xxs/xys
					cvAbs(vx, vx);
					cvAbs(vy, vy);
	
					CvScalar s1= cvScalar(delta_iter,0, 0, 0);
					cvSubS(vx, s1, tmp1);
					cvSubS(vy, s1, tmp2);

					cvSign(vx, temp1);
					cvSign(vy, temp2);

					cvMaxS(tmp1, 0, tmp1);
					cvMaxS(tmp2, 0, tmp2);

					cvMul(tmp1, temp1, xxs);
					cvMul(tmp2, temp2, xys);
					double a = cvNorm(xxs, 0, CV_L2);
					double b = cvNorm(xys, 0, CV_L2);

					//计算y - x \oplus k
					cvFilter2D(xxs, tmp1, ks);
					cvFilter2D(xys, tmp2, ks);
					cvSub(tmp1, yxs, tmp1);
					cvSub(tmp2, yys, tmp2);
					//计算新的代价
					//测试
					double c = cvNorm(tmp1, 0, CV_L2);
					double d = cvNorm(tmp2, 0, CV_L2);


					lcost[totiter] = (lambdas / 2)*(cvNorm(tmp1, 0, CV_L2)*cvNorm(tmp1, 0, CV_L2) + cvNorm(tmp2, 0, CV_L2)*cvNorm(tmp2, 0, CV_L2));
					pcost[totiter] = cvNorm(xxs, 0, CV_L1) / cvNorm(xxs, 0, CV_L2) + cvNorm(xys, 0, CV_L1) / cvNorm(xys, 0, CV_L2);
				}
			}
			cost_after_x = lcost[totiter] + pcost[totiter];//在更新x后的函数代价
			if (cost_after_x > 3 * cost_before_x)//如果代价反而增大很多，则减小delta，重新迭代
			{
				totiter = totiter_before_x;
				cvCopy(xx2, xxs);
				cvCopy(xy2, xys);
				delta_iter = delta_iter / 2;
			}
			else
				break;
		}

		//开始更新k
		cvCopy(ks, kprev);
		pcgKernelIRLS(ks, xxs, xys, yxs, yys, pcg_tol, pcg_its, k_reg_wt);
	}

	cvReleaseMat(&tmp1);
	cvReleaseMat(&tmp2);
	cvReleaseMat(&temp1);
	cvReleaseMat(&temp2);
	cvReleaseMat(&kprev);
	cvReleaseMat(&xx2);
	cvReleaseMat(&xy2);
	cvReleaseMat(&kt);
	cvReleaseMat(&kprev);
	cvReleaseMat(&vx);
	cvReleaseMat(&vy);
	cvReleaseMat(&xxprev);
	cvReleaseMat(&xyprev);
}


/************************************************************************
* @函数名称：
*	cvSign()
* @输入参数:
*	CvMat* input				- 输入图像矩阵
* @输出：
*   CvMat* output			    - 输出符号矩阵
* @说明:
*	 获得矩阵的对应位置的符号
************************************************************************/
void cvSign(CvMat* input, CvMat* output)
{
	int i = 0, j = 0;
	for (i = 0; i < input->height; i++)
	{
		double* pivalue = (double*)(input->data.ptr + i*input->step);
		double* povalue = (double*)(output->data.ptr + i*output->step);

		for (j = 0; j < input->cols;j++)
		{
			if (pivalue[j]>0)
			{
				povalue[j] = 1;
			}
			if (pivalue[j] == 0)
			{
				povalue[j] = 0;
			}
			povalue[j] = -1;
		}
	}
}


/************************************************************************
* @函数名称：
*	pcgKernelIRLS()
* @输入参数:
*	CvMat* ks				- 上次估计的模糊核k矩阵
*	CvMat* xxs				- 当前估计的x方向梯度图像矩阵
*	CvMat* xys				- 当前估计的y方向梯度图像矩阵
*	CvMat* yxs				- 当前尺度模糊图像的水平梯度矩阵
*	CvMat* yys				- 当前尺度模糊图像的垂直梯度矩阵
*	double pcg_tol			- PCG算法的停止条件
*	double pcg_its			- PCG算法内部迭代次数
*	double k_reg_wt			- 对模糊核的正则化参数，一般取0.01-1之间
* @输出：
*	CvMat* ks				- 本次估计的模糊核k矩阵
* @说明:
*	加权最小二乘算法求解模糊核
*	min 1/2\|Xk - Y\|^2 + \lambda \|k\|_1
************************************************************************/
void pcgKernelIRLS(CvMat* ks, CvMat* xxs, CvMat* xys, CvMat* yxs, CvMat* yys,
				   double pcg_tol, double pcg_its, double k_reg_wt)
{
	CvSize isize = cvGetSize(xxs);
	CvSize ksize = cvGetSize(ks);
	CvScalar s1 = cvScalarAll(0);

	int khs = cvFloor(ksize.height / 2);
	int i = 0, j = 0;
	int iter = 1;
	double lambda = k_reg_wt;
	double rho = 0;
	double rho_1 = 0;
	double beta = 0;
	double alpha = 0;

	CvMat* tempx= cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* tempy = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* xxt = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* xyt = cvCreateMat(isize.height, isize.width, CV_64FC1);
	CvMat* kprev = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* weight_l1 = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* ktemp = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* rhs = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* rhsx = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* rhsy = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* Ak = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* r = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* p = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* Ap = cvCreateMat(ks->rows, ks->cols, CV_64FC1);
	CvMat* yx2 = cvCreateMat(isize.height-ksize.height, isize.width-ksize.width, CV_64FC1);
	CvMat* yy2 = cvCreateMat(isize.height - ksize.height, isize.width - ksize.width, CV_64FC1);
	
	CvMat tmpx;
	CvMat tmpy;
	//计算X^T Y项，并将结果尺寸调整至模糊核尺寸
	cvGetSubRect(yxs, &tmpx, cvRect(khs, khs, isize.width - ksize.width, isize.height - ksize.height));
	cvGetSubRect(yys, &tmpy, cvRect(khs, khs, isize.width - ksize.width, isize.height - ksize.height));
	cvCopy(&tmpx, yx2);
	cvCopy(&tmpy, yy2);

	cvFlip(xxs, xxt, -1);
	cvFlip(xys, xyt, -1);//旋转图像180度，傅里叶变换后为共轭转置关系

	cvFilter2D(xxt, tempx, yx2);
	cvFilter2D(xyt, tempy, yy2);

	cvGetSubRect(tempx, &tmpx, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
	cvGetSubRect(tempy, &tmpy, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));

	cvCopy(&tmpx, rhsx);
	cvCopy(&tmpy, rhsy);
	cvAdd(rhsx, rhsy, rhs);//计算完成

	//外层循环
	for (i = 0; i < iter; i++)
	{
		cvCopy(ks, kprev);
		//计算加权迭代最小二乘算法的权重对角阵
		cvAbs(kprev, weight_l1);
		cvMaxS(weight_l1, 0.0001, weight_l1);
		cvPow(weight_l1, weight_l1, -1);//这里的指数为exp_a-2,exp_a为对应的a范数
		cvScale(weight_l1, weight_l1, lambda, 0);
		//预处理共轭梯度法求解X^T Xk+weight_l1 k=X^T Y=rhs
		//计算Xk
		cvFilter2D(xxs, tempx, kprev);
		cvFilter2D(xys, tempy, kprev);
		cvGetSubRect(tempx, &tmpx, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
		cvGetSubRect(tempx, &tmpy, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
		cvCopy(&tmpx, rhsx);
		cvCopy(&tmpy, rhsy);
		//计算X^T Xk
		cvFilter2D(xxt, tempx, rhsx);
		cvFilter2D(xyt, tempy, rhsy);
		cvGetSubRect(tempx, &tmpx, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
		cvGetSubRect(tempx, &tmpy, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
		cvCopy(&tmpx, rhsx);
		cvCopy(&tmpy, rhsy);
		//计算weight_l1 k
		cvMul(kprev, weight_l1, ktemp);

		cvAdd(rhsx, rhsy, Ak);
		cvAdd(Ak, ktemp, Ak);

		cvSub(rhs, Ak, r);

		//内层循环
		for (j = 0; j < pcg_its; j++)
		{
			cvMul(r, r, ktemp);
			s1=cvSum(ktemp);
			rho = s1.val[0];

			if (j>0)
			{
				beta = rho / rho_1;
				cvScale(p, p, beta, 0);
				cvAdd(p, r, p);
			}
			else
			{
				cvCopy(r, p);
			}
			//计算Ap
			//计算Xp
			cvFilter2D(xxs, tempx, p);
			cvFilter2D(xys, tempy, p);
			cvGetSubRect(tempx, &tmpx, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
			cvGetSubRect(tempx, &tmpy, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
			cvCopy(&tmpx, rhsx);
			cvCopy(&tmpy, rhsy);
			//计算X^T Xp
			cvFilter2D(xxt, tempx, rhsx);
			cvFilter2D(xyt, tempy, rhsy);
			cvGetSubRect(tempx, &tmpx, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
			cvGetSubRect(tempx, &tmpy, cvRect(cvFloor((isize.width - ksize.width) / 2), cvFloor((isize.height - ksize.height) / 2), ksize.width, ksize.height));
			cvCopy(&tmpx, rhsx);
			cvCopy(&tmpy, rhsy);
			//计算weight_l1 p
			cvMul(p, weight_l1, ktemp);

			cvAdd(rhsx, rhsy, Ap);
			cvAdd(Ap, ktemp, Ap);

			cvMul(p, Ap, ktemp);
			s1 = cvSum(ktemp);
			alpha = rho/s1.val[0];//alpha更新

			cvScale(p, ktemp, alpha, 0);
			cvAdd(ks, ktemp, ks);//更新k
			cvScale(Ap, ktemp, alpha, 0);
			cvSub(r, ktemp, r);//更新r,残差矩阵
			rho_1 = rho;

			if (rho < pcg_tol)
			{
				break;
			}
		}
	}

	cvReleaseMat(&tempx);
	cvReleaseMat(&tempy); 
	cvReleaseMat(&xxt); 
	cvReleaseMat(&xyt); 
	cvReleaseMat(&kprev);
	cvReleaseMat(&weight_l1);
	cvReleaseMat(&ktemp);
	cvReleaseMat(&rhs);
	cvReleaseMat(&rhsx);
	cvReleaseMat(&rhsy);
	cvReleaseMat(&Ak);
	cvReleaseMat(&r);
	cvReleaseMat(&p);
	cvReleaseMat(&Ap);
	cvReleaseMat(&yx2);
	cvReleaseMat(&yy2);
}


/************************************************************************
* @函数名称：
*	cvMatFilterConv2()
* @输入参数:
*	const CvMat* input				- 输入的矩阵
*	const CvMat* kernel				- 输入的卷积核
*	ConvolutionType type			- 卷积类型
* @输出：
*	CvMat* dest						- 卷积后得到的矩阵
* @说明:
*	用cvFilter2D实现指定类型的卷积操作
*	CONVOLUTION_FULL/CONVOLUTION_FULL/CONVOLUTION_FULL
************************************************************************/
CvMat* cvMatFilterConv2(const CvMat* input, const CvMat* kernel, ConvolutionType type)
{
	CvMat* dest=0;
	CvMat* tmp=0;
	CvMat* ikernel=0;
	CvMat* source=0;
	/*如果要求返回全部卷积结果或者输入矩阵大小，需要先进行零填充*/
	if ((CONVOLUTION_FULL == type) || (CONVOLUTION_SAME == type))
	{
		const int additionalRows = kernel->rows - 1, additionalCols = kernel->cols - 1;
		source = cvCreateMat(input->height + additionalRows, input->width + additionalCols, CV_64FC1);
		cvCopyMakeBorder(input, source, cvPoint(additionalCols / 2, additionalRows / 2), IPL_BORDER_CONSTANT, cvScalar(0));
	}
	else
	{
		source=cvCloneMat(input);
	}

	dest = cvCloneMat(source);
	ikernel = cvCloneMat(kernel);

	cvFlip(kernel, ikernel, -1);//需要翻转模糊核
	cvFilter2D(source, dest, ikernel);
	cvReleaseMat(&source);
	cvReleaseMat(&ikernel);

	/*如果要求只返回输入矩阵大小，需要对卷积结果进行剪裁，不能直接返回cvFilter2D结果，因为这个函数采用邻近插值而非零填充处理边界*/
	if (CONVOLUTION_SAME == type)
	{
		CvMat* dst = cvCreateMat(input->cols, input->rows, CV_64FC1);//为了能返回指针变量，必须手动开辟内存
		cvGetSubRect(dest, dst, cvRect(kernel->cols / 2, kernel->rows / 2, input->cols, input->rows));

		return dst;
	}

	/*如果要求只返回没有用到填充值的部分，需要对卷积结果进行剪裁*/
	if (CONVOLUTION_VALID == type)
	{
		CvMat* dst = cvCreateMat(input->rows - kernel->rows + 1, input->cols - kernel->cols + 1, CV_64FC1);
		dst = cvGetSubRect(dest, dst, cvRect(kernel->cols / 2, kernel->rows / 2, input->cols - kernel->cols + 1, input->rows - kernel->rows + 1));

		return dst;
	}
	return dest;
}

/************************************************************************
* @函数名称：
*	cvMatDftConv2()
* @输入参数:
*	const CvMat* input				- 输入的矩阵
*	const CvMat* kernel				- 输入的卷积核
*	ConvolutionType type			- 卷积类型
* @输出：
*	CvMat* dest						- 卷积后得到的矩阵
* @说明:
*	用DFT实现指定类型的卷积操作
*	CONVOLUTION_FULL/CONVOLUTION_FULL/CONVOLUTION_FULL
************************************************************************/
CvMat* cvMatDftConv2(const CvMat* input, const CvMat* kernel, ConvolutionType type)
{
	CvMat* dest = 0;
	CvMat* tmp = 0;
	CvMat* source = 0;

	dest = dftCoreConv2(input, kernel);
	/*默认返回全部卷积结果*/

	/*如果要求只返回输入矩阵大小，需要对卷积结果进行剪裁*/
	if (CONVOLUTION_SAME == type)
	{
		CvMat* dst = cvCreateMat(input->rows, input->cols, CV_64FC1);//为了能返回指针变量，必须手动开辟内存
		dst = cvGetSubRect(dest, dst, cvRect(kernel->cols / 2, kernel->rows / 2, input->cols, input->rows));//注意图像坐标轴

		return dst;
	}

	/*如果要求只返回没有用到填充值的部分，需要对卷积结果进行剪裁*/
	if (CONVOLUTION_VALID == type)
	{
		CvMat* dst = cvCreateMat(input->rows - kernel->rows + 1, input->cols - kernel->cols + 1, CV_64FC1);
		dst = cvGetSubRect(dest, dst, cvRect(kernel->cols - 1, kernel->rows - 1, input->cols - kernel->cols + 1, input->rows - kernel->rows + 1));

		return dst;
	}
	return dest;
}



/************************************************************************
* @函数名称：
*	dftCoreConv2()
* @输入参数:
*	const CvMat* input				- 输入的矩阵
*	const CvMat* kernel				- 输入的卷积核
*	ConvolutionType type			- 卷积类型
* @输出：
*	CvMat* dest						- 卷积后得到的矩阵
* @说明:
*	用FFT实现卷积,频域相乘,全尺寸输出
************************************************************************/
CvMat* dftCoreConv2(const CvMat* input,const CvMat* kernel)
{
	int dft_M = cvGetOptimalDFTSize(input->rows + kernel->rows - 1);
	int dft_N = cvGetOptimalDFTSize(input->cols + kernel->cols - 1);

	CvMat *dft_full = cvCreateMat(input->rows + kernel->rows - 1, input->cols + kernel->cols - 1, input->type);
	CvMat *dft_input = cvCreateMat(dft_M, dft_N, input->type);
	CvMat *dft_kernel = cvCreateMat(dft_M, dft_N, kernel->type);
	CvMat tmp;

	//输入矩阵零填充
	cvGetSubRect(dft_input, &tmp, cvRect(0, 0, input->cols, input->rows));
	cvCopy(input, &tmp);
	cvGetSubRect(dft_input, &tmp, cvRect(input->cols, 0, dft_input->cols - input->cols, input->rows));
	cvZero(&tmp);

	//只需要优化列数
	cvDFT(dft_input, dft_input, CV_DXT_FORWARD, input->rows);

	//输入卷积核零填充
	cvGetSubRect(dft_kernel, &tmp, cvRect(0, 0, kernel->cols, kernel->rows));
	cvCopy(kernel, &tmp);
	cvGetSubRect(dft_kernel, &tmp, cvRect(kernel->cols, 0, dft_kernel->cols - kernel->cols, kernel->rows));
	cvZero(&tmp);

	//只需要优化列数
	cvDFT(dft_kernel, dft_kernel, CV_DXT_FORWARD, kernel->rows);

	//复数相乘实现卷积或者用CV_DXT_MUL_CONJ来实现相关 
	cvMulSpectrums(dft_input, dft_kernel, dft_input, 0);

	//输出全部大小
	cvDFT(dft_input, dft_input, CV_DXT_INV_SCALE, dft_full->rows);
	cvGetSubRect(dft_input, &tmp, cvRect(0, 0, dft_full->cols, dft_full->rows));
	cvCopy(&tmp, dft_full);

	cvReleaseMat(&dft_input);
	cvReleaseMat(&dft_kernel);

	return dft_full;
}