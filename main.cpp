#include "basicOperation.h"
#include "assessment.h"
#include "restore.h"

int main(int argc, char* argv[])
{
	IplImage* input1;
	IplImage* dst;

	if ((input1 = cvLoadImage("redTree.png", 0)) == NULL)//keeper_256_320.jpg、lena_circular_10_50.bmp、keeper_256_320_10_50_blur.jpg 0-读灰度图，1-读彩色图，-1-读原通道
	{	
		printf("Input1 no image data ！\n");
		return -1;
	}

	CvMat* psf = cvCreateMat(21, 21, CV_64FC1);
	CvMat* image = cvCreateMat(input1->height,input1->width, CV_64FC1);
	CvMat* dest = cvCreateMat(input1->height, input1->width, CV_64FC1);

	IplImage* psf1 = cvCreateImage(cvSize(21, 21),IPL_DEPTH_32F,1);
	IplImage* out = cvCloneImage(input1);

	cvZero(psf1);

	int psf_max_size = 21;
	double t1 = (double)cvGetTickCount();
	uchar f = blurIdentify(input1);
	double INRSS = calINRSS(input1);

	printf("INRSS=%f\n", INRSS);

	//genaratePsf(psf1, 10, 50);
	//cvScale(psf1, psf, 1, 0);
	//cvScale(input1, image, 1, 0);

	//dest = deconvBregman(image, psf, 3000, 1);
	//cvScale(dest, out, 255, 0);
	//printf("out=");
	//for (int i = 0; i <3; i++)
	//{
	//	uchar* pvalue = (uchar*)(out->imageData + i*out->widthStep);
	//	for (int j = 0; j < out->width; j++)
	//	{
	//		printf("%d ", pvalue[j]);
	//	}
	//	printf("\n");
	//}

	//psf = blindEstKernel(input1, psf_max_size);
	//cvScale(psf, psf1, 1, 0);
//dst=deconvRL1(input1, psf,30);
//dst = deconvRL(input1, psf1, 30);


	//t1 = (double)cvGetTickCount() - t1;
	//t1 = t1 / (cvGetTickFrequency() * 1000000);
	//printf("time=%fs\n", t1);
	////IplImage* psf1 = cvCreateImage(cvSize(21,21), IPL_DEPTH_32F, 1);
	////cvScale(psf, psf1, 1, 0);
	////cvScale(input1, input, 1, 0);

	////genaratePsf(psf1, 10, 50);
	////deconvRL(input, psf1, input, 30);

	//cvNamedWindow("input1", 1);
	//cvShowImage("input1", input1);
	//cvNamedWindow("psf",0);
	//cvShowImage("psf", psf);
	//cvNamedWindow("dst", 1);
	//cvShowImage("dst", out);

	//double a[4] = {1, 0, 3, 2 };
	//double b[4] = { 3, 1, 2, 0 };
	//double c[4] = { 2, 1, 3, 1 };
	//double d[4] = { 3, 2, 0, 1 };


	////double dx[2] = { 1,-1 };
	//////double dy[4] = { -1, 0, 1, 0 };

	//////CvMat* r;

	//CvMat aa = cvMat(2,2, CV_64FC1,a);
	//CvMat bb = cvMat(2,2, CV_64FC1, b);
	//CvMat cc = cvMat(2, 2, CV_64FC1, c);
	//CvMat dd = cvMat(2, 2, CV_64FC1, d);
	//CvMat* i1 = cvCreateMat(2, 2, CV_64FC2);
	//CvMat* i2 = cvCreateMat(2, 2, CV_64FC2);
	//cvMerge(&aa, &bb, 0, 0,i1);
	//cvMerge(&cc, &dd, 0, 0, i2);
	//CvMat* r = complexMatrixDivide(i1, i2);


	//cvSplit(r, &aa, &bb, 0, 0);
	//printf("aa=");
	//for (int i = 0; i < aa.rows; i++)
	//{
	//	double* pvalue = (double*)(aa.data.ptr + i*aa.step);
	//	for (int j = 0; j < aa.cols; j++)
	//	{
	//		printf("%f ", pvalue[j]);
	//	}
	//	printf("\n");
	//}
	//printf("bb=");
	//for (int i = 0; i <2; i++)
	//{
	//	double* pvalue = (double*)(bb.data.ptr + i*bb.step);
	//	//double* pvalue = (double*)(dest->data.ptr + i*dest->step);

	//	for (int j = 0; j < 2; j++)
	//	{
	//		printf("%f ", pvalue[j]);
	//	}
	//	printf("\n");
	//}

//	//CvMat* ap = cvCloneMat(&aa);
////	CvMat* aip = cvCloneMat(&aa);
//	CvMat*dest = cvMatDftConv2(&aa, &bb, CONVOLUTION_FULL);

	//CvMat* ap = psf2otfMat(&aa, cvSize(5, 5));
	////cvMulSpectrums(ap, ap, ap, CV_DXT_MUL_CONJ);
	//cvDFT(&aa, ap, CV_DXT_FORWARD, 3);
//	cvScale(ap, ap, 1.5, 3);
	//cvDFT(&aa, aip, CV_DXT_INV_SCALE, 3);
	//CvMat* pad = padMat(&aa, 3, 3, PAD_CONSTANT, PAD_POST);
	//CvMat* otf = circularShift(pad, -1,-1);

	//printf("aip=");
	//for (int i = 0; i < aip->rows; i++)
	//{
	//	//double* pvalue = (double*)(bb.data.ptr + i*bb.step);
	//	double* pvalue = (double*)(aip->data.ptr + i*aip->step);

	//	for (int j = 0; j < aip->cols; j++)
	//	{
	//		printf("%f ", pvalue[j]);
	//	}
	//	printf("\n");
	//}
	//double t1 = (double)cvGetTickCount();
	//r = cvMatFilterConv2(&aa, &bb, CONVOLUTION_VALID);
	//t1 = (double)cvGetTickCount() - t1;
	//t1 = t1 / (cvGetTickFrequency() * 1000000);


	////double t2 = (double)cvGetTickCount();
	////r = cvMatFilterConv2(&aa, &bb, CONVOLUTION_FULL);
	////t2 = (double)cvGetTickCount() - t2;
	////t2 = t2 / (cvGetTickFrequency()*1000000);

	//printf("r=");
	//for (int i = 0; i < r->rows; i++)
	//{
	//	double* pvalue = (double*)(r->data.ptr + i*r->step);
	//	for (int j = 0; j < r->cols; j++)
	//	{
	//		printf("%f ", pvalue[j]);
	//	}
	//	printf("\n");
	//}

	//printf("t1=%fs\n",t1);
	//printf("t2=%fs\n",t2);


	//cvReleaseMat(&r);
	//double m,M;
	//cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
	//printf("min=%f max=%f\n", m, M);
	//printf(" input1=");
	//for (int i = 0; i < 3; i++)
	//{
	//	//uchar* pvalue = (uchar*)(blurImage->imageData + i* blurImage->widthStep);
	//	uchar* pvalue = (uchar*)(input1->imageData + i*  input1->widthStep);

	//	for (int j = 0; j < input1->width; j++)
	//	{
	//		printf("%d ", pvalue[j]);
	//	}
	//	printf("\n");
	//}

	//cvNamedWindow("blur", 1);
	//cvShowImage("blur", image_char);

	//double angle = doubFreqAngle(image_char);
	//double len1 = autocorrLen(image_char, angle);
	////double len2 = cepstrumLen(image_char, angle);
	//printf("angle=%f\n", angle);
	//printf("len1=%f\n", len1);

	//genaratePsf(epsf, len1, angle);
	//deconvRL(image_Re,psf, image_Re, 50);
	//blindEstKernel(image_Re,psf);

	//double m,M;
	//cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
	//printf("min=%f max=%f\n", m, M);
	//printf("CC=");
	//for (int i = 0; i < 3; i++)
	//{
	//	//uchar* pvalue = (uchar*)(blurImage->imageData + i* blurImage->widthStep);
	//	double* pvalue = (double*)(image_Re->imageData + i* image_Re->widthStep);
	//	//double* pvalue = (double*)(YF->data.ptr + i*  YF->step);


	//	for (int j = 0; j < image_Re->width; j++)
	//	{
	//		printf("%f ", pvalue[j]);
	////	}
	////}
	////cvScale(image_Re, image_char, 1, 0);
	//cvNamedWindow("kernel", 1);
	//cvShowImage("kernel", psf);


	cvWaitKey(0);

	cvDestroyAllWindows();
	//cvReleaseImage(&input1);
	//cvReleaseMat(&psf);
	//cvReleaseImage(&psf1);

	//cvReleaseImage(&dst);

	//cvReleaseImage(&epsf);
	//cvReleaseImage(&image_Re);
	//cvReleaseImage(&image_Im);
	//cvReleaseImage(&image_char);
	//cvReleaseMat(&psf1);
	//cvReleaseMat(&blur);

	return 0;
}
