#ifndef __BASICOPERATION_H__
#define __BASICOPERATION_H__

#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <math.h>


#define PI 3.141592653 
#define EPS 2.2204e-016 
#define max(a,b)    (((a) > (b)) ? (a) : (b))  
#define min(a,b)    (((a) < (b)) ? (a) : (b)) 


void meshgrid(int x0, int xt, int y0, int yt, CvMat* X, CvMat* Y);
void genaratePsf(IplImage* psf, double len, double angle);
void mat2Image(CvMat* input, IplImage* output);
void image2Mat(IplImage* input, CvMat* output);
void showDFT2(IplImage* input);
void DFT2(IplImage* input, CvMat* output, int flag);
void shiftDFT(CvArr * input, CvArr * output);
void psf2otf(IplImage* psf, CvMat* otf);
void calConv(IplImage* input, IplImage* kernal, CvMat* output);


CvMat* psf2otf(CvMat* psf, CvSize size);
CvMat* circularShift(const CvMat* input, int yshift, int xshift);
CvMat* complexMatrixDivide(const CvMat* m, const CvMat* n);

CvRect kcvRectIntersection(CvRect rect1, CvRect rect2);
CvRect kcvRectFromCenterAndSize(int cx, int cy, int w, int h = 0);
void sort(uchar a[], int n);
double clip(double p, double a, double b);
uchar percentileValue(IplImage* img, CvRect rect, double p);
void percentileFilter(IplImage* src, IplImage* dst, double p, int kwidth, int kheight = 0);
uchar percentileValue2(IplImage* img, CvRect rect, double p);
void percentileFilter2(IplImage* src, IplImage* dst, double p, int kwidth, int kheight = 0);
void adpMedian(IplImage* src, IplImage* dst, int Smax);

#endif