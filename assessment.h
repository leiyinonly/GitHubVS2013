#ifndef __ASSESSMENT_H__
#define __ASSESSMENT_H__

#include <cv.h>
#include <highgui.h>

uchar blurIdentify(const IplImage* input);
double calGMG(const IplImage* input);
double calLuminanceSim(const IplImage* input1, const IplImage* input2);
double calContrastSim(const IplImage* input1, const IplImage* input2);
double calStructSim(const IplImage* input1, const IplImage* input2);
double calGradSim(const IplImage* image1, const IplImage* image2);

double calMISSIM(const IplImage* image1, const IplImage* image2, int n);
double calISSIM(const IplImage* image1, const IplImage* image2);
double calINRSS(const IplImage* input);

IplImage* gradientImage(const IplImage* input);

#endif