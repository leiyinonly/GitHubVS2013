#ifndef __RESTORE_H__
#define __RESTORE_H__

//�����ά������͵�ö����������
enum ConvolutionType {
	/* ����ȫ�����ݣ������õ����߽������ */
	CONVOLUTION_FULL,
	/* ���غ�ԭͼ��Сһ�������� */
	CONVOLUTION_SAME,
	/* ֻ����û���õ�������ݵľ����� */
	CONVOLUTION_VALID
};

double doubFreqAngle(IplImage* input);
void edgetaper(IplImage* input, IplImage* psf, IplImage* output);
double cepstrumLen(IplImage* input, double angle);
void imRotate(IplImage* input, IplImage* output, double angle);
void deconvRL(IplImage* input, IplImage* kernal, IplImage* output, int num);
void corelucy(IplImage* input_Y, CvMat* input_H, IplImage* input_g, CvMat* output);
double autocorrLen(IplImage* input, double angle);

IplImage* blindEstKernel(const IplImage* blur, const int psf_max_size);

void coreBlindEstKernel(CvMat* yx, CvMat* yy, CvMat* xxs, CvMat*xys, CvMat* ks, double lambdas, double delta,
	double k_reg_wt, int x_in_iter, int x_out_iter, int xk_iter);
void cvSign(CvMat* input, CvMat* output);
void pcgKernelIRLS(CvMat* ks, CvMat* xxs, CvMat* xys, CvMat* yxs, CvMat* yys,
	double pcg_tol, double pcg_its, double k_reg_wt);

CvMat* cvMatFilterConv2(const CvMat* input, const CvMat* kernel, ConvolutionType type);
CvMat* cvMatDftConv2(const CvMat* input, const CvMat* kernel, ConvolutionType type);
CvMat* dftCoreConv2(const CvMat* input, const CvMat* kernel);


#endif