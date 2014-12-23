#ifndef __RESTORE_H__
#define __RESTORE_H__

//�����ά�������͵�ö����������
enum ConvolutionType {
	/* ����ȫ�����ݣ������õ����߽������ */
	CONVOLUTION_FULL,
	/* ���غ�ԭͼ��Сһ�������� */
	CONVOLUTION_SAME,
	/* ֻ����û���õ�������ݵľ������ */
	CONVOLUTION_VALID
};

enum PadType {
	/* �ھ���ǰ����� */
	PAD_PRE,
	/* �ھ���ǰ�˺ͺ��ͬʱ��� */
	PAD_BOTH,
	/* �ھ�������� */
	PAD_POST,
	/* �������(��)��� */
	PAD_CONSTANT,
	/* �߽縴����� */
	PAD_REPLICATE
};

enum DftType {
	/* ���ص�ͨ��CCS���� */
	DFT_CCS,
	/* ����˫ͨ���������� */
	DFT_COMPLEX,
	/* ���ص�ͨ��ʵ������ */
	PAD_REAL,
};

double doubFreqAngle(IplImage* input);
double cepstrumLen(IplImage* input, double angle);
void imRotate(IplImage* input, IplImage* output, double angle);

IplImage* deconvRL1(const IplImage* input, const CvMat* kernel, const int num);
IplImage* deconvRL(IplImage* input, IplImage* kernel, int num);
IplImage* edgetaper1(const IplImage* input, const CvMat* psf);
void edgetaper(IplImage* input, IplImage* psf, IplImage* output);

void corelucy(IplImage* input_Y, CvMat* input_H, IplImage* input_g, CvMat* output);
double autocorrLen(IplImage* input, double angle);

CvMat* blindEstKernel(const IplImage* blur, const int psf_max_size);

void coreBlindEstKernel(CvMat* yx, CvMat* yy, CvMat* xxs, CvMat*xys, CvMat* ks, double lambdas, double delta,
	double k_reg_wt, int x_in_iter, int x_out_iter, int xk_iter);
void cvSign(CvMat* input, CvMat* output);
void pcgKernelIRLS(CvMat* ks, CvMat* xxs, CvMat* xys, CvMat* yxs, CvMat* yys,
	double pcg_tol, double pcg_its, double k_reg_wt);

CvMat* cvMatFilterConv2(const CvMat* input, const CvMat* kernel, ConvolutionType type);
CvMat* cvMatDftConv2(const CvMat* input, const CvMat* kernel, ConvolutionType type);
CvMat* dftCoreConv2(const CvMat* input, const CvMat* kernel);
CvMat* centerKernel(const CvMat* ks);
CvMat* padMat(const CvMat* input, int rows, int cols, PadType type1, PadType type2);
CvMat* deconvBregman(const CvMat* input, const CvMat* kernel, double lambda,double alpha);
CvMat* psf2otfMat(const CvMat* psf, CvSize size, DftType type);
CvMat* dftMat(const CvMat* input, DftType type);

#endif