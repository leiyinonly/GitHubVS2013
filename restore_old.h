#ifndef __RESTORE_H__
#define __RESTORE_H__

//定义二维卷积类型的枚举数据类型
enum ConvolutionType {
	/* 返回全部数据，包含用到填充边界的数据 */
	CONVOLUTION_FULL,
	/* 返回和原图大小一样的数据 */
	CONVOLUTION_SAME,
	/* 只返回没有用到填充数据的卷积结果 */
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