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

enum PadType {
	/* 在矩阵前端填充 */
	PAD_PRE,
	/* 在矩阵前端和后端同时填充 */
	PAD_BOTH,
	/* 在矩阵后端填充 */
	PAD_POST,
	/* 常数填充(零)填充 */
	PAD_CONSTANT,
	/* 边界复制填充 */
	PAD_REPLICATE
};

enum DftType {
	/* 返回单通道CCS矩阵 */
	DFT_CCS,
	/* 返回双通道复数矩阵 */
	DFT_COMPLEX,
	/* 返回单通道实数矩阵 */
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