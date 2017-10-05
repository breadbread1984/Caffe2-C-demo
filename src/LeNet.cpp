#include <algorithm>
#include "LeNet.h" 

using namespace std;

LeNet::LeNet(string initNet,string predictNet,string param)
:Caffe2Net(initNet,predictNet,param)
{
}

LeNet::~LeNet()
{
}

TensorCPU LeNet::preProcess(Mat img)
{
	assert(img.channels() == 1);
	assert(img.rows == 28);
	assert(img.cols == 28);
	vector<TIndex> dims({1, img.channels(), img.rows, img.cols});
	vector<float> data(1 * 1 * 28 * 28);
	
	img.convertTo(img, CV_32FC1, 1.0/256,0);
	copy((float *)img.datastart, (float *)img.dataend,data.begin());
	
	return TensorCPU(dims, data, NULL);
}

vector<float> LeNet::postProcess(TensorCPU output)
{
	const float * probs = output.data<float>();
	vector<TIndex> dims = output.dims();
	//检查输出的dims是否正确
	assert(2 == output.ndim());
	assert(1 == dims[0]);
	assert(10 == dims[1]);
	vector<float> retVal(dims[1]);
	copy(probs,probs+dims[1],retVal.begin());
	return retVal;
}
