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

vector<float> LeNet::predict(Mat img)
{
	TensorCPU output = predict_(img);
	const float * probs = output.data<float>();
	vector<TIndex> dims = output.dims();
	assert(1 == dims[0]);
	//batchsize=1
	vector<float> retVal;
	for(int i = 0 ; i < dims[1] * dims[2] * dims[3] ; i++)
		retVal.push_back(probs[i]);
	return retVal;
}

TensorCPU LeNet::preprocess(Mat img)
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
