#ifndef LENET_H
#define LENET_H

#include "Caffe2Net.h"

using namespace std;
using namespace cv;
using namespace caffe2;

class LeNet : public Caffe2Net {
public:
	LeNet(string initNet,string predictNet);
	virtual ~LeNet();
protected:
	virtual TensorCPU preProcess(Mat img);
	virtual vector<float> postProcess(TensorCPU output);
};

#endif
