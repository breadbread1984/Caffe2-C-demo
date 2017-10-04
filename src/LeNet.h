#ifndef LENET_H
#define LENET_H

#include "Caffe2Net.h"

using namespace std;
using namespace cv;
using namespace caffe2;

class LeNet : public Caffe2Net {
public:
	LeNet(string initNet,string predictNet,string param);
	virtual ~LeNet();
	virtual vector<float> predict(Mat img);
protected:
	virtual TensorCPU preprocess(Mat img);
};

#endif
