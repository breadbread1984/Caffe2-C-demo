#ifndef CAFFE2NET_H
#define CAFFE2NET_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

using namespace std;
using namespace cv;
using namespace caffe2;

class Caffe2Net {
public:
	Caffe2Net(string initNet,string predictNet,string param);
	virtual ~Caffe2Net() = 0;
	vector<float> predict(Mat img);
protected:
	virtual TensorCPU preProcess(Mat img) = 0;
	virtual vector<float> postProcess(TensorCPU output) = 0;

	NetDef init_net, predict_net;
	auto_ptr<Predictor> predictor;
};

#endif
