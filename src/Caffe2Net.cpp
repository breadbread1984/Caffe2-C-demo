#include "Caffe2Net.h"

Caffe2Net::Caffe2Net(string initNet,string predictNet)
{
	//载入部署模型
	CAFFE_ENFORCE(ReadProtoFromFile(initNet, &init_net));
	CAFFE_ENFORCE(ReadProtoFromFile(predictNet, &predict_net));
	//创建判别器
	predictor = auto_ptr<Predictor>(new Predictor(init_net, predict_net));
}

Caffe2Net::~Caffe2Net()
{
}

vector<float> Caffe2Net::predict(Mat img)
{
	TensorCPU context = preProcess(img);
	Predictor::TensorVector input({&context}),output;
	predictor->run(input,&output);
	return postProcess(*(output[0]));
}

TensorCPU Caffe2Net::preProcess(Mat img)
{
}

vector<float> Caffe2Net::postProcess(TensorCPU output)
{
}
