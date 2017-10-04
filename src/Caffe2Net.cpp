#include "Caffe2Net.h"

Caffe2Net::Caffe2Net(string initNet,string predictNet,string param)
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
}

TensorCPU Caffe2Net::preprocess(Mat img)
{
}

TensorCPU Caffe2Net::predict_(Mat img)
{
	TensorCPU context = preprocess(img);
	Predictor::TensorVector input({&context}),output;
	predictor->run(input,&output);
	return *(output[0]);
}
