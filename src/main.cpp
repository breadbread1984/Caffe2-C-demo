#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "LeNet.h"

using namespace std;
using namespace boost::program_options;
using namespace cv;

int main(int argc,char ** argv)
{
	string img_path;
	options_description desc;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&img_path),"输入图片路径");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || 1 != vm.count("input") || 1 == vm.count("help")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	Mat img = imread(img_path,IMREAD_GRAYSCALE);
	if(true == img.empty()) {
		cerr<<"图片无法打开！"<<endl;
		return EXIT_FAILURE;
	}
	
	LeNet lenet("deploy_models/mnist_init_net.pbtxt","deploy_models/mnist_predict_net.pbtxt");
	vector<float> result = lenet.predict(img);
	vector<float>::iterator max_iter = max_element(result.begin(),result.end());
	cout<<max_iter - result.begin()<<endl;
	
	return EXIT_SUCCESS;
}
