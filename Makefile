CAFFE2_PREFIX=/home/xieyi/opt/caffe2
CXXFLAGS = `pkg-config --cflags opencv eigen3` -Isrc -I${CAFFE2_PREFIX}/include -std=c++14
LIBS=`pkg-config --libs opencv eigen3` -lboost_program_options -L${CAFFE2_PREFIX}/lib -lcaffe2_gpu -lcaffe2 -lglog
OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

all: train test predictor

predictor: $(OBJS)
	$(CXX) $^ $(LIBS) -o ${@}

.PHONY: train test dataset
dataset:
	mkdir tmp
	curl --progress-bar http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > tmp/train-images-idx3-ubyte
	curl --progress-bar http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > tmp/train-labels-idx1-ubyte
	make_mnist_db --image_file=tmp/train-images-idx3-ubyte --label_file=tmp/train-labels-idx1-ubyte --output_file=mnist-train-nchw-leveldb --channel_first --db leveldb
	curl --progress-bar http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > tmp/t10k-images-idx3-ubyte
	curl --progress-bar http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > tmp/t10k-labels-idx1-ubyte
	make_mnist_db --image_file=tmp/t10k-images-idx3-ubyte --label_file=tmp/t10k-labels-idx1-ubyte --output_file=mnist-test-nchw-leveldb --channel_first --db leveldb
	$(RM) -r tmp

train: train_plan.pbtxt
	run_plan --plan $^
	
test: test_plan.pbtxt
	run_plan --plan $^
	
clean:
	$(RM) *.log *.summary
	$(RM) -r LeNet_params
	$(RM) predictor $(OBJS)
