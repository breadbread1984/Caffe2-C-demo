# Caffe2 C++ demo
### Introduction

This demo shows how Caffe2 train models, save params and works with C++ language. Generally, Caffe2 has a tool called run_plan which is a counterpart of the executable caffe of Caffe1. We can train model with run_plan in command line and use the trained params in C++. The process is almost the same as Caffe1.

### modify Makefile
change the value of CAFFE2_PREFIX in the makefile according to your customized installation location.

### download mnist dataset and convert it into leveldb format
```Shell
make dataset
```

### train on mnist dataset and save params into lmdb format
```Shell
make clean && make train
```

### test on mnist dataset
```Shell
make test
```
check out accuracy.txt to verify whether the model is properly trained.

### make handwriting predictor
```Shell
make predictor
```
you can use the predictor to classify single channel 20x20 mnist like handwriting pics. There are some samples extracted from MNIST. You can test on them with the following command.
```Shell
./predictor -i imgs/0.bmp
```
