using QARBoM
using MLDatasets

trainset = MNIST(:train)

X_train = trainset[1]
y_train = trainset[2]
