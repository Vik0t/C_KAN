//
// Created by viktor on 5/4/26.
//
#include "model.h"
#include "math.h"

double RELU(double x) {
    return x > 0 ? x : 0;
}

double Sigmoid(double x) {
    return 1.0 / exp(-x);
}
