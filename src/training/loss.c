//
// Created by viktor on 5/4/26.
//

#include "loss.h"
#include "math.h"



double mse(const double* pred, const double* target, int size) {
    double mse = 0;
    for (int i = 0; i < size; i++) {
        mse += (pred[i] - target[i]) * (pred[i] - target[i]);
    }
    return mse / size;
}

double mae(const double* pred, const double* target, int size) {
    double mae = 0;
    for (int i = 0; i < size; i++) {
        mae += fabs(pred[i] - target[i]);
    }
    return mae / size;
}

