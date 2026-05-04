//
// Created by viktor on 5/4/26.
//

#include "loss.h"
#include <stdlib.h>

double mse(double* pred, double* target, int size) {
    double mse = 0;
    for (int i = 0; i < size; i++) {
        mse += (pred[i] - target[i]) * (pred[i] - target[i]);
    }
    return mse / size;
}

double mae(double* pred, double* target, int size) {
    double mae = 0;
    for (int i = 0; i < size; i++) {
        mae += abs(pred[i] - target[i]);
    }
    return mae / size;
}

