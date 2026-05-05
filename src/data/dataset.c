//
// Created by viktor on 5/4/26.
//
#include "dataset.h"
#include <math.h>


Dataset* dataset_create_function_1d(double (*f)(double), double from, double to, int n);

void dataset_free(Dataset* data);


double sin_func(double x) {
    return sin(x);
}

