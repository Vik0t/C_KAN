//
// Created by viktor on 5/4/26.
//

#ifndef DATASET_H
#define DATASET_H

typedef struct {
    int size;
    int input_dim;
    int output_dim;

    double** X;
    double** Y;

} Dataset;


Dataset* dataset_create_function_1d(double (*f)(double), double from, double to, int n);
Dataset* dataset_create_function_2d(double (*f)(double, double), double from, double to, int n);
void dataset_free(Dataset* data);
int dataset_to_csv(const Dataset* data, const char* filename);

double sin_func(double x);

#include <math.h>

double bespalov(double x1, double x2);

#endif



