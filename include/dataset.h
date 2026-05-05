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
void dataset_free(Dataset* data);

double sin_func(double x);
#endif



