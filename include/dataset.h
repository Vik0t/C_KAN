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

Dataset* create_function_dataset(double (*f)(double), int n);

double sin_func(double x);
#endif



