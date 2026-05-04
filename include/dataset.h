//
// Created by viktor on 5/4/26.
//

#ifndef C_KAN_CKAN_H

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

#endif

#define C_KAN_CKAN_H

#endif //C_KAN_CKAN_H