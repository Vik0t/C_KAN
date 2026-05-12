//
// Created by viktor on 5/4/26.
//



#ifndef KAN_H
#define KAN_H

#include "model.h"


typedef struct {
    int input_dim;
    int output_dim;
    int grid_size;

    double grid_min;
    double grid_max;

    double* tables;
    double* grad_tables;

    double* biases;
    double* grad_biases;

    int* last_left;
    int* last_right;
    double* last_t;

} KANLayer;


typedef struct {
    int layer_count;
    int *sizes;
    KANLayer* layers;
    double **activations;

} KAN;

Model* kan_create(int *sizes, int layer_count, int grid_size);

#endif

