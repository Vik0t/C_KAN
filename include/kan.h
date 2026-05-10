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

    double* table;
    double* grad_table;

    double bias;
    double grad_bias;

    int last_left_index;
    int last_right_index;
    double last_t;

} KAN;

Model* kan_create(int input_size, int grid_size, int output_size);

#endif

