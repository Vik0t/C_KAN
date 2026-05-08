//
// Created by viktor on 5/4/26.
//



#ifndef KAN_H
#define KAN_H

#include "model.h"

typedef struct {
    int input_dim;
    int grid_size;

    double** tables; // [input_dim][grid_size]

} KAN;

Model* kan_create(int input_size, int hidden_size, int output_size);

#endif

