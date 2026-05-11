//
// Created by viktor on 5/4/26.
//


#ifndef MLP_H
#define MLP_H
#include "model.h"

Model* mlp_create(int *sizes, int layer_count);

typedef struct {
    int layer_count;
    int* sizes;

    double** weights;
    double** biases;

    double** grad_weights;
    double** grad_biases;

    double** activations;
} MLP;


#endif

