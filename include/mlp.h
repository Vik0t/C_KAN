//
// Created by viktor on 5/4/26.
//


#ifndef MLP_H
#define MLP_H
#include "model.h"

Model* mlp_create(int input_size, int hidden_size, int output_size);

typedef struct {
    int in, hidden, out;

    double* W1;
    double* b1;

    double* W2;
    double* b2;

    double* hidden_activations;
    double* output;

} MLP;


#endif

