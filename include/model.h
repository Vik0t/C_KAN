//
// Created by viktor on 5/4/26.
//


#ifndef MODEL_H
#define MODEL_H

typedef struct Model {
    void* impl;
    int input_dim;
    int output_dim;

    void (*forward)(void* impl, const double* input, double* output);
    void (*backward)(void* impl, const double* input, const double* grad_output);
    void (*update)(void* impl, double lr);
    void (*zero_grad)(void* impl);
    void (*free)(void* impl);
    int (*parameter_count)(void* impl);
} Model;

typedef double (*ActivationFunc)(double, double);

double RELU(double x);
double Sigmoid(double x);
double Tanh(double x);

#endif