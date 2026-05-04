//
// Created by viktor on 5/4/26.
//

#ifndef MATH_H
#define MATH_H
#endif

#ifndef MODEL_H
#define MODEL_H

typedef struct {
    void* impl;

    void (*forward)(void* impl, double* input, double* output);
    void (*backward)(void* impl, double* grad_output);
    void (*update)(void* impl, double lr);
    void (*free)(void* impl);

} Model;

typedef double (*ActivationFunc)(double, double);

double RELU(double x);
double Sigmoid(double x);

#endif