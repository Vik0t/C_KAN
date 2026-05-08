//
// Created by viktor on 5/4/26.
//

#include "model.h"
#include "mlp.h"
#include <stdlib.h>


static void mlp_backward(void* impl, const double* input, const double* grad_output)
{

}

static void mlp_update(void* impl, double lr)
{

}

static void mlp_zero_grad(void* impl)
{

}

static int mlp_parameter_count(void* impl)
{
    MLP* mlp = (MLP*)impl;
    return mlp->in * mlp->hidden + mlp->hidden
         + mlp->hidden * mlp->out + mlp->out;
}

static void mlp_forward(void* impl, const double* input, double* output)
{
    MLP* mlp = (MLP*)impl;
}

static void mlp_free(void* impl)
{
    MLP* mlp = (MLP*)impl;

    free(mlp->W1);
    free(mlp->b1);
    free(mlp->W2);
    free(mlp->b2);
    free(mlp->hidden_activations);
    free(mlp->output);

    free(mlp);
}

Model* mlp_create(int input_size, int hidden_size, int output_size)
{
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->in = input_size;
    mlp->hidden = hidden_size;
    mlp->out = output_size;

    mlp->W1 = malloc(sizeof(double) * input_size * hidden_size);
    mlp->b1 = malloc(sizeof(double) * hidden_size);

    mlp->W2 = malloc(sizeof(double) * output_size * hidden_size);
    mlp->b2 = malloc(sizeof(double) * output_size);

    mlp->hidden_activations = malloc(sizeof(double) * hidden_size);
    mlp->output = malloc(sizeof(double) * output_size);


    // щас реально оборачиваем

    Model* model = (Model*)malloc(sizeof(Model));

    model->impl = mlp;
    model->input_dim = input_size;
    model->output_dim = output_size;


    model->forward = mlp_forward;
    model->backward = mlp_backward;
    model->update = mlp_update;
    model->free = mlp_free;
    model->parameter_count = mlp_parameter_count;
    model->zero_grad = mlp_zero_grad;

    return model;

}

