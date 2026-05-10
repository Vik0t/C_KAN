//
// Created by viktor on 5/4/26.
//

#include "model.h"
#include "kan.h"

#include <math.h>
#include <stdlib.h>
#include "random.h"



static void kan_forward(void* impl, const double* input, double* output)
{
    KAN* kan = (KAN*)impl;
    int left=0, right = 0;
    double t = 0, position = 0;
    double x = input[0];

    if (x <= kan->grid_min)
    {
        left = 0;
        right = 0;
        t = 0;
    }


    else if (x >= kan->grid_max)
    {
        left= kan->grid_size-1;
        right = kan->grid_size-1;
    }

    else
    {
        position = (x - kan->grid_min) / (kan->grid_max - kan->grid_min) * (kan->grid_size - 1);
        left = floor(position);
        right = left +1;
        t = position - left;
    }

    kan->last_left_index = left;
    kan->last_right_index = right;
    kan->last_t = t;


    double y = (1-t) * kan->table[left] + t * kan->table[right] + kan->bias;

    output[0] = y;

}

static void kan_free(void* impl)
{
    KAN* kan = (KAN*)impl;
    free(kan->table);
    free(kan->grad_table);
    free(kan);

}

static void kan_zero_grad(void* impl)
{
    KAN* kan = (KAN*)impl;

    for (int i = 0; i < kan->grid_size; i++)
    {
        kan->grad_table[i] = 0;
    }

    kan->grad_bias = 0;
}


static int kan_parameter_count(void* impl)
{
    KAN* kan = (KAN*)impl;
    return kan->grid_size + 1;
}

static void kan_backward(void* impl, const double* input, const double* grad_output)
{
    KAN* kan = (KAN*)impl;

    kan_zero_grad(impl);



    kan->grad_table[kan->last_left_index] += (1 - kan->last_t) * grad_output[0];
    kan->grad_table[kan->last_right_index] += kan->last_t * grad_output[0];
    kan->grad_bias += grad_output[0];

}

static void kan_update(void* impl, double lr)
{
    KAN* kan = (KAN*)impl;

    for (int i = 0; i < kan->grid_size; i++)
    {
        kan->table[i] -= lr * kan->grad_table[i];
    }
    kan->bias -= lr * kan->grad_bias;
}



Model* kan_create(int input_size, int grid_size, int output_size)
{
    KAN* kan = (KAN*)malloc(sizeof(KAN));

    kan->input_dim = input_size;
    kan->output_dim = output_size;
    kan->grid_size = grid_size;

    kan->bias = 0;
    kan->grad_bias = 0;

    kan->grid_min = -1;
    kan->grid_max = 1;

    kan->grad_table = calloc(grid_size, sizeof(double));
    kan->table = (double*)malloc(sizeof(double) * grid_size);
    random_fill_uniform(kan->table, grid_size, -0.1, 0.1);


    Model* model = (Model*)malloc(sizeof(Model));
    model->impl = kan;
    model->input_dim = input_size;
    model->output_dim = output_size;


    model->forward = kan_forward;
    model->backward = kan_backward;
    model->update = kan_update;
    model->free = kan_free;
    model->parameter_count = kan_parameter_count;
    model->zero_grad = kan_zero_grad;

    return model;
}