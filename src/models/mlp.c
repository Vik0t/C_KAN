//
// Created by viktor on 5/4/26.
//

#include "model.h"
#include "mlp.h"
#include <stdlib.h>
#include "random.h"

static void mlp_zero_grad(void* impl)
{
    MLP* mlp = (MLP*)impl;

    for (int j = 0; j < mlp->hidden; j++)
    {
        mlp->db1[j] = 0;
    }

    for (int i = 0; i < mlp->in; i++)
    {
        for (int j = 0; j < mlp->hidden; j++)
        {
            mlp->dW1[i*mlp->hidden+j] = 0;
        }
    }

    for (int k = 0; k < mlp->out; k++)
    {
        mlp->db2[k] = 0;
        for (int j = 0; j < mlp->hidden; j++)
        {
            mlp->dW2[j*mlp->out+k] = 0;
        }
    }
}

static void mlp_backward(void* impl, const double* input, const double* grad_output)
{
    MLP* mlp = (MLP*)impl;
    mlp_zero_grad(impl);
    double *grad_hidden = calloc(mlp->hidden, sizeof(double));

    for (int k = 0; k < mlp->out; k++)
    {
        mlp->db2[k] = grad_output[k];
        for (int j = 0; j < mlp->hidden; j++)
        {
            mlp->dW2[j*mlp->out+k] = mlp->hidden_activations[j]*grad_output[k];
            grad_hidden[j] += grad_output[k]*mlp->W2[j*mlp->out+k];
        }
    }

    for (int j = 0; j < mlp->hidden; j++)
    {
        double z = grad_hidden[j] * (1 - mlp->hidden_activations[j]*mlp->hidden_activations[j]);
        mlp->db1[j] = z;
        for (int i = 0; i < mlp->in; i++)
        {
            mlp->dW1[i*mlp->hidden+j] = input[i]*z;
        }

    }
    free(grad_hidden);
}

static void mlp_update(void* impl, double lr)
{
    MLP* mlp = (MLP*)impl;
    for (int i = 0; i < mlp->in; i++)
    {
        for (int j = 0; j < mlp->hidden; j++)
        {
            mlp->W1[i*mlp->hidden+j] -= mlp->dW1[i*mlp->hidden+j]*lr;
        }
    }

    for (int k = 0; k < mlp->hidden; k++)
    {
        mlp->b1[k] -= mlp->db1[k]*lr;
    }

    for (int i = 0; i < mlp->out; i++)
    {
        mlp->b2[i] -= mlp->db2[i]*lr;
    }


    for (int j = 0; j < mlp->hidden; j++)
    {

        for (int k = 0; k < mlp->out; k++)
        {
             mlp->W2[j*mlp->out+k] -= mlp->dW2[j*mlp->out+k]*lr;
        }
    }
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


    for (int j = 0; j < mlp->hidden; j++)
    {
        double z = mlp->b1[j];
        for (int i = 0; i < mlp->in; i++)
        {
            z += mlp->W1[i*mlp->hidden + j] * input[i];
        }
        mlp->hidden_activations[j] = Tanh(z);
    }

    for (int k = 0; k < mlp->out; k++)
    {
        double z = mlp->b2[k];
        for (int j = 0; j < mlp->hidden; j++)
        {
            z += mlp->hidden_activations[j] * mlp->W2[j*mlp->out + k];
        }
        output[k] = z;
        mlp->output[k] = z;
    }


}

static void mlp_free(void* impl)
{
    MLP* mlp = (MLP*)impl;

    free(mlp->W1);
    free(mlp->b1);
    free(mlp->db1);
    free(mlp->dW1);

    free(mlp->W2);
    free(mlp->b2);
    free(mlp->db2);
    free(mlp->dW2);


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
    random_fill_uniform(mlp->W1, input_size * hidden_size, -0.5, 0.5);
    mlp->dW1 = calloc(input_size * hidden_size,sizeof(double));

    mlp->b1 = calloc(hidden_size , sizeof(double));
    mlp->db1 = calloc(hidden_size , sizeof(double));


    mlp->W2 = malloc(sizeof(double) * output_size * hidden_size);
    random_fill_uniform(mlp->W2, output_size * hidden_size, -0.5, 0.5);
    mlp->dW2 = calloc(output_size * hidden_size,sizeof(double));

    mlp->b2 = calloc(output_size , sizeof(double));
    mlp->db2 = calloc(output_size , sizeof(double));

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

