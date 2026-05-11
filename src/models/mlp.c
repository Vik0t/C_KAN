//
// Created by viktor on 5/4/26.
//

#include "model.h"
#include "mlp.h"
#include "random.h"

#include <stdlib.h>

static void mlp_zero_grad(void* impl)
{
    MLP* mlp = (MLP*)impl;

    for (int l = 0; l < mlp->layer_count - 1; l++)
    {
        int current_size = mlp->sizes[l];
        int next_size = mlp->sizes[l + 1];

        for (int j = 0; j < next_size; j++)
        {
            mlp->grad_biases[l][j] = 0.0;
        }

        for (int i = 0; i < current_size; i++)
        {
            for (int j = 0; j < next_size; j++)
            {
                mlp->grad_weights[l][i * next_size + j] = 0.0;
            }
        }
    }
}

static void mlp_backward(void* impl, const double* input, const double* grad_output)
{
    (void)input;

    MLP* mlp = (MLP*)impl;
    int last_layer = mlp->layer_count - 1;

    mlp_zero_grad(impl);

    double** deltas = malloc(sizeof(double*) * mlp->layer_count);
    for (int l = 0; l < mlp->layer_count; l++)
    {
        deltas[l] = calloc(mlp->sizes[l], sizeof(double));
    }

    for (int j = 0; j < mlp->sizes[last_layer]; j++)
    {
        deltas[last_layer][j] = grad_output[j];
    }

    for (int l = last_layer - 1; l >= 1; l--)
    {
        int current_size = mlp->sizes[l];
        int next_size = mlp->sizes[l + 1];

        for (int i = 0; i < current_size; i++)
        {
            double grad = 0.0;
            for (int j = 0; j < next_size; j++)
            {
                grad += mlp->weights[l][i * next_size + j] * deltas[l + 1][j];
            }

            double activation = mlp->activations[l][i];
            deltas[l][i] = grad * (1.0 - activation * activation);
        }
    }

    for (int l = 0; l < mlp->layer_count - 1; l++)
    {
        int current_size = mlp->sizes[l];
        int next_size = mlp->sizes[l + 1];

        for (int j = 0; j < next_size; j++)
        {
            mlp->grad_biases[l][j] = deltas[l + 1][j];
        }

        for (int i = 0; i < current_size; i++)
        {
            for (int j = 0; j < next_size; j++)
            {
                mlp->grad_weights[l][i * next_size + j] =
                    mlp->activations[l][i] * deltas[l + 1][j];
            }
        }
    }

    for (int l = 0; l < mlp->layer_count; l++)
    {
        free(deltas[l]);
    }
    free(deltas);
}

static void mlp_update(void* impl, double lr)
{
    MLP* mlp = (MLP*)impl;

    for (int l = 0; l < mlp->layer_count - 1; l++)
    {
        int current_size = mlp->sizes[l];
        int next_size = mlp->sizes[l + 1];

        for (int j = 0; j < next_size; j++)
        {
            mlp->biases[l][j] -= lr * mlp->grad_biases[l][j];
        }

        for (int i = 0; i < current_size; i++)
        {
            for (int j = 0; j < next_size; j++)
            {
                int index = i * next_size + j;
                mlp->weights[l][index] -= lr * mlp->grad_weights[l][index];
            }
        }
    }
}

static int mlp_parameter_count(void* impl)
{
    MLP* mlp = (MLP*)impl;

    int count = 0;
    for (int l = 0; l < mlp->layer_count - 1; l++)
    {
        count += mlp->sizes[l] * mlp->sizes[l + 1] + mlp->sizes[l + 1];
    }

    return count;
}

static void mlp_forward(void* impl, const double* input, double* output)
{
    MLP* mlp = (MLP*)impl;

    for (int i = 0; i < mlp->sizes[0]; i++)
    {
        mlp->activations[0][i] = input[i];
    }

    for (int l = 0; l < mlp->layer_count - 1; l++)
    {
        int current_size = mlp->sizes[l];
        int next_size = mlp->sizes[l + 1];

        for (int j = 0; j < next_size; j++)
        {
            double z = mlp->biases[l][j];
            for (int i = 0; i < current_size; i++)
            {
                z += mlp->activations[l][i] * mlp->weights[l][i * next_size + j];
            }

            if (l == mlp->layer_count - 2)
            {
                mlp->activations[l + 1][j] = z;
            }
            else
            {
                mlp->activations[l + 1][j] = Tanh(z);
            }
        }
    }

    int output_layer = mlp->layer_count - 1;
    for (int i = 0; i < mlp->sizes[output_layer]; i++)
    {
        output[i] = mlp->activations[output_layer][i];
    }
}

static void mlp_free(void* impl)
{
    MLP* mlp = (MLP*)impl;

    for (int l = 0; l < mlp->layer_count - 1; l++)
    {
        free(mlp->weights[l]);
        free(mlp->grad_weights[l]);
        free(mlp->biases[l]);
        free(mlp->grad_biases[l]);
    }

    for (int l = 0; l < mlp->layer_count; l++)
    {
        free(mlp->activations[l]);
    }

    free(mlp->weights);
    free(mlp->biases);
    free(mlp->grad_weights);
    free(mlp->grad_biases);
    free(mlp->activations);
    free(mlp->sizes);
    free(mlp);
}

Model* mlp_create(int* sizes, int layer_count)
{
    int weight_layer_count = layer_count - 1;

    MLP* mlp = malloc(sizeof(MLP));
    mlp->layer_count = layer_count;

    mlp->sizes = malloc(sizeof(int) * layer_count);
    for (int i = 0; i < layer_count; i++)
    {
        mlp->sizes[i] = sizes[i];
    }

    mlp->weights = malloc(sizeof(double*) * weight_layer_count);
    mlp->biases = malloc(sizeof(double*) * weight_layer_count);
    mlp->grad_weights = malloc(sizeof(double*) * weight_layer_count);
    mlp->grad_biases = malloc(sizeof(double*) * weight_layer_count);
    mlp->activations = malloc(sizeof(double*) * layer_count);

    for (int l = 0; l < layer_count; l++)
    {
        mlp->activations[l] = calloc(mlp->sizes[l], sizeof(double));
    }

    for (int l = 0; l < weight_layer_count; l++)
    {
        int weight_count = mlp->sizes[l] * mlp->sizes[l + 1];
        int bias_count = mlp->sizes[l + 1];

        mlp->weights[l] = malloc(sizeof(double) * weight_count);
        random_fill_uniform(mlp->weights[l], weight_count, -0.1, 0.1);

        mlp->grad_weights[l] = calloc(weight_count, sizeof(double));
        mlp->biases[l] = calloc(bias_count, sizeof(double));
        mlp->grad_biases[l] = calloc(bias_count, sizeof(double));
    }

    Model* model = malloc(sizeof(Model));

    model->impl = mlp;
    model->input_dim = mlp->sizes[0];
    model->output_dim = mlp->sizes[layer_count - 1];

    model->forward = mlp_forward;
    model->backward = mlp_backward;
    model->update = mlp_update;
    model->free = mlp_free;
    model->parameter_count = mlp_parameter_count;
    model->zero_grad = mlp_zero_grad;

    return model;
}
