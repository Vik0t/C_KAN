//
// Created by viktor on 5/4/26.
//

#include "model.h"
#include "kan.h"

#include <math.h>
#include <stdlib.h>
#include "random.h"



static void kan_layer_forward(void* impl, int layer_num, const double* input, double* output)
{
    KAN* kan = (KAN*)impl;
    int left=0, right = 0;
    double t = 0, position = 0;

    for (int j = 0; j < kan->layers[layer_num].output_dim; j++)
    {
        double sum = kan->layers[layer_num].biases[j];

        for (int i = 0; i < kan->layers[layer_num].input_dim; i++)
        {
            double x = input[i];

            int edge = i * kan->layers[layer_num].output_dim + j;
            int base = edge * kan->layers[layer_num].grid_size;

            if (x <= kan->layers[layer_num].grid_min)
            {
                left = 0;
                right = 0;
                t = 0;
            }

            else if (x >= kan->layers[layer_num].grid_max)
            {
                left= kan->layers[layer_num].grid_size-1;
                right = kan->layers[layer_num].grid_size-1;
            }

            else
            {
                position = (x - kan->layers[layer_num].grid_min) /
                    (kan->layers[layer_num].grid_max - kan->layers[layer_num].grid_min)
                    * ( kan->layers[layer_num].grid_size - 1);
                left = floor(position);
                right = left +1;
                t = position - left;
            }

            kan->layers[layer_num].last_left[edge] = left;
            kan->layers[layer_num].last_right[edge] = right;
            kan->layers[layer_num].last_t[edge] = t;


            double y = (1-t) * kan->layers[layer_num].tables[base+left] + t *
                kan->layers[layer_num].tables[base+right];
            sum += y;
        }
        output[j] = sum;
    }
}

static void kan_forward(void* impl, const double* input, double* output)
{
    KAN* kan = (KAN*)impl;

    for (int i = 0; i < kan->layers[0].input_dim; i++)
    {
        kan->activations[0][i] = input[i];
    }

    for (int i = 0; i < kan->layer_count-1; i++)
    {
        kan_layer_forward(kan, i, kan->activations[i], kan->activations[i+1]);
    }

    for (int i = 0; i < kan->sizes[kan->layer_count-1]; i++)
    {
        output[i] = kan->activations[kan->layer_count-1][i];
    }
}

static void kan_free(void* impl)
{
    KAN* kan = (KAN*)impl;

    for (int k = 0; k < kan->layer_count-1; k++)
    {
        free(kan->layers[k].last_left);
        free(kan->layers[k].last_right);
        free(kan->layers[k].last_t);
        free(kan->layers[k].grad_biases);
        free(kan->layers[k].grad_tables);
        free(kan->layers[k].tables);
        free(kan->layers[k].biases);
        free(kan->activations[k]);
    }

    free(kan->activations[kan->layer_count-1]);
    free(kan->activations);
    free(kan->layers);
    free(kan->sizes);
    free(kan);

}

static void kan_zero_grad(void* impl)
{
    KAN* kan = (KAN*)impl;

    for (int k = 0; k < kan->layer_count-1; k++)
    {
        int table_count = kan->layers[k].input_dim*kan->layers[k].output_dim*kan->layers[k].grid_size;
        for (int i = 0; i < table_count; i++)
        {
            kan->layers[k].grad_tables[i] = 0;
        }
        for (int i = 0; i < kan->layers[k].output_dim; i++)
        {
            kan->layers[k].grad_biases[i] = 0;
        }
    }
}


static int kan_parameter_count(void* impl)
{
    KAN* kan = (KAN*)impl;
    int size = 0;
    for (int i = 0; i < kan->layer_count-1; i++)
    {
        size += kan->layers[i].output_dim
            * kan->layers[i].input_dim
            * kan->layers[i].grid_size
            + kan->layers[i].output_dim;
    }
    return size;
}



static void kan_backward(void* impl, const double* input, const double* grad_output)
{
    KAN* kan = (KAN*)impl;
    kan_zero_grad(impl);
    (void)input; // чтоб не ругался

    double** deltas = malloc(sizeof(double*) * kan->layer_count);
    for (int l = 0; l < kan->layer_count; l++)
    {
        deltas[l] = calloc(kan->sizes[l], sizeof(double));
    }

    for (int j = 0; j < kan->sizes[kan->layer_count-1]; j++)
    {
        deltas[kan->layer_count-1][j] = grad_output[j];
    }

    for (int l = kan->layer_count-2; l >= 1; l--)
    {
        KANLayer* layer = &kan->layers[l];

        for (int i = 0; i < layer->input_dim; i++)
        {
            double grad = 0.0;

            for (int j = 0; j < layer->output_dim; j++)
            {
                double derivative = 0;

                int edge = i * layer->output_dim + j;
                int base = edge * layer->grid_size;

                int left = layer->last_left[edge];
                int right = layer->last_right[edge];

                if (left == right)
                {
                    derivative= 0.0;
                }

                else
                {
                    double step = (layer->grid_max - layer->grid_min) / (layer->grid_size - 1);

                    double left_value = layer->tables[base + left];
                    double right_value = layer->tables[base + right];

                    derivative = (right_value - left_value) / step;
                }

                grad += deltas[l + 1][j] * derivative;
            }

            deltas[l][i] = grad;
        }
    }


    for (int l = 0; l < kan->layer_count - 1; l++)
    {
        KANLayer* layer = &kan->layers[l];

        for (int j = 0; j < layer->output_dim; j++)
        {
            layer->grad_biases[j] += deltas[l + 1][j];
        }

        for (int i = 0; i < layer->input_dim; i++)
        {
            for (int j = 0; j < layer->output_dim; j++)
            {
                int edge = i * layer->output_dim + j;
                int base = edge * layer->grid_size;

                int left = layer->last_left[edge];
                int right = layer->last_right[edge];
                double t = layer->last_t[edge];
                double delta = deltas[l + 1][j];

                layer->grad_tables[base + left] += (1.0 - t) * delta;
                layer->grad_tables[base + right] += t * delta;
            }
        }
    }

    for (int i = 0; i < kan->layer_count; i++)
    {
        free(deltas[i]);
    }
    free(deltas);
}

static void kan_update(void* impl, double lr)
{
    KAN* kan = (KAN*)impl;

    for (int k = 0; k < kan->layer_count-1; k++)
    {
        int table_count = kan->layers[k].input_dim*kan->layers[k].output_dim*kan->layers[k].grid_size;
        for (int i = 0; i < table_count; i++)
        {
            kan->layers[k].tables[i] -= lr * kan->layers[k].grad_tables[i];
        }
        for (int i = 0; i < kan->layers[k].output_dim; i++)
        {
            kan->layers[k].biases[i] -= lr * kan->layers[k].grad_biases[i];
        }
    }

}



Model* kan_create(int *sizes, int layer_count, int grid_size)
{
    KAN* kan = (KAN*)malloc(sizeof(KAN));

    kan->layer_count = layer_count;
    kan->sizes = malloc(sizeof(int) * layer_count);

    for (int i = 0; i < layer_count; i++)
    {
        kan->sizes[i] = sizes[i];
    }

    kan->layers = malloc(sizeof(KANLayer) * (layer_count-1));
    kan->activations = malloc(sizeof(double*) * layer_count);

    for (int i = 0; i < kan->layer_count-1; i++)
    {

        kan->activations[i] = malloc(sizeof(double) * sizes[i]);

        int edge_count = sizes[i]*sizes[i+1];
        kan->layers[i].input_dim = sizes[i];
        kan->layers[i].output_dim = sizes[i+1];
        kan->layers[i].grid_size = grid_size;

        kan->layers[i].biases = calloc(sizes[i+1], sizeof(double));
        kan->layers[i].grad_biases = calloc(sizes[i+1], sizeof(double));

        kan->layers[i].last_left = malloc(sizeof(int) * edge_count);
        kan->layers[i].last_right = malloc(sizeof(int) * edge_count);
        kan->layers[i].last_t = malloc(sizeof(double) * edge_count);

        kan->layers[i].tables = malloc(sizeof(double) * edge_count*grid_size);
        kan->layers[i].grad_tables = calloc(edge_count*grid_size, sizeof(double));

        kan->layers[i].grid_min = -1;
        kan->layers[i].grid_max = 1;


        random_fill_uniform(kan->layers[i].tables,
                edge_count*grid_size, -0.1, 0.1);

    }
    kan->activations[layer_count-1] = malloc(sizeof(double) * sizes[layer_count-1]);

    Model* model = (Model*)malloc(sizeof(Model));
    model->impl = kan;
    model->input_dim = sizes[0];
    model->output_dim = sizes[layer_count-1];


    model->forward = kan_forward;
    model->backward = kan_backward;
    model->update = kan_update;
    model->free = kan_free;
    model->parameter_count = kan_parameter_count;
    model->zero_grad = kan_zero_grad;

    return model;
}