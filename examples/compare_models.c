//
// Created by viktor on 5/4/26.
//
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "ckan.h"

int main(void)
{
    random_seed(42);

    int epochs = 300;
    int sample_count = 1000;
    double lr = 1e-3;

    Model* mlp = mlp_create(1, 16, 1);
    Model* kan = kan_create(1, 20, 1);

    Dataset* dataset = dataset_create_function_1d(sin_func, -1, 1, sample_count);
    assert(dataset != NULL);

    train(mlp, dataset, epochs, lr, 0);
    train(kan, dataset, epochs, lr, 0);

    printf("Task: sin(x), range [-1, 1], samples: %d\n", sample_count);
    printf("Model  Parameters  Epochs  LR        MSE\n");
    printf("MLP    %10d  %6d  %.6f  %.6f\n", mlp->parameter_count(mlp->impl), epochs, lr, validate(mlp, dataset, mse));
    printf("KAN    %10d  %6d  %.6f  %.6f\n", kan->parameter_count(kan->impl), epochs, lr, validate(kan, dataset, mse));

    dataset_free(dataset);

    kan->free(kan->impl);
    free(kan);

    mlp->free(mlp->impl);
    free(mlp);

    return 0;
}
