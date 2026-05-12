//
// Created by viktor on 5/4/26.
//
#include <assert.h>
#include <math.h>

#include "ckan.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{

    random_seed(42);


    int sizes[] = {1, 16, 1};
    Model* kan = kan_create(sizes, 3, 20);

    double x[1] = {0.5};
    double y[1];

    kan->forward(kan->impl, x, y);

    assert(isfinite(y[0]));

    printf("%f\n", y[0]);

    Dataset *dataset;
    dataset = dataset_create_function_1d(sin_func, -1, 1, 1000);
    assert(dataset != NULL);

    double loss_before = validate(kan, dataset, mse);
    train(kan, dataset, 1000, 1e-3, 0);
    double loss_after = validate(kan, dataset, mse);

    printf("loss before: %f\n", loss_before);
    printf("loss after:  %f\n", loss_after);
    assert(loss_after < loss_before);


    dataset_free(dataset);
    kan->free(kan->impl);
    free(kan);
    return 0;
}
