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

    Model* kan = kan_create(1, 20, 1);

    double x[1] = {0.5};
    double y[1];

    kan->forward(kan->impl, x, y);

    assert(isfinite(y[0]));

    printf("%f\n", y[0]);

    Dataset *dataset;
    dataset = dataset_create_function_1d(sin_func, -1, 1, 1000);
    assert(dataset != NULL);

    train(kan, dataset, 1000, 1e-3, 0);


    dataset_free(dataset);
    kan->free(kan->impl);
    free(kan);
    return 0;
}
