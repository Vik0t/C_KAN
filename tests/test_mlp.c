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

    Model* mlp = mlp_create(1, 16, 1);

    double x[1] = {0.5};
    double y[1];

    mlp->forward(mlp->impl, x, y);

    assert(isfinite(y[0]));

    printf("%f\n", y[0]);

    Dataset *dataset;
    dataset = dataset_create_function_1d(sin_func, -1, 1, 100);
    assert(dataset != NULL);

    train(mlp, dataset, 100, 1e-3);


    dataset_free(dataset);
    mlp->free(mlp->impl);
    free(mlp);
    return 0;
}
