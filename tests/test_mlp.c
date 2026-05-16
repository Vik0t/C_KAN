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
    Model *mlp = mlp_create(sizes, 3);

    double x[1] = {0.5};
    double y[1];

    mlp->forward(mlp->impl, x, y);

    assert(isfinite(y[0]));

    printf("%f\n", y[0]);

    Dataset *dataset;
    dataset = dataset_create_function_1d(sin_func, -1, 1, 100);
    assert(dataset != NULL);
    
    double loss_before = validate(mlp, dataset, mse);
    train(mlp, dataset, 1000, 1e-3, 0);
    double loss_after = validate(mlp, dataset, mse);

    printf("loss before: %f\n", loss_before);
    printf("loss after:  %f\n", loss_after);
    assert(loss_after < loss_before);


    dataset_free(dataset);
    mlp->free(mlp->impl);
    free(mlp);
    return 0;
}
