//
// Created by viktor on 5/4/26.
//

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

    printf("%f\n", y[0]);

    mlp->free(mlp->impl);
    free(mlp);
    return 0;
}
