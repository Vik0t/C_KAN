//
// Created by Vitya on 05.05.2026.
//
#include "ckan.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main()
{

    Dataset *dataset;
    dataset = dataset_create_function_1d(sin_func, -1, 1, 100);

    assert(dataset != NULL);
    assert(dataset->size == 100);
    assert(dataset->input_dim == 1);
    assert(dataset->output_dim == 1);
    assert(dataset->X[0][0] == -1.0);
    assert(fabs(dataset->X[0][99] - 1.0) < 1e-9);


    assert(dataset_to_csv(dataset,
        "C:/Users/Vitya/CLionProjects/C_KAN/tests/data.csv") == 0);

    dataset_free(dataset);

    return 0;
}