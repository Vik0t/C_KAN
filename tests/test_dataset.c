//
// Created by Vitya on 05.05.2026.
//
#include "ckan.h"
#include <stdio.h>

int main()
{

    Dataset *dataset;
    dataset = dataset_create_function_1d(sin_func, -1, 1, 100);

    if (dataset_to_csv(dataset,
        "C:/Users/Vitya/CLionProjects/C_KAN/tests/data.csv") == 0)
    {
        printf("success");
    }
    else
    {
        printf("fail");
    }

    return 0;
}