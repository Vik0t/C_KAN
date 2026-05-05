//
// Created by viktor on 5/4/26.
//
#include "random.h"
#include <stdlib.h>

void random_seed(unsigned int seed)
{
    srand(seed);
}

double random_uniform(double min, double max)
{
    double t = (double)rand() / (double)RAND_MAX;
    return min + t * (max - min);
}

void random_fill_uniform(double* array, int size, double min, double max)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = random_uniform(min, max);
    }
}

void random_fill_zero(double* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = 0.0;
    }
}