//
// Created by Vitya on 05.05.2026.
//
#include <fftw3.h>
#include "ckan.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main()
{

    int n = 64;

    double* input = fftw_malloc(sizeof(double) * n * n);
    fftw_complex* freq = fftw_malloc(sizeof(fftw_complex) * n * (n / 2 + 1));
    double* output = fftw_malloc(sizeof(double) * n * n);


    Dataset *dataset;
    dataset = dataset_create_grid_2d(bespalov, -1, 1, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int index = i * n + j;
            input[index] = dataset->Y[0][index];
        }

    }

    fftw_plan forward = fftw_plan_dft_r2c_2d(n, n, input, freq, FFTW_ESTIMATE);
    fftw_execute(forward);


    fftw_plan backward = fftw_plan_dft_c2r_2d(n, n, freq, output, FFTW_ESTIMATE);
    fftw_execute(backward);

    for (int i = 0; i < n*n; i++)
    {
        output[i] /= n*n;
    }

    double error = mse(dataset->Y[0], output, n*n);
    printf("%.15e\n", error);


    dataset_free(dataset);
    fftw_destroy_plan(forward);
    fftw_destroy_plan(backward);
    fftw_free(input);
    fftw_free(freq);
    fftw_free(output);

    return 0;
}