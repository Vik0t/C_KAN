//
// Created by viktor on 5/4/26.
//
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

#include "ckan.h"

int main(void)
{
    random_seed(42);

    int epochs = 100;
    int n = 5;
    double lr = 1e-3;


    int sizes[] = {2, 16, 1};
    int mlp_sizes[] = {2, 16, 1};

    Model *mlp = mlp_create(mlp_sizes, 3);
    Model* kan = kan_create(sizes, 3, 16);

    Dataset* dataset = dataset_create_grid_2d(bespalov, -1, 1, n);
    assert(dataset != NULL);

    double* input = fftw_malloc(sizeof(double) * n * n);
    fftw_complex* freq = fftw_malloc(sizeof(fftw_complex) * n * (n / 2 + 1));
    double* output = fftw_malloc(sizeof(double) * n * n);

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


    train(mlp, dataset, epochs, lr, 0);
    train(kan, dataset, epochs, lr, 0);

    printf("Task: sin(x), range [-1, 1], samples: %d\n", n);
    printf("Model  Parameters  Epochs  LR        MSE\n");
    printf("MLP    %10d  %6d  %.6f  %.6f\n", mlp->parameter_count(mlp->impl), epochs, lr, validate(mlp, dataset, mse));
    printf("KAN    %10d  %6d  %.6f  %.6f\n", kan->parameter_count(kan->impl), epochs, lr, validate(kan, dataset, mse));
    printf("FFT    %10d  %6d  %.6f  %.6f\n", -1, epochs, lr, mse(dataset->Y[0], output, n*n));

    dataset_free(dataset);

    kan->free(kan->impl);
    free(kan);

    mlp->free(mlp->impl);
    free(mlp);

    return 0;
}
