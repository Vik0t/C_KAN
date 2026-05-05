//
// Created by viktor on 5/4/26.
//
#include "dataset.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


Dataset* dataset_create_function_1d(double (*f)(double), double from, double to, int n)
{
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->size = n;
    dataset->input_dim = 1;
    dataset->output_dim = 1;

    dataset->X = malloc(sizeof(double*) * dataset->input_dim);
    dataset->Y = malloc(sizeof(double*) * dataset->output_dim);

    dataset->X[0] = malloc(sizeof(double) * n);
    dataset->Y[0] = malloc(sizeof(double) * n);

    dataset->X[0][0] = from;
    dataset->X[0][n-1] = to;

    dataset->Y[0][0] = f(from);
    dataset->Y[0][n-1] = f(to);

    const double step = (to-from)/(double)n;

    for (int i = 1; i < n-1; i++)
    {
        dataset->X[0][i] = dataset->X[0][i-1] + step;
        dataset->Y[0][i] = f(dataset->X[0][i]);
    }

    return dataset;
}

void dataset_free(Dataset* data)
{
    for (int i = 0; i < data->input_dim; i++)
    {
        free(data->X[i]);
    }

    for (int i = 0; i < data->output_dim; i++)
    {
        free(data->Y[i]);
    }

    free(data->X);
    free(data->Y);
    free(data);
}

int dataset_to_csv(const Dataset* data, const char* filename)
{
    FILE *fpt = fopen(filename, "w");
    if (fpt == NULL) {
        printf("Ошибка открытия файла\n");
        return 1;
    }

    for (int i = 0; i < data->input_dim; i++)
    {
        fprintf(fpt, "input_%d,", i);
    }

    for (int i = 0; i < data->input_dim-1; i++)
    {
        fprintf(fpt, "target_%d,", i);
    }
    fprintf(fpt, "target_%d\n", data->input_dim-1);

    for (int i = 0; i < data->size; i++)
    {
        for (int j = 0; j < data->input_dim; j++)
        {
            fprintf(fpt, "%f,", data->X[j][i]);
        }

        for (int j = 0; j < data->output_dim-1; j++)
        {
            fprintf(fpt, "%f,", data->Y[j][i]);
        }
        fprintf(fpt, "%f", data->Y[data->output_dim-1][i]);

        fprintf(fpt, "\n");
    }



    fclose(fpt);
    return 0;


}


double sin_func(double x) {
    return sin(x);
}

