//
// Created by viktor on 5/4/26.
//
#include "trainer.h"

#include <stdio.h>
#include "loss.h"

void train_epoch(Model* model, Dataset* data, double lr) {
    for (int i = 0; i < data->size; i++) {

        double x[data->input_dim];
        double y[data->output_dim];
        double pred[data->output_dim];
        double grad[data->output_dim];

        for (int j = 0; j < data->input_dim; j++)
        {
            x[j] = data->X[j][i];
        }
        for (int k = 0; k < data->output_dim; k++)
        {
            y[k] = data->Y[k][i];
        }

        model->forward(model->impl, x, pred);

        for (int k = 0; k < data->output_dim; k++)
        {
            grad[k] = pred[k] - y[k];
        }

        model->backward(model->impl, x,  grad);
        model->update(model->impl, lr);

    }
}

double validate(Model* model, Dataset* data, double(*loss)(const double*, const double*, int)) {
    double x[data->input_dim];
    double y[data->output_dim];
    double pred[data->output_dim];


    double total_loss = 0;

    for (int i = 0; i < data->size; i++) {

        for (int j = 0; j < data->input_dim; j++)
        {
            x[j] = data->X[j][i];
        }

        for (int k = 0; k < data->output_dim; k++)
        {
            y[k] = data->Y[k][i];
        }

        model->forward(model->impl, x, pred);
        total_loss += loss(pred, y, data->output_dim);
    }

    return total_loss / data->size;
}


void train(Model* model, Dataset* data, int epochs, double lr, int verbose) {
    for (int e = 0; e < epochs; e++) {
        train_epoch(model, data, lr);
        double loss = validate(model, data, mse);
        if (verbose)
        {
            printf("epoch:%3d  loss: %5f\n", e, loss);
        }

    }
}

