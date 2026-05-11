//
// Created by viktor on 5/4/26.
//
#include "trainer.h"

#include <stdio.h>
#include "loss.h"

void train_epoch(Model* model, Dataset* data, double lr) {
    double pred[data->size];
    double y[data->size];
    for (int i = 0; i < data->size; i++) {
        double x[1];

        x[0] = data->X[0][i];
        y[i] = data->Y[0][i];

        model->forward(model->impl, x, &pred[i]);

        double grad[1];
        grad[0] = pred[i] - y[i];

        model->backward(model->impl, x,  grad);
        model->update(model->impl, lr);
    }
}

double validate(Model* model, Dataset* data, double(*loss)(const double*, const double*, int)) {
    double pred[data->size];
    double y[data->size];
    double x[1];
    for (int i = 0; i < data->size; i++) {
        x[0] = data->X[0][i];
        y[i] = data->Y[0][i];

        model->forward(model->impl, x, &pred[i]);
    }
    return loss(pred, y, data->size);

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

