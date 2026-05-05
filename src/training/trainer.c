//
// Created by viktor on 5/4/26.
//
#include "trainer.h"

void train(Model* model, Dataset* data, int epochs, double lr) {
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < data->size; i++) {
            double* x = data->X[i];
            double* y = data->Y[i];

            double pred[1];

            model->forward(model->impl, x, pred);

            double grad[1];
            grad[0] = pred[0] - y[0];

            //model->backward(model->impl, grad);
            model->update(model->impl, lr);
        }
    }
}