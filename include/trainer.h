//
// Created by viktor on 5/4/26.
//



#ifndef TRAINER_H
#define TRAINER_H

#include "model.h"
#include "dataset.h"

void train(Model* model, Dataset* data, int epochs, double lr, int verbose);
double validate(Model* model, Dataset* data, double(*loss)(const double*, const double*, int));
#endif



