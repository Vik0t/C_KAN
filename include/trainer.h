//
// Created by viktor on 5/4/26.
//



#ifndef TRAINER_H
#define TRAINER_H

#include "model.h"
#include "dataset.h"

void train(Model* model, Dataset* data, int epochs, double lr);

#endif



