//
// Created by Vitya on 05.05.2026.
//

#ifndef RANDOM_H
#define RANDOM_H


void random_seed(unsigned int seed);
double random_uniform(double min, double max);
void random_fill_uniform(double* array, int size, double min, double max);
void random_fill_zero(double* array, int size);


#endif //RANDOM_H
