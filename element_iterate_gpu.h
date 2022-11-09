#pragma once
#include"element_handle.h"
#include"model.h"
#include"SolverInterface.h"
extern "C"
{
void deltaS_iterate_gpu(ElementGroup& Egold,ElementGroup& Egnew,const double dt);

void theta_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew, const double dt);

void H_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew, const double dt);

void deltaS_theta_H_synchronize(ElementGroup& Egnew);

void K_iterate_gpu(ElementGroup& Egnew);

//void density_iterate(ElementGroup&);

void bodyforce_compute_gpu(ElementGroup& Egnew);

void bodyforce_K_synchronize(ElementGroup& Egnew);

void omega_velocity_iterate_gpu(ElementGroup& Egnew, ModelConf& model, SolverInterface* handle);

}