#pragma once
#include"element_handle.h"
#include"element_iterate.h"
#include"model.h"
#include"SolverInterface.h"
#include"CusolverRfHandle.h"



extern "C"
{
bool ElongateGpu(ElementGroup& egold, ElementGroup& egnew, ModelConf& model);

void deltaS_iterate_gpu(ElementGroup& Egold,ElementGroup& Egnew,const double dt);

void theta_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew, const double dt);

void H_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew, const double dt);

void deltaS_theta_H_synchronize(ElementGroup& Egnew);

void K_iterate_gpu(ElementGroup& Egnew);

void density_iterate_gpu(ElementGroup& Egnew, ModelConf& model);

void bodyforce_compute_gpu(ElementGroup& Egnew);

void K_density_bodyforce_synchronize(ElementGroup& Egnew);

void surface_force_iterate_gpu(ElementGroup& Egnew,ModelConf& model,int iterating);

void Omega_Delta_iterate_gpu(ElementGroup& Egnew,ModelConf& model,SolverInterface* SolverHandle,bool ResetMatrix);

void omega_velocity_iterate_gpu(ElementGroup& Egnew, ModelConf& model, SolverInterface* handle);

void ClampedFreeGpu(ElementGroup& eg, CuVector<double>& vals);

void BodyForceGpu(ElementGroup& eg, CuVector<double>& b);

void SurfaceForceGpu(ElementGroup& eg, CuVector<double>& b);

}