#pragma once

extern "C"
{
void deltaS_iterate_gpu(double* deltaS_new,const double* deltaS_old,const double* velocity_old,const long long size,const double dt);

void theta_iterate_gpu(double* theta_new,const double* theta_old,const double* omega_old,const long long size,const double dt);

void H_iterate_gpu(double* H_new,const double* H_old,const double* Delta_old,const long long size,const double dt);

void K_iterate_gpu(double* K, const double* deltaS, const double* theta, const long long size);

//void density_iterate(double* density, const double g);

void bodyforce_compute_gpu(double* Gravity, double* GravityCos, double* GravitySin, const double* density, const double* H, const double* theta, const double g, const long long size);

void omega_iterate_gpu(double* omega, const double* Omega, const double* deltaS, const long long size);

void velocity_iterate_gpu(double* velocity, const double* Delta, const double* deltaS, const long long size);
}