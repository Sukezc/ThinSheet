#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<sm_60_atomic_functions.h>
#include"element_iterate_gpu.h"
#include<thrust/device_vector.h>
#include<thrust/execution_policy.h>
#include<thrust/transform_scan.h>
#include<cmath>


constexpr int threads = 128;

__global__ void deltaS_iterate_kernel(double* deltaS_new, const double* deltaS_old, const double* velocity_old, const long long size, const double dt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i>0 && i < size) deltaS_new[i] = deltaS_old[i] + dt * (velocity_old[i - 1] - velocity_old[i]);
}

__global__ void theta_iterate_kernel(double* theta_new, const double* theta_old, const double* omega_old, const long long size, const double dt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) theta_new[i] = theta_old[i] + dt * omega_old[i];
}

__global__ void H_iterate_kernel(double* H_new, const double* H_old, const double* Delta_old, const long long size, const double dt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		double H_temp = H_old[i], Delta_temp = Delta_old[i];
		double k1 = -H_temp * Delta_temp;
		double k2 = -(H_temp + dt / 2.0 * k1) * Delta_temp;
		double k3 = -(H_temp + dt / 2.0 * k2) * Delta_temp;
		double k4 = -(H_temp + dt * k3) * Delta_temp;
		H_new[i] = H_temp + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
	}
}

__global__ void K_iterate_kernel(double* K, const double* deltaS, const double* theta, const long long size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > 0 && i < size - 1)
	{
		double dSj = deltaS[i];
		double dSj_1 = deltaS[i + 1];
		K[i] = theta[i + 1] * (-dSj) / (dSj_1 * (dSj_1 + dSj)) +
			theta[i] * (dSj - dSj_1) / (dSj * dSj_1) +
			theta[i - 1] * (dSj_1) / ((dSj + dSj_1) * dSj);
	}
}

template<typename Func>
__global__ void bodyforce_compute_kernel(double* GravityBase,const double* density, const double* H, const double* theta, const double g, const long long size, Func func)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) GravityBase[i] = H[i] * density[i] * g * func(theta[i]);
}

__global__ void omega_iterate_kernel(double* omega, double* omega_local, const double* Omega, const double* deltaS,const long long size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

}

__global__ void velocity_iterate_kernel(double* velocity, double* velocity_local,const double* Delta, const double* deltaS, const long long size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

}

extern "C"
{
	void deltaS_iterate_gpu(ElementGroup& Egold,ElementGroup& Egnew,const double dt)
	{
		Egold.deltaSGroup.send(); Egold.velocityGroup.send();
		deltaS_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.deltaSGroup.data(CuVecDev),Egold.deltaSGroup.data(CuVecDev), Egold.velocityGroup.data(CuVecDev),Egold.size,dt);
	}

	void theta_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew, const double dt)
	{
		Egold.thetaGroup.send(); Egold.omegaGroup.send();
		theta_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.thetaGroup.data(CuVecDev),Egold.thetaGroup.data(CuVecDev),Egold.omegaGroup.data(CuVecDev),Egold.size - 1,dt);
	}

	void H_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew,const double dt)
	{
		Egold.HGroup.send(); Egold.DeltaGroup.send();
		H_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.HGroup.data(CuVecDev), Egold.HGroup.data(CuVecDev), Egold.DeltaGroup.data(CuVecDev),Egold.size, dt);
	}

	void deltaS_theta_H_synchronize(ElementGroup& Egnew)
	{
		cudaDeviceSynchronize();
		Egnew.deltaSGroup.fetch(); Egnew.thetaGroup.fetch(); Egnew.HGroup.fetch();
	}

	void K_iterate_gpu(double* K, const double* deltaS, const double* theta, const long long size)
	{
		//compute the outside point 
		double dSn_1 = deltaS[1];
		double dSn_2 = deltaS[2];
		K[0] = theta[2] * dSn_1 / (dSn_2 * (dSn_2 + dSn_1)) +
			theta[1] * (-dSn_1 - dSn_2) / (dSn_1 * dSn_2) +
			theta[0] * (dSn_2 + 2.0 * dSn_1) / ((dSn_1 + dSn_2) * dSn_1);

		//compute the inner point
		double dS0 = deltaS[size - 1];
		double dS1 = deltaS[size - 2];
		K[size - 1] = theta[size - 1] * (-2.0 * dS0 - dS1) / ((dS0 + dS1) * dS0) +
			theta[size - 2] * (dS0 + dS1) / (dS0 * dS1) +
			theta[size - 3] * (-dS0) / ((dS1 + dS0) * dS1);

		K_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (K, deltaS, theta, size);
	}

	void bodyforce_compute_gpu(double* Gravity, double* GravityCos, double* GravitySin, const double* density, const double* H, const double* theta, const double g, const long long size)
	{
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (Gravity, density, H, theta, g, size, []__device__(double i) { return 1.0; });
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (GravityCos, density, H, theta, g, size, [=] __device__(double i) { return cos(i); });
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (GravitySin, density, H, theta, g, size, [=] __device__(double i) { return sin(i); });
	}

	void omega_velocity_iterate_gpu(double* omega, double omega_local, double* velocity, double* velocity_local, const double* Omega, const double* Delta, const double* deltaS, const long long size)
	{
		omega[size - 1] = 0.0;
		omega[size - 2] = -(Omega[size - 1] + Omega[size - 2]) * deltaS[size - 1] / 2.0;
		velocity[size - 1] = 0.0;
		velocity[size - 2] = (Delta[size - 1] + Delta[size - 2]) * deltaS[size - 1] / 2.0;
		omega_iterate_kernel << <(size + threads - 1) / threads, threads >> > (omega,omega_local,Omega,deltaS, size);
		velocity_iterate_kernel << <(size + threads - 1) / threads, threads >> > (velocity,velocity_local,Delta, deltaS,size);
		cudaDeviceSynchronize();

	}

}