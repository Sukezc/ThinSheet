#include"element_iterate_gpu.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<sm_60_atomic_functions.h>
#include<thrust/device_vector.h>
#include<thrust/execution_policy.h>
#include<thrust/transform_scan.h>
#include<thrust/reverse.h>
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
	if (i>0 && i < size - 1)
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

__global__ void omega_iterate_kernel(double* omega, const double* Omega, const double* deltaS,const long long size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > 0 && i < size - 1)
	{
		double dSj = deltaS[i], dSj_1 = deltaS[i + 1];
		omega[i-1] = Omega[i + 1] * (dSj + dSj_1) * (2.0 * dSj_1 - dSj) / 6.0 / dSj_1 +
			Omega[i] * (dSj + dSj_1) * (dSj + dSj_1) * (dSj + dSj_1) / 6.0 / dSj / dSj_1 +
			Omega[i - 1] * (dSj + dSj_1) * (2.0 * dSj - dSj_1) / 6.0 / dSj;
	}
}

__global__ void velocity_iterate_kernel(double* velocity,const double* Delta, const double* deltaS, const long long size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > 0 && i < size - 1)
	{
		double dSj = deltaS[i], dSj_1 = deltaS[i + 1];
		velocity[i-1] = Delta[i + 1] * (dSj + dSj_1) * (2.0 * dSj_1 - dSj) / 6.0 / dSj_1 +
			Delta[i] * (dSj + dSj_1) * (dSj + dSj_1) * (dSj + dSj_1) / 6.0 / dSj / dSj_1 +
			Delta[i - 1] * (dSj + dSj_1) * (2.0 * dSj - dSj_1) / 6.0 / dSj;
	}
}

__global__ void velocity_omega_iterate_kernel(double* velocity_or_omega, const double* Delta_or_Omega, const double* deltaS, const long long size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > 0 && i < size - 1)
	{
		double dSj = deltaS[i], dSj_1 = deltaS[i + 1];
		velocity_or_omega[i - 1] = Delta_or_Omega[i + 1] * (dSj + dSj_1) * (2.0 * dSj_1 - dSj) / 6.0 / dSj_1 +
			Delta_or_Omega[i] * (dSj + dSj_1) * (dSj + dSj_1) * (dSj + dSj_1) / 6.0 / dSj / dSj_1 +
			Delta_or_Omega[i - 1] * (dSj + dSj_1) * (2.0 * dSj - dSj_1) / 6.0 / dSj;
	}
}

template<typename Func>
__global__ void velocity_omega_aux_kernel(double* velocity_or_omega, const double* Delta_or_Omega, const double* deltaS, const long long size, Func func)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size-2)
	{
		double temp = (Delta_or_Omega[size - 1] + Delta_or_Omega[size - 2]) / 2.0 * deltaS[size - 1] + (Delta_or_Omega[i] + Delta_or_Omega[i + 1]) / 2.0 * deltaS[i + 1];
		velocity_or_omega[i] = func(velocity_or_omega[i], temp) / 2.0;
	}
}

__global__ void surface_force_sdirection_aux(double* b,const double * H,const double* Pup, const double* Pdown, const double * Tup, const double * Tdown,const double* deltaS,const long long size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	long long n = size - 1;
	if (i > 0 && i < n)
	{
		double Hi = H[i], Hi_1 = H[i + 1], Pip = Pup[i], Pin = Pdown[i], Tip = Tup[i], Tin = Tdown[i],
			Pi_1p = Pup[i + 1], Pi_1n = Pdown[i + 1], Ti_1p = Tup[i + 1], Ti_1n = Tdown[i + 1],
			dSi_1 = deltaS[i + 1];
		b[2 * n + 1 - i] += -Tip + Tin - Hi / 2.0 / dSi_1 * (Pip + Pin - Pi_1p - Pi_1n);
	}
}


__global__ void surface_force_zdirection_aux(double* b, const double* H, const double* Pup, const double* Pdown, const double* Tup, const double* Tdown, const double* deltaS,const long long size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	long long n = size - 1;
	if (i > 0 && i < n)
	{
		double Hi = H[i], Hi_1 = H[i + 1], Pip = Pup[i], Pin = Pdown[i], Tip = Tup[i], Tin = Tdown[i],
			Pi_1p = Pup[i + 1], Pi_1n = Pdown[i + 1], Ti_1p = Tup[i + 1], Ti_1n = Tdown[i + 1],
			dSi_1 = deltaS[i + 1];
		b[n - i] += -Pip + Pin - Hi / 2.0 / dSi_1 * (Tip + Tin - Ti_1p - Ti_1n) - (Hi - Hi_1) / dSi_1 * (Tip + Tin);
	}
}

__global__ void clampedfree_valsA_kernel(double* vals, const double* deltaS, const double* K, const double miu, const int n, const int logic)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	// i = [n-1,1]
	// data : i =        [n-1:1]
	//result: i = 4||6 * [0:n-2]
	if (i && i < n)
	{
		if (!logic)
		{

			double dSi = deltaS[i], dSi_1 = deltaS[i + 1], Ki = K[i];// miu = eg.viscosity;
			vals[4 * (n - 1 - i)] = miu / 3.0 * 2.0 / dSi_1 / (dSi + dSi_1);
			vals[4 * (n - 1 - i) + 1] = -miu / 3.0 * 2.0 / dSi / dSi_1 + 5.0 * miu / 6.0 * Ki * Ki;
			vals[4 * (n - 1 - i) + 2] = miu / 3.0 * 2.0 / dSi / (dSi + dSi_1);
			vals[4 * (n - 1 - i) + 3] = 4.0 * miu * Ki;
		}
	
		else
		{
		
			double dSi = deltaS[i], dSi_1 = deltaS[i + 1], Ki = K[i], Ki_1 = K[i + 1], Kip1 = K[i - 1];// miu = eg.viscosity;
			vals[6 * (n - 1 - i)] = -miu / 2.0 * Ki * dSi / dSi_1 / (dSi_1 + dSi);
			vals[6 * (n - 1 - i) + 1] = miu / 2.0 * Ki * (dSi - dSi_1) / dSi / dSi_1 + 5.0 * miu / 6.0 / dSi / dSi_1 / (dSi_1 + dSi) * (Kip1 * dSi_1 * dSi_1 - Ki_1 * dSi * dSi + Ki * (dSi * dSi - dSi_1 * dSi_1));
			vals[6 * (n - 1 - i) + 2] = miu / 2.0 * Ki * dSi_1 / dSi / (dSi_1 + dSi);

			vals[6 * (n - 1 - i) + 3] = -4.0 * miu * dSi / dSi_1 / (dSi + dSi_1);
			vals[6 * (n - 1 - i) + 4] = 4.0 * miu * (dSi - dSi_1) / dSi / dSi_1;
			vals[6 * (n - 1 - i) + 5] = 4.0 * miu * dSi_1 / dSi / (dSi_1 + dSi);
		
		}
	}
}

extern "C"
{
	
	bool ElongateGpu(ElementGroup& egold, ElementGroup& egnew, ModelConf& model)
	{
		if (model.extrudepolicy.policy == ExtrudePolicy::Sparse)
		{
			if (!(model.extrudepolicy.iterating % model.extrudepolicy.SparseNum))
			{
				egold.elongateGpu(model.extrudepolicy.DsEnd, model.H, model.velocity);
				egnew.elongateGpu(model.extrudepolicy.DsEnd, model.H, model.velocity);
				model.extrudepolicy.iterating++;
				model.grid_num = egold.size;
				model.Standardize();
				return true;
			}
			model.extrudepolicy.iterating++;
			return false;
		}
		else if (model.extrudepolicy.policy == ExtrudePolicy::Dense)
		{

			egold.elongateGpu(model.extrudepolicy.Ds, model.H, model.velocity, model.extrudepolicy.DenseNum);
			egold.elongateGpu(model.extrudepolicy.DsEnd, model.H, model.velocity);


			egnew.elongateGpu(model.extrudepolicy.Ds, model.H, model.velocity, model.extrudepolicy.DenseNum);
			egnew.elongateGpu(model.extrudepolicy.DsEnd, model.H, model.velocity);

			model.grid_num = egold.size;
			model.Standardize();
			return true;
		}
	}

	void deltaS_iterate_gpu(ElementGroup& Egold,ElementGroup& Egnew,const double dt)
	{
		auto size = Egnew.deltaSGroup.size(CVD);
		//Egnew.deltaSGroup.SyncSize(HostToDevice()); Egold.deltaSGroup.send(); Egold.velocityGroup.send();
		deltaS_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.deltaSGroup.data(CVD),Egold.deltaSGroup.data(CVD), Egold.velocityGroup.data(CVD),size,dt);
	}

	void theta_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew, const double dt)
	{
		auto size = Egnew.thetaGroup.size(CVD);
		//Egnew.thetaGroup.SyncSize(HostToDevice()); Egold.thetaGroup.send(); Egold.omegaGroup.send();
		theta_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.thetaGroup.data(CVD),Egold.thetaGroup.data(CVD),Egold.omegaGroup.data(CVD),size - 1,dt);
	}

	void H_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew,const double dt)
	{
		auto size = Egnew.HGroup.size(CVD);
		//Egnew.HGroup.SyncSize(HostToDevice()); Egold.HGroup.send(); Egold.DeltaGroup.send();
		H_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.HGroup.data(CVD), Egold.HGroup.data(CVD), Egold.DeltaGroup.data(CVD),size, dt);
	}

	void deltaS_theta_H_synchronize(ElementGroup& Egnew)
	{
		cudaDeviceSynchronize();
		//Egnew.deltaSGroup.fetch(); Egnew.thetaGroup.fetch(); Egnew.HGroup.fetch();
		Egnew.thetaGroup.back(CVD) = 0.0; Egnew.deltaSGroup(0) = Egnew.deltaSGroup(1);
	}

	void K_iterate_gpu(ElementGroup& Egnew)
	{
		//Egnew.KGroup.SyncSize(HostToDevice());
		auto size = Egnew.KGroup.size(CVD);
		//compute the outside point 
		double dSn_1 = Egnew.deltaSGroup(1);
		double dSn_2 = Egnew.deltaSGroup(2);
		Egnew.KGroup(0) = Egnew.thetaGroup(2) * dSn_1 / (dSn_2 * (dSn_2 + dSn_1)) +
			Egnew.thetaGroup(1) * (-dSn_1 - dSn_2) / (dSn_1 * dSn_2) +
			Egnew.thetaGroup(0) * (dSn_2 + 2.0 * dSn_1) / ((dSn_1 + dSn_2) * dSn_1);

		//compute the inner point
		double dS0 = Egnew.deltaSGroup(size - 1);
		double dS1 = Egnew.deltaSGroup(size - 2);
		Egnew.KGroup(size - 1) = Egnew.thetaGroup(size - 1) * (-2.0 * dS0 - dS1) / ((dS0 + dS1) * dS0) +
			Egnew.thetaGroup(size - 2) * (dS0 + dS1) / (dS0 * dS1) +
			Egnew.thetaGroup(size - 3) * (-dS0) / ((dS1 + dS0) * dS1);

		K_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.KGroup.data(CVD), Egnew.deltaSGroup.data(CVD), Egnew.thetaGroup.data(CVD),size);
	}

	void density_iterate_gpu(ElementGroup& Egnew,ModelConf& model)
	{
		//Egnew.densityGroup.send();
	}

	void bodyforce_compute_gpu(ElementGroup& Egnew)
	{
		auto size = Egnew.GravityGroup.size(CVD);
		//Egnew.GravityGroup.SyncSize(HostToDevice()); Egnew.GravityGroupCos.SyncSize(HostToDevice()); Egnew.GravityGroupSin.SyncSize(HostToDevice());
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.GravityGroup.data(CVD), Egnew.densityGroup.data(CVD), Egnew.HGroup.data(CVD), Egnew.thetaGroup.data(CVD), Egnew.g, size, []__device__(double i) { return 1.0; });
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.GravityGroupCos.data(CVD), Egnew.densityGroup.data(CVD), Egnew.HGroup.data(CVD), Egnew.thetaGroup.data(CVD), Egnew.g, size, [=] __device__(double i) { return cos(i); });
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.GravityGroupSin.data(CVD), Egnew.densityGroup.data(CVD), Egnew.HGroup.data(CVD), Egnew.thetaGroup.data(CVD), Egnew.g, size, [=] __device__(double i) { return sin(i); });
	}

	void K_density_bodyforce_synchronize(ElementGroup& Egnew)
	{
		cudaDeviceSynchronize();
		//Egnew.KGroup.trans(Egnew.KGroup.begin(CVD) + 1, Egnew.KGroup.end(CVD) - 1, Egnew.KGroup.begin() + 1); Egnew.GravityGroup.fetch(); Egnew.GravityGroupCos.fetch(); Egnew.GravityGroupSin.fetch();
	}

	void surface_force_iterate_gpu(ElementGroup& Egnew, ModelConf& model, int iterating)
	{

	}

	void Omega_Delta_iterate_gpu(ElementGroup& Egnew, ModelConf& model, SolverInterface* SolverHandle, bool ResetMatrix)
	{
		//the number of length element
		long long n = Egnew.size - 1;
		static CuVector<double> vals;  static CuVector<double> b; static CuVector<int> rowPtr; static CuVector<int> colInd;
		
		switch (model.boundaryCondition)
		{
		case BoundaryCondition::ClampedFree:
			if (ResetMatrix) { 
				Egnew.deltaSGroup.fetch(); Egnew.HGroup.fetch(); Egnew.KGroup.fetch();
				ClampedFree(Egnew, vals, rowPtr, colInd); 
			}
			else ClampedFreeGpu(Egnew, vals); 
			break;
		case BoundaryCondition::ClampedBoth:
			//ClampedBoth(Egnew, vals, rowPtr, colInd); 
			break;
		default:
			break;
		}
		

		if (ResetMatrix)
		{
			SolverHandle->Reset();
			SolverHandle->Initialize(vals.data(), rowPtr.data(), colInd.data(),vals.size(),rowPtr.size());
		}
		else
		{
			reinterpret_cast<CusolverRfHandle*>(SolverHandle)->ResetAGpu(vals.data(CVD));
		}
		CuVector<double>& X = *static_cast<CuVector<double>*>(SolverHandle->getContainer());
		thrust::for_each(X.begin(CVD), X.end(CVD), []__device__(auto & it) { it = 0.0; });

		switch (model.forceCondition)
		{
		case ForceCondition::BodyForceOnly:
			BodyForceGpu(Egnew,X); break;
		case ForceCondition::SurfaceAndBodyForce:
			SurfaceForceGpu(Egnew,X); BodyForceGpu(Egnew,X); break;
		case ForceCondition::SurfaceForceOnly:
			SurfaceForceGpu(Egnew,X); break;
		default:
			break;
		}

		SolverHandle->solve();
	}

	void omega_velocity_iterate_gpu(ElementGroup& Egnew, ModelConf& model,SolverInterface* handle)
	{
		CuVector<double>& X = *static_cast<CuVector<double>*>(handle->getContainer());
		auto size = X.size(CVD)/2;
		//Egnew.omegaGroup.SyncSize(HostToDevice());
		//Egnew.velocityGroup.SyncSize(HostToDevice());
		
		thrust::reverse(X.begin(CVD), X.end(CVD));
		thrust::transform(X.begin(CVD), X.begin(CVD) + size, Egnew.HGroup.begin(CVD), X.begin(CVD), []__device__(auto & it1, auto & it2) { return it1 / it2; });
		thrust::transform(X.begin(CVD) + size, X.end(CVD), Egnew.HGroup.begin(CVD), X.begin(CVD) + size, []__device__(auto & it1, auto & it2) { return it1 / it2 / it2 / it2; });
		thrust::copy(X.begin(CVD), X.begin(CVD) + size, Egnew.DeltaGroup.begin(CVD));
		thrust::copy(X.begin(CVD) + size, X.end(CVD), Egnew.OmegaGroup.begin(CVD));
		
		Egnew.omegaGroup(size - 1) = 0.0;
		Egnew.omegaGroup(size - 2) = -(Egnew.OmegaGroup(size - 1) + Egnew.OmegaGroup(size - 2)) * Egnew.deltaSGroup(size - 1) / 2.0;
		Egnew.velocityGroup(size - 1) = 0.0;
		Egnew.velocityGroup(size - 2) = (Egnew.DeltaGroup(size - 1) + Egnew.DeltaGroup(size - 2)) * Egnew.deltaSGroup(size - 1) / 2.0;
		
		omega_iterate_kernel << <(size + threads - 1) / threads, threads >> > (Egnew.omegaGroup.data(CVD),Egnew.OmegaGroup.data(CVD),Egnew.deltaSGroup.data(CVD),size);
		velocity_iterate_kernel << <(size + threads - 1) / threads, threads >> > (Egnew.velocityGroup.data(CVD),Egnew.DeltaGroup.data(CVD), Egnew.deltaSGroup.data(CVD),size);
		cudaDeviceSynchronize();
		//0~size-3
		
		thrust::transform_inclusive_scan(Egnew.omegaGroup.rbegin(CVD)+2, Egnew.omegaGroup.rend(CVD), Egnew.omegaGroup.rbegin(CVD)+2, thrust::negate<double>(), thrust::plus<double>());
		thrust::inclusive_scan(Egnew.velocityGroup.rbegin(CVD) + 2, Egnew.velocityGroup.rend(CVD), Egnew.velocityGroup.rbegin(CVD) + 2, thrust::plus<double>());
		velocity_omega_aux_kernel << < (size + threads - 1) / threads, threads >> > (Egnew.velocityGroup.data(CVD), Egnew.DeltaGroup.data(CVD), Egnew.deltaSGroup.data(CVD), size, thrust::plus<double>());
		velocity_omega_aux_kernel << < (size + threads - 1) / threads, threads >> > (Egnew.omegaGroup.data(CVD), Egnew.OmegaGroup.data(CVD), Egnew.deltaSGroup.data(CVD), size, thrust::minus<double>());
		cudaDeviceSynchronize();
		
		double C;
		//if(model.omegaStandard.first < size - 2)
		C = model.omegaStandard.second - Egnew.omegaGroup(model.omegaStandard.first);
		//else 
		//	C = model.omegaStandard.second - Egnew.omegaGroup[model.omegaStandard.first];
		thrust::for_each(Egnew.omegaGroup.begin(CVD), Egnew.omegaGroup.end(CVD), [=]__device__(double& it) {it += C; });
		//Egnew.omegaGroup[size - 1] += C; Egnew.omegaGroup[size - 2] += C;
		//if(model.velocityStandard.first < size - 2)
		C = model.velocityStandard.second - Egnew.velocityGroup(model.velocityStandard.first);
		//else
		//	C = model.velocityStandard.second - Egnew.velocityGroup[model.velocityStandard.first];
		thrust::for_each(Egnew.velocityGroup.begin(CVD), Egnew.velocityGroup.end(CVD), [=]__device__(double& it) {it += C; });
		//Egnew.velocityGroup[size - 1] += C; Egnew.velocityGroup[size - 2] += C;
		
		//Egnew.velocityGroup.trans(Egnew.velocityGroup.begin(CVD), Egnew.velocityGroup.end(CVD) - 2, Egnew.velocityGroup.begin());
		//Egnew.omegaGroup.trans(Egnew.omegaGroup.begin(CVD), Egnew.omegaGroup.end(CVD) - 2, Egnew.omegaGroup.begin());
		
	}

	void ClampedFreeGpu(ElementGroup& eg, CuVector<double>& vals)
	{
	
		long long n = eg.size - 1;
		vals.resize(10 * n - 2); 

		clampedfree_valsA_kernel << <(n + threads - 1) / threads, threads >> > (vals.data(CVD) + 3, eg.deltaSGroup.data(CVD), eg.KGroup.data(CVD), eg.viscosity, n, 0);
		clampedfree_valsA_kernel << <(n + threads - 1) / threads, threads >> > (vals.data(CVD) + 4 * n + 3, eg.deltaSGroup.data(CVD), eg.KGroup.data(CVD), eg.viscosity, n, 1);
		double dSn_2 = eg.deltaSGroup(2), dSn_1 = eg.deltaSGroup(1), H0 = eg.HGroup(0), H1 = eg.HGroup(1), H2 = eg.HGroup(2);
		vals(0) = dSn_1 / dSn_2 / (dSn_1 + dSn_2) / H2;
		vals(1) = (-dSn_1 - dSn_2) / dSn_1 / dSn_2 / H1;
		vals(2) = (dSn_2 + 2.0 * dSn_1) / dSn_1 / (dSn_1 + dSn_2) / H0;

		// 4 * (n - 1) + 3 = 4 * n - 1
		vals(4 * n - 1) = dSn_1 / dSn_2 / (dSn_1 + dSn_2) / H2 / H2 / H2;
		vals(4 * n) = (-dSn_1 - dSn_2) / dSn_1 / dSn_2 / H1 / H1 / H1;
		vals(4 * n + 1) = (dSn_2 + 2.0 * dSn_1) / dSn_1 / (dSn_1 + dSn_2) / H0 / H0 / H0;

		vals(4 * n + 2) = 1.0;
		vals(10 * n - 3) = 1.0;
		cudaDeviceSynchronize();
	}

	void BodyForceGpu(ElementGroup& eg, CuVector<double>& b)
	{
		long long n = b.size(CVD) / 2 - 1;
		thrust::transform(eg.GravityGroupCos.rbegin(CVD) + 1, eg.GravityGroupCos.rend(CVD) - 1, b.begin(CVD) + 1, b.begin(CVD) + 1,thrust::plus<double>());
		thrust::transform(eg.GravityGroupSin.rbegin(CVD) + 1, eg.GravityGroupSin.rend(CVD) - 1, b.begin(CVD) + 2 + n, b.begin(CVD) + 2 + n,thrust::plus<double>());
		//b(0) = 0.0; b(n) = 0.0; b(n + 1) = 0.0; b(2 * n + 1) = 0.0;
	}

	void SurfaceForceGpu(ElementGroup& eg, CuVector<double>& b)
	{
		auto size = b.size(CVD)/2;
		surface_force_sdirection_aux << < (size + threads - 1) / threads, threads >> > (b.data(CVD), eg.HGroup.data(CVD), eg.PupGroup.data(CVD), eg.PdownGroup.data(CVD), eg.TupGroup.data(CVD), eg.TdownGroup.data(CVD), eg.deltaSGroup.data(CVD), size);
		surface_force_zdirection_aux << < (size + threads - 1) / threads, threads >> > (b.data(CVD), eg.HGroup.data(CVD), eg.PupGroup.data(CVD), eg.PdownGroup.data(CVD), eg.TupGroup.data(CVD), eg.TdownGroup.data(CVD), eg.deltaSGroup.data(CVD), size);
		cudaDeviceSynchronize();
	}


}//end of extern "C"