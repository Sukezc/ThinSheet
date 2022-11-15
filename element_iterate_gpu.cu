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
		auto size = Egnew.size;
		//Egnew.deltaSGroup.SyncSize(HostToDevice()); Egold.deltaSGroup.send(); Egold.velocityGroup.send();
		deltaS_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.deltaSGroup.data(CVD),Egold.deltaSGroup.data(CVD), Egold.velocityGroup.data(CVD),Egold.size,dt);
	}

	void theta_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew, const double dt)
	{
		auto size = Egnew.size;
		//Egnew.thetaGroup.SyncSize(HostToDevice()); Egold.thetaGroup.send(); Egold.omegaGroup.send();
		theta_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.thetaGroup.data(CVD),Egold.thetaGroup.data(CVD),Egold.omegaGroup.data(CVD),Egold.size - 1,dt);
	}

	void H_iterate_gpu(ElementGroup& Egold, ElementGroup& Egnew,const double dt)
	{
		auto size = Egnew.size;
		//Egnew.HGroup.SyncSize(HostToDevice()); Egold.HGroup.send(); Egold.DeltaGroup.send();
		H_iterate_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.HGroup.data(CVD), Egold.HGroup.data(CVD), Egold.DeltaGroup.data(CVD),Egold.size, dt);
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
		auto size = Egnew.size;
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
		auto size = Egnew.size;
		//Egnew.GravityGroup.SyncSize(HostToDevice()); Egnew.GravityGroupCos.SyncSize(HostToDevice()); Egnew.GravityGroupSin.SyncSize(HostToDevice());
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.GravityGroup.data(CVD), Egnew.densityGroup.data(CVD), Egnew.HGroup.data(CVD), Egnew.thetaGroup.data(CVD), Egnew.g, Egnew.size, []__device__(double i) { return 1.0; });
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.GravityGroupCos.data(CVD), Egnew.densityGroup.data(CVD), Egnew.HGroup.data(CVD), Egnew.thetaGroup.data(CVD), Egnew.g, Egnew.size, [=] __device__(double i) { return cos(i); });
		bodyforce_compute_kernel << <(size +  threads - 1) / threads, threads >> > (Egnew.GravityGroupSin.data(CVD), Egnew.densityGroup.data(CVD), Egnew.HGroup.data(CVD), Egnew.thetaGroup.data(CVD), Egnew.g, Egnew.size, [=] __device__(double i) { return sin(i); });
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
			if (ResetMatrix) ClampedFree(Egnew, vals, rowPtr, colInd); else ClampedFreeGpu(Egnew, vals, rowPtr, colInd); break;
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
			reinterpret_cast<CusolverRfHandle*>(SolverHandle)->ResetAGpu(vals.data(CVD), rowPtr.data(CVD), colInd.data(CVD),vals.size(CVD),rowPtr.size(CVD));
		}
		
		thrust::for_each(SolverHandle->X.begin(CVD), SolverHandle->X.end(CVD), []__device__(auto & it) { it = 0.0; });

		switch (model.forceCondition)
		{
		case ForceCondition::BodyForceOnly:
			BodyForceGpu(Egnew, SolverHandle->X); break;
		case ForceCondition::SurfaceAndBodyForce:
			SurfaceForceGpu(Egnew, SolverHandle->X); BodyForceGpu(Egnew, SolverHandle->X); break;
		case ForceCondition::SurfaceForceOnly:
			SurfaceForceGpu(Egnew, SolverHandle->X); break;
		default:
			break;
		}

		SolverHandle->solve();
	}

	void omega_velocity_iterate_gpu(ElementGroup& Egnew, ModelConf& model,SolverInterface* handle)
	{
		auto size = Egnew.size;
		//Egnew.omegaGroup.SyncSize(HostToDevice());
		//Egnew.velocityGroup.SyncSize(HostToDevice());
		
		thrust::reverse(thrust::device, handle->X.begin(CVD), handle->X.end(CVD));
		thrust::transform(thrust::device, handle->X.begin(CVD), handle->X.begin(CVD) + size, Egnew.HGroup.begin(CVD), handle->X.begin(CVD), []__device__(auto & it1, auto & it2) { return it1 / it2; });
		thrust::transform(thrust::device, handle->X.begin(CVD) + size, handle->X.end(CVD), Egnew.HGroup.begin(CVD), handle->X.begin(CVD) + size, []__device__(auto & it1, auto & it2) { return it1 / it2 / it2 / it2; });
		//thrust::copy(handle->X.begin(CVD), handle->X.begin(CVD) + size, Egnew.DeltaGroup.data());
		//thrust::copy(handle->X.begin(CVD) + size, handle->X.end(CVD), Egnew.OmegaGroup.data());
		
		Egnew.omegaGroup(size - 1) = 0.0;
		Egnew.omegaGroup(size - 2) = -(Egnew.OmegaGroup(size - 1) + Egnew.OmegaGroup(size - 2)) * Egnew.deltaSGroup(size - 1) / 2.0;
		Egnew.velocityGroup(size - 1) = 0.0;
		Egnew.velocityGroup(size - 2) = (Egnew.DeltaGroup(size - 1) + Egnew.DeltaGroup(size - 2)) * Egnew.deltaSGroup(size - 1) / 2.0;
		velocity_omega_iterate_kernel << <(size + threads - 1) / threads, threads >> > (Egnew.omegaGroup.data(CVD),handle->X.data(CVD) + size,Egnew.deltaSGroup.data(CVD),size);
		velocity_omega_iterate_kernel << <(size + threads - 1) / threads, threads >> > (Egnew.velocityGroup.data(CVD),handle->X.data(CVD), Egnew.deltaSGroup.data(CVD),size);
		cudaDeviceSynchronize();
		//0~size-3
		
		thrust::transform_inclusive_scan(Egnew.omegaGroup.rbegin(CVD)+2, Egnew.omegaGroup.rend(CVD), Egnew.omegaGroup.rbegin(CVD)+2, thrust::negate<double>(), thrust::plus<double>());
		thrust::inclusive_scan(Egnew.velocityGroup.rbegin(CVD) + 2, Egnew.velocityGroup.rend(CVD), Egnew.velocityGroup.rbegin(CVD) + 2, thrust::plus<double>());
		velocity_omega_aux_kernel << < (size + threads - 1) / threads, threads >> > (Egnew.velocityGroup.data(CVD), handle->X.data(CVD), Egnew.deltaSGroup.data(CVD), size, thrust::plus<double>());
		velocity_omega_aux_kernel << < (size + threads - 1) / threads, threads >> > (Egnew.omegaGroup.data(CVD), handle->X.data(CVD) + size, Egnew.deltaSGroup.data(CVD), size, thrust::minus<double>());
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

	void ClampedFreeGpu(ElementGroup& eg, CuVector<double>& vals, CuVector<int>& rowPtr, CuVector<int>& colInd)
	{
	
		long long n = eg.size - 1;
		vals.erase(CVD); colInd.erase(CVD); rowPtr.erase(CVD);
		vals.resize(10 * n - 2); colInd.resize(10 * n - 2); rowPtr.resize(2 * n + 3);
		
		
		colInd.push_back(2 * n - 1);
		colInd.push_back(2 * n);
		colInd.push_back(2 * n + 1);
		for (long long i = 1; i < n; i++)
		{
			colInd.push_back(i - 1);
			colInd.push_back(i);
			colInd.push_back(i + 1);

			colInd.push_back(i + n + 1);

		}
		colInd.push_back(n - 2);
		colInd.push_back(n - 1);
		colInd.push_back(n);

		//colInd.push_back(n + 1);
		colInd.push_back(n);
		for (long long i = 1; i < n; i++)
		{
			colInd.push_back(i - 1);
			colInd.push_back(i);
			colInd.push_back(i + 1);
			//

			colInd.push_back(i + n);
			colInd.push_back(i + n + 1);
			colInd.push_back(i + n + 2);
		}
		colInd.push_back(2 * n + 1);


		long long sum = 0;
		rowPtr.push_back(0);
		sum += 3;
		rowPtr.push_back(sum);
		for (long long i = 0; i < n - 1; i++)
		{
			sum += 4;
			rowPtr.push_back(sum);
		}
		sum += 3;
		rowPtr.push_back(sum);

		sum += 1;
		rowPtr.push_back(sum);
		for (long long i = 0; i < n - 1; i++)
		{
			//
			sum += 6;
			rowPtr.push_back(sum);
		}
		sum += 1;
		rowPtr.push_back(sum);
		

		double dSn_2 = eg.deltaSGroup[2], dSn_1 = eg.deltaSGroup[1];
		vals.push_back(dSn_1 / dSn_2 / (dSn_1 + dSn_2) / eg.HGroup[2]);
		vals.push_back((-dSn_1 - dSn_2) / dSn_1 / dSn_2 / eg.HGroup[1]);
		vals.push_back((dSn_2 + 2.0 * dSn_1) / dSn_1 / (dSn_1 + dSn_2) / eg.HGroup[0]);
		for (long long i = n - 1; i > 0; i--)
		{
			double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], Ki = eg.KGroup[i], miu = eg.viscosity;
			vals.push_back(miu / 3.0 * 2.0 / dSi_1 / (dSi + dSi_1));
			vals.push_back(-miu / 3.0 * 2.0 / dSi / dSi_1 + 5.0 * miu / 6.0 * Ki * Ki);
			vals.push_back(miu / 3.0 * 2.0 / dSi / (dSi + dSi_1));
			vals.push_back(4.0 * miu * Ki);
		}


		vals.push_back(dSn_1 / dSn_2 / (dSn_1 + dSn_2) / eg.HGroup[2] / eg.HGroup[2] / eg.HGroup[2]);
		vals.push_back((-dSn_1 - dSn_2) / dSn_1 / dSn_2 / eg.HGroup[1] / eg.HGroup[1] / eg.HGroup[1]);
		vals.push_back((dSn_2 + 2.0 * dSn_1) / dSn_1 / (dSn_1 + dSn_2) / eg.HGroup[0] / eg.HGroup[0] / eg.HGroup[0]);

		vals.push_back(1.0);
		for (long long i = n - 1; i > 0; i--)
		{
			double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], Ki = eg.KGroup[i], Ki_1 = eg.KGroup[i + 1], Kip1 = eg.KGroup[i - 1], miu = eg.viscosity;
			vals.push_back(-miu / 2.0 * Ki * dSi / dSi_1 / (dSi_1 + dSi));
			vals.push_back(miu / 2.0 * Ki * (dSi - dSi_1) / dSi / dSi_1 + 5.0 * miu / 6.0 / dSi / dSi_1 / (dSi_1 + dSi) * (Kip1 * dSi_1 * dSi_1 - Ki_1 * dSi * dSi + Ki * (dSi * dSi - dSi_1 * dSi_1)));
			vals.push_back(miu / 2.0 * Ki * dSi_1 / dSi / (dSi_1 + dSi));

			vals.push_back(-4.0 * miu * dSi / dSi_1 / (dSi + dSi_1));
			vals.push_back(4.0 * miu * (dSi - dSi_1) / dSi / dSi_1);
			vals.push_back(4.0 * miu * dSi_1 / dSi / (dSi_1 + dSi));
		}
		vals.push_back(1.0);

	}

	void BodyForceGpu(ElementGroup& eg, CuVector<double>& b)
	{
		long long n = eg.size - 1;
		thrust::transform(eg.GravityGroupCos.rbegin(CVD) + 1, eg.GravityGroupCos.rend(CVD) - 1, b.begin(CVD) + 1, b.begin(CVD) + 1,thrust::plus<double>());
		thrust::transform(eg.GravityGroupSin.rbegin(CVD) + 1, eg.GravityGroupSin.rend(CVD) - 1, b.begin(CVD) + 2 + n, b.begin(CVD) + 2 + n,thrust::plus<double>());
		//b(0) = 0.0; b(n) = 0.0; b(n + 1) = 0.0; b(2 * n + 1) = 0.0;
	}

	void SurfaceForceGpu(ElementGroup& eg, CuVector<double>& b)
	{
		auto size = eg.size;
		surface_force_sdirection_aux << < (size + threads - 1) / threads, threads >> > (b.data(CVD), eg.HGroup.data(CVD), eg.PupGroup.data(CVD), eg.PdownGroup.data(CVD), eg.TupGroup.data(CVD), eg.TdownGroup.data(CVD), eg.deltaSGroup.data(CVD), eg.size);
		surface_force_zdirection_aux << < (size + threads - 1) / threads, threads >> > (b.data(CVD), eg.HGroup.data(CVD), eg.PupGroup.data(CVD), eg.PdownGroup.data(CVD), eg.TupGroup.data(CVD), eg.TdownGroup.data(CVD), eg.deltaSGroup.data(CVD), eg.size);
		cudaDeviceSynchronize();
	}


}//end of extern "C"