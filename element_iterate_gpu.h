#pragma once
#include"element_handle.h"
#include"element_iterate.h"
#include"model.h"
#include"SolverInterface.h"
#include"CusolverRfHandle.h"


//template<typename Container1, typename Container2, typename Container3>
//void ClampedBoth(ElementGroup& eg, Container1& vals, Container2& rowPtr, Container3& colInd)
//{
//	long long n = eg.size - 1;
//	vals.erase(vals.begin(), vals.end()); colInd.erase(colInd.begin(), colInd.end()); rowPtr.erase(rowPtr.begin(), rowPtr.end());
//
//	for (int i = 0; i < 2 * n + 2; i++)colInd.push_back(i);
//
//	for (int i = 1; i < n; i++)
//	{
//		colInd.push_back(i - 1);
//		colInd.push_back(i);
//		colInd.push_back(i + 1);
//
//		colInd.push_back(i + n + 1);
//
//	}
//
//	//for (int i = 0; i < n; i++)colInd.push_back(i);
//	//for (int i = n + 1; i < 2 * n + 2; i++)colInd.push_back(i);
//	for (int i = 0; i < 2 * n + 2; i++)colInd.push_back(i);
//
//
//	//colInd.push_back(n + 1);
//	for (int i = 0; i < n + 1; i++)colInd.push_back(i);
//
//	for (int i = 1; i < n; i++)
//	{
//		colInd.push_back(i - 1);
//		colInd.push_back(i);
//		colInd.push_back(i + 1);
//
//		//
//		colInd.push_back(i + n);
//		colInd.push_back(i + n + 1);
//		colInd.push_back(i + n + 2);
//
//	}
//	colInd.push_back(2 * n + 1);
//
//
//	int sum = 0;
//	rowPtr.push_back(0);
//	sum += 2 * n + 2;
//	rowPtr.push_back(sum);
//	for (int i = 0; i < n - 1; i++)
//	{
//		sum += 4;
//		rowPtr.push_back(sum);
//	}
//	sum += 2 * n + 2;
//	rowPtr.push_back(sum);
//
//	sum += n + 1;
//	rowPtr.push_back(sum);
//	for (int i = 0; i < n - 1; i++)
//	{
//		//
//		sum += 6;
//		rowPtr.push_back(sum);
//	}
//	sum += 1;
//	rowPtr.push_back(sum);
//
//	std::vector<double> SjCos(n + 1, 0); std::vector<double> SjSin(n + 1, 0);
//	//
//	for (int i = 1; i < n + 1; i++)
//	{
//		SjCos[i] = SjCos[i - 1] + eg.deltaSGroup[i - 1] * cos(eg.thetaGroup[i - 1]);
//		SjSin[i] = SjSin[i - 1] + eg.deltaSGroup[i - 1] * sin(eg.thetaGroup[i - 1]);
//	}
//
//	//
//	for (long long i = n; i > -1; i--)
//	{
//		if (i == n)
//		{
//			vals.push_back(-0.5 * eg.deltaSGroup[n] * SjCos[i] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}//
//		else if (i == 0)
//		{
//			vals.push_back(-0.5 * eg.deltaSGroup[0] * cos(eg.thetaGroup[0]) * eg.deltaSGroup[0] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}
//		else
//		{
//			double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
//			vals.push_back((-0.5 * dSi * dSi_1 * cos(theta) - SjCos[i] * 0.5 * (dSi + dSi_1)) / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}
//	}
//
//	for (long long i = n; i > -1; i--)
//	{
//		if (i == n)
//		{
//			vals.push_back(0.5 * eg.deltaSGroup[n] * sin(eg.thetaGroup[n]) / eg.HGroup[i]);
//		}
//		else if (i == 0)
//		{
//			vals.push_back(0.5 * eg.deltaSGroup[1] * sin(eg.thetaGroup[1]) / eg.HGroup[i]);
//		}
//		else
//		{
//			vals.push_back((0.5 * (eg.deltaSGroup[i] * sin(eg.thetaGroup[i]) + eg.deltaSGroup[i + 1] * sin(eg.thetaGroup[i + 1]))) / eg.HGroup[i]);
//		}
//	}
//
//
//	for (long long i = n - 1; i > 0; i--)
//	{
//		double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], Ki = eg.KGroup[i], miu = eg.viscosity;
//		vals.push_back(miu / 3.0 * 2.0 / dSi_1 / (dSi + dSi_1));
//		vals.push_back(-miu / 3.0 * 2.0 / dSi / dSi_1 + 5.0 * miu / 6.0 * Ki * Ki);
//		vals.push_back(miu / 3.0 * 2.0 / dSi / (dSi + dSi_1));
//		vals.push_back(4.0 * miu * Ki);
//	}
//
//	//
//	for (long long i = n; i > -1; i--)
//	{
//		if (i == n)
//		{
//			vals.push_back(0.5 * eg.deltaSGroup[n] * SjSin[i] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}//
//		else if (i == 0)
//		{
//			vals.push_back(0.5 * eg.deltaSGroup[0] * sin(eg.thetaGroup[0]) * eg.deltaSGroup[0] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}
//		else
//		{
//			double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
//			vals.push_back((0.5 * dSi * dSi_1 * sin(theta) + SjSin[i] * 0.5 * (dSi + dSi_1)) / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}
//	}
//
//	for (long long i = n; i > -1; i--)
//	{
//		if (i == n)
//		{
//			vals.push_back(0.5 * eg.deltaSGroup[n] * cos(eg.thetaGroup[n]) / eg.HGroup[i]);
//		}
//		else if (i == 0)
//		{
//			vals.push_back(0.5 * eg.deltaSGroup[1] * cos(eg.thetaGroup[1]) / eg.HGroup[i]);
//		}
//		else
//		{
//			vals.push_back((0.5 * (eg.deltaSGroup[i] * cos(eg.thetaGroup[i]) + eg.deltaSGroup[i + 1] * cos(eg.thetaGroup[i + 1]))) / eg.HGroup[i]);
//		}
//	}
//
//
//	for (int i = n; i > -1; i--)
//	{
//		if (i == n)
//		{
//			vals.push_back(0.5 * eg.deltaSGroup[i] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}
//		else if (i == 0)
//		{
//			vals.push_back(0.5 * eg.deltaSGroup[1] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}
//		else
//		{
//			vals.push_back(0.5 * (eg.deltaSGroup[i] + eg.deltaSGroup[i + 1]) / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
//		}
//
//	}
//	for (long long i = n - 1; i > 0; i--)
//	{
//		double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], Ki = eg.KGroup[i], Ki_1 = eg.KGroup[i + 1], Kip1 = eg.KGroup[i - 1], miu = eg.viscosity;
//		vals.push_back(-miu / 2.0 * Ki * dSi / dSi_1 / (dSi_1 + dSi));
//		vals.push_back(miu / 2.0 * Ki * (dSi - dSi_1) / dSi / dSi_1 + 5.0 * miu / 6.0 / dSi / dSi_1 / (dSi_1 + dSi) * (Kip1 * dSi_1 * dSi_1 - Ki_1 * dSi * dSi + Ki * (dSi * dSi - dSi_1 * dSi_1)));
//		vals.push_back(miu / 2.0 * Ki * dSi_1 / dSi / (dSi_1 + dSi));
//
//		//vals.push_back(-4.0 * miu / dSi_1);
//		//vals.push_back(4.0 * miu / dSi_1);
//		vals.push_back(-4.0 * miu * dSi / dSi_1 / (dSi + dSi_1));
//		vals.push_back(4.0 * miu * (dSi - dSi_1) / dSi / dSi_1);
//		vals.push_back(4.0 * miu * dSi_1 / dSi / (dSi_1 + dSi));
//	}
//	vals.push_back(1.0);
//
//}
//

//
//template<typename Container>
//void SurfaceForceOnly(ElementGroup& eg, Container& b)
//{
//	long long n = eg.size - 1;
//	b.erase(b.begin(), b.end());
//	b.push_back(0.0);
//	for (long long i = n - 1; i > 0; i--)
//	{
//		double Hi = eg.HGroup[i], Hi_1 = eg.HGroup[i + 1],
//			Pip = eg.PupGroup[i], Pin = eg.PdownGroup[i], Tip = eg.TupGroup[i], Tin = eg.TdownGroup[i],
//			Pi_1p = eg.PupGroup[i + 1], Pi_1n = eg.PdownGroup[i + 1], Ti_1p = eg.TupGroup[i + 1], Ti_1n = eg.TdownGroup[i + 1],
//			dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
//		b.push_back(-Pip + Pin - Hi / 2.0 / dSi_1 * (Tip + Tin - Ti_1p - Ti_1n) - (Hi - Hi_1) / dSi_1 * (Tip + Tin));
//		//          +                         -     +
//		//          -                         +     -  
//
//	}
//	b.push_back(0.0);
//	b.push_back(0.0);
//	for (long long i = n - 1; i > 0; i--)
//	{
//		double Hi = eg.HGroup[i], Hi_1 = eg.HGroup[i + 1],
//			Pip = eg.PupGroup[i], Pin = eg.PdownGroup[i], Tip = eg.TupGroup[i], Tin = eg.TdownGroup[i],
//			Pi_1p = eg.PupGroup[i + 1], Pi_1n = eg.PdownGroup[i + 1], Ti_1p = eg.TupGroup[i + 1], Ti_1n = eg.TdownGroup[i + 1],
//			dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
//		b.push_back(-Tip + Tin - Hi / 2.0 / dSi_1 * (Pip + Pin - Pi_1p - Pi_1n));
//		//          +                         -     +     -
//		//          -                         +     -     + 
//
//	}
//	b.push_back(0.0);
//}
//
//template<typename Container>
//void SurfaceAndBodyForce(ElementGroup& eg, Container& b)
//{
//	long long n = eg.size - 1;
//	b.erase(b.begin(), b.end());
//	b.push_back(0.0);
//	for (long long i = n - 1; i > 0; i--)
//	{
//		double Hi = eg.HGroup[i], Hi_1 = eg.HGroup[i + 1], rou = eg.density, g = eg.g,
//			Pip = eg.PupGroup[i], Pin = eg.PdownGroup[i], Tip = eg.TupGroup[i], Tin = eg.TdownGroup[i],
//			Pi_1p = eg.PupGroup[i + 1], Pi_1n = eg.PdownGroup[i + 1], Ti_1p = eg.TupGroup[i + 1], Ti_1n = eg.TdownGroup[i + 1],
//			dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
//		b.push_back(eg.GravityGroupCos[i] - Pip + Pin - Hi / 2.0 / dSi_1 * (Tip + Tin - Ti_1p - Ti_1n) - (Hi - Hi_1) / dSi_1 * (Tip + Tin));
//		//          +                         -     +
//		//          -                         +     -  
//
//	}
//	b.push_back(0.0);
//	b.push_back(0.0);
//	for (long long i = n - 1; i > 0; i--)
//	{
//		double Hi = eg.HGroup[i], Hi_1 = eg.HGroup[i + 1], rou = eg.density, g = eg.g,
//			Pip = eg.PupGroup[i], Pin = eg.PdownGroup[i], Tip = eg.TupGroup[i], Tin = eg.TdownGroup[i],
//			Pi_1p = eg.PupGroup[i + 1], Pi_1n = eg.PdownGroup[i + 1], Ti_1p = eg.TupGroup[i + 1], Ti_1n = eg.TdownGroup[i + 1],
//			dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
//		b.push_back(eg.GravityGroupSin[i] - Tip + Tin - Hi / 2.0 / dSi_1 * (Pip + Pin - Pi_1p - Pi_1n));
//		//          +                         -     +     -
//		//          -                         +     -     + 
//	}
//	b.push_back(0.0);
//}


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

void ClampedFreeGpu(ElementGroup& eg, CuVector<double>& vals, CuVector<int>& rowPtr, CuVector<int>& colInd);

void BodyForceGpu(ElementGroup& eg, CuVector<double>& b);

void SurfaceForceGpu(ElementGroup& eg, CuVector<double>& b);

}