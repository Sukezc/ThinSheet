#pragma once
#include"model.h"
#include"element_handle.h"
#include"element_iterate_gpu.h"
#include<cmath>
#include<algorithm>
#include<iostream>

/*

used for physics field information updating (computing actually)

*/

//0.0174532922 equals to 1бу

bool Elongate(ElementGroup& egold, ElementGroup& egnew, ModelConf& model);

//{1,2,3}
void deltaS_iterate(ElementGroup& egold, ElementGroup& egnew, double dt);

//{1,2,3}
void theta_iterate(ElementGroup& egold, ElementGroup& egnew, double dt);

//{1,2,3}
void H_iterate(ElementGroup& egold, ElementGroup& egnew, double dt);

//4
void K_iterate(ElementGroup& eg);

template<typename dampfunc>
void fluids_pressure_set(ElementGroup& eg,ModelConf& model,long long start,dampfunc& CornerFlowdampingfunc)
{
	double criticalAngle = fabs(model.criticalangle);
	for (long long i = start; i > 0; i--)
	{
		double theta = eg.thetaGroup[i]; theta = theta >= -criticalAngle ? -criticalAngle : theta;
		double pressure_intensity_up = 2.0 * model.mantleViscosity * (model.velocity)*sin(theta) * sin(theta) / (theta * theta - sin(theta) * sin(theta));
		double pressure_intensity_down = 2.0 * model.mantleViscosity * (model.velocity)*sin(-theta) / (3.1415926535 + theta + sin(-theta));
		double distance_r = 0.0;


		for (long long j = start + 1; j > i; j--)
		{
			distance_r += eg.deltaSGroup[j];
		}

		eg.PupGroup[i] = pressure_intensity_up / distance_r * (CornerFlowdampingfunc(eg.YGroup[i]));
		eg.PdownGroup[i] = -pressure_intensity_down / distance_r * (CornerFlowdampingfunc(eg.YGroup[i]));
	}
	eg.PupGroup[0] = eg.PupGroup[1]; eg.PdownGroup[0] = eg.PdownGroup[1];
}

//5-
void surface_force_iterate(ElementGroup& eg, ModelConf& model, int iterating);

void density_iterate(ElementGroup& eg, ModelConf& model);

void bodyforce_compute(ElementGroup& eg);

template<typename Container1, typename Container2, typename Container3>
void ClampedFree(ElementGroup& eg, Container1& vals, Container2& rowPtr, Container3& colInd)
{
	long long n = eg.size - 1;
	vals.erase(vals.begin(), vals.end()); colInd.erase(colInd.begin(), colInd.end()); rowPtr.erase(rowPtr.begin(), rowPtr.end());

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

		vals.push_back(- 4.0 * miu * dSi / dSi_1 / (dSi + dSi_1));
		vals.push_back(4.0 * miu * (dSi - dSi_1) / dSi / dSi_1);
		vals.push_back(4.0 * miu * dSi_1 / dSi / (dSi_1 + dSi));
	}
	vals.push_back(1.0);

}

template<typename Container1, typename Container2, typename Container3>
void ClampedBoth(ElementGroup& eg, Container1& vals, Container2& rowPtr, Container3& colInd)
{
	long long n = eg.size - 1;
	vals.erase(vals.begin(), vals.end()); colInd.erase(colInd.begin(), colInd.end()); rowPtr.erase(rowPtr.begin(), rowPtr.end());

	for (int i = 0; i < 2 * n + 2; i++)colInd.push_back(i);

	for (int i = 1; i < n; i++)
	{
		colInd.push_back(i - 1);
		colInd.push_back(i);
		colInd.push_back(i + 1);

		colInd.push_back(i + n + 1);

	}
	
	//for (int i = 0; i < n; i++)colInd.push_back(i);
	//for (int i = n + 1; i < 2 * n + 2; i++)colInd.push_back(i);
	for (int i = 0; i < 2 * n + 2; i++)colInd.push_back(i);


	//colInd.push_back(n + 1);
	for (int i = 0; i < n + 1; i++)colInd.push_back(i);

	for (int i = 1; i < n; i++)
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


	int sum = 0;
	rowPtr.push_back(0);
	sum += 2 * n + 2;
	rowPtr.push_back(sum);
	for (int i = 0; i < n - 1; i++)
	{
		sum += 4;
		rowPtr.push_back(sum);
	}
	sum += 2 * n + 2;
	rowPtr.push_back(sum);

	sum += n + 1;
	rowPtr.push_back(sum);
	for (int i = 0; i < n - 1; i++)
	{
		//
		sum += 6;
		rowPtr.push_back(sum);
	}
	sum += 1;
	rowPtr.push_back(sum);

	std::vector<double> SjCos(n + 1, 0); std::vector<double> SjSin(n + 1, 0);
	//
	for (int i = 1; i < n + 1; i++)
	{
		SjCos[i] = SjCos[i - 1] + eg.deltaSGroup[i - 1] * cos(eg.thetaGroup[i - 1]);
		SjSin[i] = SjSin[i - 1] + eg.deltaSGroup[i - 1] * sin(eg.thetaGroup[i - 1]);
	}

	//
	for (long long i = n; i > -1; i--)
	{
		if (i == n)
		{
			vals.push_back(-0.5 * eg.deltaSGroup[n] * SjCos[i] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}//
		else if (i == 0)
		{
			vals.push_back(-0.5 * eg.deltaSGroup[0] * cos(eg.thetaGroup[0]) * eg.deltaSGroup[0] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}
		else
		{
			double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
			vals.push_back((-0.5 * dSi * dSi_1 * cos(theta) - SjCos[i] * 0.5 * (dSi + dSi_1)) / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}
	}

	for (long long i = n; i > -1; i--)
	{
		if (i == n)
		{
			vals.push_back(0.5 * eg.deltaSGroup[n] * sin(eg.thetaGroup[n]) / eg.HGroup[i]);
		}
		else if (i == 0)
		{
			vals.push_back(0.5 * eg.deltaSGroup[1] * sin(eg.thetaGroup[1]) / eg.HGroup[i]);
		}
		else
		{
			vals.push_back((0.5 * (eg.deltaSGroup[i] * sin(eg.thetaGroup[i]) + eg.deltaSGroup[i + 1] * sin(eg.thetaGroup[i + 1]))) / eg.HGroup[i]);
		}
	}

	
	for (long long i = n - 1; i > 0; i--)
	{
		double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], Ki = eg.KGroup[i], miu = eg.viscosity;
		vals.push_back(miu / 3.0 * 2.0 / dSi_1 / (dSi + dSi_1));
		vals.push_back(-miu / 3.0 * 2.0 / dSi / dSi_1 + 5.0 * miu / 6.0 * Ki * Ki);
		vals.push_back(miu / 3.0 * 2.0 / dSi / (dSi + dSi_1));
		vals.push_back(4.0 * miu * Ki);
	}
	
	//
	for (long long i = n; i > -1; i--)
	{
		if (i == n)
		{
			vals.push_back(0.5 * eg.deltaSGroup[n] * SjSin[i] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}//
		else if (i == 0)
		{
			vals.push_back(0.5 * eg.deltaSGroup[0] * sin(eg.thetaGroup[0]) * eg.deltaSGroup[0] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}
		else
		{
			double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
			vals.push_back((0.5 * dSi * dSi_1 * sin(theta) + SjSin[i] * 0.5 * (dSi + dSi_1)) / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}
	}

	for (long long i = n; i > -1; i--)
	{
		if (i == n)
		{
			vals.push_back(0.5 * eg.deltaSGroup[n] * cos(eg.thetaGroup[n]) / eg.HGroup[i]);
		}
		else if (i == 0)
		{
			vals.push_back(0.5 * eg.deltaSGroup[1] * cos(eg.thetaGroup[1]) / eg.HGroup[i]);
		}
		else
		{
			vals.push_back((0.5 * (eg.deltaSGroup[i] * cos(eg.thetaGroup[i]) + eg.deltaSGroup[i + 1] * cos(eg.thetaGroup[i + 1]))) / eg.HGroup[i]);
		}
	}

	
	for (int i = n; i > -1; i--)
	{
		if (i == n)
		{
			vals.push_back(0.5 * eg.deltaSGroup[i] /eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}
		else if (i == 0)
		{
			vals.push_back(0.5 * eg.deltaSGroup[1] / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}
		else
		{
			vals.push_back(0.5 * (eg.deltaSGroup[i] + eg.deltaSGroup[i + 1]) / eg.HGroup[i] / eg.HGroup[i] / eg.HGroup[i]);
		}

	}
	for (long long i = n - 1; i > 0; i--)
	{
		double dSi = eg.deltaSGroup[i], dSi_1 = eg.deltaSGroup[i + 1], Ki = eg.KGroup[i], Ki_1 = eg.KGroup[i + 1], Kip1 = eg.KGroup[i - 1], miu = eg.viscosity;
		vals.push_back(-miu / 2.0 * Ki * dSi / dSi_1 / (dSi_1 + dSi));
		vals.push_back(miu / 2.0 * Ki * (dSi - dSi_1) / dSi / dSi_1 + 5.0 * miu / 6.0 / dSi / dSi_1 / (dSi_1 + dSi) * (Kip1 * dSi_1 * dSi_1 - Ki_1 * dSi * dSi + Ki * (dSi * dSi - dSi_1 * dSi_1)));
		vals.push_back(miu / 2.0 * Ki * dSi_1 / dSi / (dSi_1 + dSi));

		//vals.push_back(-4.0 * miu / dSi_1);
		//vals.push_back(4.0 * miu / dSi_1);
		vals.push_back(- 4.0 * miu * dSi / dSi_1 / (dSi + dSi_1));
		vals.push_back(4.0 * miu * (dSi - dSi_1) / dSi / dSi_1);
		vals.push_back(4.0 * miu * dSi_1 / dSi / (dSi_1 + dSi));
	}
	vals.push_back(1.0);

}

template<typename Container>
void BodyForceOnly(ElementGroup& eg, Container& b)
{
	long long n = eg.size - 1;
	b.erase(b.begin(), b.end());
	b.push_back(0.0);
	for (long long i = n - 1; i > 0; i--)
	{
		b.push_back(eg.GravityGroupCos[i]);
		//          +                         -     +
		//          -                         +     -  
	}
	b.push_back(0.0);
	b.push_back(0.0);
	for (long long i = n - 1; i > 0; i--)
	{
		b.push_back(eg.GravityGroupSin[i]);
		//          +                         -     +     -
		//          -                         +     -     + 
	}
	b.push_back(0.0);
}

template<typename Container>
void SurfaceForceOnly(ElementGroup& eg, Container& b)
{
	long long n = eg.size - 1;
	b.erase(b.begin(), b.end());
	b.push_back(0.0);
	for (long long i = n - 1; i > 0; i--)
	{
		double Hi = eg.HGroup[i], Hi_1 = eg.HGroup[i + 1],
			Pip = eg.PupGroup[i], Pin = eg.PdownGroup[i], Tip = eg.TupGroup[i], Tin = eg.TdownGroup[i],
			Pi_1p = eg.PupGroup[i + 1], Pi_1n = eg.PdownGroup[i + 1], Ti_1p = eg.TupGroup[i + 1], Ti_1n = eg.TdownGroup[i + 1],
			dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
		b.push_back(-Pip + Pin - Hi / 2.0 / dSi_1 * (Tip + Tin - Ti_1p - Ti_1n) - (Hi - Hi_1) / dSi_1 * (Tip + Tin));
		//          +                         -     +
		//          -                         +     -  

	}
	b.push_back(0.0);
	b.push_back(0.0);
	for (long long i = n - 1; i > 0; i--)
	{
		double Hi = eg.HGroup[i], Hi_1 = eg.HGroup[i + 1],
			Pip = eg.PupGroup[i], Pin = eg.PdownGroup[i], Tip = eg.TupGroup[i], Tin = eg.TdownGroup[i],
			Pi_1p = eg.PupGroup[i + 1], Pi_1n = eg.PdownGroup[i + 1], Ti_1p = eg.TupGroup[i + 1], Ti_1n = eg.TdownGroup[i + 1],
			dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
		b.push_back(-Tip + Tin - Hi / 2.0 / dSi_1 * (Pip + Pin - Pi_1p - Pi_1n));
		//          +                         -     +     -
		//          -                         +     -     + 

	}
	b.push_back(0.0);
}

template<typename Container>
void SurfaceAndBodyForce(ElementGroup& eg, Container& b)
{
	long long n = eg.size - 1;
	b.erase(b.begin(), b.end());
	b.push_back(0.0);
	for (long long i = n - 1; i > 0; i--)
	{
		double Hi = eg.HGroup[i], Hi_1 = eg.HGroup[i + 1], rou = eg.density, g = eg.g,
			Pip = eg.PupGroup[i], Pin = eg.PdownGroup[i], Tip = eg.TupGroup[i], Tin = eg.TdownGroup[i],
			Pi_1p = eg.PupGroup[i + 1], Pi_1n = eg.PdownGroup[i + 1], Ti_1p = eg.TupGroup[i + 1], Ti_1n = eg.TdownGroup[i + 1],
			dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
		b.push_back(eg.GravityGroupCos[i] - Pip + Pin - Hi / 2.0 / dSi_1 * (Tip + Tin - Ti_1p - Ti_1n) - (Hi - Hi_1) / dSi_1 * (Tip + Tin));
		//          +                         -     +
		//          -                         +     -  

	}
	b.push_back(0.0);
	b.push_back(0.0);
	for (long long i = n - 1; i > 0; i--)
	{
		double Hi = eg.HGroup[i], Hi_1 = eg.HGroup[i + 1], rou = eg.density, g = eg.g,
			Pip = eg.PupGroup[i], Pin = eg.PdownGroup[i], Tip = eg.TupGroup[i], Tin = eg.TdownGroup[i],
			Pi_1p = eg.PupGroup[i + 1], Pi_1n = eg.PdownGroup[i + 1], Ti_1p = eg.TupGroup[i + 1], Ti_1n = eg.TdownGroup[i + 1],
			dSi_1 = eg.deltaSGroup[i + 1], theta = eg.thetaGroup[i];
		b.push_back(eg.GravityGroupSin[i] - Tip + Tin - Hi / 2.0 / dSi_1 * (Pip + Pin - Pi_1p - Pi_1n));
		//          +                         -     +     -
		//          -                         +     -     + 
	}
	b.push_back(0.0);
}

//5
template<class handle>
void Omega_Delta_iterate(ElementGroup& eg,ModelConf& model, handle& SolverHandle, bool ResetMatrix)
{
	//the number of length element
	long long n = eg.size - 1;
	static std::vector<double> vals;  static std::vector<double> b; static std::vector<int> rowPtr; static std::vector<int> colInd;

	switch (model.boundaryCondition)
	{
	case BoundaryCondition::ClampedFree:
		ClampedFree(eg, vals, rowPtr, colInd); break;
	case BoundaryCondition::ClampedBoth:
		ClampedBoth(eg, vals, rowPtr, colInd); break;
	default:
		break;
	}
	switch (model.forceCondition)
	{
	case ForceCondition::BodyForceOnly:
		BodyForceOnly(eg, b); break;
	case ForceCondition::SurfaceAndBodyForce:
		SurfaceAndBodyForce(eg, b); break;
	case ForceCondition::SurfaceForceOnly:
		SurfaceForceOnly(eg, b); break;
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
		SolverHandle->ResetA(vals.data(), rowPtr.data(), colInd.data(),vals.size(),rowPtr.size());
	}
	SolverHandle->loadB(b.data(),b.size());
	SolverHandle->solve();
	SolverHandle->X.fetch();
	for (long long i = n; i >= 0; i--)
	{
		long long j = n - i; double H = eg.HGroup[i];
		eg.OmegaGroup[i] = SolverHandle->X[j] / H / H / H;
		eg.DeltaGroup[i] = SolverHandle->X[j + n + 1] / H;
	}
}

//{6,7}
void omega_iterate(ElementGroup& eg, ModelConf& model);

//{6,7}
void velocity_iterate(ElementGroup& eg, ModelConf& model);