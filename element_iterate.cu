#include"element_iterate.h"
#include<algorithm>

bool Elongate(ElementGroup& egold, ElementGroup& egnew, ModelConf& model)
{
	if (model.extrudepolicy.policy == ExtrudePolicy::Sparse)
	{
		if (!(model.extrudepolicy.iterating % model.extrudepolicy.SparseNum))
		{
			egold.elongate(model.extrudepolicy.DsEnd, model.H, model.velocity);
			egnew.elongate(model.extrudepolicy.DsEnd, model.H, model.velocity);
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

		egold.elongate(model.extrudepolicy.Ds, model.H, model.velocity, model.extrudepolicy.DenseNum);
		egold.elongate(model.extrudepolicy.DsEnd, model.H, model.velocity);


		egnew.elongate(model.extrudepolicy.Ds, model.H, model.velocity, model.extrudepolicy.DenseNum);
		egnew.elongate(model.extrudepolicy.DsEnd, model.H, model.velocity);

		model.grid_num = egold.size;
		model.Standardize();
		return true;
	}
}

//{1,2,3}
void deltaS_iterate(ElementGroup& egold, ElementGroup& egnew, double dt)
{
	for (long long i = 1; i < egnew.size; i++)
	{
		egnew.deltaSGroup[i] = egold.deltaSGroup[i] + dt * (egold.velocityGroup[i - 1] - egold.velocityGroup[i]);
	}
	//egnew.deltaSGroup[0] = egnew.deltaSGroup[1];
}

//{1,2,3}
void theta_iterate(ElementGroup& egold, ElementGroup& egnew, double dt)
{
	for (long long i = 0; i < egnew.size - 1; i++)
	{
		egnew.thetaGroup[i] = egold.thetaGroup[i] + dt * egold.omegaGroup[i];
	}
}

//{1,2,3}
void H_iterate(ElementGroup& egold, ElementGroup& egnew, double dt)
{
	for (long long i = 0; i < egnew.size; i++)
	{
		double k1 = -egold.HGroup[i] * egold.DeltaGroup[i];
		double k2 = -(egold.HGroup[i] + dt / 2.0 * k1) * egold.DeltaGroup[i];
		double k3 = -(egold.HGroup[i] + dt / 2.0 * k2) * egold.DeltaGroup[i];
		double k4 = -(egold.HGroup[i] + dt * k3) * egold.DeltaGroup[i];

		egnew.HGroup[i] = egold.HGroup[i] + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
	}

}

//4
void K_iterate(ElementGroup& eg)
{
	//compute the middle point
	for (long long i = 1; i < eg.size - 1; i++)
	{
		double dSj = eg.deltaSGroup[i];
		double dSj_1 = eg.deltaSGroup[i + 1];
		eg.KGroup[i] = eg.thetaGroup[i + 1] * (-dSj) / (dSj_1 * (dSj_1 + dSj)) +
			eg.thetaGroup[i] * (dSj - dSj_1) / (dSj * dSj_1) +
			eg.thetaGroup[i - 1] * (dSj_1) / ((dSj + dSj_1) * dSj);
	}

	//compute the outside point 
	double dSn_1 = eg.deltaSGroup[1];
	double dSn_2 = eg.deltaSGroup[2];
	eg.KGroup[0] = eg.thetaGroup[2] * dSn_1 / (dSn_2 * (dSn_2 + dSn_1)) +
		eg.thetaGroup[1] * (-dSn_1 - dSn_2) / (dSn_1 * dSn_2) +
		eg.thetaGroup[0] * (dSn_2 + 2.0 * dSn_1) / ((dSn_1 + dSn_2) * dSn_1);

	//compute the inner point
	double dS0 = eg.deltaSGroup[eg.size - 1];
	double dS1 = eg.deltaSGroup[eg.size - 2];
	eg.KGroup[eg.size - 1] = eg.thetaGroup[eg.size - 1] * (-2.0 * dS0 - dS1) / ((dS0 + dS1) * dS0) +
		eg.thetaGroup[eg.size - 2] * (dS0 + dS1) / (dS0 * dS1) +
		eg.thetaGroup[eg.size - 3] * (-dS0) / ((dS1 + dS0) * dS1);

}

void shear_stress_set(ElementGroup* eg, ModelConf& model)
{

}

//5-
void surface_force_iterate(ElementGroup& eg, ModelConf& model, int iterating)
{
	//for(auto it = eg.TdownGroup.begin(); it != eg.TdownGroup.end(); it++)* it = 5.0 * model.density * model.g * model.H;
	//for (auto it = eg.PdownGroup.begin(); it != eg.PdownGroup.end(); it++)*it = - 0.5 *model.density * model.g * model.H;
	//for (auto it = eg.PdownGroup.begin(); it != eg.PdownGroup.end(); it++)*it = 0.0;

	const double TopDepth = model.lithosphereTop, BottomDepth = model.lithosphereBottom;
	long long pipePIdxMax = 0, pipePIdxTop = 0, pipePIdxBottom = 0;
	double Depth = 0.0, DepthMax = 0.0, U0 = model.velocity;//, cornerlengthrate = 0.0;// , UV = model.upperVelocity;
	double dampingRate = model.dampingrate, criticalAngle = fabs(model.criticalangle);
	std::for_each(eg.PupGroup.begin(), eg.PupGroup.end(), [](auto& it) {it = 0.0; });
	std::for_each(eg.PdownGroup.begin(), eg.PdownGroup.end(), [](auto& it) {it = 0.0; });
	std::for_each(eg.TupGroup.begin(), eg.TupGroup.end(), [](auto& it) {it = 0.0; });
	std::for_each(eg.TdownGroup.begin(), eg.TdownGroup.end(), [](auto& it) {it = 0.0; });
	eg.ComputeXY();

	static long long start = 0;
	if (!iterating) start = 0;

	auto CornerFlowdampingfunc = [=, &dampingRate](double y)-> double {return 1.0 / (1.0 + exp(dampingRate * (y - BottomDepth))); };
	auto Lithospheredampingfunc = [=, &dampingRate](double y)-> double {return 1.0 / (1.0 + exp(dampingRate * (BottomDepth - y))); };

	for (long long i = eg.size - 2; i > -1; i--)
	{
		Depth = eg.YGroup[i];
		if (Depth < DepthMax)
		{
			DepthMax = Depth;
			pipePIdxMax = i;
			if (Depth < eg.YGroup[i - 1])break;
		}
	}

	Depth = 0.0;
	for (long long i = eg.size - 1; i >= pipePIdxMax; i--)
	{
		Depth = eg.YGroup[i];
		if (Depth > TopDepth)
		{
			pipePIdxTop = i;
		}
		if (Depth > BottomDepth)
		{
			pipePIdxBottom = i;
		}
	}

	if(std::max(pipePIdxBottom, pipePIdxMax) > start)start = std::max(pipePIdxBottom, pipePIdxMax);
	
	/*const gsl_multiroot_fsolver_type* T = gsl_multiroot_fsolver_hybrid;
	gsl_multiroot_fsolver* s = gsl_multiroot_fsolver_alloc(T, 2);*/


	//DEPRECATED
	double pressure = model.pipePressure; for (long long i = pipePIdxTop; i >= pipePIdxBottom; i--) { eg.PupGroup[i] = pressure; }

	fluids_pressure_set(eg, model, start, CornerFlowdampingfunc);
}

void density_iterate(ElementGroup& eg, ModelConf& model)
{

}

void bodyforce_compute(ElementGroup& eg)
{
	eg.ComputeGravityAll();
}

//5
void Omega_Delta_iterate(ElementGroup& eg, ModelConf& model, SolverInterface* SolverHandle, bool ResetMatrix)
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
		SolverHandle->Initialize(vals.data(), rowPtr.data(), colInd.data(), vals.size(), rowPtr.size());
	}
	else
	{
		SolverHandle->ResetA(vals.data());
	}
	SolverHandle->loadB(b.data());
	SolverHandle->solve();
	void* pContainer = SolverHandle->getContainer();
	CuVector<double>& X = *static_cast<CuVector<double>*>(pContainer);
	X.fetch();
	for (long long i = n; i >= 0; i--)
	{
		long long j = n - i; double H = eg.HGroup[i];
		eg.OmegaGroup[i] = X[j] / H / H / H;
		eg.DeltaGroup[i] = X[j + n + 1] / H;
	}
}

//{6,7}
void omega_iterate(ElementGroup& eg, ModelConf& model)
{

	long long size = eg.size;
	eg.omegaGroup[size - 1] = 0.0;
	eg.omegaGroup[size - 2] = -(eg.OmegaGroup[size - 1] + eg.OmegaGroup[size - 2]) * eg.deltaSGroup[size - 1] / 2.0;

	for (long long i = size - 3; i > -1; i--)
	{
		double omega_local = 0.0;
		omega_local -= (eg.OmegaGroup[size - 1] + eg.OmegaGroup[size - 2]) / 2.0 * eg.deltaSGroup[size - 1] + (eg.OmegaGroup[i] + eg.OmegaGroup[i + 1]) / 2.0 * eg.deltaSGroup[i + 1];
		for (long long j = size - 2; j > i; j--)
		{
			double dSj = eg.deltaSGroup[j], dSj_1 = eg.deltaSGroup[j + 1];
			omega_local -= eg.OmegaGroup[j + 1] * (dSj + dSj_1) * (2.0 * dSj_1 - dSj) / 6.0 / dSj_1 +
				eg.OmegaGroup[j] * (dSj + dSj_1) * (dSj + dSj_1) * (dSj + dSj_1) / 6.0 / dSj / dSj_1 +
				eg.OmegaGroup[j - 1] * (dSj + dSj_1) * (2.0 * dSj - dSj_1) / 6.0 / dSj;
		}
		eg.omegaGroup[i] = omega_local / 2.0;
	}

	double C = model.omegaStandard.second - eg.omegaGroup[model.omegaStandard.first];
	std::for_each(eg.omegaGroup.begin(), eg.omegaGroup.end(), [=](double& it) {it += C; });

	/*for (long long i = size - 3; i > -1; i--)
	{
		eg.omegaGroup[i] = 0.0;
		for (long long j = size - 1; j > i; j--)
		{
			eg.omegaGroup[i] -= (eg.OmegaGroup[j] + eg.OmegaGroup[j - 1]) * eg.deltaSGroup[j] / 2.0;
		}
	}*/
}

//{6,7}
void velocity_iterate(ElementGroup& eg, ModelConf& model)
{
	long long size = eg.size;
	eg.velocityGroup[size - 1] = 0.0;
	eg.velocityGroup[size - 2] = (eg.DeltaGroup[size - 1] + eg.DeltaGroup[size - 2]) * eg.deltaSGroup[size - 1] / 2.0;

	for (long long i = size - 3; i > -1; i--)
	{
		double u_local = 0.0;
		u_local += (eg.DeltaGroup[size - 1] + eg.DeltaGroup[size - 2]) / 2.0 * eg.deltaSGroup[size - 1] + (eg.DeltaGroup[i] + eg.DeltaGroup[i + 1]) / 2.0 * eg.deltaSGroup[i + 1];
		for (long long j = size - 2; j > i; j--)
		{
			double dSj = eg.deltaSGroup[j], dSj_1 = eg.deltaSGroup[j + 1];
			u_local += eg.DeltaGroup[j + 1] * (dSj + dSj_1) * (2.0 * dSj_1 - dSj) / 6.0 / dSj_1 +
				eg.DeltaGroup[j] * (dSj + dSj_1) * (dSj + dSj_1) * (dSj + dSj_1) / 6.0 / dSj / dSj_1 +
				eg.DeltaGroup[j - 1] * (dSj + dSj_1) * (2.0 * dSj - dSj_1) / 6.0 / dSj;
		}
		eg.velocityGroup[i] = u_local / 2.0;

	}

	double C = model.velocityStandard.second - eg.velocityGroup[model.velocityStandard.first];
	std::for_each(eg.velocityGroup.begin(), eg.velocityGroup.end(), [=](double& it) {it += C; });

	/*for (long long i = size - 3; i > -1; i--)
	{
		eg.velocityGroup[i] = U0;
		for (long long j = size - 1; j > i; j--)
		{
			eg.velocityGroup[i] += (eg.DeltaGroup[j] + eg.DeltaGroup[j - 1]) * eg.deltaSGroup[j] / 2.0;
		}
	}*/
}