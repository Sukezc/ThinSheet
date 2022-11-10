#include"computeWidget.h"
#include"Xml.h"
#include<iostream>
#include<sstream>


void computeKernel(bool saveFlag, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force)
{
	ElementGroup::InitializeTorqueGroup(model.num_iterate);
	bool ResetMatrix;
	Egold = ElementGroup(model); Egnew = ElementGroup(model);
	for (int i = model.extrudepolicy.iterating; i < model.num_iterate; i++)
	{
		ResetMatrix = Elongate(Egold, Egnew, model);
		deltaS_iterate(Egold, Egnew, model.dt);
		theta_iterate(Egold, Egnew, model.dt);
		H_iterate(Egold, Egnew, model.dt);
		K_iterate(Egnew);
		density_iterate(Egnew, model);
		bodyforce_compute(Egnew);
		surface_force_iterate(Egnew, model, i);
		Omega_Delta_iterate(Egnew, model, SolverHandle, ResetMatrix);
		omega_iterate(Egnew, model);
		velocity_iterate(Egnew, model);
		if (model.SaveEveryXY)
		{
			Egnew.ComputeXY().ComputeGravityTorque(i).ComputePforceTorque(i);
			if (saveFlag) Egnew.SaveXY(outfile_xy).SaveForce(outfile_force);
		}
		else if (!(i % (model.num_iterate / model.num_XY)))
		{
			Egnew.ComputeXY().ComputeGravityTorque(i).ComputePforceTorque(i);
			if (saveFlag) Egnew.SaveXY(outfile_xy).SaveForce(outfile_force);
		}
		std::swap(Egold, Egnew);
	}
	model.ResetGridAndIterating();
}

void computeKernelGpu(bool saveFlag, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force)
{
	ElementGroup::InitializeTorqueGroup(model.num_iterate);
	bool ResetMatrix;
	Egold = ElementGroup(model); Egnew = ElementGroup(model);
	for (int i = model.extrudepolicy.iterating; i < model.num_iterate; i++)
	{
		if (model.grid_num < 128)
		{
			ResetMatrix = Elongate(Egold, Egnew, model);
			deltaS_iterate(Egold, Egnew, model.dt);
			theta_iterate(Egold, Egnew, model.dt);
			H_iterate(Egold, Egnew, model.dt);
			K_iterate(Egnew);
			density_iterate(Egnew, model);
			bodyforce_compute(Egnew);
			surface_force_iterate(Egnew, model, i);
			Omega_Delta_iterate(Egnew, model, SolverHandle, ResetMatrix);
			omega_iterate(Egnew, model);
			velocity_iterate(Egnew, model);
		}
		else
		{
			ResetMatrix = Elongate(Egold, Egnew, model);
			
			deltaS_iterate_gpu(Egold, Egnew, model.dt);
			theta_iterate_gpu(Egold, Egnew, model.dt);
			H_iterate_gpu(Egold, Egnew, model.dt);
			deltaS_theta_H_synchronize(Egnew);

			K_iterate_gpu(Egnew);
			density_iterate_gpu(Egnew, model);
			bodyforce_compute_gpu(Egnew);
			K_density_bodyforce_synchronize(Egnew);

			surface_force_iterate(Egnew, model, i);
			Omega_Delta_iterate_gpu(Egnew, model, SolverHandle, ResetMatrix);
			omega_velocity_iterate_gpu(Egnew, model,SolverHandle);
		}
		//std::cout << Egnew.velocityGroup.back() << " " << Egnew.velocityGroup.Dvec.back() << std::endl;
		if (model.SaveEveryXY)
		{
			Egnew.ComputeXY().ComputeGravityTorque(i).ComputePforceTorque(i);
			if (saveFlag) Egnew.SaveXY(outfile_xy).SaveForce(outfile_force);
		}
		else if (!(i % (model.num_iterate / model.num_XY)))
		{
			Egnew.ComputeXY().ComputeGravityTorque(i).ComputePforceTorque(i);
			if (saveFlag) Egnew.SaveXY(outfile_xy).SaveForce(outfile_force);
		}
		std::swap(Egold, Egnew);
	}
	model.ResetGridAndIterating();
}


void computeInnerProductRegression(bool saveOrNot, std::pair<double, double>& dampRate_innerProduct, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force)
{ 
	//innerProduct resolvent
	static std::vector<double> resolventInnerProduct;
	resolventInnerProduct.clear(); resolventInnerProduct.resize(model.dampSearchResolventCount, 0.0);
	std::cout << "----------DampRate Resolvent----------" << std::endl;
	std::ofstream outfile_placehold;
	for(int resolvent = 0;resolvent < model.dampSearchResolventCount;resolvent++)
	{
		computeKernel(false, Egold, Egnew, SolverHandle, model, outfile_placehold, outfile_placehold);
		resolventInnerProduct[resolvent] = ElementGroup::ComputeGravityPforceTorqueInnerProduct();
		std::cout << "DampRate: " << model.dampingrate << " InnerProduct: " << resolventInnerProduct[resolvent] << "\n";
		model.dampingrate /= 10.0;
	}
	if (model.dampSearchResolventCount > 0)
	{
		int num = std::distance(resolventInnerProduct.begin(), std::min_element(resolventInnerProduct.begin(), resolventInnerProduct.end()));
		model.dampingrate = model.dampingrate_copy;
		for (int i = 0; i < num; i++)model.dampingrate /= 10.0;
		std::cout << "Start DampRate: " << model.dampingrate << std::endl;
	}
	

	for (int dampsearchcount = 0; dampsearchcount < model.dampSearch; dampsearchcount++)
	{
		std::cout << "DampSearchCount  " << dampsearchcount + 1 << " : " << model.dampSearch << std::endl;
		bool saveFlag;

		if (dampsearchcount == model.dampSearch - 1)saveFlag = true;
		else saveFlag = false;

		computeKernel(saveOrNot && saveFlag, Egold, Egnew, SolverHandle, model, outfile_xy, outfile_force);
		std::cout << "dampingRate:" << model.dampingrate << "\n";
		std::cout << "innerProduct:" << ElementGroup::ComputeGravityPforceTorqueInnerProduct() << "\n";
		static double dampingRateinterval = 0.0; static int DRchange = 0;

		if (!dampsearchcount)
		{
			dampRate_innerProduct.first = model.dampingrate;
			dampRate_innerProduct.second = ElementGroup::ComputeGravityPforceTorqueInnerProduct();
			dampingRateinterval = model.dampingrate * 0.27;
			DRchange = 0;
			model.dampingrate *= 0.73;
		}
		else if (dampsearchcount == model.dampSearch - 1)break;
		else
		{
			std::pair<double, double> tempDamprate_innerProduct;
			tempDamprate_innerProduct.first = model.dampingrate;
			tempDamprate_innerProduct.second = ElementGroup::ComputeGravityPforceTorqueInnerProduct();
			if (tempDamprate_innerProduct.first < 0.0 || tempDamprate_innerProduct.second / dampRate_innerProduct.second > 1.0)
			{
				dampingRateinterval *= 0.5;
				model.dampingrate = dampRate_innerProduct.first - dampingRateinterval;
				dampsearchcount--;
				DRchange++;
				if (DRchange == model.dampSearch)
				{
					dampsearchcount = model.dampSearch - 2;
					model.dampingrate = dampRate_innerProduct.first;
				}
			}
			else if (fabs(tempDamprate_innerProduct.first - dampRate_innerProduct.first) < 1e-8)
			{
				dampsearchcount = model.dampSearch - 2;
				model.dampingrate = dampRate_innerProduct.first;
			}
			else
			{
				if (dampsearchcount == model.dampSearch - 2) model.dampingrate = dampRate_innerProduct.first;
				dampRate_innerProduct.first = tempDamprate_innerProduct.first;
				dampRate_innerProduct.second = tempDamprate_innerProduct.second;
				model.dampingrate -= dampingRateinterval;
			}
		}
	}
	model.ResetDampRate();
}

void computeCriticalAngleRegressionBasedOnInnerProduct(const std::string& filename,std::pair<double, double>& dampRate_innerProduct, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model)
{
#ifdef MY_WINDOWS
	std::string mkdir = "md";
	std::string copy = "copy";
#else
	std::string mkdir = "mkdir";
	std::string copy = "cp";
#endif // MY_WINDOWS
	double InnerProductLeft = 0.0, InnerProductRight = 0.0,tempInnerProduct = 0.0;
	double CriticalAngleLeft, CriticalAngleRight, CAinterval = 0.0;
	CAinterval = fabs(fabs(model.criticalAngleRange.second) - fabs(model.criticalAngleRange.first))/ model.criticalAngleSegment;
	CriticalAngleLeft = model.criticalAngleRange.first; CriticalAngleRight = CriticalAngleLeft + CAinterval;
	for (int CASegment = 0; CASegment < model.criticalAngleSegment; CASegment++)
	{
		model.criticalAngleRange.first = CriticalAngleLeft;
		model.criticalAngleRange.second = CriticalAngleRight;
		CriticalAngleLeft += CAinterval; CriticalAngleRight += CAinterval;
		for (int anglesearchcount = 0; anglesearchcount < model.criticalRangeCount; anglesearchcount++)
		{
			std::ofstream outfile_placehold;
			if (model.criticalRangeCount > 1 && !anglesearchcount)
			{
				for (int p = 0; p < 2; p++)
				{
					if (!p && !CASegment)
					{
						model.criticalangle = model.criticalAngleRange.first;
						computeInnerProductRegression(false, dampRate_innerProduct, Egold, Egnew, SolverHandle, model, outfile_placehold, outfile_placehold);
						InnerProductLeft = dampRate_innerProduct.second;
					}
					else if (!p)
					{
						InnerProductLeft = tempInnerProduct;
					}
					else
					{
						model.criticalangle = model.criticalAngleRange.second;
						computeInnerProductRegression(false, dampRate_innerProduct, Egold, Egnew, SolverHandle, model, outfile_placehold, outfile_placehold);
						tempInnerProduct = InnerProductRight = dampRate_innerProduct.second;
					}
				}
				model.criticalangle = (model.criticalAngleRange.first + model.criticalAngleRange.second) / 2.0;
			}
			std::cout << "CriticalAngle:" << fabs(model.criticalangle) << std::endl;
			std::string dir = std::to_string(fabs(model.criticalangle));
			system((mkdir + " " + dir).c_str());
			system((copy + " " + filename + " " + dir).c_str());
			std::ofstream outfile_xy(dir + "/" + model.XYaddress);
			std::ofstream outfile_force(dir + "/" + model.Forceaddress);
			std::ofstream outfile_Torque(dir + "/" + model.Torqueaddress);

			computeInnerProductRegression(true, dampRate_innerProduct, Egold, Egnew, SolverHandle, model, outfile_xy, outfile_force);

			vector_save(ElementGroup::GravityTorqueGroup, outfile_Torque);
			vector_save(ElementGroup::PforceTorqueGroup, outfile_Torque);


			std::ifstream fin(dir + "/" + filename);
			std::stringstream ss;
			ss << fin.rdbuf();
			const std::string& str = ss.str();
			xml::Xml root;
			root.parse(str);
			fin.close();
			root["CriticalAngle"].text(std::to_string(model.criticalangle));
			root["DampingRate"].text(std::to_string(dampRate_innerProduct.first));
			root.append(xml::Xml("Varience", std::to_string(dampRate_innerProduct.second)));
			root.save(dir + "/Conf.xml");
			outfile_xy.close();
			outfile_force.close();
			outfile_Torque.close();

			if (model.criticalRangeCount > 1)
			{
				double tempCriticalAngle = model.criticalangle;
				if (fabs(dampRate_innerProduct.second - InnerProductLeft) > fabs(dampRate_innerProduct.second - InnerProductRight))
				{
					InnerProductLeft = dampRate_innerProduct.second;
					model.criticalAngleRange.first = model.criticalangle;
				}
				else
				{
					InnerProductRight = dampRate_innerProduct.second;
					model.criticalAngleRange.second = model.criticalangle;
				}
				model.criticalangle = (model.criticalAngleRange.first + model.criticalAngleRange.second) / 2.0;
			}
		}
	}
}

void computeCreateAngleInitalFile(double Angle)
{

}

void computeLoadAngleInitalFile(const std::string& FileName)
{

}
