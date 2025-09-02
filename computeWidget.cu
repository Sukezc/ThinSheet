#include"computeWidget.h"
#include"Xml.h"
#include<iostream>
#include<sstream>
#define PI 3.14159265358979

void computeKernel(bool saveFlag, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force)
{
	ElementGroup::InitializeTorqueGroup(model.num_iterate);
	//Egold = ElementGroup(model); Egnew = ElementGroup(model);
	bool ResetMatrix;
	for (int i = model.extrudepolicy.iterating; i < model.num_iterate; i++)
	{
		std::cout << i << std::endl;
		ResetMatrix = Elongate(Egold, Egnew, model);
		if (i == model.extrudepolicy.iterating - 1)ResetMatrix = true;
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
		else if (!(i % model.recordInterval))
		{
			Egnew.ComputeXY().ComputeGravityTorque(i).ComputePforceTorque(i);
			if (saveFlag) Egnew.SaveXY(outfile_xy).SaveForce(outfile_force);
		}
		std::swap(Egold, Egnew);
	}
	model.ResetGridAndIterating();
}

void computeKernel_FibreStress(bool saveFlag, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force,std::ofstream& outfile_fibrestress)
{
	ElementGroup::InitializeTorqueGroup(model.num_iterate);
	//Egold = ElementGroup(model); Egnew = ElementGroup(model);
	bool ResetMatrix;
	for (int i = model.extrudepolicy.iterating; i < model.num_iterate; i++)
	{
		std::cout << i  << std::endl;
		ResetMatrix = Elongate(Egold, Egnew, model);
		if (i == model.extrudepolicy.iterating - 1)ResetMatrix = true;
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
			Egnew.ComputeXY().ComputeGravityTorque(i).ComputePforceTorque(i).ComputeFibreStress();
			if (saveFlag) Egnew.SaveXY(outfile_xy).SaveForce(outfile_force).SaveFibreStress(outfile_fibrestress);
		}
		else if (!(i % model.recordInterval))
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
	//Egold = ElementGroup(model); Egnew = ElementGroup(model);
	bool GpuSwitch = true;
	
	for (int i = model.extrudepolicy.iterating; i < model.num_iterate; i++)
	{
		std::cout << i << std::endl;
		if (model.grid_num < 256)
		{
			ResetMatrix = Elongate(Egold, Egnew, model);
			if (i == model.extrudepolicy.iterating - 1)ResetMatrix = true;
			deltaS_iterate(Egold, Egnew, model.dt);
			theta_iterate(Egold, Egnew, model.dt);
			//std::cout << Egnew.ComputeAverageTheta() << std::endl;
			H_iterate(Egold, Egnew, model.dt);
			K_iterate(Egnew);
			density_iterate(Egnew, model);
			bodyforce_compute(Egnew);
			surface_force_iterate(Egnew, model, i);
			Omega_Delta_iterate(Egnew, model, SolverHandle, ResetMatrix);
			omega_iterate(Egnew, model);
			velocity_iterate(Egnew, model);
			if (saveFlag) { 
				Egnew.ComputeXY().ComputeGravityTorque(i).ComputePforceTorque(i);
				computeSave(Egnew, model, outfile_xy, outfile_force); 
			}
		}
		else
		{
			if (GpuSwitch)
			{
				Egnew.sendAll(); Egold.sendAll();
				ElementGroup::GravityTorqueGroup.send();
				ElementGroup::PforceTorqueGroup.send();
				GpuSwitch = false;
				std::cout << "change platform" << std::endl;
			}
			ResetMatrix = ElongateGpu(Egold, Egnew, model);
			if (i == model.extrudepolicy.iterating - 1)ResetMatrix = true;
			
			deltaS_iterate_gpu(Egold, Egnew, model.dt);
			theta_iterate_gpu(Egold, Egnew, model.dt);
			H_iterate_gpu(Egold, Egnew, model.dt);
			deltaS_theta_H_synchronize(Egnew);
			
			
			//std::cout << Egnew.ComputeAverageTheta(CVD) << std::endl;
			K_iterate_gpu(Egnew);
			density_iterate_gpu(Egnew, model);
			bodyforce_compute_gpu(Egnew);
			K_density_bodyforce_synchronize(Egnew);
			
			surface_force_iterate_gpu(Egnew, model, i);
			Omega_Delta_iterate_gpu(Egnew, model, SolverHandle, ResetMatrix);
			
			omega_velocity_iterate_gpu(Egnew, model,SolverHandle);
			//std::cout << i << std::endl;
			if (saveFlag) { 
				Egnew.ComputeGravityTorque(i, CVD).ComputePforceTorque(i, CVD);
				computeSaveGpu(Egnew, model, outfile_xy, outfile_force); 
			}
		}
		std::swap(Egold, Egnew);
	}
	/*std::cout << thrust::reduce(ElementGroup::PforceTorqueGroup.begin(CVD), ElementGroup::PforceTorqueGroup.end(CVD)) << "\n";
	std::cout << thrust::reduce(ElementGroup::GravityTorqueGroup.begin(CVD), ElementGroup::GravityTorqueGroup.end(CVD));*/
	if (!GpuSwitch)
	{
		ElementGroup::PforceTorqueGroup.fetch();
		ElementGroup::GravityTorqueGroup.fetch();
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

void computeCreateAngleInitFile(double Angle,double LengthExpected, SolverInterface* SolverHandle, ModelConf& model,const std::string& FileName)
{
	const double velocityInterval = 5e-12;
	bool ResetMatrix; bool EndFlag = false;
	ElementGroup Egold; ElementGroup Egnew;
	
	for (int num = 0;;num++)
	{
		Egold = ElementGroup(model); Egnew = ElementGroup(model);
		std::cout << "num: " << num << " velocity: " << model.velocity << std::endl;
		model.Debug();
		for (int i = model.extrudepolicy.iterating;; i++)
		{
			//std::cout << "i: " << i << std::endl;
			ResetMatrix = Elongate(Egold, Egnew, model);
			if (i == model.extrudepolicy.iterating - 1)ResetMatrix = true;
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
			Egnew.ComputeSlabLength();
			
			if (Egnew.slabLength > LengthExpected)
			{
				double thetaAverage = fabs(Egnew.ComputeAverageTheta()) / PI * 180.0;
				std::cout << "thetaAverage: " << thetaAverage << "\n" << std::endl;
				if (thetaAverage > Angle + 2.0)
				{
					model.velocity += velocityInterval;
					model.process_parameter();
				}
				else if (thetaAverage < Angle - 2.0)
				{
					model.velocity -= velocityInterval;
					model.process_parameter();
				}
				else
				{
					ElementGroup::SaveState(&Egnew, &Egold, &model, FileName);
					EndFlag = true;
				}
				break;
			}
			std::swap(Egold, Egnew);
		}
		if (EndFlag)break;
		model.ResetGridAndIterating();
	}
}

void computeLoadAngleInitFile(double Angle,ElementGroup& Egold, ElementGroup& Egnew, ModelConf& model,const std::string& FileName)
{
	ModelConf tempModel;
	ElementGroup::LoadState(&Egold, &Egnew, &tempModel, FileName);
	auto theta_it = std::find_if(Egold.thetaGroup.rbegin(), Egold.thetaGroup.rend(), [=](auto& it) {return fabs(it) / PI * 180 > Angle; });
	auto velocity_it = Egold.velocityGroup.rbegin() + std::distance(Egold.thetaGroup.rbegin(), theta_it);
	std::for_each(theta_it + 1, Egold.thetaGroup.rend(), [=](auto& it) {it = *theta_it; });
	std::for_each(velocity_it + 1, Egold.velocityGroup.rend(), [=](auto& it) {it = *velocity_it; });
	double velocity_rate = model.velocity / Egold.velocityGroup.back();
	std::for_each(Egold.velocityGroup.begin(), Egold.velocityGroup.end(), [=](auto& it) {it *= velocity_rate; });
	model.grid_num = tempModel.grid_num;
	model.grid_num_copy = tempModel.grid_num_copy;
	model.slabLength = Egold.ComputeSlabLength().slabLength;
	model.deltaS = model.slabLength / size_t(model.grid_num) - 1;
	model.process_parameter();
}

void computeSave(ElementGroup& Egnew, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force)
{
	if (model.SaveEveryXY)
	{
		Egnew.SaveXY(outfile_xy).SaveForce(outfile_force);
	}
	else if (!((model.extrudepolicy.iterating - 1) % model.recordInterval))
	{
		Egnew.SaveXY(outfile_xy).SaveForce(outfile_force);
	}
}

void computeSaveGpu(ElementGroup& Egnew, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force)
{
	if (model.SaveEveryXY)
	{
		Egnew.SaveXY(outfile_xy,CVD).SaveForce(outfile_force,CVD);
	}
	else if (!((model.extrudepolicy.iterating - 1) % model.recordInterval))
	{
		Egnew.SaveXY(outfile_xy,CVD).SaveForce(outfile_force,CVD);
	}
}
