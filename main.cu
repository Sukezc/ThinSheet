/*
author:zhang_chuanzhe
coding with: utf-8
email:191830203@smail.nju.edu.cn
*/


/*
基于一维随体坐标系的二维薄板片运动模拟
稀疏矩阵库采用英伟达官方提供的cusolver库
*/


#include"element_handle.h"
#include"CusolverSpHandle.h"
#include"CusolverRfHandle.h"
#include"ObjFactory.h"
#include"computeWidget.h"
#include<cstdlib>
#include<ctime>
#include<iostream>
#include<memory>

int main(int argc, char* argv[])
{
	ModelConf model;
	if (argc < 2) 
	{
		std::cout << "Wrong arguments!" << std::endl;
		exit(-1); 
		
	}
	std::ifstream fin(argv[1]);
	model.load_parameter(fin);
	model.process_parameter();
	ElementGroup Egold(model); ElementGroup Egnew(model);
	fin.close();

	
	
	REGISTER(CusolverRfHandle)
	//REGISTER(CusolverSpHandle)
	SolverInterface* SolverHandle = nullptr;
	
	if (model.solver == Solver::RF) SolverHandle = ObjFactory::Instance().CreateObj<SolverInterface>("CusolverRfHandle");
	/*else if (model.solver == Solver::SP) 
	{
		SolverHandle = ObjFactory::Instance().CreateObj<SolverInterface>("CusolverSpHandle");
		reinterpret_cast<CusolverSpHandle*>(SolverHandle)->solution = model.solution;
	}*/
	
#ifdef MY_WINDOWS
	std::string mkdir = "md";
	std::string copy = "copy";
#else
	std::string mkdir = "mkdir";
	std::string copy = "cp";
#endif // MY_WINDOWS
	//std::pair<double, double> dampRate_innerProduct;
	//computeCriticalAngleRegressionBasedOnInnerProduct(argv[1],dampRate_innerProduct,Egold,Egnew,SolverHandle,model);
	//std::string dir = std::to_string(fabs(model.criticalangle));
	
	std::string dir = "outdata_1";
	system((mkdir + " " + dir).c_str());
	system((copy + " " + "Conf.xml" + " " + dir).c_str());
	std::ofstream outfile_xy(dir + "/" + model.XYaddress);
	std::ofstream outfile_force(dir + "/" + model.Forceaddress);
	std::ofstream outfile_Torque(dir + "/" + model.Torqueaddress);
	std::ofstream outfile_FibreStress(dir + "/FibreStress.csv");
	//computeKernelGpu(true, Egold, Egnew, SolverHandle, model,outfile_xy, outfile_force);
	//computeCreateAngleInitFile(60.0, 6.5e5, SolverHandle, model, "60.dat");
	if (argc > 2)
	{
		std::string angle = argv[2];
		computeLoadAngleInitFile(std::stod(angle), Egold, Egnew, model,angle + ".dat");
	}
	//std::ofstream outfile_placehold;
	//std::cout << Egold.velocityGroup.front() << "  " << Egold.velocityGroup.back() << std::endl;
	//Egold = ElementGroup(model); Egnew = ElementGroup(model);
	//std::cout << model.grid_num << std::endl;
	model.Debug();
	
	computeKernel(true, Egold, Egnew, SolverHandle, model, outfile_xy, outfile_force);
	//computeKernel_FibreStress(true, Egold, Egnew, SolverHandle, model, outfile_xy, outfile_force,outfile_FibreStress);
	
	vector_save(ElementGroup::GravityTorqueGroup, outfile_Torque);
	vector_save(ElementGroup::PforceTorqueGroup, outfile_Torque);
	//fin.close();
	
	outfile_xy.close();
	outfile_force.close();
	outfile_Torque.close();
	outfile_FibreStress.close();
	//std::cout << "out" << std::endl;
	delete SolverHandle;
	
	return 0;

}



