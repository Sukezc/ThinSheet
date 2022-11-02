/*
author:zhang_chuanzhe
coding with: utf-8
email:191830203@smail.nju.edu.cn
last modify date:2022.10.31
*/


/*
基于一维随体坐标系的二维薄板片运动模拟
稀疏矩阵库采用英伟达官方提供的cusolver库
*/


#include"element_handle.h"
#include"CusolverSpHandle.h"
#include"CusolverRfHandle.h"
#include"element_iterate.h"
#include"ObjFactory.h"
#include"computeWidget.h"
#include<cstdlib>
#include<ctime>
#include<iostream>

//int main()
//{
//	double* test = NULL;
//	const double* consttest = NULL;
//
//	bodyforce_compute_gpu(test, test, test, consttest, consttest, consttest, 9.8, 10);
//	return 0;
//}


int main(int argc, char* argv[])
{

	ModelConf model;
	if (argc != 2) 
	{ 
		exit(-1); 
		std::cout << "Wrong arguments!" << std::endl;
	}
	std::ifstream fin(argv[1]);
	model.load_parameter(fin);
	ElementGroup Egold; ElementGroup Egnew;
	fin.close();

	model.Debug();
	
	REGISTER(CusolverRfHandle)
	REGISTER(CusolverSpHandle)
	SolverInterface* SolverHandle = NULL;

	if (model.solver == Solver::RF) SolverHandle = ObjFactory::Instance().CreateObj<SolverInterface>("CusolverRfHandle");
	else if (model.solver == Solver::SP) 
	{
		SolverHandle = ObjFactory::Instance().CreateObj<SolverInterface>("CusolverSpHandle");
		reinterpret_cast<CusolverSpHandle*>(SolverHandle)->solution = model.solution;
	}
	
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
	std::string dir = "AngleFileTest";
	system((mkdir + " " + dir).c_str());
	system((copy + " " + "Conf.xml" + " " + dir).c_str());
	std::ofstream outfile_xy(dir + "/" + model.XYaddress);
	std::ofstream outfile_force(dir + "/" + model.Forceaddress);
	std::ofstream outfile_Torque(dir + "/" + model.Torqueaddress);
	computeKernel(true, Egold, Egnew, SolverHandle, model,outfile_xy, outfile_force);
	vector_save(ElementGroup::GravityTorqueGroup, outfile_Torque);
	vector_save(ElementGroup::PforceTorqueGroup, outfile_Torque);
	
	delete SolverHandle;
	return 0;
}



