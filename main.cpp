/*
author:zhang_chuanzhe
coding with: utf-8
email:191830203@smail.nju.edu.cn
last modify date:2022.8.1
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
//	ElementGroup Egold; ElementGroup Egnew; ModelConf model;
//	model.criticalangle = 22.0;
//	Egold.XGroup.resize(3, 2.0); Egold.YGroup.resize(3, 1.0);
//	Egnew.XGroup.resize(3, 3.0); Egnew.YGroup.resize(3, 4.0);
//
//	Egold.g = 5.02; Egnew.density = -0.02;
//	ElementGroup::SaveState(&Egold, &Egnew, &model, "test");
//	ElementGroup::LoadState(&Egold, &Egnew, &model, "test");
//	std::cout << model.criticalangle << std::endl;
//	for (auto& it : Egold.XGroup)
//	{
//		std::cout << it << std::endl;
//	}
//	for (auto& it : Egnew.YGroup)
//	{
//		std::cout << it << std::endl;
//	}
//	for (auto& it : Egold.YGroup)
//	{
//		std::cout << it << std::endl;
//	}
//	for (auto& it : Egnew.XGroup)
//	{
//		std::cout << it << std::endl;
//	}
//	std::cout << Egold.g << " " << Egnew.density << std::endl;
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
	
	std::pair<double, double> dampRate_innerProduct;
	computeCriticalAngleRegressionBasedOnInnerProduct(argv[1],dampRate_innerProduct,Egold,Egnew,SolverHandle,model);

	delete SolverHandle;
	return 0;
}



