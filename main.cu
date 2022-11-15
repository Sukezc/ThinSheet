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
#include"ObjFactory.h"
#include"computeWidget.h"
#include<cstdlib>
#include<ctime>
#include<iostream>

//int main()
//{
//	
//	//CuVector<int> a(10,10);
//	//a.resize(10, 10);
//	//a.erase(a.begin(), a.end());
//	////CuVector<int> a(10,1);
//	////a.resizeWithSend(20, 20);
//	////int i = thrust::count(a.begin(0), a.end(0), 1);
//	////for (auto it = test.begin(); it != test.end(); it++)cout << *it << std::endl;
//	//return 0;
//
//	CusolverRfHandle solver;
//	std::vector<int> csrrowptrA = {0,2,5,7,9};
//	std::vector<int> csrcolindA = {0,3,1,2,3,0,1,2,3};
//	std::vector<double> csrvalA = {1,2,3,4,1,1,2,1,2};
//	std::vector<double> b = { 1,2,3,4 };
//	solver.Initialize(csrvalA, csrrowptrA, csrcolindA);
//	solver.loadB(b);
//	solver.solve();
//	csrvalA = { 1,2,3,4,1,1,2,10,12 };
//	solver.ResetA(csrvalA, csrrowptrA, csrcolindA);
//	solver.loadB(b);
//	solver.solve();
//	solver.Reset();
//	solver.Initialize(csrvalA, csrrowptrA, csrcolindA);
//	solver.loadB(b);
//	solver.solve();
//	for (auto& i : solver.X.h_vector)std::cout << i << std::endl;
//
//	return 0;
//}

int main()
{
	
	thrust::device_vector<int> d_a(10,10);
	thrust::device_vector<int> d_b(10, 2);
	thrust::copy(thrust::device, d_a.data(), d_a.data() + 5, d_b.begin());
	for (int i = 0; i < 10; i++)std::cout << d_b[i] << std::endl;
	return 0;
}

int mn(int argc, char* argv[])
{
	ModelConf model;
	if (argc != 2) 
	{ 
		exit(-1); 
		std::cout << "Wrong arguments!" << std::endl;
	}
	std::ifstream fin(argv[1]);
	model.load_parameter(fin);
	model.process_parameter();
	ElementGroup Egold; ElementGroup Egnew;
	fin.close();

	model.Debug();
	
	REGISTER(CusolverRfHandle)
	//REGISTER(CusolverSpHandle)
	SolverInterface* SolverHandle = NULL;

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
	
	std::string dir = "AngleFileTest";
	system((mkdir + " " + dir).c_str());
	system((copy + " " + "Conf.xml" + " " + dir).c_str());
	std::ofstream outfile_xy(dir + "/" + model.XYaddress);
	std::ofstream outfile_force(dir + "/" + model.Forceaddress);
	std::ofstream outfile_Torque(dir + "/" + model.Torqueaddress);
	//computeKernelGpu(true, Egold, Egnew, SolverHandle, model,outfile_xy, outfile_force);
	//computeCreateAngleInitFile(20.0, 6.5e5, SolverHandle, model, "20.dat");
	fin.open(argv[1]);
	computeLoadAngleInitFile(20.0,Egold, Egnew, model, fin, "20.dat");
	//std::cout << Egold.velocityGroup.front() << "  " << Egold.velocityGroup.back() << std::endl;
	computeKernel(true, Egold, Egnew, SolverHandle, model, outfile_xy, outfile_force);
	vector_save(ElementGroup::GravityTorqueGroup, outfile_Torque);
	vector_save(ElementGroup::PforceTorqueGroup, outfile_Torque);
	fin.close();
	delete SolverHandle;
	return 0;

}



