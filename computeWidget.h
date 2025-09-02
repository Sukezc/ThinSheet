#pragma once
#include"element_handle.h"
#include"element_iterate.h"
#include"element_iterate_gpu.h"
#include"SolverInterface.h"
#define MY_WINDOWS 

void computeKernel(bool saveFlag, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force);

void computeKernel_FibreStress(bool saveFlag, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force, std::ofstream& outfile_fibrestress);

void computeKernelGpu(bool saveFlag, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force);

void computeInnerProductRegression(bool saveOrNot, std::pair<double, double>& dampRate_innerProduct, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force);

void computeCriticalAngleRegressionBasedOnInnerProduct(const std::string& configuration, std::pair<double, double>& dampRate_innerProduct, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model);

//Angle +,бу not radius
void computeCreateAngleInitFile(double Angle, double LengthExpected, SolverInterface* SolverHandle, ModelConf& model, const std::string& FileName);

void computeLoadAngleInitFile(double Angle,ElementGroup& Egold, ElementGroup& Egnew, ModelConf& model,const std::string& FileName);

void computeSave(ElementGroup& Egnew, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force);

void computeSaveGpu(ElementGroup& Egnew, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force);