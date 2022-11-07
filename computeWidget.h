#pragma once
#include"element_handle.h"
#include"element_iterate.h"
#include"SolverInterface.h"
#define MY_WINDOWS 

void computeKernel(bool saveFlag, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force);

void computeInnerProductRegression(bool saveOrNot, std::pair<double, double>& dampRate_innerProduct, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model, std::ofstream& outfile_xy, std::ofstream& outfile_force);

void computeCriticalAngleRegressionBasedOnInnerProduct(const std::string& configuration, std::pair<double, double>& dampRate_innerProduct, ElementGroup& Egold, ElementGroup& Egnew, SolverInterface* SolverHandle, ModelConf& model);

void computeCreateAngleInitalFile(double Angle);

void computeLoadAngleInitalFile(const std::string& FileName);