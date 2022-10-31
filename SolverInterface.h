#pragma once
#include<vector>
#include"CudaAllocator.h"
class SolverInterface
{
public:
	SolverInterface(){}
	virtual ~SolverInterface(){}
	virtual void Initialize(const std::vector<double>& csrval, const std::vector<int>& csrrowptr, const std::vector<int>& csrcolind) = 0;
	virtual void solve() = 0;
	virtual void loadB(const std::vector<double>& b) = 0;
	virtual void ResetA(const std::vector<double>& csrval, const std::vector<int>& csrrowptr, const std::vector<int>& csrcolind) = 0;
	virtual void Reset() = 0;
	std::vector<double, CudaAllocator<double>> X;
};

