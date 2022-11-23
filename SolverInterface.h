#pragma once
#include<vector>
#include"CuVector.h"

class SolverInterface
{
public:
	SolverInterface(){}
	virtual ~SolverInterface(){}
	virtual void Initialize(const double* csrval, const int* csrrowptr, const int* csrcolind, const long long vals_size, const int row_size) = 0;
	virtual void solve() = 0;
	virtual void loadB(const double* b) = 0;
	virtual void ResetA(const double* csrval) = 0;
	virtual void Reset() = 0;
	CuVector<double> X;
};

