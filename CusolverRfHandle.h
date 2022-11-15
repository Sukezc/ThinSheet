#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cusolverRf.h"
#include "cusolverSp.h"
#include "cusolverSp_LOWLEVEL_PREVIEW.h"
#include"CuVector.h"
#include"SolverInterface.h"
#include"CudaAllocator.h"
#include<vector>
#include<type_traits>
#include<algorithm>
#include<thrust/execution_policy.h>


/*

used for sparse matrix solving

*/

class CusolverRfHandle : public SolverInterface
{
public:

	cusolverRfHandle_t cusolverRfH;
	cusolverSpHandle_t cusolverSpH;
	cusparseMatDescr_t descrA;
	csrluInfoHost_t info;


public:

	std::vector<int> csrRowPtrL;
	std::vector<int> csrColIndL;
	std::vector<double> csrValL;

	std::vector<int> csrRowPtrU;
	std::vector<int> csrColIndU;
	std::vector<double> csrValU;

	//std::vector<int, CudaAllocator<int>> csrRowPtrA;
	//std::vector<int, CudaAllocator<int>> csrColIndA;
	//std::vector<double, CudaAllocator<double>> csrValA;
	CuVector<int> csrRowPtrA;
	CuVector<int> csrColIndA;
	CuVector<double> csrValA;

	std::vector<char> buffer_cpu;
	CuVector<double> T;
	//std::vector<double,CudaAllocator<double>> T;
	//std::vector<double, CudaAllocator<double>> X;

	CuVector<int> P;
	CuVector<int> Q;
	//std::vector<int,CudaAllocator<int>> P;
	//std::vector<int,CudaAllocator<int>> Q;

	int n;
	int nzero, nboost;
	cusolverRfFactorization_t fact_alg;
	cusolverRfTriangularSolve_t solve_alg;

public:
	CusolverRfHandle(double nzero = 0.0,double nboost = 0.0, 
		cusolverRfFactorization_t fact_alg = CUSOLVERRF_FACTORIZATION_ALG0,
		cusolverRfTriangularSolve_t solve_alg = CUSOLVERRF_TRIANGULAR_SOLVE_ALG1
	):cusolverRfH(NULL),cusolverSpH(NULL),descrA(NULL),info(NULL),n(0),nzero(nzero), nboost(nboost),fact_alg(fact_alg),solve_alg(solve_alg)
	{
		checkCudaErrors(cusolverSpCreate(&cusolverSpH));
		checkCudaErrors(cusparseCreateMatDescr(&descrA));
		checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
		checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
		checkCudaErrors(cusolverSpCreateCsrluInfoHost(&info));

		checkCudaErrors(cusolverRfCreate(&cusolverRfH));
		checkCudaErrors(cusolverRfSetNumericProperties(cusolverRfH, nzero, nboost));
		checkCudaErrors(cusolverRfSetAlgs(cusolverRfH, fact_alg, solve_alg));
		checkCudaErrors(cusolverRfSetMatrixFormat(cusolverRfH, CUSOLVERRF_MATRIX_FORMAT_CSR, CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));
		checkCudaErrors(cusolverRfSetResetValuesFastMode(cusolverRfH, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON));
	}

	~CusolverRfHandle()
	{
		if (cusolverRfH) { checkCudaErrors(cusolverRfDestroy(cusolverRfH)); }
		if (cusolverSpH) { checkCudaErrors(cusolverSpDestroy(cusolverSpH)); }
		if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
		if (info) { checkCudaErrors(cusolverSpDestroyCsrluInfoHost(info)); }
	}

	void Initialize(const double* csrval, const int* csrrowptr, const int* csrcolind, const long long vals_size, const int row_size)
	{
		csrValA.resize(vals_size);
		std::copy(csrval, csrval + vals_size, csrValA.begin());
		csrRowPtrA.resize(row_size);
		std::copy(csrrowptr , csrrowptr + row_size, csrRowPtrA.begin());
		csrColIndA.resize(vals_size);
		std::copy(csrcolind , csrcolind + vals_size, csrColIndA.begin());
		n = csrRowPtrA.size() - 1;

		P.resize(n); 	
		Q.resize(n); 
		checkCudaErrors(cusolverSpXcsrluAnalysisHost(cusolverSpH, n, csrValA.size(),descrA, csrRowPtrA.data(), csrColIndA.data(),info));
		size_t size_internal, size_lu;
		checkCudaErrors(cusolverSpDcsrluBufferInfoHost(cusolverSpH, n, csrValA.size(),descrA, csrValA.data(), csrRowPtrA.data(), 
			csrColIndA.data(),info,&size_internal,&size_lu));
		buffer_cpu.resize(size_lu);
		checkCudaErrors(cusolverSpDcsrluFactorHost(cusolverSpH, n, csrValA.size(),
			descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(),info, 1.0,(void*)buffer_cpu.data()));
		int singularity;
		cusolverSpDcsrluZeroPivotHost(cusolverSpH, info, 1e-15, &singularity);

#ifndef RELEASED
		if (singularity >= 0) printf("singularity=%d (singularity>=0 indices that A is not invertible)\n", singularity);
#endif // !RELEASED
		
		int nnzL, nnzU;
		checkCudaErrors(cusolverSpXcsrluNnzHost(cusolverSpH,&nnzL,&nnzU,info));
		csrValL.resize(nnzL); csrColIndL.resize(nnzL); csrValU.resize(nnzU); csrColIndU.resize(nnzU);
		csrRowPtrL.resize(size_t(n) + 1); csrRowPtrU.resize(size_t(n) + 1);
		checkCudaErrors(cusolverSpDcsrluExtractHost(cusolverSpH,
			P.data(),Q.data(),
			descrA,
			csrValL.data(),csrRowPtrL.data(),csrColIndL.data(),
			descrA,
			csrValU.data(),csrRowPtrU.data(),csrColIndU.data(),
			info,
			buffer_cpu.data()));

		checkCudaErrors(cusolverRfSetupHost(
			n, csrValA.size(),
			csrRowPtrA.data(), csrColIndA.data(), csrValA.data(),
			nnzL,
			csrRowPtrL.data(), csrColIndL.data(), csrValL.data(),
			nnzU,
			csrRowPtrU.data(), csrColIndU.data(), csrValU.data(),
			P.data(),
			Q.data(),
			cusolverRfH));
		//checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cusolverRfAnalyze(cusolverRfH));
		T.resize(n); X.resize(n); P.send(); Q.send();
	}

	
	void solve()
	{
		checkCudaErrors(cusolverRfRefactor(cusolverRfH));
		//checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cusolverRfSolve(cusolverRfH, P.data(CVD), Q.data(CVD), 1, T.data(CVD), n, X.data(CVD), n));
		//X.fetch();
		//checkCudaErrors(cudaDeviceSynchronize());
	}

	
	void loadB(const double* b, const long long size)
	{
		cudaMemcpy(X.data(CVD), b, size * sizeof(double), cudaMemcpyHostToDevice);
	}

	void ResetA(const double* csrval, const int* csrrowptr, const int* csrcolind, const long long vals_size, const int row_size)
	{
		cudaMemcpy(csrValA.data(CVD), csrval, vals_size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(csrRowPtrA.data(CVD), csrrowptr, row_size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(csrColIndA.data(CVD), csrcolind, vals_size * sizeof(double), cudaMemcpyHostToDevice);

		checkCudaErrors(cusolverRfResetValues(
			n, csrValA.size(CVD),
			csrRowPtrA.data(CVD), csrColIndA.data(CVD), csrValA.data(CVD),
			P.data(CVD),
			Q.data(CVD),
			cusolverRfH));
	}
	
	void ResetAGpu(const double* csrval, const int* csrrowptr, const int* csrcolind, const long long vals_size, const int row_size)
	{
		thrust::copy(thrust::device,csrval, csrval + vals_size, csrValA.data(CVD));
		thrust::copy(thrust::device,csrrowptr, csrrowptr + row_size, csrRowPtrA.data(CVD));
		thrust::copy(thrust::device,csrcolind, csrcolind + vals_size, csrColIndA.data(CVD));
		
		checkCudaErrors(cusolverRfResetValues(
			n, csrValA.size(CVD),
			csrRowPtrA.data(CVD), csrColIndA.data(CVD), csrValA.data(CVD),
			P.data(CVD),
			Q.data(CVD),
			cusolverRfH));
	}

	void Reset()
	{
		if (cusolverRfH) { checkCudaErrors(cusolverRfDestroy(cusolverRfH)); }
		//if (cusolverSpH) { checkCudaErrors(cusolverSpDestroy(cusolverSpH)); }
		//if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
		if (info) { checkCudaErrors(cusolverSpDestroyCsrluInfoHost(info)); }

		//checkCudaErrors(cusolverSpCreate(&cusolverSpH));
		//checkCudaErrors(cusparseCreateMatDescr(&descrA));
		//checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
		//checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
		checkCudaErrors(cusolverSpCreateCsrluInfoHost(&info));

		checkCudaErrors(cusolverRfCreate(&cusolverRfH));
		checkCudaErrors(cusolverRfSetNumericProperties(cusolverRfH, nzero, nboost));
		checkCudaErrors(cusolverRfSetAlgs(cusolverRfH, fact_alg, solve_alg));
		checkCudaErrors(cusolverRfSetMatrixFormat(cusolverRfH, CUSOLVERRF_MATRIX_FORMAT_CSR, CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L));
		checkCudaErrors(cusolverRfSetResetValuesFastMode(cusolverRfH, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON));
	}

};

