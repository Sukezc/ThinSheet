#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cusolverSp.h"
#include<vector>
#include<type_traits>
#include<algorithm>
#include"CudaAllocator.h"
#include"model.h"
#include"SolverInterface.h"
/*

used for sparse matrix solving 

*/




//选择计算平台
#define CPU_MODE
#ifndef CPU_MODE
#define GPU_MODE
#endif // !CPU_MODE

class CusolverSpHandle : public SolverInterface
{
public:
    /// <summary>
    /// <param name="m"> represent the number of rows of A matrix </param>
    /// <param name="n"> represent the number of columns of A matrix </param>
    /// <param name="nnzA"> represent the number of non-zero value in the matrix </param>
    /// <param name="tol"> represent the tolerance which is used for judging singularity of the matrix </param>
    /// <param name="reorder">  </param>
    /// <param name="length_x"> represent the length of the output vector x </param>
    /// <param name="singularity"> represent the matrix A's singularity(-1 if A is invertible) </param>
    /// <param name="rankA"> represent the rank of matrix A for some configure  </param>
    /// <param name="min_norm">  </param>
    /// </summary>
    
    int m;
    int n;
    int nnz;

    double tol;
    int reorder;
    int length_x;
    int singularity;
    int rankA;
    cusolverSpHandle_t handle;
    cusparseMatDescr_t descrA;
    double min_norm;
    Solution solution;
    
#ifdef CPU_MODE
    std::vector<double> csrValA;//const double* csrValA;
    std::vector<int> csrRowPtrA;//const int* csrRowPtrA;
    std::vector<int> csrColIndA;//const int* csrColIndA;
    std::vector<double> B;// const double* b;
    //std::vector<double> X;//double* x;
    std::vector<int> p;//double* p;
#endif // CPU_MODE

#ifdef GPU_MODE
    //cusparseHandle_t cusparseHandle;
    //cudaStream_t stream;

    std::vector<double,CudaAllocator<double>> csrValA;
    std::vector<int, CudaAllocator<int>> csrRowPtrA;
    std::vector<int, CudaAllocator<int>> csrColIndA;
    std::vector<double, CudaAllocator<double>> B;
    std::vector<double, CudaAllocator<double>> X;
#endif // GPU_MODE

    

public:
    CusolverSpHandle():singularity(-1), reorder(0), tol(1.e-12), length_x(0), rankA(0), min_norm(0),solution(Solution::LU),
        m(0),n(0),nnz(0)
    {
        checkCudaErrors(cusolverSpCreate(&handle));
        checkCudaErrors(cusparseCreateMatDescr(&descrA));
        checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }
#ifdef CPU_MODE
    CusolverSpHandle(int _m,int _n, int _nnz, int _reorder = 0, double _tol = 1.e-12) :
        csrValA(_nnz),csrRowPtrA(_m + size_t(1)), csrColIndA(_nnz), 
        B(_m),p(_n),m(_m),n(_n),nnz(_nnz),
        singularity(-1), reorder(_reorder), tol(_tol), length_x(0),rankA(0),min_norm(0), solution(Solution::LU)
    {
        checkCudaErrors(cusolverSpCreate(&handle));
        checkCudaErrors(cusparseCreateMatDescr(&descrA));
        checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }

    ~CusolverSpHandle()
    {
        if (handle) { checkCudaErrors(cusolverSpDestroy(handle)); }
        if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
    }
#endif // CPU_MODE

#ifdef GPU_MODE
    CusolverSpHandle(int _m,int _n, int _nnz, int _reorder = 0, double _tol = 1.e-12) :
        csrValA(_nnz),csrRowPtrA(_m + size_t(1)), csrColIndA(_nnz),
        B(_m), X(_n),
        singularity(0), reorder(_reorder), tol(_tol), length_x(0),rankA(0), min_norm(0)
    {

        checkCudaErrors(cusolverSpCreate(&handle));
        checkCudaErrors(cusparseCreateMatDescr(&descrA));
        //checkCudaErrors(cusparseCreate(&cusparseHandle));
        //checkCudaErrors(cudaStreamCreate(&stream));

        //checkCudaErrors(cusolverSpSetStream(handle, stream));
        //checkCudaErrors(cusparseSetStream(cusparseHandle, stream));
        checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }

    ~CusolverSpHandle()
    {
        if (handle) { checkCudaErrors(cusolverSpDestroy(handle)); }
        if (descrA) { checkCudaErrors(cusparseDestroyMatDescr(descrA)); }
        //if (cusparseHandle) {checkCudaErrors(cusparseDestroy(cusparseHandle));}
        //if (stream) {checkCudaErrors(cudaStreamDestroy(stream));}
    }
#endif // GPU_MODE

//CPU端成员函数
#ifdef CPU_MODE
    
    void solve()
    {
        //printf("SP\n");
        if (X.size() < n) X.resize(n);

        if (solution == Solution::LU) { solvelslu(m,nnz); }

        else if (solution == Solution::QR) { solvelsqr(m, nnz); }

        else if (solution == Solution::QQR) { solvelsqqr(m, n, nnz); }

        length_x = n;
        //if(singularity >= 0)
        //printf("singularity=%d\n", singularity);
    }
    
    void solvelslu(int num_of_rows_and_cols, int number_of_non_zero_value){
        checkCudaErrors(cusolverSpDcsrlsvluHost(handle, num_of_rows_and_cols, number_of_non_zero_value, descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(), B.data(), tol, reorder, X.data(), &singularity));
    }

    void solvelsqr(int num_of_rows_and_cols, int number_of_non_zero_value){
        checkCudaErrors(cusolverSpDcsrlsvqrHost(handle, num_of_rows_and_cols, number_of_non_zero_value, descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(), B.data(), tol, reorder, X.data(), &singularity));
    }

    void solvelsqqr(int num_of_rows,int num_of_cols, int number_of_non_zero_value){
        if(p.size()!=n) p.resize(n);
        checkCudaErrors(cusolverSpDcsrlsqvqrHost(handle, num_of_rows, num_of_cols, number_of_non_zero_value, descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(), B.data(), tol, &rankA, X.data(), p.data(), &min_norm));
    }

    

#endif // CPU_MODE
    
///GPU端成员函数
#ifdef GPU_MODE
    template<typename functype>
    void solve(int num_of_rows,int num_of_cols, int num_of_non_zero_value, functype, int dstDevice = 0)
    {
        //经过测试，本人的nvidia 3050Laptop显卡不支持异步预取功能
        //文档说设备必须具有cudaDevAttrConcurrentManagedAccess属性为非零
        
        
        if (X.size() < num_of_cols) X.resize(num_of_cols);

        if constexpr (std::is_same_v<functype, LSQR>) { solvelsqr(num_of_rows,num_of_non_zero_value); }

        if constexpr (std::is_same_v<functype, LSCHOL>) { solvelschol(num_of_rows, num_of_non_zero_value); }
        
        length_x = num_of_cols;
        //cudaMemPrefetchAsyncDtoH(stream, X.data(), num_of_rows_and_cols);
    }

    void solvelsqr(int num_of_rows_and_cols, int num_of_non_zero_value, int dstDevice = 0){
        checkCudaErrors(cusolverSpDcsrlsvqr(handle, num_of_rows_and_cols, num_of_non_zero_value, descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(), B.data(), tol, reorder, X.data(), &singularity));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void solvelschol(int num_of_rows_and_cols, int num_of_non_zero_value, int dstDevice = 0){
        checkCudaErrors(cusolverSpDcsrlsvchol(handle, num_of_rows_and_cols, num_of_non_zero_value, descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(), B.data(), tol, reorder, X.data(), &singularity));
        //checkCudaErrors(cudaDeviceSynchronize());
    }
    

#endif // GPU_MODE

    void Initialize(const std::vector<double>& csrval, const std::vector<int>& csrrowptr, const std::vector<int>& csrcolind)
    {
        csrValA.resize(csrval.size());
        std::copy(csrval.begin(), csrval.end(), csrValA.begin());
        csrRowPtrA.resize(csrrowptr.size());
        std::copy(csrrowptr.begin(), csrrowptr.end(), csrRowPtrA.begin());
        csrColIndA.resize(csrcolind.size());
        std::copy(csrcolind.begin(), csrcolind.end(), csrColIndA.begin());

        nnz = csrval.size();
        m = csrrowptr.size() - 1;
        n = m;
    }

    void ResetA(const std::vector<double>& csrval, const std::vector<int>& csrrowptr, const std::vector<int>& csrcolind)
    {
        std::copy(csrval.begin(), csrval.end(), csrValA.begin());
        std::copy(csrrowptr.begin(), csrrowptr.end(), csrRowPtrA.begin());
        std::copy(csrcolind.begin(), csrcolind.end(), csrColIndA.begin());
    }

    void loadB(const std::vector<double>& b)
    {
        if (b.size() <= B.size()); else B.resize(b.size());
        std::copy(b.begin(), b.end(), B.begin());
    }

    void Reset()
    {
       
    }
};