#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<vector>
#include<type_traits>
#define RELEASED 1

#ifndef RELEASED 
#define DEBUG
#include"helper_cuda.h"
#endif // !RELEASED

#ifdef RELEASED
#define checkCudaErrors(expr) expr
#endif // RELEASED


template<typename Type>
class CudaAllocator
{
public:
	using value_type = Type;
	using pointer = Type*;
	using const_pointer = const Type*;
	using reference = Type&;
	using const_reference = const Type&;
	using size_type = size_t;
	using difference_type = ptrdiff_t;

	template <class U>
	struct rebind
	{
		using other = CudaAllocator<U>;
	};

	CudaAllocator() noexcept {};
	~CudaAllocator() noexcept {};

	pointer allocate(size_t _Count)
	{
		Type* ptr = nullptr;
		checkCudaErrors(cudaMallocManaged(&ptr, sizeof(Type) * _Count));
		return ptr;
	}

	const_pointer address(const_reference _Val) const noexcept
	{
		return &_Val;
	}

	void deallocate(Type* ptr, size_t)
	{
		checkCudaErrors(cudaFree(ptr));
	}

	template<typename... Args>
	void construct(Type* ptr, Args && ... args)
	{
		if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<Type>))
			:: new((void*)ptr) Type(std::forward<Args>(args)...);
	}

	void destroy(Type* ptr)
	{
		if constexpr (!(std::is_pod_v<Type> || std::is_trivially_destructible_v<Type>))
			ptr->~Type();
	}

	CudaAllocator<Type>& operator=(const CudaAllocator<Type>&) = default;

};
