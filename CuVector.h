#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/functional.h>
#include<type_traits>

#define EXPAND(x) x

#define PARE(...) __VA_ARGS__ // UNPAIR((double) x) => PARE(double) x => double x
#define UNPAIR(x) PARE x

#define EAT(...)
#define STRIP(x) EAT x // STRIP((double) x) => EAT(double) x => x

#define __COUNT_ARGS(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_n,...) _n
#define COUNT_ARGS(...) EXPAND(__COUNT_ARGS(__VA_ARGS__,10,9,8,7,6,5,4,3,2,1))

#define PARAMETER_EXPAND_1(func,arg) func(arg)
#define PARAMETER_EXPAND_2(func,arg,...) func(arg),PARAMETER_EXPAND_1(func,__VA_ARGS__) 
#define PARAMETER_EXPAND_3(func,arg,...) func(arg),PARAMETER_EXPAND_2(func,__VA_ARGS__)
#define PARAMETER_EXPAND_4(func,arg,...) func(arg),PARAMETER_EXPAND_3(func,__VA_ARGS__)
#define PARAMETER_EXPAND_5(func,arg,...) func(arg),PARAMETER_EXPAND_4(func,__VA_ARGS__)
#define PARAMETER_EXPAND_6(func,arg,...) func(arg),PARAMETER_EXPAND_5(func,__VA_ARGS__)
#define PARAMETER_EXPAND_7(func,arg,...) func(arg),PARAMETER_EXPAND_6(func,__VA_ARGS__)
#define PARAMETER_EXPAND_8(func,arg,...) func(arg),PARAMETER_EXPAND_7(func,__VA_ARGS__)
#define PARAMETER_EXPAND_9(func,arg,...) func(arg),PARAMETER_EXPAND_8(func,__VA_ARGS__)
#define PARAMETER_EXPAND_10(func,arg,...) func(arg),PARAMETER_EXPAND_9(func,__VA_ARGS__)

#define _PASTE(x, y) x##y
#define PASTE(x, y) _PASTE(x, y)

#define CLASS_DEFAULT_CONSTRUCT(className)\
	className() = default; \
	~className() = default; \
	className(const className& other) = default;\
	className(className&& other) = default;\
	className& operator=(const className& other) = default;\
	className& operator=(className&& other) = default;


//#define CUVECTOR_CLASS_BEGIN(dataType) \
//template<typename dataType> \
//class CuVector \
//{ \
//public: \
//	thrust::host_vector<dataType> h_vector; \
//	thrust::device_vector<dataType> d_vector;\
//public: \
//	CLASS_DEFAULT_CONSTRUCT(CuVector)\
//	template<typename ... Args>\
//	CuVector(Args&&... args) {\
//		h_vector = thrust::host_vector<dataType>(std::forward<Args>(args)...);\
//		d_vector = h_vector;\
//	}\
//	\
//	void send(){ \
//		d_vector = h_vector;\
//	}\
//	\
//	void fetch(){ \
//		h_vector = d_vector;\
//	}\
//	\
//	using value_type = dataType;\
//	using size_type = size_t;
//
//
//#define CUVECTOR_CREATE_EMPTY_FUNCTION_BIND(function,method) \
//	void function() { \
//		h_vector.method();\
//		d_vector.method();\
//}
//
//
//#define CUVECTOR_CREATE_FUNCTION_BIND(function,method, ...) \
//    void function(PASTE(PARAMETER_EXPAND_,COUNT_ARGS(__VA_ARGS__) (UNPAIR,__VA_ARGS__))) {\
//		h_vector.method(PASTE(PARAMETER_EXPAND_,COUNT_ARGS(__VA_ARGS__) (STRIP,__VA_ARGS__)));\
//		d_vector.method(PASTE(PARAMETER_EXPAND_,COUNT_ARGS(__VA_ARGS__) (STRIP,__VA_ARGS__)));\
//}
//
//#define CUVECTOR_CREATE_EMPTY_FUNCTION_BEGIN(function, returnType) \
//	returnType function(){
//
//#define CUVECTOR_CREATE_FUNCTION_BEGIN(function, returnType,...) \
//	returnType function(PASTE(PARAMETER_EXPAND_,COUNT_ARGS(__VA_ARGS__) (UNPAIR,__VA_ARGS__))){
//
//#define TEMPLATE_BEGIN template<
//#define TEMPLATE_END >
//#define TYPENAME(type) typename type
//
//#define CUVECTOR_CREATE_FUNCTION_END(...) }
//
//#define CUVECTOR_CLASS_END(...) };
//
//
////CUVECTOR_CREATE_FUNCTION_BEGIN()
//// 
////CUVECTOR_CREATE_FUNCTION_END()

#define CuVecDev ((int)0)
#define CVD CuVecDev
struct HostToDevice {};
struct DeviceToHost {};

template<typename dataType> 
class CuVector
{	
public:
	thrust::host_vector<dataType> Hvec;
	thrust::device_vector<dataType> Dvec;
public:
	CLASS_DEFAULT_CONSTRUCT(CuVector)
		template<typename ... Args>
	CuVector(Args&&... args) {
		Hvec = thrust::host_vector<dataType>(std::forward<Args>(args)...);
		Dvec = Hvec;
	}

	using value_type = dataType;
	using size_type = size_t;

	void send() {
		Dvec = Hvec;
	}


	void fetch() {
		Hvec = Dvec;
	}

	template<typename Iterator1, typename Iterater2>
	void trans(Iterator1 input1, Iterator1 input2, Iterater2 output)
	{
		thrust::copy(input1, input2, output);
	}

	template<typename KeyType>
	void trans(KeyType input1, KeyType input2, KeyType output,HostToDevice)
	{
		thrust::copy(Hvec.begin() + input1, Hvec.begin() + input2, Dvec.begin() + output);
	}

	template<typename KeyType>
	void trans(KeyType input1, KeyType input2, KeyType output, DeviceToHost)
	{
		thrust::copy(Dvec.begin() + input1, Dvec.begin() + input2, Hvec.begin() + output);
	}

	void resize(size_type size)
	{
		Hvec.resize(size);
		Dvec.resize(size);
	}

	void resize(size_type size, const dataType& vals)
	{
		Hvec.resize(size, vals);
		Dvec.resize(size, vals);
	}

	dataType* data() noexcept
	{
		return Hvec.data();
	}

	dataType* data(int) noexcept
	{
		return Dvec.data().get();
	}

	dataType& operator[](size_type pos)
	{
		return Hvec[pos];
	}

	auto begin() noexcept{return Hvec.begin();}

	auto end() noexcept{return Hvec.end();}

	auto cbegin() const noexcept{ return Hvec.cbegin(); }

	auto cend() noexcept{ return Hvec.cend(); }

	auto rbegin() noexcept{return Hvec.rbegin();}

	auto rend() noexcept{return Hvec.rend();}

	auto crbegin() noexcept {return Hvec.crbegin();}

	auto crend() noexcept {return Hvec.crend();}

	auto begin(int) noexcept{return Dvec.begin();}

	auto end(int) noexcept{return Dvec.end();}

	auto cbegin(int) const noexcept{return Dvec.cbegin();}

	auto cend(int) const noexcept{return Dvec.cend();}

	auto rbegin(int) noexcept{return Dvec.rbegin();}

	auto rend(int) noexcept{return Dvec.rend();}

	auto crbegin(int)const noexcept {return Dvec.crbegin();}

	auto crend(int) const noexcept{return Dvec.crend();}

	size_type size() noexcept
	{
		return Hvec.size();
	}

	size_type size(int) noexcept
	{
		return Dvec.size();
	}

	void push_back(const dataType& vals)
	{
		Hvec.push_back(vals);
	}

	void push_back(const dataType& vals,int)
	{
		Dvec.push_back(vals);
	}

	dataType& back()
	{
		return Hvec.back();
	}

	auto back(int)
	{
		return Dvec.back();
	}

	void clear() noexcept
	{
		Hvec.clear();
	}


	template<typename Iterater>
	Iterater erase(Iterater left, Iterater right)
	{
		return Hvec.erase(left, right);
	}

	auto erase()
	{
		return Hvec.erase(Hvec.begin(), Hvec.end());
	}

	void SyncSize(HostToDevice)
	{
		Dvec.resize(Hvec.size());
	}

	void SyncSize(DeviceToHost)
	{
		Hvec.resize(Dvec.size());
	}

};