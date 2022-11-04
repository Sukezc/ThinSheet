#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

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


#define CUVECTOR_CLASS_BEGIN(dataType) \
template<typename dataType> \
class CuVector \
{ \
public: \
	thrust::host_vector<dataType> h_vector; \
	thrust::device_vector<dataType> d_vector;\
public: \
	CLASS_DEFAULT_CONSTRUCT(CuVector)\
	template<typename ... Args>\
	CuVector(Args&&... args) {\
	d_vector(args...);\
	h_vector(std::forward<Args>(args)...);\
	}\
	\
	void send(){ \
		d_vector = h_vecotr;\
	}\
	\
	void fetch(){ \
		h_vector = d_vector;\
	}


#define CUVECTOR_CREATE_EMPTY_FUNCTION_BIND(function,method) \
	void function() { \
		h_vector.method();\
		d_vector.method();\
}


#define CUVECTOR_CREATE_FUNCTION_BIND(function,method, ...) \
    void function(PASTE(PARAMETER_EXPAND_,COUNT_ARGS(__VA_ARGS__) (UNPAIR,__VA_ARGS__))) {\
		h_vector.method(PASTE(PARAMETER_EXPAND_,COUNT_ARGS(__VA_ARGS__) (STRIP,__VA_ARGS__)));\
		d_vector.method(PASTE(PARAMETER_EXPAND_,COUNT_ARGS(__VA_ARGS__) (STRIP,__VA_ARGS__)));\
}

#define CUVECTOR_CREATE_EMPTY_FUNCTION_BEGIN(function, returnType) \
	returnType function(){

#define CUVECTOR_CREATE_FUNCTION_BEGIN(function, returnType,...) \
	returnType function(PASTE(PARAMETER_EXPAND_,COUNT_ARGS(__VA_ARGS__) (UNPAIR,__VA_ARGS__))){

#define TEMPLATE_BEGIN template<
#define TEMPLATE_END >
#define TYPENAME(type) typename type

#define CUVECTOR_CREATE_FUNCTION_END(...) }

#define CUVECTOR_CLASS_END(...) };

CUVECTOR_CLASS_BEGIN(dataType)
CUVECTOR_CREATE_FUNCTION_BIND(resize, resize, (size_t)size)
CUVECTOR_CREATE_FUNCTION_BIND(resize, resize, (size_t)size, (dataType)vals)

CUVECTOR_CREATE_FUNCTION_BEGIN(data, dataType*, (int))
return d_vector.data().get();
CUVECTOR_CREATE_FUNCTION_END(data, dataType*, (int))

CUVECTOR_CREATE_EMPTY_FUNCTION_BEGIN(data, dataType*)
return h_vector.data();
CUVECTOR_CREATE_FUNCTION_END(data, dataType*)

CUVECTOR_CLASS_END(dataType)
