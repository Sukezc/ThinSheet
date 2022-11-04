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

#define CLASS_DEFAULT_CONSTRUCT(className)\
	className() = default; \
	~className() = default; \
	className(const className& other) = default;\
	className(className&& other) = default;\
	className& operator=(const className& other) = default;\
	className& operator=(className&& other) = default;


#define CUVECTOR_CLASS_BEGIN(dataType) \
template<typename dataType> \
class CuVector<dataType> \
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
	

#define CUVECTOR_CREATE_EMPTY_PARAMETER_FUNCTION_HOST(function,method,returnType)\
	returnType function() {\
		return h_vector.method();\
}

#define CUVECTOR_CREATE_FUNCTION_HOST(function,method,returnType,...)\
	returnType function(__VA_ARGS__) {\
		return h_vector.method(__VA_ARGS__);\
}

#define CUVECTOR_CREATE_EMPTY_PARAMETER_FUNCTION_BIND(function,method) \
	void function() { \
		h_vector.method();\
		d_vector.method();\
}


#define CUVECTOR_CREATE_FUNCTION_BIND(function,method, ...) \
    void function(__VA_ARGS__) {\
		h_vector.method(__VA_ARGS__);\
		d_vector.method(__VA_ARGS__);\
}

#define CUVECTOR_CREATE_EMPTY_PARAMETER_FUNCTION_BEGIN(function, returnType) \
	returnType function(){

#define CUVECTOR_CREATE_FUNCTION_BEGIN(function, returnType,...) \
	returnType function(__VA_ARGS__){

#define CUVECTOR_CREATE_FUNCTION_END(...) }

#define CUVECTOR_CLASS_END(...) }


