#pragma once
#include<string>
#include<map>

template<class Class, typename... Args>
void* __createObjFunc(Args... args)
{
	return new Class(args...);
}

#ifndef REGISTER
#define REGISTER(Class, ...)\
	ObjFactory::Instance().RegisterCreateObjFunc(#Class, (void*)&__createObjFunc<Class, ##__VA_ARGS__>);
#endif // !REGISTER


class ObjFactory
{
public:
	std::map<std::string, void*> _map;

	static ObjFactory& Instance()
	{
		static ObjFactory Fac;
		return Fac;
	}

	template<typename BaseClass, typename... Args>
	BaseClass* CreateObj(std::string name, Args... args)
	{
		using _createFactory = BaseClass * (*)(Args...);
		auto it = Map().find(name);
		if (it == Map().end())return NULL;
		else
		{
			return reinterpret_cast<_createFactory>(Map()[name])(args...);
		}
	}

	void RegisterCreateObjFunc(std::string name, void* fun)
	{
		Instance().Map().emplace(name, fun);
	}

	std::map<std::string, void*>& Map()
	{
		return _map;
	}
};
