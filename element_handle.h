#pragma once
#include<vector>
#include<string>
#include<fstream>
#include<algorithm>
#include<numeric>
#include<type_traits>
#include<cmath>
#include<iomanip>
#include<iostream>
#include"model.h"


template<typename Type>
void vector_save(Type& vec, std::ofstream& outfile, size_t Col = size_t(-1), const std::string& sep = ",")
{
	const size_t col = Col; auto size = vec.size();
	{
		for (size_t i = 0; i < Col && i < size; i++)
		{
			outfile <<std::scientific << std::setprecision(9)<< vec[size - i - 1];
			if (i != Col - 1 && i != size - 1)outfile << sep;
			else { outfile << "\n"; Col += col; }
		}
	}
}


template<typename Type, typename Pred>
void vector_save_pred(Type& vec1,Type& vec2,Pred pred,std::ofstream& outfile, size_t Col = size_t(-1), const std::string& sep = ",")
{
	const size_t col = Col; auto size = vec1.size() < vec2.size() ? vec1.size() : vec2.size();
	{
		for (size_t i = 0; i < Col && i < size; i++)
		{
			outfile <<std::scientific << std::setprecision(9) << pred(vec1[size - i - 1], vec2[size - i - 1]);
			if (i != Col - 1 && i != size - 1)outfile << sep;
			else { outfile << "\n"; Col += col; }
		}
	}
}

/*

used for information integration

*/

class ElementGroup
{
public:
	
	//rate of bending
	std::vector<double> OmegaGroup;

	//rate of stretching
	std::vector<double> DeltaGroup;

	//rate of rotation
	std::vector<double> omegaGroup;

	//curvature
	std::vector<double> KGroup;

	//thickness of slab
	std::vector<double> HGroup;

	//angle of slab
	std::vector<double> thetaGroup;

	//Length element length
	std::vector<double> deltaSGroup;

	//velocity of the middle surface
	std::vector<double> velocityGroup;

	//relative density related to the depth
	std::vector<double> densityGroup;

	//all the surface force 
	std::vector<double> PupGroup;
	std::vector<double> PdownGroup;
	std::vector<double> TupGroup;
	std::vector<double> TdownGroup;
	
	//all the bodyforce and bodyforce decompose

	std::vector<double> GravityGroup;
	std::vector<double> GravityGroupCos;
	std::vector<double> GravityGroupSin;
	
	//decribe the node space information
	std::vector<double> XGroup;
	std::vector<double> YGroup;

	//describe the Torque
	static std::vector<double> GravityTorqueGroup;
	static std::vector<double> PforceTorqueGroup;

	//describe the valid data
	long long size;

	//decribe the eigenvalue of the slab
	double 	slabLength;
	double U0;
	double viscosity;
	double g;
	double density;

	static const size_t vector_num = 18;

public:

	ElementGroup() = default;


	/*
	 
	DEPRECATED

	*/

	ElementGroup(int grid_num, double H_value, double deltaS_value, double velocity_value, double _viscosity,double _density,
		double Omega_value = 0.0, double Delta_value = 0.0, double omega_value = 0.0, double K_value = 0.0,double theta_value = 0.0):
			OmegaGroup(grid_num, Omega_value),
			DeltaGroup(grid_num,Delta_value),
			omegaGroup(grid_num, omega_value),
			KGroup(grid_num, K_value),
			HGroup(grid_num, H_value),
			thetaGroup(grid_num, theta_value),
			deltaSGroup(grid_num, deltaS_value),
			velocityGroup(grid_num, velocity_value),
			densityGroup(grid_num, _density),
			PupGroup(grid_num, 0.0),PdownGroup(grid_num,0.0),TupGroup(grid_num,0.0),TdownGroup(grid_num,0.0),
			size(grid_num),slabLength(grid_num * deltaS_value - deltaS_value),U0(velocity_value), viscosity(_viscosity),g(9.8),density(_density),
			XGroup(grid_num,0.0),YGroup(grid_num,0.0)
	{
	}


	ElementGroup(ModelConf& model):
		OmegaGroup(model.grid_num, model.Omega),
		DeltaGroup(model.grid_num, model.Delta),
		omegaGroup(model.grid_num, model.omega),
		KGroup(model.grid_num, model.K),
		HGroup(model.grid_num, model.H),
		thetaGroup(model.grid_num, model.theta),
		deltaSGroup(model.grid_num, model.deltaS),
		velocityGroup(model.grid_num, model.velocity),
		densityGroup(model.grid_num, model.density),
		GravityGroup(model.grid_num, model.density* model.H * model.g),
		GravityGroupCos(model.grid_num,model.density*model.H * model.g),
		GravityGroupSin(model.grid_num,0.0),
		PupGroup(model.grid_num, 0.0), PdownGroup(model.grid_num, 0.0), TupGroup(model.grid_num, 0.0), TdownGroup(model.grid_num, 0.0),
		size(model.grid_num), slabLength(model.grid_num* model.deltaS - model.deltaS), U0(model.velocity), viscosity(model.viscosity), g(model.g), density(model.density),
		XGroup(model.grid_num, 0.0), YGroup(model.grid_num, 0.0)
	{
	}
	

	~ElementGroup() = default;
	ElementGroup(const ElementGroup& Eg) = default;
	ElementGroup(ElementGroup&& Eg) = default;
	ElementGroup& operator=(const ElementGroup& Eg) = default;
	ElementGroup& operator=(ElementGroup&& Eg) = default;

	ElementGroup& elongate(double deltaS_value, double H_value, double velocity_value,int num = 1)
	{
		if (num < 0) throw;
		if (num == 0)return *this;
		int num_place = 0;
		num_place = num > 1 ? num + 1 : num;
		const double OmegaInterval = OmegaGroup.back()/static_cast<double>(num_place);
		const double DeltaInterval = DeltaGroup.back()/static_cast<double>(num_place);
		const double omegaInterval = omegaGroup.back()/static_cast<double>(num_place);
		const double thetaInterval = thetaGroup.back()/static_cast<double>(num_place);
		for (int i = 0; i < num; i++)
		{
			OmegaGroup.emplace_back(OmegaGroup.back() - OmegaInterval);
			DeltaGroup.emplace_back(DeltaGroup.back() - DeltaInterval);
			omegaGroup.emplace_back(omegaGroup.back() - omegaInterval);
			KGroup.emplace_back(0.0);
			HGroup.emplace_back(H_value);
			thetaGroup.emplace_back(thetaGroup.back() - thetaInterval);
			deltaSGroup.emplace_back(deltaS_value);
			velocityGroup.emplace_back(velocity_value);
			densityGroup.emplace_back(density);
			PupGroup.emplace_back(0.0); PdownGroup.emplace_back(0.0);
			TupGroup.emplace_back(0.0); TdownGroup.emplace_back(0.0);
			GravityGroup.emplace_back(0.0);
			GravityGroupCos.emplace_back(0.0); GravityGroupSin.emplace_back(0.0);
			XGroup.emplace_back(0.0); YGroup.emplace_back(0.0);
			slabLength += deltaS_value; size += 1;
		}
		
		return *this;
	}

	
	static void	InitializeTorqueGroup(int num_iterate)
	{
		ElementGroup::GravityTorqueGroup.clear();
		ElementGroup::PforceTorqueGroup.clear();
		ElementGroup::GravityTorqueGroup.resize(num_iterate, 0.0);
		ElementGroup::PforceTorqueGroup.resize(num_iterate, 0.0);
	}

	ElementGroup& ComputeGravityAll()
	{
		for (long long i = this->size - 1; i > -1; i--)
		{
			this->GravityGroup[i] = this->HGroup[i] * this->densityGroup[i] * this->g;
			this->GravityGroupCos[i] = this->HGroup[i] * this->densityGroup[i] * this->g * cos(this->thetaGroup[i]);
			this->GravityGroupSin[i] = this->HGroup[i] * this->densityGroup[i] * this->g * sin(this->thetaGroup[i]);
		}
		return *this;
	}

	ElementGroup& ComputeGravityTorque(int iterating)
	{
		double gravityTorque = 0.0;
		for (long long i = this->size - 1; i > -1; i--)
		{
			gravityTorque += GravityGroup[i] * XGroup[i];
		}
		GravityTorqueGroup[GravityTorqueGroup.size() - iterating - 1] = gravityTorque;
		//std::cout <<"gravityTorque:" << gravityTorque << std::endl;
		return *this;
	}

	double GetGravityTorque(int iterating)
	{
		return GravityTorqueGroup[GravityTorqueGroup.size() - iterating - 1];
	}

	ElementGroup& ComputePforceTorque(int iterating)
	{
		double pforceTorque = 0.0;
		for (long long i = this->size - 1; i > -1; i--)
		{
			pforceTorque += (PupGroup[i] - PdownGroup[i]) * pow(XGroup[i] * XGroup[i] + YGroup[i] * YGroup[i], 0.5);
		}
		PforceTorqueGroup[PforceTorqueGroup.size() - iterating - 1] = pforceTorque;
		//std::cout <<"pforceTorque:"<< pforceTorque << std::endl;
		return *this;
	}

	double GetPforceTorque(int iterating)
	{
		return PforceTorqueGroup[PforceTorqueGroup.size() - iterating - 1];
	}

	ElementGroup& ComputeXY()
	{
		double x = 0, y = 0;
		XGroup[size - 1] = x;
		YGroup[size - 1] = y;
		for (long long i = size-1; i > 0; i--)
		{
			x += cos(thetaGroup[i]) * deltaSGroup[i];
			y += sin(thetaGroup[i]) * deltaSGroup[i];
			XGroup[i - 1] = x; YGroup[i - 1] = y;
		}
		return *this;
	}

	ElementGroup& ComputeY()
	{
		double y = 0;
		YGroup[size - 1] = y;
		for (long long i = size - 1; i > 0; i--)
		{
			y += sin(thetaGroup[i]) * deltaSGroup[i];
			YGroup[i - 1] = y;
		}
		return *this;
	}

	ElementGroup& ComputeSlabLength()
	{
		double length = 0;
		for (long long i = 1; i < size; i++)
		{
			length += deltaSGroup[i];
		}
		slabLength = length;
		return *this;
	}

	ElementGroup& SaveXY(std::ofstream& outfile)
	{
		vector_save(XGroup, outfile);
		vector_save(YGroup, outfile);
		outfile << "\n";
		return *this;
	}

	ElementGroup& SaveForce(std::ofstream& outfile)
	{
		vector_save(GravityGroupCos, outfile);
		vector_save_pred(PupGroup, PdownGroup, [](auto& it1, auto& it2) { return fabs(it1) + fabs(it2); }, outfile);
		vector_save(PupGroup, outfile);
		vector_save(PdownGroup, outfile);
		outfile << "\n";
		return *this;
	}

	static double ComputeGravityPforceTorqueInnerProduct()
	{
		static std::vector<double> tempPforceTorque; static std::vector<double> tempGravityTorque;
		tempPforceTorque = PforceTorqueGroup; tempGravityTorque = GravityTorqueGroup;
		std::for_each(tempPforceTorque.begin(), tempPforceTorque.end(), [](auto& it) {it /= 1e15; });
		std::for_each(tempGravityTorque.begin(), tempGravityTorque.end(), [](auto& it) {it /= 1e15; });
		return std::inner_product(tempPforceTorque.begin(), tempPforceTorque.end(), tempGravityTorque.begin(), 0.0, [](auto it1, auto it2){ return it1 + it2; }, [](auto it1, auto it2) {return (it1 - it2)*(it1 - it2); });
	}
	
	static void SaveState(ElementGroup* p_Egold, ElementGroup* p_Egnew, ModelConf* p_model,const std::string& outfile_name)
	{
		std::ofstream outfileEg; std::ofstream outfileModel;
		outfileEg.open(outfile_name + "Eg.dat", std::ios::binary | std::ios::out);
		std::vector<double>::size_type BufferSize;
		for (int i = 0; i < ElementGroup::vector_num; i++)
		{
			std::vector<double>* p_vector = (std::vector<double>*)p_Egold + i;
			BufferSize = p_vector->size();
			outfileEg.write((char*)&BufferSize, sizeof(std::vector<double>::size_type));
			outfileEg.write((char*)p_vector->data(), sizeof(double) * BufferSize);
			p_vector = (std::vector<double>*)p_Egnew + i;
			outfileEg.write((char*)p_vector->data(), sizeof(double) * BufferSize);
		}
		char* ret_old = (char*)((std::vector<double>*)p_Egold + ElementGroup::vector_num);
		char* ret_new = (char*)((std::vector<double>*)p_Egnew + ElementGroup::vector_num);
		ptrdiff_t retbyte = (char*)(p_Egold + 1) - ret_old;
		outfileEg.write(ret_old, retbyte);
		outfileEg.write(ret_new, retbyte);

		outfileModel.open(outfile_name + "Model.dat", std::ios::binary | std::ios::out);
		outfileModel.write((char*)p_model, sizeof(ModelConf));

		outfileEg.close();
		outfileModel.close();
	}

	static void LoadState(ElementGroup* p_Egold, ElementGroup* p_Egnew, ModelConf* p_model,const std::string& infile_name)
	{
		std::ifstream infileEg; std::ifstream infileModel;
		infileEg.open(infile_name + "Eg.dat", std::ios::binary | std::ios::in);
		std::vector<double>::size_type BufferSize;
		for (int i = 0; i < ElementGroup::vector_num; i++)
		{
			infileEg.read((char*)&BufferSize, sizeof(BufferSize));
			std::vector<double>* p_vector_old = (std::vector<double>*)p_Egold + i;
			std::vector<double>* p_vector_new = (std::vector<double>*)p_Egnew + i;
			p_vector_old->resize(BufferSize); p_vector_new->resize(BufferSize);
			infileEg.read((char*)p_vector_old->data(), sizeof(double) * BufferSize);
			infileEg.read((char*)p_vector_new->data(), sizeof(double) * BufferSize);
		}
		char* ret_old = (char*)((std::vector<double>*)p_Egold + ElementGroup::vector_num);
		char* ret_new = (char*)((std::vector<double>*)p_Egnew + ElementGroup::vector_num);
		ptrdiff_t retbyte = (char*)(p_Egold + 1) - ret_old;
		infileEg.read(ret_old, retbyte);
		infileEg.read(ret_new, retbyte);

		infileModel.open(infile_name + "Model.dat", std::ios::binary | std::ios::in);
		infileModel.read((char*)p_model, sizeof(ModelConf));

		infileEg.close();
		infileModel.close();
	}
};
