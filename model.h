#pragma once
#include"Xml.h"
#include<string>
#include <utility>
#include<iostream>
#include<fstream>
#include<sstream>

enum class BoundaryCondition
{
	ClampedFree, ClampedBoth
};

enum class ForceCondition
{
	BodyForceOnly,SurfaceForceOnly,SurfaceAndBodyForce
};

enum class Solver
{
	SP, RF
};

enum class Solution
{
	LU, QR, QQR
};

enum class SimulateType
{
	Extrude,Bend
};

enum class ExtrudePolicy
{
	Dense, Sparse
};

struct ModelConf
{
	struct Policy
	{
		ExtrudePolicy policy;
		double Ds;
		double DsEnd;
		int SparseNum;
		int DenseNum;
		int iterating;
		Policy(ExtrudePolicy Exp):policy(Exp),Ds(0.0),DsEnd(0.0), SparseNum(0),DenseNum(0),iterating(0){}
	};

	double H;
	double deltaS;
	double velocity;
	double upperVelocity;
	double viscosity;
	double mantleViscosity;
	double density;
	double pipePressure;
	double pipePressurerate;
	double Omega;
	double omega;
	double Delta;
	double theta;
	double K;
	double dt;
	double slabLength;
	double lithosphereTop;
	double lithosphereBottom;
	double g;
	double dampingrate;
	double dampingrate_copy;
	double criticalangle;
	int grid_num;
	int grid_num_copy;
	int num_iterate;
	int deltaT_coefficient;
	bool SaveEveryXY;
	int dampSearch;
	int dampSearchResolventCount;
	int num_XY;
	int criticalRangeCount;
	int criticalAngleSegment;
	std::pair<long long,double> omegaStandard;
	std::pair<long long, double> velocityStandard;
	std::pair<double, double> criticalAngleRange;
	BoundaryCondition boundaryCondition;
	ForceCondition forceCondition;
	Solver solver;
	Solution solution;
	SimulateType simulation;
	Policy extrudepolicy;
	std::string XYaddress;
	std::string Forceaddress;
	std::string Torqueaddress;



	ModelConf(int _grid_num = 0, int _num_iterate = 0, double H_value = 0.0, double deltaS_value = 0.0, double velocity_value = 0.0, double _viscosity = 0.0, double _Mviscosity = 0.0, double _density = 0.0,
		double Omega_value = 0.0, double Delta_value = 0.0, double omega_value = 0.0, double K_value = 0.0, double theta_value = 0.0, double _dt = 0.0) :
		grid_num(_grid_num),grid_num_copy(_grid_num),H(H_value), deltaS(deltaS_value), velocity(velocity_value),upperVelocity(0.0), pipePressure(1e8), pipePressurerate(1.0),
		viscosity(_viscosity),mantleViscosity(_Mviscosity), density(_density), Omega(Omega_value), omega(omega_value),
		Delta(Delta_value), theta(theta_value), K(K_value), dt(_dt), XYaddress(), num_iterate(_num_iterate), deltaT_coefficient(200.0),
		slabLength((size_t(_grid_num) - 1)* deltaS_value),SaveEveryXY(true), num_XY(0),g(9.8),omegaStandard(_grid_num-1,0),velocityStandard(_grid_num-1,velocity),
		dampingrate(1e-3), dampingrate_copy(1e-3),criticalangle(-0.25),dampSearch(1), dampSearchResolventCount(1),criticalAngleRange(0.0,0.0), criticalRangeCount(0), criticalAngleSegment(1),
		boundaryCondition(BoundaryCondition::ClampedFree), forceCondition(ForceCondition::SurfaceAndBodyForce),
		solver(Solver::RF),solution(Solution::LU),simulation(SimulateType::Bend),extrudepolicy(ExtrudePolicy::Sparse), lithosphereTop(-1e4), lithosphereBottom(-1.3e5)
	{
	}

	void Standardize()
	{
		static int i = 0;
		if (boundaryCondition == BoundaryCondition::ClampedFree)
		{
			this->omegaStandard.first = (long long)this->grid_num - 1;
			this->velocityStandard.first = (long long)this->grid_num - 1;
			if (!(i++))
			{
				this->omegaStandard.second = 0.0;
				this->velocityStandard.second = this->velocity;
			}
		}
		else if (boundaryCondition == BoundaryCondition::ClampedBoth)
		{
			//´ý²¹³ä
		}
    }

	void ResetGridAndIterating()
	{
		grid_num = grid_num_copy;
		extrudepolicy.iterating = 0;
	}

	void ResetDampRate()
	{
		dampingrate = dampingrate_copy;
	}

	void Debug()
	{
		if (simulation == SimulateType::Extrude)
		{
			std::cout << "Simulate::Extrude" << "\n";
		}
		else if (simulation == SimulateType::Bend)
		{
			std::cout << "Simulate::Bend" << "\n";
		}

		if (extrudepolicy.policy == ExtrudePolicy::Sparse)
		{
			std::cout << "Extrude::Sparse" << "\n";
			std::cout << "SparseNum:" << extrudepolicy.SparseNum << "\n";
			std::cout << "DsEnd:" << extrudepolicy.DsEnd << "\n";
		}
		else if (extrudepolicy.policy == ExtrudePolicy::Dense)
		{
			std::cout << "Extrude::Dense" << "\n";
			std::cout << "DenseNum:" << extrudepolicy.DenseNum << "\n";
			std::cout << "Ds:" << extrudepolicy.Ds << "\n";
			std::cout << "DsEnd:" << extrudepolicy.DsEnd << "\n";
		}
		std::cout << "slabLength:" << slabLength << "\n";
		std::cout << "deltaS:" << deltaS << "\n";
		std::cout << "taub:" << dt * deltaT_coefficient << "\n";
		std::cout << "dt:" << dt << "\n";
		std::cout << "velocity:" << velocity << "\n" << std::endl;

	}

	void load_parameter(std::ifstream& file)
	{
		std::map<std::string, Solver> SolverSelecter;
		std::map<std::string, Solution> SolutionSelecter;
		std::map<std::string, BoundaryCondition> BoundaryConditionSelecter;
		std::map<std::string, SimulateType> SimulateTypeSelecter;
		std::map<std::string, ForceCondition> ForceConditionSelecter;
		SolverSelecter["RF"] = Solver::RF;
		SolverSelecter["SP"] = Solver::SP;
		SolutionSelecter["LU"] = Solution::LU;
		SolutionSelecter["QR"] = Solution::QR;
		SolutionSelecter["QQR"] = Solution::QQR;
		BoundaryConditionSelecter["ClampedFree"] = BoundaryCondition::ClampedFree;
		BoundaryConditionSelecter["ClampedBoth"] = BoundaryCondition::ClampedBoth;
		SimulateTypeSelecter["Bend"] = SimulateType::Bend;
		SimulateTypeSelecter["Extrude"] = SimulateType::Extrude;
		ForceConditionSelecter["BodyForceOnly"] = ForceCondition::BodyForceOnly;
		ForceConditionSelecter["SurfaceForceOnly"] = ForceCondition::SurfaceForceOnly;
		ForceConditionSelecter["SurfaceAndBodyForce"] = ForceCondition::SurfaceAndBodyForce;

		std::stringstream ss;
		ss << file.rdbuf();
		const std::string& str = ss.str();
		xml::Xml root;
		root.parse(str);

		grid_num = std::stoi(root["grid_num"].text());
		grid_num_copy = grid_num;
		H = std::stod(root["H"].text());
		slabLength = std::stod(root["SlabLength"].text());
		velocity = std::stod(root["velocity"].text());
		upperVelocity = std::stod(root["UpperVelocity"].text());
		viscosity = std::stod(root["viscosity"].text());
		mantleViscosity = std::stod(root["MantleViscosity"].text());
		pipePressurerate = std::stod(root["PipePressureRate"].text());
		dampingrate = std::stod(root["DampingRate"].text());
		dampingrate_copy = dampingrate;
		density = std::stod(root["density"].text());
		lithosphereTop = std::stod(root["LithosphereTop"].text());
		lithosphereBottom = std::stod(root["LithosphereBottom"].text());
		theta = std::stod(root["theta"].text());
		XYaddress = root["XYaddress"].text();
		Forceaddress = root["ForceAddress"].text();
		Torqueaddress = root["TorqueAddress"].text();
		dampSearch = std::stoi(root["DampSearch"].text());
		dampSearchResolventCount = std::stoi(root["DampSearchResolventCount"].text());
		num_iterate = std::stoi(root["num_iterate"].text());
		deltaT_coefficient = std::stoi(root["deltaT_coefficient"].text());
		
		solver = SolverSelecter[root["Solver"].text()];
		if (solver == Solver::SP)solution = SolutionSelecter[root["Solution"].text()];
		boundaryCondition = BoundaryConditionSelecter[root["BoundaryCondition"].text()];
		simulation = SimulateTypeSelecter[root["StimulateType"].text()];
		forceCondition = ForceConditionSelecter[root["ForceCondition"].text()];

		deltaS = slabLength / (size_t(grid_num) - 1);
		criticalangle = std::stod(root["CriticalAngle"].text());
		criticalRangeCount = std::stoi(root["CriticalAngleRangeCount"].text());
		criticalAngleRange.first = std::stod(root["CriticalAngleRangeLeft"].text());
		criticalAngleRange.second = std::stod(root["CriticalAngleRangeRight"].text());
		criticalAngleSegment = std::stoi(root["CriticalAngleSegment"].text());


		if (criticalRangeCount > 1) criticalangle = (criticalAngleRange.first + criticalAngleRange.second) / 2.0;

		if (simulation == SimulateType::Bend)
		{
			dt = viscosity * H * H / g / density / (double)deltaT_coefficient / slabLength / slabLength / slabLength;
		}
		else if (simulation == SimulateType::Extrude)
		{
			//velocity = pow(20.0, 4) * H * H * g * density / viscosity;
			//slabLength = 0.1*pow(viscosity * velocity * H * H / g / density, 0.25);
			deltaS = slabLength / (size_t(grid_num) - 1);
			//dt = pow(viscosity * H * H / velocity / velocity / velocity / g / density, 0.25) / (double)deltaT_coefficient;
			//dt = slabLength / velocity / (double)deltaT_coefficient;
			//velocity += upperVelocity;
			dt = pow(viscosity * H * H / (velocity + upperVelocity) / (velocity + upperVelocity) / (velocity + upperVelocity) / g / density, 0.25) / (double)deltaT_coefficient;
		}
		velocity += upperVelocity;

		double IntervalMove = velocity * dt;

		if (IntervalMove * 2.0 < deltaS)
		{
			if (velocity < 1e-12)
			{
				extrudepolicy.SparseNum = num_iterate + 1;
			}
			else
			{
				extrudepolicy.policy = ExtrudePolicy::Sparse;
				extrudepolicy.SparseNum = int(deltaS / IntervalMove);
				extrudepolicy.DsEnd = extrudepolicy.SparseNum * IntervalMove;
			}
		}
		else
		{
			extrudepolicy.policy = ExtrudePolicy::Dense;
			if (IntervalMove > 1.5 * deltaS)
			{
				extrudepolicy.DenseNum = int(IntervalMove / deltaS);
				double midterm = fabs(extrudepolicy.DenseNum - IntervalMove / deltaS);
				extrudepolicy.Ds = deltaS;
				extrudepolicy.DsEnd = deltaS * midterm;
				if (midterm < 0.5)
				{
					extrudepolicy.DsEnd += deltaS;
					extrudepolicy.DenseNum--;
				}
			}
			else
			{
				extrudepolicy.DenseNum = 0;
				extrudepolicy.DsEnd = IntervalMove;
			}
		}
		pipePressure = pipePressurerate * density * g * H;

		Standardize();
	}

};
