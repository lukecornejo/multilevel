#ifndef LOMLNDA_H // Include guard
#define LOMLNDA_H


#include <vector>

using namespace std;

class HOSolver;


class Triplet {
	int column, row;
	double value;
	
	public:
	//Triplet ();
	//Triplet (int, int, double);
	
	Triplet() {
	  column = 0;
	  row = 0;
	  value = 0.0;
	}

	Triplet(int i, int j, double input) {
	  row = i;
	  column = j;
	  value = input;
	}
	
	int Column() { return column; }
	int Row() { return row; }
	double Value() { return value; }
};

//======================================================================================//
//++ Cross Section Class ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
class LOXS {
	private:
	// Variables
	bool write_xs;
	vector< vector<int> > mIndex, material; // material #
	vector< vector<double> > sigma_gT, sigma_gF, nu_gF, nuSigma_gF, chi_g, s_gext, D_g; // material # arrays
	vector< vector< vector<double> > > sigma_gS; // material # arrays
	//======================================================================================//
	// variables
	public:
	// variables
	// Solution Grid
	int eta_star, Nx, Ny;
	vector<int> Ng;
	vector< vector<int> > omegaP;
	std::vector<double> x, y, xe, ye, hx, hy;
	
	vector<int> Ndown;
	vector< vector< vector< vector<double> > > > sigmaT, nuSigmaF, chi, s_ext; // material arrays
	vector< vector< vector< vector< vector<double> > > > > sigmaS; // material arrays
	vector<int> matnum; // material #
	vector<string> xsname; // material name
	// methods
	int initializeLOXS(HOSolver &);
	void writeLOMatXS(ofstream&);
	double Total(int, int, int, int);
	double Fission(int, int, int, int);
	double Chi(int, int, int, int);
	double Source(int, int, int, int);
	double Scatter(int, int, int, int, int);
	
	inline double Diffusion(int g, int i, int j) { return D_g[mIndex[i][j]][g]; };
	
	inline double Total0(int g, int i, int j) { return sigma_gT[mIndex[i][j]][g]; };
	inline double Fission0(int g, int i, int j) { return nuSigma_gF[mIndex[i][j]][g]; };
	inline double Chi0(int g, int i, int j) { return chi_g[mIndex[i][j]][g]; };
	inline double Source0(int g, int i, int j) { return s_gext[mIndex[i][j]][g]; };
	inline double Scatter0(int g, int gg, int i, int j) { return sigma_gS[mIndex[i][j]][g][gg]; };
	
	vector< vector<double> > Total(int, int);
	vector< vector<double> > Fission(int, int);
	vector< vector<double> > Chi(int, int);
	vector< vector<double> > Source(int, int);
	vector< vector<double> > Scatter(int, int, int);
	vector< vector<double> > Diffusion(int);
	
	void writeLOXSFile(std::string);
	void writeMaterial(std::ofstream&);
};
//======================================================================================//

//======================================================================================//
//++++++++++++++++++++++++++++++++ Low Order Solution ++++++++++++++++++++++++++++++++++//
//======================================================================================//
class LOSolution : public LOXS {
	private:
	//int Nx, Ny, eta_star;
	//std::vector<int> Ng;
	//std::vector< std::vector<int> > omegaP;
	//std::vector<double> x, y, hx, hy;
	
	public:
	bool writeOutput;                   // Write LO Solution
	bool NDASolution;
	//++ NDA Solution and Factors +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	std::vector< std::vector< std::vector<double> > > phiL, phiR, phiB, phiT;           // Boundary scalar flux from NDA
	//++ Low Order Factors +++++++++++++++++++++++++++++++++++++++++++
	std::vector< std::vector< std::vector< std::vector<double> > > > D_xP, D_xN, D_yP, D_yN; // Positive and Negative Corrected Diffusion Coefficients
	std::vector< std::vector< std::vector<double> > > D_x, D_y;                         // Boundary Diffusion coefficients
	//++ QD Solution and Factors ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	std::vector< std::vector< std::vector< std::vector<double> > > > phi_x, phi_y;          // NDA scalar flux and current solution
	//++ Low Order Factors +++++++++++++++++++++++++++++++++++++++++++
	std::vector< std::vector< std::vector< std::vector<double> > > > D_xxC, D_yyC;
	std::vector< std::vector< std::vector< std::vector<double> > > > D_xxL, D_yyL, D_xyL, D_xxR, D_yyR, D_xyR;
	std::vector< std::vector< std::vector< std::vector<double> > > > D_xxB, D_yyB, D_xyB, D_xxT, D_yyT, D_xyT;
	
	//++ Common Variables +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	std::vector< std::vector< std::vector< std::vector<double> > > > phi, phiLast, j_x, j_y;          // scalar flux and current solution
	std::vector< std::vector< std::vector<double> > >     FL,     FR,     FB,     FT;                   // Boundary Factors
	std::vector< std::vector< std::vector<double> > >   jInL,   jInR,   jInB,   jInT;                   // Boundary Factors
	std::vector< std::vector< std::vector<double> > > phiInL, phiInR, phiInB, phiInT;                   // Boundary Factors
	
	std::vector< std::vector< std::vector<double> > > sourceScatter, sourceFission, kappaLast;
	
	LOSolution() {
		writeOutput=true;
		NDASolution=true;
	}
	
	//++ Initialize Low Order Flux and Factors Arrays ++++++++++++++++++++++++++++++++
	void initializeSolution();
	//++ Calculate Low Order Factor From HO Factors and Solution +++++++++++++++++++++++++++//
	void LOFromHOFactors(HOSolver &);
	//++ Average Low Order Factors +++++++++++++++++++++++++++++++++
	void averageFactors(int);
	//++ Collapse Solution to a coarser grid
	void collapseSolution(int);
	//++ Normalize the Eigenfunction to The Area ++++++++++++++++++++++++++++++++++++++++++++++++++//
	void normalizeEigen(int);
	//++ Calculate Currents on Grid etaL
	void calculateGroupCurrents(int, int);
	//++ Calculate L# norm of Newton Residual
	double NewtonLNorm(int);
	
	//++ Write Low-order Method ++++++
	std::string writeLOSolverType();
	//++ Write Low-order Residuals +++
	void writeResiduals(int, ofstream&);
	//++ Write Consistency between Energy grids
	void writeConsistency(std::ofstream&);
	//++ Write Consistency between HO and LO solution
	void consistencyBetweenLOAndHO(HOSolver &, std::ofstream&);
	//++ Write LO solution to ".out" file ++++++
	void writeSolutionOut(std::ofstream&);
	//++ Write Low Order Solution and Factors ++++++++++++++++++++++++++++++++++++++++++++++
	void writeSolutionDat(std::string);
	
	//=======================================================================================================
};

//======================================================================================//
//+++++++++++++++++++++++++++++++++ Low Order Solver ++++++++++++++++++++++++++++++++++//
//======================================================================================//
class LOSolver : public LOSolution {
	private:
	// Varables
	vector< vector<double> > sigmaWS;
	// Methods
	//++ Recursive Function to Correct flux +++++++++++++++++++++++++++++++++++++++++++++++//
	void fCorrection(int, int, int, int, double, int);
	void oneLevelCorrection(int, int, int, int, double, int);
	public:
	// Variables
	// Options
	bool gaussSeidel;
	bool greyNewton;                      // Use Newton method in One-group problem
	bool solveK, fixK;                    // Solve for k-Eigenvalue, option to stop solving k after it converges
	bool trackFactorConvergence;           // Track Convergence of D_Tilde Factors
	bool relativeConvergence;             // Converge Using Relative Convergence Criteria
	vector<bool> wSolve;                  // Solve LO problem on prolongation step
	double initialK;                      // Give a fix value of k in the input
	double delta, tolNewtonR, tolNewtonA; // Grey solution variables
	int stop_solver;                      // Maximum number of solutions on low order problem
	double epsilon_solver;                // Tolerance of Solver
	int num_sol;
	vector< vector<double> > epsilon_phi, epsilon_keff;
	vector<double> resNewton, resInitialNewton;
	vector<int> stop_phi, relaxation, preconditioner_type; // limit number of iterations
	// Iteration histories
	vector< vector<double> > rho_phi, rho_phiH, rho_keff, rho_kappa, norm_phi, norm_phiH, norm_keff, norm_kappa, k_keff, err_lo, dt, dt_pc;
	vector<double> norm_DTLI, norm_DTL2;
	vector<int> num_mtot;
	vector< vector<int> > num_losi, num_logm, num_grid;
	// residual data
	vector<double> res_bal, res_mbal, res_Rmatrix, res_Amatrix;
	vector<int> i_bal, j_bal, g_bal, i_mbal, j_mbal, g_mbal, g_Rmatrix, g_Amatrix;
	// Methods
	LOSolver() {
		gaussSeidel=true;
		greyNewton=false;                      // Use Newton method in One-group problem
		fixK=false;                    // Solve for k-Eigenvalue, option to stop solving k after it converges
		trackFactorConvergence=false;           // Track Convergence of D_Tilde Factors
		relativeConvergence=false;             // Converge Using Relative Convergence Criteria
		initialK=1.0;                      // Give a fix value of k in the input
		delta=1e-3; tolNewtonR=0.0; tolNewtonA=0.0; // Grey solution variables
		stop_solver=1000;                      // Maximum number of solutions on low order problem
		epsilon_solver=0.0;                // Tolerance of Solver
	}
	void checkPreconditioner();
	int initializeLO(HOSolver &);
	//======================================================================================//
	//++ Find group average cross sections +++++++++++++++++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	void averageXS(int);
	//======================================================================================//
	//++ Calculate the Correction Factors and corrected flux +++++++++++++++++++++++++++++++//
	//======================================================================================//
	void correctFlux(int, std::vector< std::vector< std::vector<double> > > &, std::vector< std::vector< std::vector<double> > > &);
	void correctFlux(int, std::vector< std::vector<double> > &, std::vector< std::vector<double> > &);
	
	double residualIterative(int, int &, int &, int &);
	double findNormKappa(std::vector< std::vector<double> > &);
	
	double greyEigenvalue();
	double multigroupEigenvalue(int);
	//======================================================================================//
	void logIteration(int, double, double, double, double, double, double, double, double, double);
	
	bool greySolutionPositive();
	
	
	void constructAndSolve(int, int, int, std::vector<double> &, std::vector<double> &, std::vector<Triplet> &);
	
	//======================================================================================//
	//+++++++++++++++++++++++++++++++ Low Order Factors ++++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	
	void calculateGridCurrents(int);
	
	//======================================================================================//
	//+++++++++++++++++++++++++++++++ Low Order Solution +++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	
	//++ function to solve Grey Newton problem +++++++++++++++++++++++++++++++++++++++++++++++++++++//
	bool greyNewtonSolution(int, vector< vector<double> >&);
	//++ function to solve Fixed Source problem +++++++++++++++++++++++++++++++++++++++++++++++++++++//
	void fixedSourceSolution(int, int, int, std::vector< std::vector<double> >&, std::vector< std::vector<double> >&);
	
	//++ function to solve Grey Newton problem +++++++++++++++++++++++++++++++++++++++++++++++++++++//
	bool NDAgreyNewtonSolution(int, vector< vector<double> >&);
	//++ function to solve Fixed Source problem +++++++++++++++++++++++++++++++++++++++++++++++++++++//
	void NDAfixedSourceSolution(int, int, int, std::vector< std::vector<double> >&, std::vector< std::vector<double> >&);
	//++ function to solve Grey Newton problem +++++++++++++++++++++++++++++++++++++++++++++++++++++//
	bool QDgreyNewtonSolution(int, vector< vector<double> >&);
	//++ function to solve Fixed Source problem +++++++++++++++++++++++++++++++++++++++++++++++++++++//
	void QDfixedSourceSolution(int, int, int, std::vector< std::vector<double> >&, std::vector< std::vector<double> >&);
	
	//======================================================================================//
	//++++++++++++++++++++++++++++ Method Specific Functions +++++++++++++++++++++++++++++++//
	//======================================================================================//
	
	void edgeFlux(int, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &);
	
	std::string writeLinearSolverType();
	//======================================================================================//
	//++++++++++++++++++++++++++++ Input / Output Methods ++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	
	int readLO(std::vector<std::string>&, HOSolver&);
	
	void writeLO(std::ofstream&);
	
	//++ Function to Recursively Write Iteration data ++++++++++++++++++++++++++++++++++++++//
	void rec_write_iteration_out(int, int, ofstream&);
	//======================================================================================//
	void rec_write_iteration_dat(int, int, ofstream&);
	//======================================================================================//
	void rec_write_iteration_long_dat(int, int, ofstream&);
	//======================================================================================//
	void writeLOSolution(std::string);
	
	void writeSpatialGrid(ofstream&);
};


/*
Triplet::Triplet() {
  column = 0;
  row = 0;
  value = 0.0;
}

Triplet::Triplet(int i, int j, double input) {
  column = i;
  row = j;
  value = input;
}
*/
#endif // LOMLNDA_H


