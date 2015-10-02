#ifndef HOMLNDA_H // Include guard
#define HOMLNDA_H

#include <vector>
#include <time.h>  /* clock_t, clock, CLOCKS_PER_SEC */

using namespace std;


//======================================================================================//
//++ Cross Section Class ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
class HOXS {
	private:
	// Methods
	//++ Find Material Index ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
	int materialIndex(std::vector<int>&, std::vector<int>&, std::vector<int>&, int, int, std::vector< std::vector<double> >&);
	//++ Read XS Values ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
	int XSinput(int, int);
	//======================================================================================//
	public:
	// Variables
	bool write_xs;
	vector< vector<int> > mIndex, material; // material #
	vector< vector<double> > sigma_gT, sigma_gF, nu_gF, nuSigma_gF, chi_g, s_gext, D_g; // material # arrays
	vector< vector< vector<double> > > sigma_gS; // material # arrays
	// variables
	// Solution Grid
	int Ng, Nx, Ny, Sn;
	std::vector<double> x, y, xe, ye, hx, hy;                                 // Solution Grid
	
	vector<int> matnum; // material #
	vector<string> xsname; // material name
	// methods
	
	inline double Diffusion(int g, int i, int j) { return D_g[mIndex[i][j]][g]; };
	
	inline double Total(int g, int i, int j) { return sigma_gT[mIndex[i][j]][g]; };
	inline double Fission(int g, int i, int j) { return nuSigma_gF[mIndex[i][j]][g]; };
	inline double Chi(int g, int i, int j) { return chi_g[mIndex[i][j]][g]; };
	inline double Source(int g, int i, int j) { return s_gext[mIndex[i][j]][g]; };
	inline double Scatter(int g, int gg, int i, int j) { return sigma_gS[mIndex[i][j]][g][gg]; };
	
	vector< vector<double> > Total(int);
	vector< vector<double> > Fission(int);
	vector< vector<double> > Chi(int);
	vector< vector<double> > Source(int);
	vector< vector<double> > Scatter(int, int);
	vector< vector<double> > Diffusion(int);
	//======================================================================================//
	//++ Find group average cross sections +++++++++++++++++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	int readHOXS(std::vector<std::string> &, std::vector<double> &, std::vector<double> &, int);
	void writeHOXS(ofstream&);
	void writeMatFile(string);
	
	void writeHOXSFile(std::string);
	void writeMaterial(std::ofstream&);
};
//======================================================================================//



//======================================================================================//
//+++++++++++++++++++++++++++++++ High Order Factors +++++++++++++++++++++++++++++++++++//
//======================================================================================//
class HOSolution : public HOXS {
	public:
	bool writeOutput;
	bool NDASolution;
	double pi;
	vector< vector< vector<double> > > phi, phi_x, phi_y, j_x, j_y;      // scalar flux and current from transport
	vector< vector< vector< vector<double> > > > psiL, psiR, psiB, psiT;
	//++ NDA Factors +++++++++++++++++++++++++++++++++++++++++++++++++++++++
	std::vector< std::vector< std::vector<double> > > D_x, D_y, D_xT, D_yT;  // Consistency terms
	//++ QD Factors ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	std::vector< std::vector< std::vector<double> > > E_xx, E_yy;  // Cell Average Eddington Factors
	std::vector< std::vector< std::vector<double> > > E_xx_x, E_yy_x, E_xy_x;  // Consistency terms
	std::vector< std::vector< std::vector<double> > > E_xx_y, E_yy_y, E_xy_y;  // Consistency terms
	//++ Default Constructor ++++++++++++++++++++++++++++++++
	HOSolution() {
		writeOutput=true;
		NDASolution=true;
		pi=3.14159265358979323846;
	}
	//++ Initialize values ++++++++++++++++++++++++++++++++++
	void initializeSolution();
	//++ Zero Factors before each transport iteration
	void zeroSolution();
	//++ Tally Consistency Factors +++++++++++++++++++++++++++
	void tallySolution(int, std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, std::vector< std::vector<double> > &, double, double, double);
	//++ Calculate Factors
	void calculateFactors();
	//++ Write Factors to ".ho.csv" file
	void writeSolutionDat(std::string);
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++
};


class HOSolver : public HOSolution {
	private:
	// Variables
	bool o_angular; // option variables
	bool boundary_corrections;
	int N;
	vector<double> mu, eta, xi, w;   // quadrature
	// Methods
	//======================================================================================//
	//++ Function to solve transport in a single general cell ++++++++++++++++++++++++++++++//
	//======================================================================================//
	void cellSolution(double, double, double, double, int, int, int, double&, double&, double&);
	//======================================================================================//
	//======================================================================================//
	//++ sweep through angles and cells in each angular quadrant +++++++++++++++++++++++++++//
	//======================================================================================//
	void quad1(int, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &);
	//======================================================================================//
	void quad2(int, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &);
	//======================================================================================//
	void quad3(int, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &);
	//======================================================================================//
	void quad4(int, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &, vector< vector<double> > &);
	//======================================================================================//
	
	public:
	// Classes
	// variables
	std::vector< double > epsilon_phi, epsilon_keff; // Convergence Critera
	int stop_phi;
	int n_iterations;
	std::vector< std::vector<double> > bcL, bcR, bcB, bcT;
	// History
	vector<double> rho_phi, rho_phiH, rho_keff, rho_kappa, norm_phi, norm_phiH, norm_keff, norm_kappa, k_keff, dt, norm_inf;
	vector< vector<double> > norm_gphi;
	vector< vector<int> > i_gphi, j_gphi; 
	std::vector<int> i_phi, j_phi, g_phi;
	std::vector<double> bal_res;
	std::vector<int> i_res, j_res, g_res;
	// residual data
	std::vector<double> res_ho;
	std::vector<int> i_ho, j_ho, g_ho;
	clock_t t_ho;
	// Methods
	HOSolver() {
		stop_phi=1000;
		o_angular=false; // option variables
		boundary_corrections=true;
	}
	int initializeHO();
	//======================================================================================//
	//++ function to find the quadrature +++++++++++++++++++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	int quadSet();
	//======================================================================================//
	void reflectiveBCIterations(int, vector< vector< vector<double> > >&, vector< vector<double> >&, vector< vector<double> >&, vector< vector<double> >&, vector< vector<double> >&);
	//======================================================================================//
	void logIteration(double, double, double, double, double, double, double, double, double);
	//======================================================================================//
	double residual0(std::vector< std::vector< std::vector<double> > > &, int &, int &, int &);
	double residualIterative(int &, int &, int &);
	//======================================================================================//
	//++++++++++++++++++++++++++++++ High Order Solution ++++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	
	//++ Initial Transport Solution ++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
	void initialAngleSweep();
	//++ Determine how to sweep thought cells ++++++++++++++++++++++++++++++++++++++++++++++//
	void angleSweep(int, vector< vector< vector<double> > >&, vector< vector<double> >&, vector< vector<double> >&, vector< vector<double> >&, vector< vector<double> >&);
	
	void normalizeEigen();
	
	//======================================================================================//
	//++++++++++++++++++++++++++++++ High Order Factors ++++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	
	
	//======================================================================================//
	//++++++++++++++++++++++++++++ Input / Output Methods ++++++++++++++++++++++++++++++++++//
	//======================================================================================//
	
	int readGrid(std::ifstream& infile);
	//++ Read High Order Data
	int readHO(std::vector<std::string>&);
	//++ Write High Order Data
	void writeHO(std::ofstream&);
	//++ Output Quadrature Set
	void outputQuadrature(ofstream&);
	//++ Write Iteration Data
	std::string writeIteration(int);
	//++
	void writeSpatialGrid(ofstream&);
	void writeSpatialGridDat(ofstream&);
	
};

#endif // HOMLNDA_H

