#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <time.h>  /* clock_t, clock, CLOCKS_PER_SEC */
#include <vector>
#include "HO_1.6.h"
#include "LO_1.6.h"
#include "IO_1.6.h"
#include <stdlib.h>
#include <omp.h>
using namespace std;

/*
  2D Transport Solution With Step Characteristics and Non-linear Diffusion Acceleration
  
  Programer : Luke Cornejo
  Version   : 2.5
  Date      : 11-16-14
  
                         Changes
  ******************************************************
  Used Seldon Linear Algebra package
  
  To complile
  Windows
c++ -I .\eigen SC_MLNDA2.6.cpp ioMLNDA2.6.cpp loMLNDA2.6.cpp hoMLNDA2.6.cpp classMLNDA2.6.cpp -o SC_MLNDA2.6.exe -std=c++0x
  Linux
Seldon
c++ -I ./seldon-5.2 -O3 SC_MLNDA2.6.cpp ioMLNDA2.6.cpp loMLNDA2.6s.cpp hoMLNDA2.6.cpp classMLNDA2.6.cpp -o SC_MLNDA2.6s.exe
c++ -I ./seldon-5.2 -fopenmp -O3 SC_MLNDA2.6.cpp ioMLNDA2.6.cpp loMLNDA2.6s.cpp hoMLNDA2.6.cpp classMLNDA2.6.cpp -o SC_MLNDA2.6s.exe
Eigen
c++ -I ./eigen -O3 SC_MLNDA2.7.cpp ioMLNDA2.7.cpp hoMLNDA2.7.cpp xsMLNDA2.7.cpp loMLNDA2.7main.cpp loMLNDA2.7e.cpp -o SC_MLNDA2.7e.exe
c++ -I ./eigen -fopenmp -O3 SC_MLNDA2.7.cpp ioMLNDA2.7.cpp hoMLNDA2.7.cpp xsMLNDA2.7.cpp loMLNDA2.7main.cpp loMLNDA2.7e.cpp -o SC_MLNDA2.7e.exe
  
  hpc
-O
-O2
-O3 better
-Os
-O3 -ffast-math


-O1
-O2
-O3 best
-xO
-fast

  Boundary type options
  1: incoming according to side
  2: incoming according to angle
  3: reflective on all sides
  11 or 4: reflective on Left and Bottom, incoming on Right and Top according to side
  12     : reflective on Right and Bottom, incoming on Left and Top according to side
  13 or 5: reflective on Right and Top, incoming on Left and Bottom according to side
  14     : reflective on Left and Top, incoming on Right and Bottom according to side
  21 : Reflective on Left
  22 : Reflective on Bottom
  23 : Reflective on Right
  24 : Reflective on Top
  31 : Vacuum on Right , Reflective on Left, Top and Bottom
  32 : Vacuum on Top   , Reflective on Bottom, Left and Right
  33 : Vacuum on Left  , Reflective on Right, Top and Bottom
  34 : Vacuum on Bottom, Reflective on Top, Left and Right
  
  BC input order
  according to side : Left, Bottom, Right, Top
  according to angle: quad 1, quad 2, quad 3, quad 4
  
  Quadratures
  S4, S6, S8, S12, S16
  Q20, Q36
  

*/

string test_name;

bool KE_problem=false; // option variables
int    kbc; // Kind of BC
bool reflectiveB, reflectiveT, reflectiveL, reflectiveR;
vector< vector<double> > source, loss;
// Solution
double k_eff=1.0; // K-effective
ofstream temfile;

HOSolver ho;
LOSolver lo;

inline double LInfNorm(vector< vector<double> > &phi, double norm_p) {
	int nx=phi.size(), ny=phi[0].size();
	//# pragma omp parallel for
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			if ( abs(phi[i][j])>norm_p ) {
				norm_p=abs(phi[i][j]);
			}
		}
	}
	return norm_p;
}
// Find the difference norm between two arrays
inline double diffNorm(vector< vector<double> > &phi1, vector< vector<double> > &phi2, double norm_p, int &im, int &jm) {
	int nx=phi1.size(), ny=phi1[0].size();
	//# pragma omp parallel for
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			if ( abs(phi1[i][j]-phi2[i][j])>norm_p ) {
				norm_p=abs(phi1[i][j]-phi2[i][j]);
				im=i;
				jm=j;
			}
		}
	}
	return norm_p;
}
// Find the difference norm between two arrays
inline double diffL2Norm(vector< vector<double> > &phi1, vector< vector<double> > &phi2, double norm_p) {
	int nx=phi1.size(), ny=phi1[0].size();
	norm_p=norm_p*norm_p;
	//# pragma omp parallel for
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			norm_p+=(phi1[i][j]-phi2[i][j])*(phi1[i][j]-phi2[i][j]);
		}
	}
	return sqrt(norm_p);
}
// Set one array equal to another
inline void setArrayEqual(vector< vector<double> > &phi1, vector< vector<double> > &phi2) {
	//# pragma omp parallel for shared(phi1)
	for (int i=0; i<lo.Nx; i++) {
		for (int j=0; j<lo.Ny; j++) {
			phi1[i][j]=phi2[i][j];
		}
	}
}
// Find Minimum of two doubles
inline double min(double a, double b) {
	return (a < b) ? a : b;
}

int initialize() {
	try {
		int quad_err=ho.quadSet(); // find quadrature 
		if ( ho.quadSet()!=0 )        throw "Fatal Error In Initalization Function 'ho.quadSet()' \n";
		if ( ho.initializeHO()!=0 )   throw "Fatal Error In Initalization Function 'ho.initializeHO()' \n";
		if ( lo.initializeLO(ho)!=0 ) throw "Fatal Error In Initalization Function 'lo.initializeLO(ho)' \n";
	}
	catch(std::string error) {
		cout<<"Initialization Error >> "<<error;
		return 1;
	}
	
	
	int Nx=ho.Nx, Ny=ho.Ny;
	
	source.resize(Nx);
	loss.resize(Nx);
	for (int i=0; i<Nx; i++) {
		source[i].resize(Ny);
		loss[i].resize(Ny);
	}
	
	return 0;
}


//======================================================================================//
//++ Recursive function to solve One Group +++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void oneGroupSolFS(int s) {
	int etaL, g=0; 
	int Nx=lo.Nx, Ny=lo.Ny;
	etaL=lo.eta_star-1;
	double res;
	int im, jm, gm;
	vector< vector<double> > &phiG=lo.phi[etaL][g], &phi_gL=lo.phiLast[etaL][g], &j_xG=lo.j_x[etaL][g], &j_yG=lo.j_y[etaL][g];
	vector< vector<double> > &sigmaTG=lo.sigmaT[etaL][g], &sigmaSG=lo.sigmaS[etaL][g][g], &nuSigmaFG=lo.nuSigmaF[etaL][g], &s_extG=lo.s_ext[etaL][g];
	vector< vector<double> > &kappaL=lo.kappaLast[etaL]; // One-group flux and scalar flux from last iteration
	
	double temp=lo.findNormKappa(kappaL);
	
	lo.res_mbal[etaL]=0.0;
	
	cout<<string((etaL+1)*5,' ')<<"Solve Low-order One-group Eq. \n";
	// Solve NDA on each energy group
	// create source and loss matrices
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) loss[i][j]=(sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/k_eff);
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) source[i][j]=s_extG[i][j];
	
	//write_grid_average_dat(phi, 16, temfile);
	lo.fixedSourceSolution(lo.preconditioner_type[etaL], etaL, g, source, loss); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//write_grid_average_dat(phi, 16, temfile);
	
	double norm_p=diffNorm(phiG, phi_gL, 0.0, im, jm);
	
	double norm_kap=lo.findNormKappa(kappaL);
	
	// Calculate Equation Residuals
	lo.res_bal[etaL]=0.0;
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) { // Balance in Cell
			res=abs((lo.j_x[etaL][g][i+1][j]-lo.j_x[etaL][g][i][j])/lo.hx[i]+(lo.j_y[etaL][g][i][j+1]-lo.j_y[etaL][g][i][j])/lo.hy[j]+
			loss[i][j]*phiG[i][j]-source[i][j]);
			if ( res>lo.res_bal[etaL] ) { lo.res_bal[etaL]=res; lo.i_bal[etaL]=i; lo.j_bal[etaL]=j;  lo.g_bal[etaL]=g; }
		}
	}
	
	lo.num_grid[etaL].back()++;
	lo.num_losi[etaL].push_back(1);
	
	lo.logIteration(etaL, norm_p, 0.0, norm_p, 0.0, k_eff, 0.0, 0.0, norm_kap, 0.0);
	
}
//======================================================================================//

//++ Weilalndt Shift Iteration
void WSIteration() {
	int etaL, g=0;
	int Nx=lo.Nx, Ny=lo.Ny;
	etaL=lo.eta_star-1; 
	vector< vector<double> > &phiG=lo.phi[etaL][g], &j_xG=lo.j_x[etaL][g], &j_yG=lo.j_y[etaL][g];
	vector< vector<double> > &sigmaTG=lo.sigmaT[etaL][g], &sigmaSG=lo.sigmaS[etaL][g][g], &nuSigmaFG=lo.nuSigmaF[etaL][g];
	// create source and loss matrices
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) {
			double sigmaWS=(1.0-lo.delta)*min(sigmaTG[i][j]-sigmaSG[i][j],nuSigmaFG[i][j]/k_eff);
			loss[i][j]=sigmaTG[i][j]-sigmaSG[i][j]-sigmaWS;
			source[i][j]=(nuSigmaFG[i][j]/k_eff-sigmaWS)*phiG[i][j];
		}
	}
	
	// Solve NDA on each energy group
	lo.fixedSourceSolution(lo.preconditioner_type[etaL], etaL, g, source, loss); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	
	if ( lo.solveK ) k_eff=lo.greyEigenvalue();
	
	// Calculate Equation Residuals
	lo.res_bal[etaL]=0.0;
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) { // Balance in Cell
			double res=abs((j_xG[i+1][j]-j_xG[i][j])/lo.hx[i]+(j_yG[i][j+1]-j_yG[i][j])/lo.hy[j]+loss[i][j]*phiG[i][j]-source[i][j]);
			if ( res>lo.res_bal[etaL] ) { lo.res_bal[etaL]=res; lo.i_bal[etaL]=i; lo.j_bal[etaL]=j;  lo.g_bal[etaL]=g; }
		}
	}
	
	// Normalize the solution to 1
	lo.normalizeEigen(etaL);
}

//======================================================================================//
//++ Recursive function to solve One Group +++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void oneGroupSolKE(int s) {
	int etaL, g=0;
	int Nx=lo.Nx, Ny=lo.Ny;
	etaL=lo.eta_star-1; 
	int im, jm, gm;
	bool phiConverged=false, kConverged=false;
	double rho_p, rho_k, rho_kap, norm_pL, norm_kL, norm_kapL, k_effL, res;
	vector< vector<double> > &phiG=lo.phi[etaL][g];
	vector< vector<double> > &kappaL=lo.kappaLast[etaL]; // One-group flux and scalar flux from last iteration
	
	double temp=lo.findNormKappa(kappaL);
	
	
	vector< vector<double> > phi_gL=phiG;
	
	cout<<string((etaL+1)*5,' ')<<"Solve Low-order One-group Eq. \n";
	double norm_p=lo.norm_phi[etaL].back();
	double norm_k=lo.norm_keff[etaL].back();
	double norm_kap=lo.norm_kappa[etaL].back();
	int l=0;
	do { // Do While Loop
		k_effL=k_eff;
		setArrayEqual(phi_gL, phiG);
		norm_pL=norm_p;
		norm_kL=norm_k;
		norm_kapL=norm_kap;
		
		WSIteration();
		
		norm_k=abs(k_eff-k_effL);
		norm_p=diffNorm(phiG, phi_gL, 0.0, im, jm);
		norm_kap=lo.findNormKappa(kappaL);
		rho_p=norm_p/norm_pL;
		rho_k=norm_k/norm_kL;
		
		lo.logIteration(etaL, norm_p, rho_p, norm_p, rho_p, k_eff, norm_k, rho_k, norm_kap, norm_kap/norm_kapL);
		
		l++;
		lo.num_grid[etaL].back()++;
		// Check Flux Convergence
		double max_p=0.0;
		max_p=LInfNorm(phiG, 0.0);
		if ( lo.epsilon_phi[etaL].size()==2 ) phiConverged=( norm_p      <=lo.epsilon_phi[etaL][0]*max_p+lo.epsilon_phi[etaL][1] );
		else if ( lo.relativeConvergence )    phiConverged=( norm_p/max_p<=lo.epsilon_phi[etaL][0]*(1/rho_p-1) );
		else                                  phiConverged=( norm_p      <=lo.epsilon_phi[etaL][0]*(1/rho_p-1) );
		// Check K Convergence
		if ( lo.solveK ) {
			if ( lo.epsilon_phi[etaL].size()==2 ) kConverged=( norm_k<=lo.epsilon_keff[etaL][0]*k_eff+lo.epsilon_keff[etaL][1] );
			else                                  kConverged=( norm_k<=lo.epsilon_keff[etaL][0]*(1/rho_k-1) );
		}
		else             kConverged=true;
	} while ( ( not phiConverged or not kConverged ) and l<lo.stop_phi[etaL] );
	
	
	lo.num_losi[etaL].push_back(l);
}
//======================================================================================//

//======================================================================================//
//++ Recursive function to solve One Group +++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void oneGroupNewton(int ll, int s) {
	int etaL, g=0;
	int Nx=lo.Nx, Ny=lo.Ny;
	int im, jm, gm;
	etaL=lo.eta_star-1; 
	double deltaN=1e-3;
	double rho_p, rho_k, rho_kap, norm_pL, norm_kL, norm_kapL, k_effL, res;
	vector< vector<double> > &phiG=lo.phi[etaL][g], &phi_gL=lo.phiLast[etaL][g], &j_xG=lo.j_x[etaL][g], &j_yG=lo.j_y[etaL][g];
	vector< vector<double> > &sigmaTG=lo.sigmaT[etaL][g], &sigmaSG=lo.sigmaS[etaL][g][g], &nuSigmaFG=lo.nuSigmaF[etaL][g];
	
	int l=0;
	if ( lo.num_grid[etaL][0]==0 ) {
		
		lo.collapseSolution(etaL);
		
		if ( lo.solveK ) k_eff=lo.greyEigenvalue();
		lo.normalizeEigen(etaL);
		k_effL=k_eff;
		setArrayEqual(phi_gL, phiG);
		
		WSIteration();
		
		cout<<"diff "<<diffNorm(phiG, phi_gL, 0.0, im, jm)<<endl;
		
	}
	// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
	lo.calculateGridCurrents(etaL);
	
	//write_cell_dat(j_xG, lo.x, lo.y, 16, temfile);
	
	vector< vector<double> > &kappaL=lo.kappaLast[etaL]; // One-group flux and scalar flux from last iteration
	
	double temp=lo.findNormKappa(kappaL);
	
	// Find Infinity Norm of Newton Residual
	double res_norm;
	double res_initial=lo.NewtonLNorm(0);
	//cout<<" res_initial "<<res_initial<<endl;
	lo.resInitialNewton.push_back(res_initial);
	
	bool newtonWorked=true;
	cout<<string((etaL+1)*5,' ')<<"Solve Low-order One-group Eq. \n";
	double norm_p=lo.norm_phi[etaL].back();
	double norm_k=lo.norm_keff[etaL].back();
	double norm_kap=lo.norm_kappa[etaL].back();
	bool converged=false;
	do { // Do While Loop
		k_effL=k_eff;
		setArrayEqual(phi_gL, phiG);
		norm_pL=norm_p;
		norm_kL=norm_k;
		norm_kapL=norm_kap;
		//cout<<"l "<<l<<endl;
		
		//if ( s==0 and l>2 ) newtonWorked=false;
		if ( newtonWorked ) {
			// Perform Newton Iteration
			newtonWorked=lo.greyNewtonSolution(lo.preconditioner_type[etaL], phi_gL); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			// Find Infinity Norm of Newton Residual
			res_norm=lo.NewtonLNorm(0);
			
			// Calculate Equation Residuals
			lo.res_bal[etaL]=0.0;
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) { // Balance in Cell
					res=abs((j_xG[i+1][j]-j_xG[i][j])/lo.hx[i]+(j_yG[i][j+1]-j_yG[i][j])/lo.hy[j]+(sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/k_eff)*phiG[i][j]);
					if ( res>lo.res_bal[etaL] ) { lo.res_bal[etaL]=res; lo.i_bal[etaL]=i; lo.j_bal[etaL]=j;  lo.g_bal[etaL]=g; }
				}
			}
			//cout<<"res "<<res_bal[etaL]<<endl;
			converged=( res_norm<=lo.epsilon_phi[etaL][0]*res_initial+lo.epsilon_phi[etaL][1] );
		}
		
		//write_cell_dat(phiG, 16, temfile);
		
		norm_k=abs(k_eff-k_effL);
		norm_p=diffNorm(phiG, phi_gL, 0.0, im, jm);
		norm_kap=lo.findNormKappa(kappaL);
		rho_p=norm_p/norm_pL;
		rho_k=norm_k/norm_kL;
		
		lo.logIteration(etaL, norm_p, rho_p, norm_p, rho_p, k_eff, norm_k, rho_k, norm_kap, norm_kap/norm_kapL);
		
		//cout<<" res_norm "<<res_norm<<endl;
		lo.resNewton.push_back(res_norm);
		
		if ( not newtonWorked ) converged=true;
		
		//write_cell_dat(phiG, 16, temfile);
		
		l++;
		lo.num_grid[etaL].back()++;
	} while ( not converged and l<lo.stop_phi[etaL] );
	//} while ( ( norm_p>lo.epsilon_phi[etaL]*(1/rho_p-1) or norm_k>lo.epsilon_keff[etaL]*(1/rho_k-1) ) and l<lo.stop_phi[etaL] );
	
	if ( !lo.greySolutionPositive() ) cout<<"Negative Solution \n";
	
	lo.num_losi[etaL].push_back(l);
}
//======================================================================================//

void sweepGaussSeidel(int etaL, vector< vector<double> > &S_f, double &norm_p) {
	//cout<<"Solve Multigroup\n";
	int Ng=lo.Ng[etaL], Nx=lo.Nx, Ny=lo.Ny;
	int im, jm, gm;
	double res, tempSource;
	vector< vector< vector<double> > > &phiG=lo.phi[etaL], &phi_gL=lo.phiLast[etaL];
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) {
			S_f[i][j]=0.0; for (int g=0; g<Ng; g++) S_f[i][j]+=lo.Fission(etaL,g,i,j)*phiG[g][i][j];
		}
	}
	
	norm_p=0.0;
	lo.res_bal[etaL]=0.0;
	
	for (int g=0; g<Ng; g++) setArrayEqual(phi_gL[g], phiG[g]); // Set phi_gL to previous iteration
	
	for (int g=0; g<lo.Ndown[etaL]; g++) {
		
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				source[i][j]=(lo.Chi(etaL,g,i,j)*S_f[i][j]/k_eff+lo.Source(etaL,g,i,j));
				for (int gg=g+1; gg<Ng; gg++) source[i][j]+=lo.Scatter(etaL,gg,g,i,j)*phiG[gg][i][j];
				for (int gg=0; gg<g; gg++)    source[i][j]+=lo.Scatter(etaL,gg,g,i,j)*phiG[gg][i][j];
			}
		}
		//temfile<<"source \n";
		//write_cell_dat(source, 16, temfile);
		for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) loss[i][j]=(lo.Total(etaL,g,i,j)-lo.Scatter(etaL,g,g,i,j));
		
		// Solve NDA on each energy group
		lo.fixedSourceSolution(lo.preconditioner_type[etaL], etaL, g, source, loss); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		
		
		// Calculate Equation Residuals
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) { // Residual Balance in Cell
				double res=abs((lo.j_x[etaL][g][i+1][j]-lo.j_x[etaL][g][i][j])/lo.hx[i]+(lo.j_y[etaL][g][i][j+1]-lo.j_y[etaL][g][i][j])/lo.hy[j]+
				loss[i][j]*phiG[g][i][j]-source[i][j]);
				if ( res>lo.res_bal[etaL] ) { lo.res_bal[etaL]=res; lo.i_bal[etaL]=i; lo.j_bal[etaL]=j; lo.g_bal[etaL]=g; }
			}
		}
		
	}
	int t=0;
	double relative_residual=1.0;
	while ( t < lo.relaxation[etaL] and relative_residual > lo.epsilon_phi[etaL][0] ) {
		relative_residual=0.0;
		for (int g=lo.Ndown[etaL]; g<Ng; g++) {
			
			//# pragma omp parallel for
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					source[i][j]=(lo.Chi(etaL,g,i,j)*S_f[i][j]/k_eff+lo.Source(etaL,g,i,j));
					for (int gg=g+1; gg<Ng; gg++) source[i][j]+=lo.Scatter(etaL,gg,g,i,j)*phiG[gg][i][j];
					for (int gg=0; gg<g; gg++)    source[i][j]+=lo.Scatter(etaL,gg,g,i,j)*phiG[gg][i][j];
				}
			}
			//temfile<<"source \n";
			//write_cell_dat(source, 16, temfile);
			for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) loss[i][j]=(lo.Total(etaL,g,i,j)-lo.Scatter(etaL,g,g,i,j));
			
			
			
			// Solve NDA on each energy group
			lo.fixedSourceSolution(lo.preconditioner_type[etaL], etaL, g, source, loss); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			
			
			
			
			// Calculate Equation Residuals
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) { // Residual Balance in Cell
					double res=abs((lo.j_x[etaL][g][i+1][j]-lo.j_x[etaL][g][i][j])/lo.hx[i]+(lo.j_y[etaL][g][i][j+1]-lo.j_y[etaL][g][i][j])/lo.hy[j]+
					loss[i][j]*phiG[g][i][j]-source[i][j]);
					if ( res>lo.res_bal[etaL] ) { lo.res_bal[etaL]=res; lo.i_bal[etaL]=i; lo.j_bal[etaL]=j; lo.g_bal[etaL]=g; }
				}
			}
			// Calculate Jacobi Residual
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					tempSource=(lo.Chi(etaL,g,i,j)*S_f[i][j]+lo.Source(etaL,g,i,j));
					for (int gg=0; gg<Ng; gg++) tempSource+=lo.Scatter(etaL,gg,g,i,j)*phiG[gg][i][j];
					res=abs((lo.j_x[etaL][g][i+1][j]-lo.j_x[etaL][g][i][j])/lo.hx[i]+(lo.j_y[etaL][g][i][j+1]-lo.j_y[etaL][g][i][j])/lo.hy[j]+
					lo.Total(etaL,g,i,j)*phiG[g][i][j]-tempSource)/phiG[g][i][j];
					if ( res > relative_residual ) relative_residual=res;
				}
			}
		}
		t++;
	}
	norm_p=0.0;
	for (int g=0; g<Ng; g++) norm_p=diffNorm(phiG[g], phi_gL[g], norm_p, im, jm);
}

void sweepJacobi(int etaL, vector< vector<double> > &S_f, double &norm_p) {
	//cout<<"Solve Multigroup\n";
	int Ng=lo.Ng[etaL], Nx=lo.Nx, Ny=lo.Ny;
	int im, jm, gm;
	double res, tempSource;
	vector< vector< vector<double> > > &phiG=lo.phi[etaL], &phi_gL=lo.phiLast[etaL];
	vector< vector<double> > &S_s=lo.sourceScatter[etaL];
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) {
			S_f[i][j]=0.0; for (int g=0; g<Ng; g++) S_f[i][j]+=lo.Fission(etaL,g,i,j)*phiG[g][i][j]/k_eff;
		}
	}
	
	lo.res_bal[etaL]=0.0;
	
	for (int g=0; g<Ng; g++) setArrayEqual(phi_gL[g], phiG[g]); // Set phi_gL to previous iteration
	
	int t=0;
	double relative_residual=1.0;
	while ( t < lo.relaxation[etaL] and relative_residual > lo.epsilon_phi[etaL][0] ) {
		relative_residual=0.0;
		for (int g=0; g<Ng; g++) {
			
			//# pragma omp parallel for
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					source[i][j]=(lo.Chi(etaL,g,i,j)*S_f[i][j]+lo.Source(etaL,g,i,j));
					for (int gg=0; gg<g; gg++)    source[i][j]+=lo.Scatter(etaL,gg,g,i,j)*phi_gL[gg][i][j];
					for (int gg=g+1; gg<Ng; gg++) source[i][j]+=lo.Scatter(etaL,gg,g,i,j)*phi_gL[gg][i][j];
				}
			}
			
			for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) loss[i][j]=(lo.Total(etaL,g,i,j)-lo.Scatter(etaL,g,g,i,j));
			
			// Solve NDA on each energy group
			lo.fixedSourceSolution(lo.preconditioner_type[etaL], etaL, g, source, loss); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			
			// Calculate Equation Residuals
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) { // Residual Balance in Cell
					res=abs((lo.j_x[etaL][g][i+1][j]-lo.j_x[etaL][g][i][j])/lo.hx[i]+(lo.j_y[etaL][g][i][j+1]-lo.j_y[etaL][g][i][j])/lo.hy[j]+
					loss[i][j]*phiG[g][i][j]-source[i][j]);
					if ( res>lo.res_bal[etaL] ) { lo.res_bal[etaL]=res; lo.i_bal[etaL]=i; lo.j_bal[etaL]=j; lo.g_bal[etaL]=g; }
				}
			}
			// Calculate Jacobi Residual
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					tempSource=(lo.Chi(etaL,g,i,j)*S_f[i][j]+lo.Source(etaL,g,i,j));
					for (int gg=0; gg<Ng; gg++) tempSource+=lo.Scatter(etaL,gg,g,i,j)*phiG[gg][i][j];
					res=abs((lo.j_x[etaL][g][i+1][j]-lo.j_x[etaL][g][i][j])/lo.hx[i]+(lo.j_y[etaL][g][i][j+1]-lo.j_y[etaL][g][i][j])/lo.hy[j]+
					lo.Total(etaL,g,i,j)*phiG[g][i][j]-tempSource)/phiG[g][i][j];
					if ( res > relative_residual ) relative_residual=res;
				}
			}
		}
		t++;
	}
	norm_p=0.0;
	for (int g=0; g<Ng; g++) norm_p=diffNorm(phiG[g], phi_gL[g], norm_p, im, jm);
}

//======================================================================================//
//++ Recursive function to solve Coarse Grid +++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void coarseGridSol(int etaL, int s) {
	int Nx=lo.Nx, Ny=lo.Ny;
	bool phiConverged=false, kConverged=false;
	int im, jm, gm;
	double rho_p, rho_k, k_effL;
	double norm_pL, norm_pHL, norm_kL, norm_kapL;
	vector< vector<double> > phiHL=lo.phi[lo.eta_star-1][0];

	//vector< vector<double> > phi_gL(Nx,vector<double>(Ny, 0.0));
	vector< vector<double> > &S_f=lo.sourceFission[etaL];
	
	vector< vector<double> > &kappaL=lo.kappaLast[etaL]; // One-group flux and scalar flux from last iteration
	
	if (lo.eta_star!=1) lo.findNormKappa(kappaL);
	
	lo.res_mbal[etaL]=0.0;
	
	if ( etaL!=0 and lo.num_grid[etaL][0]==0 ) {
		lo.collapseSolution(etaL);
	}
	
	cout<<string((etaL+1)*5,' ')<<"Solve Low-order Eq. on Grid # "<<etaL<<endl;
	double norm_p=lo.norm_phi[etaL].back();
	double norm_pH=lo.norm_phiH[etaL].back();
	double norm_k=lo.norm_keff[etaL].back();
	double norm_kap=lo.norm_kappa[etaL].back();
	if ( lo.eta_star!=1 and lo.wSolve[etaL] and ( lo.norm_phiH[etaL].size()>1 ) ) {
		norm_pH=lo.norm_phiH[etaL][lo.norm_phiH[etaL].size()-2];
		norm_k=lo.norm_keff[etaL][lo.norm_keff[etaL].size()-2];
		norm_kap=lo.norm_kappa[etaL][lo.norm_kappa[etaL].size()-2];
	}
	int l=0;
	do { // Do While Loop
		
		if (lo.eta_star!=1) for (int k=lo.eta_star-2; k>=etaL; k--) lo.correctFlux(k,lo.phi[k],lo.phi[k+1]);
		
		temfile<<" -- LO Grid "<<etaL<<" Iteration # "<<l<<" -- "<<endl;
		
		//write_cell_dat(S_f, 16, temfile);
		if (lo.eta_star!=1) setArrayEqual(phiHL, lo.phi[lo.eta_star-1][0]); // Set grey solution to last grey solution
		k_effL=k_eff;
		norm_pL=norm_p;
		norm_pHL=norm_pH;
		norm_kL=norm_k;
		norm_kapL=norm_kap;
		
		//cout<<"GS sweep\n";
		
		if ( lo.gaussSeidel ) sweepGaussSeidel(etaL, S_f, norm_p);
		else sweepJacobi(etaL, S_f, norm_p);
		//cout<<"norm_p "<<norm_p<<endl;
		
		//cout<<"GS sweep done\n";
		if (lo.eta_star==1) {
			if ( lo.solveK ) {
				cout<<string((etaL+2)*5,' ')<<"Multigroup Balace to find K=";
				k_eff=lo.multigroupEigenvalue(etaL); // Find
				lo.normalizeEigen(etaL);
				cout<<k_eff<<"\n";
			}
			norm_pH=0.0/0.0; // Find difference norm for grey solution
			norm_kap=0.0/0.0;
		}
		else {
			// Calculate Group Averaged values ////////////////////////////////////////////////////////////
			lo.averageFactors(etaL); // Calculate Group Averaged values ////////////////////////////////////////////////////////////
			lo.averageXS(etaL); // Calculate Group Averaged values ////////////////////////////////////////////////////////////
			// Go to the next coarser grid
			if (etaL==lo.eta_star-2) {
				if ( KE_problem ) {
					if ( lo.greyNewton and lo.solveK )    oneGroupNewton(l, s);
					else                         oneGroupSolKE(s);
				}
				else oneGroupSolFS(s);
			}
			else coarseGridSol(etaL+1,s);
			
			norm_pH=diffNorm(lo.phi[lo.eta_star-1][0], phiHL, 0.0, im, jm); // Find difference norm for grey solution
			norm_kap=lo.findNormKappa(kappaL);
		}
		// Calculate iteration data
		norm_k=abs(k_eff-k_effL);
		rho_p=norm_p/norm_pL;
		rho_k=norm_k/norm_kL;
		
		lo.logIteration(etaL, norm_p, rho_p, norm_pH, norm_pH/norm_pHL, k_eff, norm_k, rho_k, norm_kap, norm_kap/norm_kapL);
		
		l++; // increment for next loop
		lo.num_grid[etaL].back()++;
		
		double max_p=0.0;
		for (int g=0; g<lo.Ng[etaL]; g++) max_p=LInfNorm(lo.phi[etaL][g], max_p);
		// Check Phi Convergence
		if ( lo.epsilon_phi[etaL].size()==2 ) phiConverged=( norm_p      <=lo.epsilon_phi[etaL][0]*max_p+lo.epsilon_phi[etaL][1] );
		else if ( lo.relativeConvergence )    phiConverged=( norm_p/max_p<=lo.epsilon_phi[etaL][0]*(1/rho_p-1) );
		else                                  phiConverged=( norm_p      <=lo.epsilon_phi[etaL][0]*(1/rho_p-1) );
		// Check K Convergence
		if ( lo.solveK ) {
			if ( lo.epsilon_keff[etaL].size()==2 ) kConverged=( norm_k<=lo.epsilon_keff[etaL][0]*k_eff+lo.epsilon_keff[etaL][1] );
			else                                  kConverged=( norm_k<=lo.epsilon_keff[etaL][0]*(1/rho_k-1) );
		}
		else kConverged=true;
	} while ( ( not phiConverged or not kConverged ) and l<lo.stop_phi[etaL] );
	
	if ( lo.eta_star!=1 and lo.wSolve[etaL] and ( not phiConverged or not kConverged ) ) {
		for (int k=lo.eta_star-2; k>=etaL; k--) lo.correctFlux(k,lo.phi[k],lo.phi[k+1]); // calculate correction factors and update fluxes
		norm_pL=norm_p;
		if ( lo.gaussSeidel ) sweepGaussSeidel(etaL, S_f, norm_p);
		else sweepJacobi(etaL, S_f, norm_p);
		rho_p=norm_p/norm_pL;
		lo.logIteration(etaL, norm_p, rho_p, 1.234e-14, 0.0, k_eff, 1.234e-14, 0.0, 1.234e-14, 0.0);
		l++; // increment for next loop
		lo.num_grid[etaL].back()++;
		lo.num_losi[etaL+1].push_back(0);
	}
	lo.num_losi[etaL].push_back(l);
	
}
//======================================================================================//

//======================================================================================//
//++ Iterate to converge on solution +++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void Iterations() {
	int Ng=ho.Ng, Nx=ho.Nx, Ny=ho.Ny;
	int etaL=0;
	int im, jm;
	bool phiConverged=false, kConverged=false;
	if ( not KE_problem ) kConverged=true;
	if ( lo.initialK!=k_eff ) {
		kConverged=true;
		lo.solveK=false;
		k_eff=lo.initialK;
	}
	double rho_p, rho_k, k_effL;
	double norm_pL, norm_pHL, norm_kL, norm_kapL;
	vector< vector<double> > phiEL, phiER, phiEB, phiET;
	phiEL.resize(Ng);
	phiER.resize(Ng);
	phiEB.resize(Ng);
	phiET.resize(Ng);
	for(int g=0; g<Ng; g++) {
		phiEL[g].resize(Ny);
		phiER[g].resize(Ny);
		phiEB[g].resize(Nx);
		phiET[g].resize(Nx);
	}
	
	vector< vector< vector<double> > > phiL(Ng,vector< vector<double> >(Nx,vector<double>(Ny, 0.0)));
	
	vector< vector<double> > phiHL;
	if (lo.eta_star!=1) phiHL=lo.phi[lo.eta_star-1][0];
	vector< vector<double> > kappaL(Nx,vector<double>(Ny, 1.0)); // One-group flux and scalar flux from last iteration
	
	double norm_p=4.0*3.141592654/Ng, norm_pH=4.0*3.141592654, norm_k=1.0, norm_kap=1.0;
	/*
	vector< vector< vector<double> > > D_xTL, D_yTL;
	/*
	if ( lo.trackFactorConvergence ) {
		D_xTL=ho.F.D_xT; D_yTL=ho.F.D_yT;
	}
	*/
	
	int s=0;
	// begin iterations
	do { //========================================================================
		cout<<" -- Iteration # "<<s<<" -- "<<endl;
		temfile<<" -- Iteration # "<<s<<" -- "<<endl;
		
		norm_pL=norm_p; // set previous norm to normLast
		norm_pHL=norm_pH;
		norm_kL=norm_k;
		norm_kapL=norm_kap;
		k_effL=k_eff;
		clock_t iteration_time=clock();
		// set previous iteration to phiLast
		if (lo.eta_star!=1) setArrayEqual(phiHL, lo.phi[lo.eta_star-1][0]); // Set last grey solution to current grey solution
		for (int g=0; g<Ng; g++) setArrayEqual(phiL[g], ho.phi[g]); // Set last HO solution to current HO solution
		//if ( lo.trackFactorConvergence ) { D_xTL=lo.D_xT; D_yTL=lo.D_yT; }
		
		if ( s!=0 ) {
			if (lo.eta_star!=1) for (int k=lo.eta_star-2; k>=0; k--) lo.correctFlux(k,lo.phi[k],lo.phi[k+1]);   // Correct Flux using correction factors <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			lo.edgeFlux(0, phiEL, phiER, phiEB, phiET);
			ho.angleSweep(s, lo.phi[0], phiEL, phiER, phiEB, phiET);       // perform sweep through angles <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		}
		
		//cout<<"Calculate LO factors from HO factors\n";
		//write_grid_average_dat(phi, 16, temfile);
		// Calculate Low Order Factors from High Order Factors and Solution
		lo.LOFromHOFactors(ho); // Calculate diffusion factors and Boundary conditions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		
		ho.i_ho.push_back(0);
		ho.j_ho.push_back(0);
		ho.g_ho.push_back(0);
		double res_ho=ho.residual0(lo.phi[0], ho.i_ho.back(), ho.j_ho.back(), ho.g_ho.back());
		ho.res_ho.push_back(res_ho);
		
		if ( lo.trackFactorConvergence ) {
			/*
			double normLI=0.0, normL2=0.0;
			for (int g=0; g<Ng; g++) {
				normLI=diffNorm(lo.D_xT[g], D_xTL[g], normLI, im, jm);
				normLI=diffNorm(lo.D_yT[g], D_yTL[g], normLI, im, jm);
				normL2=diffL2Norm(lo.D_xT[g], D_xTL[g], normL2);
				normL2=diffL2Norm(lo.D_yT[g], D_yTL[g], normL2);
			}
			lo.norm_DTLI.push_back(normLI);
			lo.norm_DTL2.push_back(normL2);
			*/
		}
		else {
			lo.norm_DTLI.push_back(0.0/0.0);
			lo.norm_DTL2.push_back(0.0/0.0);
		}
		
		
		//cout<<"Start LO Solution\n";
		lo.num_sol=0;
		for (int k=0; k<lo.eta_star; k++) lo.num_grid[k].push_back(0);
		// Solve NDA on each grid
		clock_t t_lo=clock(); // start low-order timer
		coarseGridSol(0,s); // call MLNDA function for fine group <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		t_lo=clock()-t_lo; // stop low-order timer
		lo.dt[0].push_back(((double)t_lo)/CLOCKS_PER_SEC); // add high-order solution time to vector
		lo.num_mtot.push_back(lo.num_sol); // Total # of times LO matirx was solved
		
		//cout<<"LO Solution done\n";
		
		iteration_time=clock()-iteration_time;
		cout<<"Iteration Completed in "<<double(iteration_time)/CLOCKS_PER_SEC<<" sec\n";
		
		// Iterative Balance Residuals
		ho.i_res.push_back(0);
		ho.j_res.push_back(0);
		ho.g_res.push_back(0);
		res_ho=ho.residual0(lo.phi[0], ho.i_res.back(), ho.j_res.back(), ho.g_res.back());
		ho.bal_res.push_back(res_ho);
		
		// Find new norm
		norm_p=0.0;
		double norm_pp=0.0;
		ho.i_phi.push_back(0);
		ho.j_phi.push_back(0);
		ho.g_phi.push_back(0);
		ho.norm_gphi.push_back( std::vector<double> (Ng, 0.0) );
		ho.i_gphi.push_back( std::vector<int> (Ng, 101010) );
		ho.j_gphi.push_back( std::vector<int> (Ng, 101010) );
		for (int g=0; g<Ng; g++) {
			norm_pp=norm_p;
			ho.norm_gphi.back()[g]=diffNorm(ho.phi[g], phiL[g], ho.norm_gphi.back()[g], ho.i_gphi.back()[g], ho.j_gphi.back()[g]);
			if ( ho.norm_gphi.back()[g]>norm_pp ) {
				norm_p=ho.norm_gphi.back()[g];
				ho.i_phi.back()=ho.i_gphi.back()[g];
				ho.j_phi.back()=ho.j_gphi.back()[g];
				ho.g_phi.back()=g;
			}
		}
		if (lo.eta_star!=1) norm_pH=diffNorm(lo.phi[lo.eta_star-1][0], phiHL, 0.0, im, jm);
		else norm_pH=0.0/0.0;
		norm_k=abs(k_eff-k_effL);
		if (lo.eta_star!=1) norm_kap=lo.findNormKappa(kappaL);
		else norm_kap=0.0/0.0;
		
		rho_k=norm_k/norm_kL;
		rho_p=norm_p/norm_pL;
		
		ho.logIteration(norm_p, rho_p, norm_pH, norm_pH/norm_pHL, k_eff, norm_k, rho_k, norm_kap, norm_kap/norm_kapL);
		//cout<<"Check Convergence\n";
		// Check Convergence Criteria
		double max_p=0.0;
		for (int g=0; g<Ng; g++) max_p=LInfNorm(ho.phi[g], max_p);
		ho.norm_inf.push_back(max_p);
		// Check Flux Convergence
		if ( ho.epsilon_phi.size()==2 )    phiConverged=( norm_p      <=ho.epsilon_phi[0]*max_p+ho.epsilon_phi[1] );
		else if ( lo.relativeConvergence ) phiConverged=( norm_p/max_p<=ho.epsilon_phi[0]*(1/rho_p-1) );
		else                               phiConverged=( norm_p      <=ho.epsilon_phi[0]*(1/rho_p-1) ); // Check convergence of Flux
		// Check K convergence
		if ( lo.solveK ) {
			if ( ho.epsilon_keff.size()==2 ) kConverged=( norm_k<=ho.epsilon_keff[0]*k_eff+ho.epsilon_keff[1] );
			else if ( not (norm_k>ho.epsilon_keff[0]*(1/rho_k-1)) ) { // Check convergence of K
				kConverged=true;
				if ( lo.fixK ) lo.solveK=false; // Fix K
			}
		}
		if ( s==0 ) phiConverged=false;
		
		s++;
	} while ( ( not phiConverged or not kConverged ) and s<ho.stop_phi );
	ho.n_iterations=s;
	//write_cell_dat(lo.phi[0][0], 16, temfile);
	// for periodic case normalize solution
	if ( KE_problem ) {
		for (int k=0; k<lo.eta_star; k++) lo.normalizeEigen(k);
		ho.normalizeEigen();
	}
	//write_cell_dat(lo.phi[0][0], 16, temfile);
}
//======================================================================================//

//======================================================================================//
//++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int main (int argc, char* argv[])
{
	char fn[50], outf[50]={""};// options[50]={"\0"};
	int i, j, m;
	double nu1, c;
	string case_name, options;
	
	//omp_set_num_threads(4);
	//int n = Eigen::nbThreads();
	//cout<<" n = "<<n<<endl;
	//cout << "  Number of processors available = " << omp_get_num_procs ( ) << "\n";
	//cout << "  Number of threads =              " << omp_get_max_threads ( ) << "\n";
	
	
	strcpy (fn,argv[1]);
	cout << fn << endl;
	if ( argc<2 ) cout<<">> Error: Missing input file \n"<< "usage: " << argv[0] << " input_file_name";
	else if ( argc==3 ) {
		case_name=argv[1];
		options=argv[2];
		if (options=="check"  ) {
			cout<<"Create material view \n";
			int run_stat_input=input(case_name); // get input data
			if ( run_stat_input==0 ) {  // If there are no input errors Continue to run
				preOutput(fn);
				ho.writeMatFile(case_name);
			}
		}
		else cout<<">> Error in second argument\n";
	}
	else if ( argc==2) {
		case_name=argv[1];
		try {
			if ( input(case_name)!=0 ) throw "Fatal Error in Function 'input(case_name)' \n"; // get input data
			if ( initialize()!=0 ) throw "Fatal Error in Function 'initialize()' \n"; // initialize memory space for solution
			
			string file_name=case_name+".temp.csv";
			cout<<file_name<<endl;
			temfile.open(file_name.c_str()); // open temporary file
			cout<<"++++++++++++++++++++++\n";
			clock_t t = clock();    // start timer
			// ------------------------------------------
			Iterations(); // Call Iterations
			// ------------------------------------------
			t = clock() -t; // stop timer
			double run_time=((double)t)/CLOCKS_PER_SEC;
			cout<<"++++++++++++++++++++++\n";
			temfile.close(); // close temporary file
			preOutput(fn);
			output(case_name,run_time);     // write out solutions
		}
		catch(std::string error) {
			cout<<"Program Error >> "<<error;
		}
	}
	else cout<<">> Error: Too many  input arguments \n"<< "usage: "<<argv[0]<<" input file name + options";
	
	cout<<"Code Completed Successfully!!\n";
	
	return 0;
}
//======================================================================================//

