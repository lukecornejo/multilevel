#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <time.h>  /* clock_t, clock, CLOCKS_PER_SEC */
#include <vector>
#include <stdlib.h>
#include "HO_1.6.h"
#include "IO_1.6.h"
#include <omp.h>
//#include "H5Cpp.h"

using namespace std;

extern string test_name;
extern bool reflectiveB, reflectiveT, reflectiveL, reflectiveR;
extern double k_eff;
extern bool KE_problem; // option variables
extern int    kbc; // Kind of BC
extern ofstream temfile;

//++ Hight-Order Solution

//**************************************************************************************//
//**************************************************************************************//
//======================================================================================//
//++++++++++++++++++++++++++++++++ High Order Factors ++++++++++++++++++++++++++++++++++//
//======================================================================================//
//**************************************************************************************//
//**************************************************************************************//


//++ Initialize values ++++++++++++++++++++++++++++++++++
void HOSolution::initializeSolution() {
	double pi=3.14159265358979323846;
	
	vector< vector<double> > grid_d(Nx,vector<double>(Ny, 0.0));
	vector< vector<double> > xgrid_d(Nx+1,vector<double>(Ny, 0.0));
	vector< vector<double> > ygrid_d(Nx,vector<double>(Ny+1, 0.0));
	
	// initialize boundary conditions
	double psi_const=1.0/Ng;
	psiB.resize(Ng); 
	psiL.resize(Ng);
	for (int g=0; g<Ng; g++) {
		psiB[g].resize(Nx); 
		psiL[g].resize(Ny);
		for (int i=0; i<Nx; i++) {
			psiB[g][i].resize(Sn);
			for (int m=0; m<Sn; m++) {
				psiB[g][i][m].resize(5);
				for (int p=1; p<5; p++) psiB[g][i][m][p]=psi_const;
			}
		}
		for (int j=0; j<Ny; j++) {
			psiL[g][j].resize(Sn);
			for (int m=0; m<Sn; m++) {
				psiL[g][j][m].resize(5);
				for (int p=1; p<5; p++) psiL[g][j][m][p]=psi_const;
			}
		}
	}
	psiT=psiB;
	psiR=psiL;
	// cell average flux
	phi.resize(Ng);
	for (int g=0; g<Ng; g++) phi[g]=grid_d;
	// cell edge values on x grid
	phi_x.resize(Ng);
	for (int g=0; g<Ng; g++) phi_x[g]=xgrid_d;
	j_x=phi_x;
	// cell edge flux on y grid
	phi_y.resize(Ng);
	for (int g=0; g<Ng; g++) phi_y[g]=ygrid_d;
	j_y=phi_y;
	
	if ( NDASolution ) {
		D_x =phi_x;
		D_xT=j_x;
		D_y =phi_y;
		D_yT=j_y;
		
		// find values for D's
		for (int g=0; g<Ng; g++) {
			//# pragma omp parallel for
			for (int i=1; i<Nx; i++ ) for (int j=0; j<Ny; j++ ) D_x[g][i][j]=2*xe[i]*Diffusion(g,i-1,j)*Diffusion(g,i,j)/(Diffusion(g,i-1,j)*hx[i]+Diffusion(g,i,j)*hx[i-1]); // X Grid D
			//# pragma omp parallel for
			for (int j=0; j<Ny; j++ ) { // Top and Bottom
				D_x[g][0][j] =2*xe[0] *Diffusion(g,0,j)   /hx[0];
				D_x[g][Nx][j]=2*xe[Nx]*Diffusion(g,Nx-1,j)/hx[Nx-1];
			}
			//# pragma omp parallel for
			for (int  i=0; i<Nx; i++ ) for (int j=1; j<Ny; j++ ) D_y[g][i][j]=2*ye[j]*Diffusion(g,i,j-1)*Diffusion(g,i,j)/(Diffusion(g,i,j-1)*hy[j]+Diffusion(g,i,j)*hy[j-1]); // Y Grid D
			//# pragma omp parallel for
			for (int  i=0; i<Nx; i++ ) { // Left and Right
				D_y[g][i][0] =2*ye[0] *Diffusion(g,i,0)   /hy[0];
				D_y[g][i][Ny]=2*ye[Ny]*Diffusion(g,i,Ny-1)/hy[Ny-1];
			}
		}
	}
	else {
		// QD Factors
		E_xx=E_yy=phi;
		E_xx_x=E_yy_x=phi_x;
		E_xy_x=j_x;
		E_xx_y=E_yy_y=phi_y;
		E_xy_y=j_y;
		
		for (int g=0; g<Ng; g++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					E_xx[g][i][j]=1.0/3.0; E_yy[g][i][j]=1.0/3.0;
				}
			}
			for (int i=0; i<Nx+1; i++) {
				for (int j=0; j<Ny; j++) {
					E_xx_x[g][i][j]=1.0/3.0; E_yy_x[g][i][j]=1.0/3.0;
				}
			}
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny+1; j++) {
					E_xx_y[g][i][j]=1.0/3.0; E_yy_y[g][i][j]=1.0/3.0;
				}
			}
		}
	}
	
	// Find Initial Flux and Current from constant angular flux
	for (int g=0; g<Ng; g++) {
		# pragma omp parallel for
		for (int i=0; i<Nx; i++ ) {
			for (int j=0; j<Ny; j++ ) phi[g][i][j]=4.0*pi*psi_const;
		}
		# pragma omp parallel for
		for (int i=0; i<Nx+1; i++ ) { // X Grid
			for (int j=0; j<Ny; j++ ) phi_x[g][i][j]=4.0*pi*psi_const;
		}
		# pragma omp parallel for
		for (int i=0; i<Nx; i++ ) { // Y Grid
			for (int j=0; j<Ny+1; j++ ) phi_y[g][i][j]=4.0*pi*psi_const;
		}
	}
	
}

void HOSolution::zeroSolution() {
	
	for (int g=0; g<Ng; g++) {
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				phi[g][i][j]=0.0;
			}
		}
		//# pragma omp parallel for
		for (int i=0; i<Nx+1; i++) {
			for (int j=0; j<Ny; j++) {
				phi_x[g][i][j]=0.0; j_x[g][i][j]=0.0;
			}
		}
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny+1; j++) {
				phi_y[g][i][j]=0.0; j_y[g][i][j]=0.0;
			}
		}
	}
	
	if ( not NDASolution ) {
		// QD Factors
		for (int g=0; g<Ng; g++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					E_xx[g][i][j]=0.0; E_yy[g][i][j]=0.0;
				}
			}
			for (int i=0; i<Nx+1; i++) {
				for (int j=0; j<Ny; j++) {
					E_xx_x[g][i][j]=0.0; E_yy_x[g][i][j]=0.0; E_xy_x[g][i][j]=0.0;
				}
			}
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny+1; j++) {
					E_xx_y[g][i][j]=0.0; E_yy_y[g][i][j]=0.0; E_xy_y[g][i][j]=0.0;
				}
			}
		}
	}
}

//++ Tally Consistency Factors +++++++++++++++++++++++++++
void HOSolution::tallySolution(int g, std::vector< std::vector<double> > &psi, std::vector< std::vector<double> > &psi_x, std::vector< std::vector<double> > &psi_y, double omega_x, double omega_y, double w) {
	
	
	if ( not NDASolution ) {
		double omega_xx, omega_yy, omega_xy;
		omega_xx=omega_x*omega_x*w;
		omega_yy=omega_y*omega_y*w;
		omega_xy=omega_x*omega_y*w;
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				E_xx[g][i][j]+=omega_xx*psi[i][j]; 
				E_yy[g][i][j]+=omega_yy*psi[i][j]; 
			}
		}
		for (int i=0; i<Nx+1; i++) {
			for (int j=0; j<Ny; j++) {
				E_xx_x[g][i][j]+=omega_xx*psi_x[i][j]; 
				E_yy_x[g][i][j]+=omega_yy*psi_x[i][j];  
				E_xy_x[g][i][j]+=omega_xy*psi_x[i][j]; 
			}
		}
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny+1; j++) {
				E_xx_y[g][i][j]+=omega_xx*psi_y[i][j];  
				E_yy_y[g][i][j]+=omega_yy*psi_y[i][j];  
				E_xy_y[g][i][j]+=omega_xy*psi_y[i][j]; 
			}
		}
	}
	
	omega_x*=w;
	omega_y*=w;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			phi[g][i][j]+=psi[i][j]*w;
		}
	}
	for (int i=0; i<Nx+1; i++) {
		for (int j=0; j<Ny; j++) {
			phi_x[g][i][j]+=psi_x[i][j]*w;
			j_x[g][i][j]  +=psi_x[i][j]*omega_x;
		}
	}
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny+1; j++) {
			phi_y[g][i][j]+=psi_y[i][j]*w;
			j_y[g][i][j]  +=psi_y[i][j]*omega_y;
		}
	}
}

//++ Calculate Consistency Terms +++++++++++++++++++++++++
void HOSolution::calculateFactors() {
	if ( NDASolution ) {
		// Calculate D^tilde Factors
		//# pragma omp parallel for
		for (int g=0; g<Ng; g++) {
			for (int i=1; i<Nx; i++) for (int j=0; j<Ny; j++) 
				D_xT[g][i][j]=2*(j_x[g][i][j]+D_x[g][i][j]*(phi[g][i][j]-phi[g][i-1][j])/xe[i])/(phi[g][i][j]+phi[g][i-1][j]); // X Grid
			for (int j=0; j<Ny; j++) { // Left and Right
				D_xT[g][0][j] =2*(j_x[g][0][j] +D_x[g][0][j] *(phi[g][0][j]   -phi_x[g][0][j]) /xe[0]) /(phi[g][0][j]   +phi_x[g][0][j]); // Left Side
				D_xT[g][Nx][j]=2*(j_x[g][Nx][j]+D_x[g][Nx][j]*(phi_x[g][Nx][j]-phi[g][Nx-1][j])/xe[Nx])/(phi_x[g][Nx][j]+phi[g][Nx-1][j]); // Right Side
			}
			// D_yT
			for (int i=0; i<Nx; i++) for (int j=1; j<Ny; j++) 
				D_yT[g][i][j]=2*(j_y[g][i][j]+D_y[g][i][j]*(phi[g][i][j]-phi[g][i][j-1])/ye[j])/(phi[g][i][j]+phi[g][i][j-1]); // Y Grid
			for (int i=0; i<Nx; i++ ) { // Bottom and Top
				D_yT[g][i][0] =2*(j_y[g][i][0] +D_y[g][i][0] *(phi[g][i][0]   -phi_y[g][i][0]) /ye[0]) /(phi[g][i][0]   +phi_y[g][i][0]); // Bottom Side
				D_yT[g][i][Ny]=2*(j_y[g][i][Ny]+D_y[g][i][Ny]*(phi_y[g][i][Ny]-phi[g][i][Ny-1])/ye[Ny])/(phi_y[g][i][Ny]+phi[g][i][Ny-1]); // Top
			}
		}
	}
	else {
		// Calculate QD Factors
		for (int g=0; g<Ng; g++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					E_xx[g][i][j]/=phi[g][i][j]; 
					E_yy[g][i][j]/=phi[g][i][j];
				}
			}
			for (int i=0; i<Nx+1; i++) {
				for (int j=0; j<Ny; j++) {
					E_xx_x[g][i][j]/=phi_x[g][i][j]; 
					E_yy_x[g][i][j]/=phi_x[g][i][j]; 
					E_xy_x[g][i][j]/=phi_x[g][i][j];
				}
			}
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny+1; j++) {
					E_xx_y[g][i][j]/=phi_y[g][i][j]; 
					E_yy_y[g][i][j]/=phi_y[g][i][j]; 
					E_xy_y[g][i][j]/=phi_y[g][i][j];
				}
			}
		}
	}
}

void HOSolution::writeSolutionDat(std::string case_name) {
	int outw=16;
	
	if ( writeOutput ) {
		string ho_file=case_name+".ho.csv";
		#pragma omp critical
		cout<<ho_file<<endl;
		
		ofstream datfile (ho_file.c_str()); // open output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		// output flux
		
		datfile<<" # of x cells , # of y cells ,\n";
		datfile<<Nx<<" , "<<Ny<<", \n";
		datfile<<"x edge grid , "; for (int i=0; i<Nx+1; i++) datfile<<print_csv(x[i]); datfile<<endl;
		datfile<<"x center grid , "; for (int i=0; i<Nx; i++) datfile<<print_csv((x[i]+x[i+1])/2); datfile<<endl;
		datfile<<"y edge grid , "; for (int j=0; j<Ny+1; j++) datfile<<print_csv(y[j]); datfile<<endl;
		datfile<<"y center grid , "; for (int j=0; j<Ny; j++) datfile<<print_csv((y[j]+y[j+1])/2); datfile<<endl;
		datfile<<"Number of Energy groups, "<<Ng<<" , \n";
		
		datfile<<"\n -- Cell Averaged Scalar Flux -- \n";
		write_group_dat(phi, Ng, x, y, 0, outw, datfile); // call function to write out cell average scalar flux
		
		datfile<<"\n -- X Vertical Cell Edge Scalar Flux -- \n";
		write_group_dat(phi_x, Ng, x, y, 0, outw, datfile); // call function to write out cell edge scalar flux on x grid
		
		datfile<<"\n -- Y Horizontal Cell Edge Scalar Flux -- \n";
		write_group_dat(phi_y, Ng, x, y, 0, outw, datfile); // call function to write out cell edge scalar flux on y grid
		
		datfile<<"\n -- X Face Average Normal Current J_x -- \n";
		write_group_dat(j_x, Ng, x, y, 0, outw, datfile); // call function to write out cell edge current on x grid
		
		datfile<<"\n -- Y Face Average Normal Current J_y -- \n";
		write_group_dat(j_y, Ng, x, y, 0, outw, datfile); // call function to write out cell edge current on y grid
		
		if ( NDASolution ) {
			datfile<<endl;
			datfile<<" ---------------------------- \n";
			datfile<<" -- Diffusion Coefficients -- \n";
			datfile<<" ---------------------------- \n";
			
			datfile<<"\n -- X Grid Cell-edge Diffusion Coefficients -- \n";
			write_group_dat(D_x, Ng, x, y, 0, outw, datfile);
			
			datfile<<"\n -- Y Grid Cell-edge Diffusion Coefficients -- \n";
			write_group_dat(D_y, Ng, x, y, 0, outw, datfile);
			
			datfile<<" ----------------------- \n";
			datfile<<" -- Consistency Terms -- \n"; // 
			datfile<<" ----------------------- \n";
			
			datfile<<"\n -- X Grid Consistency Term D_xTilde -- \n";
			write_group_dat(D_xT, Ng, x, y, 0, outw, datfile);
			
			datfile<<"\n -- Y Grid Consistency Term D_yTilde -- \n";
			write_group_dat(D_yT, Ng, x, y, 0, outw, datfile);
		}
		else {
			// Write QD Eddington Factors
			datfile<<endl;
			datfile<<" ------------------------------------- \n";
			datfile<<" -- Cell Averaged Eddington Factors -- \n";
			datfile<<" ------------------------------------- \n";
			
			datfile<<"\n -- Cell E_xx Eddington Factors -- \n";
			write_group_dat(E_xx, Ng, x, y, 0, outw, datfile);
			
			datfile<<"\n -- Cell E_yy Eddington Factors -- \n";
			write_group_dat(E_yy, Ng, x, y, 0, outw, datfile);
			
			datfile<<endl;
			datfile<<" -------------------------------------------- \n";
			datfile<<" -- Y Cell Edge Averaged Eddington Factors -- \n";
			datfile<<" -------------------------------------------- \n";
			
			datfile<<"\n -- X Grid E_xx Eddington Factors -- \n";
			write_group_dat(E_xx_x, Ng, x, y, 0, outw, datfile);
			
			datfile<<"\n -- X Grid E_yy Eddington Factors -- \n";
			write_group_dat(E_yy_x, Ng, x, y, 0, outw, datfile);
			
			datfile<<"\n -- X Grid E_xy Eddington Factors -- \n";
			write_group_dat(E_xy_x, Ng, x, y, 0, outw, datfile);
			
			datfile<<endl;
			datfile<<" -------------------------------------------- \n";
			datfile<<" -- Y Cell Edge Averaged Eddington Factors -- \n";
			datfile<<" -------------------------------------------- \n";
			
			datfile<<"\n -- Y Grid E_xx Eddington Factors -- \n";
			write_group_dat(E_xx_y, Ng, x, y, 0, outw, datfile);
			
			datfile<<"\n -- Y Grid E_yy Eddington Factors -- \n";
			write_group_dat(E_yy_y, Ng, x, y, 0, outw, datfile);
			
			datfile<<"\n -- Y Grid E_xy Eddington Factors -- \n";
			write_group_dat(E_xy_y, Ng, x, y, 0, outw, datfile);
		}
		
		datfile.close(); // close output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	}
	
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//======================================================================================//
//++ Initialize HO Data ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int HOSolver::initializeHO() {

	cout<<"Initialize HO Data:";
	
	initializeSolution();
	
	dt.push_back(0.0);
	
	cout<<" Complete\n";
	return 0;
}
//======================================================================================//
void HOSolver::normalizeEigen() {
	double sum=0.0;
	
	// Find coefficient
	for (int g=0; g<Ng; g++) {
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) sum+=hx[i]*hy[j]*phi[g][i][j];
		}
	}
	sum/=(x[Nx]-x[0])*(y[Ny]-y[0]);
	// Normalize
	for (int g=0; g<Ng; g++) {
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) phi[g][i][j]/=sum;
		}
		//# pragma omp parallel for
		for (int i=0; i<Nx+1; i++) {
			for (int j=0; j<Ny; j++) {
				j_x[g][i][j]/=sum;
				phi_x[g][i][j]/=sum;
			}
		}
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny+1; j++) {
				j_y[g][i][j]/=sum;
				phi_y[g][i][j]/=sum;
			}
		}
	}
}

//======================================================================================//
void HOSolver::logIteration(double norm_p, double rho_p, double norm_pH, double rho_pH, double k, double norm_k, double rho_k, double norm_kap, double rho_kap) {
	norm_phi.push_back(norm_p);
	rho_phi.push_back(rho_p);
	norm_phiH.push_back(norm_pH);
	rho_phiH.push_back(rho_pH);
	k_keff.push_back(k_eff);
	norm_keff.push_back(norm_k);
	rho_keff.push_back(rho_k);
	norm_kappa.push_back(norm_kap);
	rho_kappa.push_back(rho_kap);
}


int HOSolver::readGrid(ifstream& infile) {
	vector<double> num_temp;
	vector<string> str_temp;
	
	vector<int> nxt, nyt;
	vector<double> xbc_e, ybc_e, bc_l, bc_r, bc_b, bc_t; // bc temp
	string line;
	int i, j, g, p;
	try {
		// read in data
		getline (infile,line); // read in name of the input
		test_name=line;
		
		// read x grid data
		getline (infile,line); // x grid edges
		vector<double> xg=parseNumber(line);
		
		getline (infile,line); // # cells in x grid
		num_temp=parseNumber(line);
		for (i=0; i<num_temp.size(); i++) nxt.push_back(int(num_temp[i]+0.5));
		if ( xg.size()-1!=nxt.size() ) throw "Fatal Error in x grid input \n";
		
		// read y grid data
		getline (infile,line); // read in grid data
		vector<double> yg=parseNumber(line);
		
		getline (infile,line); // read in grid data
		num_temp=parseNumber(line);
		for (i=0; i<num_temp.size(); i++) nyt.push_back(int(num_temp[i]+0.5));
		if ( yg.size()-1!=nyt.size() ) throw "Fatal Error in y grid input \n";
		
		// Get Type of Problem
		getline (infile,line);
		num_temp=parseNumber(line);
		KE_problem=(int)(num_temp[0]+0.5); // Eigenvalue problem ? (bool)
		
		// read in boundary conditions
		// Get Type of BC
		getline (infile,line);
		num_temp=parseNumber(line);
		kbc=int(num_temp[0]+0.5);
		
		if ( kbc==1 )       { reflectiveB=false; reflectiveT=false; reflectiveL=false; reflectiveR=false; }
		else if ( kbc==2 )  { reflectiveB=false; reflectiveT=false; reflectiveL=false; reflectiveR=false; }
		else if ( kbc==3 )  { reflectiveB=true;  reflectiveT=true;  reflectiveL=true;  reflectiveR=true; }
		else if ( kbc==4 )  { reflectiveB=true;  reflectiveT=false; reflectiveL=true;  reflectiveR=false; }
		else if ( kbc==5 )  { reflectiveB=false; reflectiveT=true;  reflectiveL=false; reflectiveR=true; }
		else if ( kbc==11 ) { reflectiveB=true;  reflectiveT=false; reflectiveL=true;  reflectiveR=false; }
		else if ( kbc==12 ) { reflectiveB=true;  reflectiveT=false; reflectiveL=false; reflectiveR=true; }
		else if ( kbc==13 ) { reflectiveB=false; reflectiveT=true;  reflectiveL=false; reflectiveR=true; }
		else if ( kbc==14 ) { reflectiveB=false; reflectiveT=true;  reflectiveL=true;  reflectiveR=false; }
		else if ( kbc==21 ) { reflectiveB=false; reflectiveT=false; reflectiveL=true;  reflectiveR=false; } // Reflective on the Left
		else if ( kbc==22 ) { reflectiveB=true;  reflectiveT=false; reflectiveL=false; reflectiveR=false; } // Reflective on the Bottom
		else if ( kbc==23 ) { reflectiveB=false; reflectiveT=false; reflectiveL=false; reflectiveR=true; }  // Reflective on the Right
		else if ( kbc==24 ) { reflectiveB=false; reflectiveT=true;  reflectiveL=false; reflectiveR=false; } // Reflective of the Top
		else if ( kbc==31 ) { reflectiveB=true;  reflectiveT=true;  reflectiveL=true;  reflectiveR=false; } // Incomming on the Right
		else if ( kbc==32 ) { reflectiveB=true;  reflectiveT=false; reflectiveL=true;  reflectiveR=true; }  // Incomming on the Top
		else if ( kbc==33 ) { reflectiveB=true;  reflectiveT=true;  reflectiveL=false; reflectiveR=true; }  // Incomming on the Left
		else if ( kbc==34 ) { reflectiveB=false; reflectiveT=true;  reflectiveL=true;  reflectiveR=true; }  // Incomming on the Bottom
		else cout<<">> Error in boundary condition type \n";
		
		// Get Bottom and Top BC
		getline (infile,line);
		num_temp=parseNumber(line);
		int xbc_n=int(num_temp[0]+0.5);
		xbc_e.push_back(0.0); for (i=1; i<xbc_n; i++) xbc_e.push_back(num_temp[i]); xbc_e.push_back(0.0);
		for (i=xbc_n; i<2*xbc_n; i++) bc_b.push_back(num_temp[i]);
		for (i=2*xbc_n; i<3*xbc_n; i++) bc_t.push_back(num_temp[i]);
		
		// Get Left and Right BC
		getline (infile,line);
		num_temp=parseNumber(line);
		int ybc_n=int(num_temp[0]+0.5);
		ybc_e.push_back(0.0); for (i=1; i<ybc_n; i++) ybc_e.push_back(num_temp[i]); ybc_e.push_back(0.0);
		for (i=ybc_n; i<2*ybc_n; i++) bc_l.push_back(num_temp[i]);
		for (i=2*ybc_n; i<3*ybc_n; i++) bc_r.push_back(num_temp[i]);
		
		getline (infile,line); // get number of energy groups for fine mesh
		num_temp=parseNumber(line);
		Ng=int(num_temp[0]+0.5);
		
		
		
		// total grid
		Nx=0; for (i=0; i<nxt.size(); i++) Nx+=nxt[i]; // find total number of cells in x grid
		Ny=0; for (i=0; i<nyt.size(); i++) Ny+=nyt[i]; // find total number of cells in y grid
		
		x.resize(Nx+1); hx.resize(Nx); xe.resize(Nx+1);
		y.resize(Ny+1); hy.resize(Ny); ye.resize(Ny+1);
		
		x[0]=xg[0]; p=0;
		for (i=0; i<nxt.size(); i++) {
			for (j=p; j<p+nxt[i]; j++) {
				hx[j]=(xg[i+1]-xg[i])/nxt[i];
				x[j+1]=x[j]+hx[j];
			}
			p+=nxt[i];
		}
		y[0]=yg[0]; p=0;
		for (i=0; i<nyt.size(); i++) {
			for (j=p; j<p+nyt[i]; j++) {
			hy[j]=(yg[i+1]-yg[i])/nyt[i];
			y[j+1]=y[j]+hy[j];
			}
			p+=nyt[i];
		}
		xe[0]=0.5*hx[0]; xe[Nx]=0.5*hx[Nx-1];
		ye[0]=0.5*hy[0]; ye[Ny]=0.5*hy[Ny-1];
		for (i=1; i<Nx; i++) xe[i]=0.5*(hx[i-1]+hx[i]);
		for (j=1; j<Ny; j++) ye[j]=0.5*(hy[j-1]+hy[j]);
		// BC
		bcB=std::vector< std::vector<double> > (Ng,vector<double>(Nx, 0.0)); // Initialize Bottom BC
		bcT=bcB; // Initialize Top BC
		bcL=std::vector< std::vector<double> > (Ng,vector<double>(Ny, 0.0)); // Initialize Left BC
		bcR=bcL; // Initialize Right BC
		if ( not KE_problem ) {
			xbc_e[0]=x[0]; xbc_e[xbc_n]=x[Nx];
			ybc_e[0]=y[0]; ybc_e[ybc_n]=y[Ny];
			
			for (i=0; i<Nx; i++) {
				double center=(x[i]+x[i+1])/2;
				for (p=0; p<xbc_n; p++) {
					if ( xbc_e[p]<center and xbc_e[p+1]>center ) {
						for (g=0; g<Ng; g++) {
							bcB[g][i]=bc_b[p];
							bcT[g][i]=bc_t[p];
						}
					}
				}
			}
			for (j=0; j<Ny; j++) {
				double center=(y[j]+y[j+1])/2;
				for (p=0; p<ybc_n; p++) {
					if ( ybc_e[p]<center and ybc_e[p+1]>center ) {
						for (g=0; g<Ng; g++) {
							bcL[g][j]=bc_l[p];
							bcR[g][j]=bc_r[p];
						}
					}
				}
			}
		}
	}
	catch (std::string error) {
		cout<<"Grid Input Error >> "<<error;
		return 1;
	}
	return 0;
}


//======================================================================================//
//++ HOSolver Reader +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int HOSolver::readHO(std::vector<std::string> &line) {
	cout<<"Read HO Data: ";
	vector<double> num_temp;
	vector<string> str_temp;
	
	num_temp=parseNumber(line[0]); // Read High order Data
	N=int(num_temp[0]+0.5); // Quadrature # (int)
		
	num_temp=parseNumber(line[1]);
	epsilon_phi=num_temp; // get flux convergence criteria (double)
	
	for (int i=2; i<line.size(); i++) {
		str_temp=parseWord(line[i]);
		num_temp=parseNumber(line[i]);
		if      (str_temp[0]=="HO_phi_epsilon"          ) epsilon_phi=num_temp;
		else if (str_temp[0]=="HO_keff_epsilon"         ) epsilon_keff=num_temp;
		else if (str_temp[0]=="HO_phi_truncation"       ) {
			if ( num_temp.size()==2 ) epsilon_phi=num_temp;
			else if ( num_temp.size()==1 ) {
				epsilon_phi.push_back(num_temp[0]);
				epsilon_phi.push_back(1e-15);
			}
			else cout<<">> Error: Wrong number of HO Phi convergence criteria \n";
		}
		else if (str_temp[0]=="HO_keff_truncation"       ) {
			if ( num_temp.size()==2 ) epsilon_keff=num_temp;
			else if ( num_temp.size()==1 ) {
				epsilon_keff.push_back(num_temp[0]);
				epsilon_keff.push_back(1e-15);
			}
			else cout<<">> Error: Wrong number of HO Keff convergence criteria \n";
		}
		else if (str_temp[0]=="HO_stopping"             ) stop_phi=int(num_temp[0]+0.5);
		else if (str_temp[0]=="Write_Angular_Flux"      ) o_angular=int(num_temp[0]+0.5);
		else if (str_temp[0]=="Boundary_Corrections_Off") boundary_corrections=false;
		else if (str_temp[0]=="Short_Output"            ) writeOutput=false;
		else cout<<">> Error: Unknown variable in HO_Data block <"<<str_temp[0]<<">\n";
	}
	// Default
	if (stop_phi==0) stop_phi=1000;
	if (epsilon_phi[0]<1e-15) { cout<<"In ReadHO>> Use defautl Convergence criteria\n"; epsilon_phi[0]=1e-5; }
	if (epsilon_keff.size()==0 ) epsilon_keff=epsilon_phi;
	
	// Input Error Handling
	if ( epsilon_phi[0]<1e-12 ) { cout<<">> Warning in HO convergence criteria input!\n"; }
	if ( N!=4 and N!=6 and N!=8 and N!=12 and N!=16 and N!=20 and N!=36 ) {
		cout<<">> Error in Quadrature input! using default q36.\n"; N=36;
	}
	
	//omp_set_num_threads(4);
	cout<<"Complete\n";
	return 0;
}
//======================================================================================//

//======================================================================================//
//++ HOSolver Writer +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void HOSolver::writeHO(ofstream& outfile) {
	outfile<<" -- High-order Solution Options -- \n";
	outfile<<"+-----------------------------------------------+\n";
	outfile<<"Transport Convergence Criteria: "; for ( int i=0; i<epsilon_phi.size(); i++ ) outfile<<epsilon_phi[i]<<" "; outfile<<endl;
	outfile<<"HO K_eff  Convergence Criteria: "; for ( int i=0; i<epsilon_keff.size(); i++ ) outfile<<epsilon_keff[i]<<" "; outfile<<endl;
	outfile<<"HO Stopping Criteria: "<<stop_phi<<endl;
	outfile<<"Angular Quadrature ";
	if ( N<20 ) outfile<<"S "<<N<<endl;
	else if ( N==20 ) outfile<<" q20 or q2468 quadruple-range quadrature\n";
	else if ( N==36 ) outfile<<" q36 or q461214 quadruple-range quadrature\n";
	else cout<<">> Error in angular quadratures\n";
	outfile<<" -- Options -- \n";
	outfile<<"Write Angular Flux   : "<<o_angular<<endl;
	outfile<<"Boundary Corrections : "<<boundary_corrections<<endl;
	if ( not writeOutput ) outfile<<"No output of HO Solution \n";
	outfile<<"+-----------------------------------------------+\n";
	//return 0;
}
//======================================================================================//


//======================================================================================//
std::string HOSolver::writeIteration(int i) {
	std::string outstr=print_out(rho_phi[i])+print_out(norm_phi[i])+print_out(k_keff[i])+print_out(rho_keff[i])+print_out(norm_keff[i])+print_out(dt[i]);
	return outstr;
}
//======================================================================================//
void HOSolver::outputQuadrature(ofstream& outfile) {
	// output quadrature
	int outw=16;
	outfile<<"\n -- Quadrature -- \n";
	if ( N==20 or N==36 ) outfile<<" Octant-Range Q";
	else outfile<<"Level Symmeteric S";
	outfile<<N<<" Normalized to Integrate to 4*pi \n";
	outfile<<"   m        mu              eta              xi            weight   \n";
	for (int m=0; m<Sn; m++) outfile<<setw(5)<<m+1<<print_out(mu[m])<<print_out(eta[m])<<print_out(xi[m])<<print_out(w[m])<<endl; // quadrature
	outfile<<endl;
}

void HOSolver::writeSpatialGrid(ofstream& outfile) {
	
	// output grid
	outfile<<"\n -- High Order Solution Grid -- \n";
	outfile<<" X grid \n"<<" Index     Cell Edge       Width Avg     Cell center      Cell Width   ";
	if ( kbc==2 ) outfile<<"  Quad 2 BC In  "<<"  Quad 4 BC In  "<<endl; // Write Bottom and Top BC
	else {
		if ( reflectiveB ) outfile<<" Bottom BC REFL ";
		else outfile<<"  Bottom BC In  ";
		if ( reflectiveT ) outfile<<" Top    BC REFL "<<endl;
		else outfile<<"  Top    BC In  "<<endl;
	}
	
	for (int i=0; i<Nx; i++) {
		outfile<<setw(6)<<i+1<<print_out(x[i])<<print_out(xe[i])<<print_out((x[i]+x[i+1])/2)<<print_out(hx[i]);
		if ( reflectiveB ) outfile<<"   reflective   ";
		else outfile<<print_out(bcB[0][i]);
		if ( reflectiveT ) outfile<<"   reflective   "<<endl;
		else outfile<<print_out(bcT[0][i])<<endl;
	}
	outfile<<setw(6)<<Nx+1<<print_out(x[Nx])<<print_out(xe[Nx])<<endl;
	
	outfile<<" Y grid \n"<<" Index     Cell Edge       Width Avg     Cell center      Cell Width   ";
	if ( kbc==2 ) outfile<<"  Quad 1 BC In  "<<"  Quad 3 BC In  "; // Write Left and Right BC
	else {
		if ( reflectiveL ) outfile<<" Left   BC REFL ";
		else outfile<<"  Left   BC In  ";
		if ( reflectiveR ) outfile<<" Right  BC REFL "<<endl;
		else outfile<<"  Right  BC In  "<<endl;
	}
	
	for (int j=0; j<Ny; j++) {
		outfile<<setw(6)<<j+1<<print_out(y[j])<<print_out(ye[j])<<print_out((y[j]+y[j+1])/2)<<print_out(hy[j]);
		if ( reflectiveL ) outfile<<"   reflective   ";
		else outfile<<print_out(bcL[0][j]);
		if ( reflectiveR ) outfile<<"   reflective   "<<endl;
		else outfile<<print_out(bcR[0][j])<<endl;
	}
	outfile<<setw(6)<<Ny+1<<print_out(y[Ny])<<print_out(ye[Ny])<<endl;
	
}

void HOSolver::writeSpatialGridDat(ofstream& datfile) {
	
	datfile<<" -- Solution Grid -- \n";
	
	datfile<<" X grid \n"<<" Index   ,  Cell Edge   ,    Width Avg   ,  Cell Center   ,   Cell Width   "<<endl;
	for (int i=0; i<Nx; i++) datfile<<setw(6)<<i+1<<","<<print_csv(x[i])<<print_csv(xe[i])<<print_csv((x[i]+x[i+1])/2)<<print_csv(hx[i])<<endl; // write x grid
	datfile<<setw(6)<<Nx+1<<","<<print_csv(x[Nx])<<print_csv(xe[Nx])<<endl;
	
	datfile<<" Y grid \n"<<" Index   ,  Cell Edge   ,    Width Avg   ,  Cell Center   ,   Cell Width   "<<endl;
	for (int j=0; j<Ny; j++) datfile<<setw(6)<<j+1<<","<<print_csv(y[j])<<print_csv(ye[j])<<print_csv((y[j]+y[j+1])/2)<<print_csv(hy[j])<<endl; // write y grid
	datfile<<setw(6)<<Ny+1<<","<<print_csv(y[Ny])<<print_csv(ye[Ny])<<endl;
	
}

//======================================================================================//


//======================================================================================//
//++ sweep through angles and cells in each angular quadrant +++++++++++++++++++++++++++//
//======================================================================================//
void HOSolver::quad1(int g, vector< vector<double> > &SA, vector< vector<double> > &psi, vector< vector<double> > &psi_x, vector< vector<double> > &psi_y) // solution in quadrant 1
{
	int i, j, outw=16;
	double omega_x, omega_y;
	
	if ( reflectiveB and not reflectiveT ) {
		for (int m=0; m<Sn; m++) for (i=0; i<Nx; i++) psiB[g][i][m][1]=psiB[g][i][m][4]; // Quad 1 Bottom Reflective BC
	}
	else if ( not reflectiveB ) {
		for (int m=0; m<Sn; m++) for (i=0; i<Nx; i++) psiB[g][i][m][1]=bcB[g][i]; // Quad 1 Bottom boundary condition
	}
	//
	if ( reflectiveL and not reflectiveR ) {
		for (int m=0; m<Sn; m++) for (j=0; j<Ny; j++) psiL[g][j][m][1]=psiL[g][j][m][2]; // Quad 1 Left Reflective BC
	}
	else if ( not reflectiveL ) {
		for (int m=0; m<Sn; m++) for (j=0; j<Ny; j++) psiL[g][j][m][1]=bcL[g][j]; // Quad 1 Left boundary condition
	}
	
	//# pragma omp parallel for
	for (int m=0; m<Sn; m++) { // first quadrant
		omega_x=mu[m]; omega_y=eta[m];
		
		for (i=0; i<Nx; i++) psi_y[i][0]=psiB[g][i][m][1]; // Bottom In BC
		for (j=0; j<Ny; j++) psi_x[0][j]=psiL[g][j][m][1]; // Left In BC
		
		for (j=0; j<Ny; j++) { // bottom to top
			for (i=0; i<Nx; i++) { // left to right
				//psiInL=psi_x[i][j]; // incoming angular flux on the left
				//psiInB=psi_y[i][j]; // incoming angular flux on the bottom
				cellSolution( psi_y[i][j], psi_x[i][j], SA[i][j], Total(g, i, j), i, j, m, psi_y[i][j+1], psi_x[i+1][j], psi[i][j] );
				//psi_x[i+1][j]=psiOutR; // outgoing angular flux on the right
				//psi_y[i][j+1]=psiOutT; // outgoing angular flux on the top
			}
		}
		
		tallySolution(g, psi, psi_x, psi_y, omega_x, omega_y, w[m]);
		
		for (i=0; i<Nx; i++) psiT[g][i][m][1]=psi_y[i][Ny]; // Top Out BC
		for (j=0; j<Ny; j++) psiR[g][j][m][1]=psi_x[Nx][j]; // Right Out BC
		
		
		// option to print out angular flux
		if ( o_angular ) {
			temfile<<" Quadrant 1 Direction # ,"<<m<<",\n";
			temfile<<"  Qmega_x ,"<<print_csv(omega_x)<<"  Omega_y ,"<<print_csv(omega_y)<<"  Weight ,"<<print_csv(w[m])<<endl;
			temfile<<" -- Cell Averaged Angular Flux -- \n";
			write_cell_dat(psi, x, y, outw, temfile); // call function to write out cell average scalar flux
			temfile<<" -- X Vertical Cell Edge Angular Flux -- \n";
			write_cell_dat(psi_x, x, y, outw, temfile); // call function to write out cell edge scalar flux on x grid
			temfile<<" -- Y Horizontal Cell Edge Angular Flux -- \n";
			write_cell_dat(psi_y, x, y, outw, temfile); // call function to write out cell edge scalar flux on y grid
		}
	}
	//temfile<<"g "<<g<<" phiT 2\n";
	//write_group_average_dat(phiT, 0, 16, temfile);
}
//======================================================================================//
void HOSolver::quad2(int g, vector< vector<double> > &SA, vector< vector<double> > &psi, vector< vector<double> > &psi_x, vector< vector<double> > &psi_y) // solution in quadrant 2
{
	int i, j, outw=16;
	double omega_x, omega_y;
	
	if ( reflectiveB and not reflectiveT ) {
		for (int m=0; m<Sn; m++) for (i=0; i<Nx; i++) psiB[g][i][m][2]=psiB[g][i][m][3]; // Quad 2 Bottom Reflective BC
	}
	else if ( not reflectiveB ) {
		for (int m=0; m<Sn; m++) for (i=0; i<Nx; i++) psiB[g][i][m][2]=bcB[g][i]; // Quad 2 Bottom boundary condition
	}
	//
	if ( reflectiveR and not reflectiveL ) {
		for (int m=0; m<Sn; m++) for (j=0; j<Ny; j++) psiR[g][j][m][2]=psiR[g][j][m][1]; // Quad 2 Right BC
	}
	else if ( not reflectiveR ) {
		for (int m=0; m<Sn; m++) for (j=0; j<Ny; j++) psiR[g][j][m][2]=bcR[g][j]; // Quad 2 Right boundary condition
	}
	
	//# pragma omp parallel for
	for (int m=0; m<Sn; m++) { // second quadrant
		omega_x=-mu[m]; omega_y=eta[m];
		
		for (i=0; i<Nx; i++) psi_y[i][0]=psiB[g][i][m][2]; // Bottom In BC
		for (j=0; j<Ny; j++) psi_x[Nx][j]=psiR[g][j][m][2]; // Right In BC
		
		for (j=0; j<Ny; j++) { // bottom to top
			for (i=Nx-1; i>=0; i--) { // right to left
				//psiInL=psi_x[i+1][j]; // incoming angular flux on the left
				//psiInB=psi_y[i][j];   // incoming angular flux on the bottom
				cellSolution( psi_y[i][j], psi_x[i+1][j], SA[i][j], Total(g, i, j), i, j, m, psi_y[i][j+1], psi_x[i][j], psi[i][j] );
				//psi_x[i][j]=psiOutR;   // outgoing angular flux on the right
				//psi_y[i][j+1]=psiOutT; // outgoing angular flux on the top
			}
		}
		
		tallySolution(g, psi, psi_x, psi_y, omega_x, omega_y, w[m]);
		
		// Out BC
		for (i=0; i<Nx; i++) psiT[g][i][m][2]=psi_y[i][Ny]; // Top Out BC
		for (j=0; j<Ny; j++) psiL[g][j][m][2]=psi_x[0][j]; // Left Out BC
		
		// option to print out angular flux
		if ( o_angular ) {
			temfile<<" Quadrant 2 Direction # ,"<<m<<",\n";
			temfile<<"  Qmega_x ,"<<print_csv(omega_x)<<"  Omega_y ,"<<print_csv(omega_y)<<"  Weight ,"<<print_csv(w[m])<<endl;
			temfile<<" -- Cell Averaged Angular Flux -- \n";
			write_cell_dat(psi, x, y, outw, temfile); // call function to write out cell average scalar flux
			temfile<<" -- X Vertical Cell Edge Angular Flux -- \n";
			write_cell_dat(psi_x, x, y, outw, temfile); // call function to write out cell edge scalar flux on x grid
			temfile<<" -- Y Horizontal Cell Edge Angular Flux -- \n";
			write_cell_dat(psi_y, x, y, outw, temfile); // call function to write out cell edge scalar flux on y grid
		}
		
	}
}
//======================================================================================//
void HOSolver::quad3(int g, vector< vector<double> > &SA, vector< vector<double> > &psi, vector< vector<double> > &psi_x, vector< vector<double> > &psi_y) // solution in quadrant 3
{
	int i, j, outw=16;
	double omega_x, omega_y;
	
	
	if ( reflectiveT and not reflectiveB ) {
		for (int m=0; m<Sn; m++) for (i=0; i<Nx; i++) psiT[g][i][m][3]=psiT[g][i][m][2]; // Quad 3 Top reflective BC
	}
	else if ( not reflectiveT ) {
		for (int m=0; m<Sn; m++) for (i=0; i<Nx; i++) psiT[g][i][m][3]=bcT[g][i]; // Quad 3 Top boundary condition
	}
	//
	if ( reflectiveR and not reflectiveL ) {
		for (int m=0; m<Sn; m++) for (j=0; j<Ny; j++) psiR[g][j][m][3]=psiR[g][j][m][4]; // Quad 3 Right reflective BC
	}
	else if ( not reflectiveR ) {
		for (int m=0; m<Sn; m++) for (j=0; j<Ny; j++) psiR[g][j][m][3]=bcR[g][j]; // Quad 3 Right boundary condition
	}
	
	//# pragma omp parallel for
	for (int m=0; m<Sn; m++) { // third quadrant
		omega_x=-mu[m]; omega_y=-eta[m];
		
		for (i=0; i<Nx; i++) psi_y[i][Ny]=psiT[g][i][m][3]; // Top In BC
		for (j=0; j<Ny; j++) psi_x[Nx][j]=psiR[g][j][m][3]; // Right In BC
		
		
		for (j=Ny-1; j>=0; j--) { // top to bottom
			for (i=Nx-1; i>=0; i--) { // right to left
				//psiInL=psi_x[i+1][j]; // incoming angular flux on the left
				//psiInB=psi_y[i][j+1]; // incoming angular flux on the bottom
				cellSolution( psi_y[i][j+1], psi_x[i+1][j], SA[i][j], Total(g, i, j), i, j, m, psi_y[i][j], psi_x[i][j], psi[i][j] );
				//psi_x[i][j]=psiOutR; // outgoing angular flux on the right
				//psi_y[i][j]=psiOutT; // outgoing angular flux on the top
			}
		}
		
		tallySolution(g, psi, psi_x, psi_y, omega_x, omega_y, w[m]);
		
		for (i=0; i<Nx; i++) psiB[g][i][m][3]=psi_y[i][0]; // Bottom Out BC
		for (j=0; j<Ny; j++) psiL[g][j][m][3]=psi_x[0][j]; // Left Out BC
		
		// option to print out angular flux
		if ( o_angular ) {
			temfile<<" Quadrant 3 Direction # ,"<<m<<",\n";
			temfile<<"  Qmega_x ,"<<print_csv(omega_x)<<"  Omega_y ,"<<print_csv(omega_y)<<"  Weight ,"<<print_csv(w[m])<<endl;
			temfile<<" -- Cell Averaged Angular Flux -- \n";
			write_cell_dat(psi, x, y, outw, temfile); // call function to write out cell average scalar flux
			temfile<<" -- X Vertical Cell Edge Angular Flux -- \n";
			write_cell_dat(psi_x, x, y, outw, temfile); // call function to write out cell edge scalar flux on x grid
			temfile<<" -- Y Horizontal Cell Edge Angular Flux -- \n";
			write_cell_dat(psi_y, x, y, outw, temfile); // call function to write out cell edge scalar flux on y grid
		}
	}
}
//======================================================================================//
void HOSolver::quad4(int g, vector< vector<double> > &SA, vector< vector<double> > &psi, vector< vector<double> > &psi_x, vector< vector<double> > &psi_y) // solution in quadrant 4
{
	int i, j, outw=16;
	double omega_x, omega_y;
	
	if ( reflectiveT and not reflectiveB ) {
		for (int m=0; m<Sn; m++) for (i=0; i<Nx; i++) psiT[g][i][m][4]=psiT[g][i][m][1]; // Quad 4 Top reflective BC
	}
	else if ( not reflectiveT ) {
		for (int m=0; m<Sn; m++) for (i=0; i<Nx; i++) psiT[g][i][m][4]=bcT[g][i]; // Quad 4 Top boundary condition
	}
	//
	if ( reflectiveL and not reflectiveR ) {
		for (int m=0; m<Sn; m++) for (j=0; j<Ny; j++) psiL[g][j][m][4]=psiL[g][j][m][3]; // Quad 4 Left reflective BC
	}
	else if ( not reflectiveL ) {
		for (int m=0; m<Sn; m++) for (j=0; j<Ny; j++) psiL[g][j][m][4]=bcL[g][j]; // Quad 4 Left boundary condition
	}
	
	//# pragma omp parallel for
	for (int m=0; m<Sn; m++) { // fourth quadrant
		omega_x=mu[m]; omega_y=-eta[m];
		
		for (i=0; i<Nx; i++) psi_y[i][Ny]=psiT[g][i][m][4]; // Top In BC
		for (j=0; j<Ny; j++) psi_x[0][j]=psiL[g][j][m][4]; // Left In BC
		
		for (j=Ny-1; j>=0; j--) { // top to bottom
			for (i=0; i<Nx; i++) { // left to right
				//psiInL=psi_x[i][j];   // incoming angular flux on the left
				//psiInB=psi_y[i][j+1]; // incoming angular flux on the bottom
				cellSolution( psi_y[i][j+1], psi_x[i][j], SA[i][j], Total(g, i, j), i, j, m, psi_y[i][j], psi_x[i+1][j], psi[i][j] );
				//psi_x[i+1][j]=psiOutR; // outgoing angular flux on the right
				//psi_y[i][j]=psiOutT;   // outgoing angular flux on the top
			}
		}
		
		tallySolution(g, psi, psi_x, psi_y, omega_x, omega_y, w[m]);
		
		for (i=0; i<Nx; i++) psiB[g][i][m][4]=psi_y[i][0]; // Bottom Out BC
		for (j=0; j<Ny; j++) psiR[g][j][m][4]=psi_x[Nx][j]; // Right Out BC
		
		// option to print out angular flux
		if ( o_angular ) {
			temfile<<" Quadrant 4 Direction # ,"<<m<<",\n";
			temfile<<"  Qmega_x ,"<<print_csv(omega_x)<<"  Omega_y ,"<<print_csv(omega_y)<<"  Weight ,"<<print_csv(w[m])<<endl;
			temfile<<" -- Cell Averaged Angular Flux -- \n";
			write_cell_dat(psi, x, y, outw, temfile); // call function to write out cell average scalar flux
			temfile<<" -- X Vertical Cell Edge Angular Flux -- \n";
			write_cell_dat(psi_x, x, y, outw, temfile); // call function to write out cell edge scalar flux on x grid
			temfile<<" -- Y Horizontal Cell Edge Angular Flux -- \n";
			write_cell_dat(psi_y, x, y, outw, temfile); // call function to write out cell edge scalar flux on y grid
		}
	}
}
//======================================================================================//

//======================================================================================//
//++ Determine how to sweep thought cells ++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void HOSolver::angleSweep(int s, vector< vector< vector<double> > > &phiG, 
vector< vector<double> > &phiLG, vector< vector<double> > &phiRG, vector< vector<double> > &phiBG, vector< vector<double> > &phiTG) {
	clock_t t_ho=clock(); // start high-order timer
	cout<<"Transport Sweep Started : ";
	temfile<<"Transport Solution \n";
	//temfile<<"phiT 1\n";
	//write_group_average_dat(phiT, 0, 16, temfile);
	vector< vector<double> > SA(Nx,vector<double>(Ny, 0.0));
	vector< vector<double> > psi(Nx,vector<double>(Ny, 0.0));
	vector< vector<double> > psi_x(Nx+1,vector<double>(Ny, 0.0));
	vector< vector<double> > psi_y(Nx,vector<double>(Ny+1, 0.0));
	
	vector< vector<double> > S_f(Nx,vector<double>(Ny, 0.0));
	
	
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) 
			for (int g=0; g<Ng; g++) S_f[i][j]+=Fission(g,i,j)*phiG[g][i][j];
	// Reflective BC
	
	// Opposing Reflective BC
	if ( reflectiveT and reflectiveB ) {
				if ( boundary_corrections or s<2 ) {
			//cout<<"HO correction \n";
			for (int g=0; g<Ng; g++) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int i=0; i<Nx; i++) {
						psiB[g][i][m][1]=psiB[g][i][m][4]*phiBG[g][i]/phi_y[g][i][0]; // Quad 1 Bottom Reflective BC
						psiB[g][i][m][2]=psiB[g][i][m][3]*phiBG[g][i]/phi_y[g][i][0]; // Quad 2 Bottom Reflective BC
						psiT[g][i][m][3]=psiT[g][i][m][2]*phiTG[g][i]/phi_y[g][i][Ny]; // Quad 3 Top Reflective BC
						psiT[g][i][m][4]=psiT[g][i][m][1]*phiTG[g][i]/phi_y[g][i][Ny]; // Quad 4 Top Reflective BC
					}
				}
			}
		}
		else {
			for (int g=0; g<Ng; g++) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int i=0; i<Nx; i++) {
						psiB[g][i][m][1]=psiB[g][i][m][4]; // Quad 1 Bottom Reflective BC
						psiB[g][i][m][2]=psiB[g][i][m][3]; // Quad 2 Bottom Reflective BC
						psiT[g][i][m][3]=psiT[g][i][m][2]; // Quad 3 Top Reflective BC
						psiT[g][i][m][4]=psiT[g][i][m][1]; // Quad 4 Top Reflective BC
					}
				}
			}
		}
	}
	
	if ( reflectiveR and reflectiveL ) {
				if ( boundary_corrections or s<2 ) {
			//cout<<"HO correction \n";
			for (int g=0; g<Ng; g++) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int j=0; j<Ny; j++) {
						psiL[g][j][m][1]=psiL[g][j][m][2]*phiLG[g][j]/phi_x[g][0][j]; // Quad 1 Left Reflective BC
						psiR[g][j][m][2]=psiR[g][j][m][1]*phiRG[g][j]/phi_x[g][Nx][j]; // Quad 2 Right Reflective BC
						psiR[g][j][m][3]=psiR[g][j][m][4]*phiRG[g][j]/phi_x[g][Nx][j]; // Quad 3 Right Reflective BC
						psiL[g][j][m][4]=psiL[g][j][m][3]*phiLG[g][j]/phi_x[g][0][j]; // Quad 4 Left Reflective BC
					}
				}
			}
		}
		else {
			for (int g=0; g<Ng; g++) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int j=0; j<Ny; j++) {
						psiL[g][j][m][1]=psiL[g][j][m][2]; // Quad 1 Left Reflective BC
						psiR[g][j][m][2]=psiR[g][j][m][1]; // Quad 2 Right Reflective BC
						psiR[g][j][m][3]=psiR[g][j][m][4]; // Quad 3 Right Reflective BC
						psiL[g][j][m][4]=psiL[g][j][m][3]; // Quad 4 Left Reflective BC
					}
				}
			}
		}
	}
	zeroSolution(); // Zero High Order Factors Before Sweep
	// Begin loop
	//# pragma omp parallel for private(SA,psi,psi_x,psi_y) shared(phi,phi_x,phi_y,j_x,j_y)
	for (int g=0; g<Ng; g++) {
		// Zero out Flux and Currents
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				SA[i][j]=Chi(g,i,j)*S_f[i][j]/k_eff+Source(g,i,j);
				for (int gg=0; gg<Ng; gg++) SA[i][j]+=Scatter(gg,g,i,j)*phiG[gg][i][j];
				SA[i][j]/=4*pi;
			}
		}
		
		//cout<<"chose\n";
		if ( kbc==1 ) {
			// Start solution sweep///////////////////////////////////////
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
		}
		else if ( kbc==3 ) {
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
		}
		else if ( kbc==11 or kbc==4 ) {
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
		}
		else if ( kbc==12 ) {
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
		}
		else if ( kbc==13 or kbc==5 ) {
			// Start solution sweep ////////////////////////////////////////////////////////
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
		}
		else if ( kbc==14 ) {
			//# pragma omp parallel for
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
		}
		else if ( kbc==21 ) {
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
		}
		else if ( kbc==22 ) {
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
		}
		else if ( kbc==23 ) {
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
		}
		else if ( kbc==24 ) {
			// Start solution sweep ////////////////////////////////////////////////////////
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
		}
		else if ( kbc==31 ) {
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
		}
		else if ( kbc==32 ) {
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
		}
		else if ( kbc==33 ) {
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
		}
		else if ( kbc==34 ) {
			// Start solution sweep ////////////////////////////////////////////////////////
			quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
			quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
			quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
		}
		else cout<<"bad boundary conditions: incorrect boundary type"<<endl;
		//temfile<<g<<" phiT \n";
		//write_cell_dat(phi[g], 16, temfile);
	}
	
	calculateFactors(); // Complete Calculations for High Order Factors
		
	//temfile<<"phiT 7\n";
	//write_group_average_dat(phiT, 0, 16, temfile);
	cout<<"Completed \n";
	t_ho=clock()-t_ho; // stop high-order timer
	dt.push_back(double(t_ho)/CLOCKS_PER_SEC); // add high-order solution time to vector
}
//======================================================================================//
//++ Determine how to sweep thought cells ++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void HOSolver::reflectiveBCIterations(int s, vector< vector< vector<double> > > &phiG, 
vector< vector<double> > &phiLG, vector< vector<double> > &phiRG, vector< vector<double> > &phiBG, vector< vector<double> > &phiTG) {
	clock_t t_ho=clock(); // start high-order timer
	cout<<"Transport Sweep Started : ";
	temfile<<"Transport Solution \n";
	//temfile<<"phiT 1\n";
	//write_group_average_dat(phiT, 0, 16, temfile);
	vector< vector<double> > SA(Nx,vector<double>(Ny, 0.0));
	vector< vector<double> > psi(Nx,vector<double>(Ny, 0.0));
	vector< vector<double> > psi_x(Nx+1,vector<double>(Ny, 0.0));
	vector< vector<double> > psi_y(Nx,vector<double>(Ny+1, 0.0));
	
	vector< vector<double> > S_f(Nx,vector<double>(Ny, 0.0));
	
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) 
			for (int g=0; g<Ng; g++) S_f[i][j]+=Fission(g,i,j)*phiG[g][i][j];
	// Reflective BC
	// Opposing Reflective BC
	if ( reflectiveT and reflectiveB ) {
				if ( boundary_corrections or s<2 ) {
			//cout<<"HO correction \n";
			for (int g=0; g<Ng; g++) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int i=0; i<Nx; i++) {
						psiB[g][i][m][1]=psiB[g][i][m][4]*phiBG[g][i]/phi_y[g][i][0]; // Quad 1 Bottom Reflective BC
						psiB[g][i][m][2]=psiB[g][i][m][3]*phiBG[g][i]/phi_y[g][i][0]; // Quad 2 Bottom Reflective BC
						psiT[g][i][m][3]=psiT[g][i][m][2]*phiTG[g][i]/phi_y[g][i][Ny]; // Quad 3 Top Reflective BC
						psiT[g][i][m][4]=psiT[g][i][m][1]*phiTG[g][i]/phi_y[g][i][Ny]; // Quad 4 Top Reflective BC
					}
				}
			}
		}
		else {
			for (int g=0; g<Ng; g++) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int i=0; i<Nx; i++) {
						psiB[g][i][m][1]=psiB[g][i][m][4]; // Quad 1 Bottom Reflective BC
						psiB[g][i][m][2]=psiB[g][i][m][3]; // Quad 2 Bottom Reflective BC
						psiT[g][i][m][3]=psiT[g][i][m][2]; // Quad 3 Top Reflective BC
						psiT[g][i][m][4]=psiT[g][i][m][1]; // Quad 4 Top Reflective BC
					}
				}
			}
		}
	}
	
	if ( reflectiveR and reflectiveL ) {
				if ( boundary_corrections or s<2 ) {
			//cout<<"HO correction \n";
			for (int g=0; g<Ng; g++) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int j=0; j<Ny; j++) {
						psiL[g][j][m][1]=psiL[g][j][m][2]*phiLG[g][j]/phi_x[g][0][j]; // Quad 1 Left Reflective BC
						psiR[g][j][m][2]=psiR[g][j][m][1]*phiRG[g][j]/phi_x[g][Nx][j]; // Quad 2 Right Reflective BC
						psiR[g][j][m][3]=psiR[g][j][m][4]*phiRG[g][j]/phi_x[g][Nx][j]; // Quad 3 Right Reflective BC
						psiL[g][j][m][4]=psiL[g][j][m][3]*phiLG[g][j]/phi_x[g][0][j]; // Quad 4 Left Reflective BC
					}
				}
			}
		}
		else {
			for (int g=0; g<Ng; g++) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int j=0; j<Ny; j++) {
						psiL[g][j][m][1]=psiL[g][j][m][2]; // Quad 1 Left Reflective BC
						psiR[g][j][m][2]=psiR[g][j][m][1]; // Quad 2 Right Reflective BC
						psiR[g][j][m][3]=psiR[g][j][m][4]; // Quad 3 Right Reflective BC
						psiL[g][j][m][4]=psiL[g][j][m][3]; // Quad 4 Left Reflective BC
					}
				}
			}
		}
	}
	
	zeroSolution();
	// Begin loop
	//# pragma omp parallel for private(SA,psi,psi_x,psi_y) shared(phi,phi_x,phi_y,j_x,j_y)
	for (int g=0; g<Ng; g++) {
		// Zero out Flux and Currents
		//cout<<"sa\n";
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				SA[i][j]=Chi(g,i,j)*S_f[i][j]/k_eff+Source(g,i,j);
				for (int gg=0; gg<Ng; gg++) SA[i][j]+=Scatter(gg,g,i,j)*phiG[gg][i][j];
				SA[i][j]/=4*pi;
			}
		}
		
		bool converged=true;
		
		do {
			//cout<<"chose\n";
			if ( kbc==3 ) {
				quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
				quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
				quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
				quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			}
			else if ( kbc==31 ) {
				quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
				quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
				quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
				quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			}
			else if ( kbc==32 ) {
				quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
				quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
				quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
				quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			}
			else if ( kbc==33 ) {
				quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
				quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
				quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
				quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
			}
			else if ( kbc==34 ) {
				// Start solution sweep ////////////////////////////////////////////////////////
				quad1(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 1
				quad2(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 2
				quad3(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 3
				quad4(g,SA,psi,psi_x,psi_y); // solution sweep through quadrant 4
			}
			else cout<<"bad boundary conditions: incorrect boundary type"<<endl;
			//temfile<<g<<" phiT \n";
			//write_cell_dat(phi[g], 16, temfile);
			
			converged=true;
			// Opposing Reflective BC
			if ( reflectiveT and reflectiveB ) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int i=0; i<Nx; i++) {
						if (epsilon_phi[0]>abs(psiB[g][i][m][1]-psiB[g][i][m][4])/psiB[g][i][m][4]) converged=false; // Quad 1 Bottom Reflective BC
						if (epsilon_phi[0]>abs(psiB[g][i][m][2]-psiB[g][i][m][3])/psiB[g][i][m][3]) converged=false; // Quad 2 Bottom Reflective BC
						if (epsilon_phi[0]>abs(psiT[g][i][m][3]-psiT[g][i][m][2])/psiT[g][i][m][2]) converged=false; // Quad 3 Top Reflective BC
						if (epsilon_phi[0]>abs(psiT[g][i][m][4]-psiT[g][i][m][1])/psiT[g][i][m][1]) converged=false; // Quad 4 Top Reflective BC
					}
				}
			}
			
			if ( reflectiveR and reflectiveL ) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int j=0; j<Ny; j++) {
						if (epsilon_phi[0]>abs(psiL[g][j][m][1]-psiL[g][j][m][2])/psiL[g][j][m][2]) converged=false; // Quad 1 Left Reflective BC
						if (epsilon_phi[0]>abs(psiR[g][j][m][2]-psiR[g][j][m][1])/psiR[g][j][m][1]) converged=false; // Quad 2 Right Reflective BC
						if (epsilon_phi[0]>abs(psiR[g][j][m][3]-psiR[g][j][m][4])/psiR[g][j][m][4]) converged=false; // Quad 3 Right Reflective BC
						if (epsilon_phi[0]>abs(psiL[g][j][m][4]-psiL[g][j][m][3])/psiL[g][j][m][3]) converged=false; // Quad 4 Left Reflective BC
					}
				}
			}
			
			// Reflective BC
			// Opposing Reflective BC
			if ( reflectiveT and reflectiveB ) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int i=0; i<Nx; i++) {
						psiB[g][i][m][1]=psiB[g][i][m][4]; // Quad 1 Bottom Reflective BC
						psiB[g][i][m][2]=psiB[g][i][m][3]; // Quad 2 Bottom Reflective BC
						psiT[g][i][m][3]=psiT[g][i][m][2]; // Quad 3 Top Reflective BC
						psiT[g][i][m][4]=psiT[g][i][m][1]; // Quad 4 Top Reflective BC
					}
				}
			}
			if ( reflectiveR and reflectiveL ) {
				//# pragma omp parallel for
				for (int m=0; m<Sn; m++) { // quad 1
					for (int j=0; j<Ny; j++) {
						psiL[g][j][m][1]=psiL[g][j][m][2]; // Quad 1 Left Reflective BC
						psiR[g][j][m][2]=psiR[g][j][m][1]; // Quad 2 Right Reflective BC
						psiR[g][j][m][3]=psiR[g][j][m][4]; // Quad 3 Right Reflective BC
						psiL[g][j][m][4]=psiL[g][j][m][3]; // Quad 4 Left Reflective BC
					}
				}
			}
		} while ( not converged );
	}
	
	calculateFactors();
	
	
	//temfile<<"phiT 7\n";
	//write_group_average_dat(phiT, 0, 16, temfile);
	cout<<"Completed \n";
	t_ho=clock()-t_ho; // stop high-order timer
	dt.push_back(double(t_ho)/CLOCKS_PER_SEC); // add high-order solution time to vector
}

//======================================================================================//
double HOSolver::residual0(std::vector< std::vector< std::vector<double> > > &phiLO, int &i_ho, int &j_ho, int &g_ho) {
	// Calculate Transport Residuals
	double S_f, S_s, res, res_ho=0.0;
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) {
			S_f=0.0;
			for (int g=0; g<Ng; g++) S_f+=Fission(g,i,j)*phiLO[g][i][j];
			for (int g=0; g<Ng; g++) {
				S_s=0.0;
				for (int gg=0; gg<Ng; gg++) S_s+=Scatter(gg,g,i,j)*phiLO[gg][i][j];
				res=abs((j_x[g][i+1][j]-j_x[g][i][j])/hx[i]+(j_y[g][i][j+1]-j_y[g][i][j])/hy[j]+
					Total(g,i,j)*phi[g][i][j]-S_s-Chi(g,i,j)*S_f/k_eff-Source(g,i,j));
				//cout<<"res g"<<g<<" "<<(j_x[g][i+1][j]-j_x[g][i][j])/hx[i]<<" "<<(j_y[g][i][j+1]-j_y[g][i][j])/hy[j]<<" "<<(Total(g,i,j)*phi[g][i][j]-S_s-Chi(g,i,j)*S_f/k_eff-Source(g,i,j))<<endl;
				if ( res>res_ho ) { res_ho=res; i_ho=i; j_ho=j; }
			}
		}
	}
	return res_ho;
}
//======================================================================================//
double HOSolver::residualIterative(int &i_ho, int &j_ho, int &g_ho) {	
	// Iterative Balance Residuals
	double S_f, S_s, res, res_ho=0.0;
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) {
			S_f=0.0;
			for (int g=0; g<Ng; g++) S_f+=Fission(g,i,j)*phi[g][i][j];
			for (int g=0; g<Ng; g++) {
				S_s=0.0;
				for (int gg=0; gg<Ng; gg++) S_s+=Scatter(gg,g,i,j)*phi[gg][i][j];
				res=abs((j_x[g][i+1][j]-j_x[g][i][j])/hx[i]+(j_y[g][i][j+1]-j_y[g][i][j])/hy[j]+
					Total(g,i,j)*phi[g][i][j]-S_s-Chi(g,i,j)*S_f/k_eff-Source(g,i,j));
				//cout<<"res "<<(j_x[g][i+1][j]-j_x[g][i][j])/hx[i]<<" "<<(j_y[g][i][j+1]-j_y[g][i][j])/hy[j]<<" "<<(Total(g,i,j)*phi[g][i][j]-S_s-Chi(g,i,j)*S_f/k_eff-Source(g,i,j))<<endl;
				if ( res>res_ho ) { res_ho=res; i_ho=i; j_ho=j; g_ho=g; }
			}
		}
	}
	return res_ho;
}
//======================================================================================//

//======================================================================================//
//++ function to find the quadrature +++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int HOSolver::quadSet()
{
	double mup[N/2], wt[8];
	int i, j, m, g, nw, nt;
	int wi[N*(N+2)/8];
	double *wp[N*(N+2)/8];
	
	nt=N*(N+2)/8;
	
	if ( N<20 ) {
		switch ( N ) {
		case 2:
			nw=1;
			wi[0]=1;
			break;
		case 4: // S4 quadrature
			nw=1;
			wi[0]=1; wi[1]=1; 
				 wi[2]=1;
			mup[0]=0.3500212;
			wt[0]=0.333333333;
			break;
		case 6: // S6 quadrature
			nw=2;
			wi[0]=1; wi[1]=2; wi[2]=1; 
				 wi[3]=2; wi[4]=2; 
					wi[5]=1;
			mup[0]=0.2666355;
			wt[0]=0.1761263; wt[1]=0.1572071;
			break;
		case 8: // S8 quadrature
			nw=3;
			wi[0]=1; wi[1]=2; wi[2]=2; wi[3]=1; 
				  wi[4]=2; wi[5]=3; wi[6]=2; 
					  wi[7]=2; wi[8]=2; 
						  wi[9]=1;
			mup[0]=0.2182179;
			wt[0]=0.1209877; wt[1]=0.0907407; wt[2]=0.0925926;
			break;
		case 10:
			nw=4;
			wi[0]=1; wi[1]=2; wi[2]=3; wi[3]=2; wi[4]=1;
				  wi[5]=2; wi[6]=4; wi[7]=4; wi[8]=2; 
					  wi[9]=3; wi[10]=4; wi[11]=3; 
						  wi[12]=2; wi[13]=2; 
							  wi[14]=1;
			break;
		case 12: // S12 quadrature
			nw=5;
			wi[0]=1; wi[1]=2; wi[2]=3; wi[3]=3; wi[4]=2; wi[5]=1;
				 wi[6]=2; wi[7]=4; wi[8]=5; wi[9]=4; wi[10]=2;
					wi[11]=3; wi[12]=5; wi[13]=5; wi[14]=3;
						 wi[15]=3; wi[16]=4; wi[17]=3;
							 wi[18]=2; wi[19]=2;
								 wi[20]=1;
			mup[0]=0.1672126;
			wt[0]=0.0707626; wt[1]=0.0558811; wt[2]=0.0373377; wt[3]=0.0502819; wt[4]=0.0258513;
			break;
		case 14:
			nw=7;
			wi[0]=1; wi[1]=2; wi[2]=3; wi[3]=4; wi[4]=3; wi[5]=2; wi[6]=1;
			   wi[7]=2; wi[8]=5; wi[9]=6; wi[10]=6; wi[11]=5; wi[12]=2;
				   wi[13]=3; wi[14]=6; wi[15]=7; wi[16]=6; wi[17]=3;
					  wi[18]=4; wi[19]=6; wi[20]=6; wi[21]=4;
						   wi[22]=3; wi[23]=5; wi[24]=3;
							   wi[25]=2; wi[26]=2;
								   wi[27]=1;
			break;
		case 16: // S16 quadrature
			nw=8;
			wi[0]=1; wi[1]=2; wi[2]=3; wi[3]=4; wi[4]=4; wi[5]=3; wi[6]=2; wi[7]=1;
			  wi[8]=2; wi[9]=5; wi[10]=6; wi[11]=7; wi[12]=6; wi[13]=5; wi[14]=2;
				  wi[15]=3; wi[16]=6; wi[17]=8; wi[18]=8; wi[19]=6; wi[20]=3;
					 wi[21]=4; wi[22]=7; wi[23]=8; wi[24]=7; wi[25]=4;
						 wi[26]=4; wi[27]=6; wi[28]=6; wi[29]=4;
							 wi[30]=3; wi[31]=5; wi[32]=3;
								 wi[33]=2; wi[34]=2;
									 wi[35]=1;
			mup[0]=0.1389568;
			wt[0]=0.0489872; wt[1]=0.0413296; wt[2]=0.0212326; wt[3]=0.0256207; wt[4]=0.0360486; wt[5]=0.0144589; wt[6]=0.0344958; wt[7]=0.0085179;
			break;
		
		default:
			cout<<"invalid quadrature\n";
			break;
		}
		
		// find weights in quadrent
		for (i=0; i<nt; i++) wp[i]=wt+wi[i]-1; // pointers to angle weights
		
		double c=2.0*(1-3*pow(mup[0],2))/(N-2);
		for (i=1; i<N/2; i++) mup[i]=sqrt(pow(mup[i-1],2)+c); // find cosines
		
		// quaderature
		Sn=N*(N+2)/8; // total number of number combinations
		mu.resize(Sn);
		eta.resize(Sn);
		xi.resize(Sn);
		w.resize(Sn);
		
		// first quadrant
		m=0;
		for (i=N/2; i>0; i--) { // first quadrant
			for (j=0; j<i; j++) {
				mu[m] = mup[i-j-1]; // cosine on x  >
				eta[m]= mup[j];     // cosine on y  <
				xi[m] = mup[N/2-i]; // cosine on z  <
				w[m]=*wp[m]*pi;
				m++;
			}
		}
		
	}
	else {
		Sn=N;
		mu.resize(Sn);
		eta.resize(Sn);
		xi.resize(Sn);
		w.resize(Sn);
		
		switch ( N ) {
		case 36: // Q36=q461214 quadrature
			
			w[ 0]=8.454511187252e-03; mu[ 0]=9.717784813336e-01; eta[ 0]=1.096881837272e-02; xi[ 0]=2.356401244281e-01;
			w[ 1]=1.913728513580e-02; mu[ 1]=9.701698603928e-01; eta[ 1]=5.695764868253e-02; xi[ 1]=2.356401244312e-01;
			w[ 2]=2.863542971348e-02; mu[ 2]=9.622473153642e-01; eta[ 2]=1.362124657777e-01; xi[ 2]=2.356401244295e-01;
			w[ 3]=3.648716160597e-02; mu[ 3]=9.410672772109e-01; eta[ 3]=2.426233944222e-01; xi[ 3]=2.356401244311e-01;
			w[ 4]=4.244873302980e-02; mu[ 4]=8.997294996538e-01; eta[ 4]=3.673697853806e-01; xi[ 4]=2.356401244316e-01;
			w[ 5]=4.642823955812e-02; mu[ 5]=8.335743322378e-01; eta[ 5]=4.996274255819e-01; xi[ 5]=2.356401244286e-01;
			w[ 6]=4.841339013884e-02; mu[ 6]=7.417637460141e-01; eta[ 6]=6.279014865859e-01; xi[ 6]=2.356401244320e-01;
			w[ 7]=4.841339013884e-02; mu[ 7]=6.279014865859e-01; eta[ 7]=7.417637460141e-01; xi[ 7]=2.356401244320e-01;
			w[ 8]=4.642823955812e-02; mu[ 8]=4.996274255819e-01; eta[ 8]=8.335743322378e-01; xi[ 8]=2.356401244286e-01;
			w[ 9]=4.244873302980e-02; mu[ 9]=3.673697853806e-01; eta[ 9]=8.997294996538e-01; xi[ 9]=2.356401244316e-01;
			w[10]=3.648716160597e-02; mu[10]=2.426233944222e-01; eta[10]=9.410672772109e-01; xi[10]=2.356401244311e-01;
			w[11]=2.863542971348e-02; mu[11]=1.362124657777e-01; eta[11]=9.622473153642e-01; xi[11]=2.356401244295e-01;
			w[12]=1.913728513580e-02; mu[12]=5.695764868253e-02; eta[12]=9.701698603928e-01; xi[12]=2.356401244312e-01;
			w[13]=8.454511187252e-03; mu[13]=1.096881837272e-02; eta[13]=9.717784813336e-01; xi[13]=2.356401244281e-01;
			w[14]=8.352354145856e-03; mu[14]=7.656319455497e-01; eta[14]=1.160393058611e-02; xi[14]=6.431742164832e-01;
			w[15]=1.873220073879e-02; mu[15]=7.633693960835e-01; eta[15]=5.995074957044e-02; xi[15]=6.431742164834e-01;
			w[16]=2.759429759588e-02; mu[16]=7.524467626583e-01; eta[16]=1.419535016004e-01; xi[16]=6.431742164829e-01;
			w[17]=3.442681426024e-02; mu[17]=7.241384940891e-01; eta[17]=2.488983098158e-01; xi[17]=6.431742164835e-01;
			w[18]=3.901232700510e-02; mu[18]=6.711819639118e-01; eta[18]=3.685670882907e-01; xi[18]=6.431742164829e-01;
			w[19]=4.130171453748e-02; mu[19]=5.909368760506e-01; eta[19]=4.869502395267e-01; xi[19]=6.431742164829e-01;
			w[20]=4.130171453748e-02; mu[20]=4.869502395267e-01; eta[20]=5.909368760506e-01; xi[20]=6.431742164829e-01;
			w[21]=3.901232700510e-02; mu[21]=3.685670882907e-01; eta[21]=6.711819639118e-01; xi[21]=6.431742164829e-01;
			w[22]=3.442681426024e-02; mu[22]=2.488983098158e-01; eta[22]=7.241384940891e-01; xi[22]=6.431742164835e-01;
			w[23]=2.759429759588e-02; mu[23]=1.419535016004e-01; eta[23]=7.524467626583e-01; xi[23]=6.431742164829e-01;
			w[24]=1.873220073879e-02; mu[24]=5.995074957044e-02; eta[24]=7.633693960835e-01; xi[24]=6.431742164834e-01;
			w[25]=8.352354145856e-03; mu[25]=1.160393058611e-02; eta[25]=7.656319455497e-01; xi[25]=6.431742164832e-01;
			w[26]=1.460888798152e-02; mu[26]=4.445439440056e-01; eta[26]=2.447911451942e-02; xi[26]=8.954225007226e-01;
			w[27]=2.995376809966e-02; mu[27]=4.288508824476e-01; eta[27]=1.196054590036e-01; xi[27]=8.954225007227e-01;
			w[28]=3.798783310581e-02; mu[28]=3.670788892962e-01; eta[28]=2.519357740235e-01; xi[28]=8.954225007226e-01;
			w[29]=3.798783310581e-02; mu[29]=2.519357740235e-01; eta[29]=3.670788892962e-01; xi[29]=8.954225007226e-01;
			w[30]=2.995376809966e-02; mu[30]=1.196054590036e-01; eta[30]=4.288508824476e-01; xi[30]=8.954225007227e-01;
			w[31]=1.460888798152e-02; mu[31]=2.447911451942e-02; eta[31]=4.445439440056e-01; xi[31]=8.954225007226e-01;
			w[32]=6.404244616724e-03; mu[32]=1.483114568272e-01; eta[32]=1.670387759191e-02; xi[32]=9.887996218887e-01;
			w[33]=1.162080754372e-02; mu[33]=1.293388490485e-01; eta[33]=7.447663982495e-02; xi[33]=9.887996218887e-01;
			w[34]=1.162080754372e-02; mu[34]=7.447663982495e-02; eta[34]=1.293388490485e-01; xi[34]=9.887996218887e-01;
			w[35]=6.404244616724e-03; mu[35]=1.670387759191e-02; eta[35]=1.483114568272e-01; xi[35]=9.887996218887e-01;
			
			break;
		case 20: // Q20=q2468 quadrature
			
			w[ 0]=2.419260514149E-02; mu[ 0]=9.713274064903E-01; eta[ 0]=3.157215799340E-02; xi[ 0]=2.356401244281E-01;
			w[ 1]=5.213067212540E-02; mu[ 1]=9.586898685237E-01; eta[ 1]=1.593344524838E-01; xi[ 1]=2.356401244307E-01;
			w[ 2]=7.185542471164E-02; mu[ 2]=9.028558915298E-01; eta[ 2]=3.596178122512E-01; xi[ 2]=2.356401244304E-01;
			w[ 3]=8.182604839076E-02; mu[ 3]=7.770210099715E-01; eta[ 3]=5.837054752370E-01; xi[ 3]=2.356401244296E-01;
			w[ 4]=8.182604839076E-02; mu[ 4]=5.837054752370E-01; eta[ 4]=7.770210099715E-01; xi[ 4]=2.356401244296E-01;
			w[ 5]=7.185542471164E-02; mu[ 5]=3.596178122512E-01; eta[ 5]=9.028558915298E-01; xi[ 5]=2.356401244304E-01;
			w[ 6]=5.213067212540E-02; mu[ 6]=1.593344524838E-01; eta[ 6]=9.586898685237E-01; xi[ 6]=2.356401244307E-01;
			w[ 7]=2.419260514149E-02; mu[ 7]=3.157215799340E-02; eta[ 7]=9.713274064903E-01; xi[ 7]=2.356401244281E-01;
			w[ 8]=2.998205782366E-02; mu[ 8]=7.645615896150E-01; eta[ 8]=4.210110375297E-02; xi[ 8]=6.431742164827E-01;
			w[ 9]=6.147460425028E-02; mu[ 9]=7.375714298063E-01; eta[ 9]=2.057068622698E-01; xi[ 9]=6.431742164831E-01;
			w[10]=7.796304620960E-02; mu[10]=6.313311043797E-01; eta[10]=4.332989313333E-01; xi[10]=6.431742164827E-01;
			w[11]=7.796304620960E-02; mu[11]=4.332989313333E-01; eta[11]=6.313311043797E-01; xi[11]=6.431742164827E-01;
			w[12]=6.147460425028E-02; mu[12]=2.057068622698E-01; eta[12]=7.375714298063E-01; xi[12]=6.431742164831E-01;
			w[13]=2.998205782366E-02; mu[13]=4.210110375297E-02; eta[13]=7.645615896150E-01; xi[13]=6.431742164827E-01;
			w[14]=2.932993043666E-02; mu[14]=4.424202396002E-01; eta[14]=4.982847370367E-02; xi[14]=8.954225007227E-01;
			w[15]=5.322055875020E-02; mu[15]=3.858240341629E-01; eta[15]=2.221674140412E-01; xi[15]=8.954225007227E-01;
			w[16]=5.322055875020E-02; mu[16]=2.221674140412E-01; eta[16]=3.858240341629E-01; xi[16]=8.954225007227E-01;
			w[17]=2.932993043666E-02; mu[17]=4.982847370367E-02; eta[17]=4.424202396002E-01; xi[17]=8.954225007227E-01;
			w[18]=1.802505216045E-02; mu[18]=1.409476441875E-01; eta[18]=4.908227124734E-02; xi[18]=9.887996218887E-01;
			w[19]=1.802505216045E-02; mu[19]=4.908227124734E-02; eta[19]=1.409476441875E-01; xi[19]=9.887996218887E-01;
			
			break;
		default:
			cout<<"invalid quadrature\n";
			break;
		}
		
		// first quadrant
		for (m=0; m<N; m++) { // first quadrant
			w[m]=w[m]*pi;
		}
		
	}
	double test=0.0;
	for (m=0; m<Sn; m++) test+=w[m];
	cout<<"weight test "<<test<<endl;
	// normalize weight
	for (m=0; m<Sn; m++) w[m]*=pi/test;
	
	return 0;
}

//======================================================================================//
//++ Function to solve transport in a single general cell ++++++++++++++++++++++++++++++//
//======================================================================================//
inline void HOSolver::cellSolution(double psiInB, double psiInL, double SA, double sigma, int i, int j, int m, double& psiOutT, double& psiOutR, double& psiA  )
{
	double epsilon, exp_epsilon, epsilon_2, du;
	double psiA1, psiA2, psiA3;
	double A1, A2;
	
	double LT=hx[i];
	double LR=hy[j];
	
	double imup=1/sqrt(1-xi[m]*xi[m]);             // 1/mu'
	double muc=LT/sqrt(LT*LT+LR*LR);        // mu of cell
	double muInPlane=mu[m]/sqrt(mu[m]*mu[m]+eta[m]*eta[m]); // projection onto x-y plane
	
	if ( abs(muInPlane-muc)<1e-15 ) { // ray passes through both corners of cell
		du=sqrt(LT*LT+LR*LR);
		if ( sigma<1e-10 ) {
			// triangle A
			psiOutT=psiInL+SA*du*imup*0.5; // find out going angular flux
			psiA1  =psiInL+SA*du*imup/3.0; // find cell angular flux
			
			// triangle C
			psiOutR=psiInB+SA*du*imup*0.5; // find out going angular flux
			psiA3  =psiInB+SA*du*imup/3.0; // find cell angular flux
			
			psiA=0.5*(psiInL+psiInB)+SA*du*imup/3.0;
		}
		else {
			epsilon=sigma*du*imup; // optical thickness
			SA/=sigma;
			exp_epsilon=exp(-epsilon); // exponent of epsilon
			epsilon_2=epsilon*epsilon; // square of epsilon
			// triangle A
			psiOutT=(psiInL*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1))/epsilon; // find out going angular flux
			psiA1=2*(psiInL*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon))/epsilon_2; // find cell angular flux
			
			// triangle C
			psiOutR=(psiInB*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1))/epsilon; // find out going angular flux
			psiA3=2*(psiInB*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon))/epsilon_2; // find cell angular flux
			
			psiA=((psiInL+psiInB)*(epsilon+exp_epsilon-1.0)+2.0*SA*(1.0+0.5*epsilon_2-exp_epsilon-epsilon))/epsilon_2;
		}
		
		// total
		//psiOutT=psiOut1;
		//psiOutR=psiOut3;
		//psiA=0.5*(psiA1+psiA3);
		
		//cout<<"balance 1"<<endl;
		//cout<<"T1 balance "<<SA*du/mup-2*(psiOutT-psiInL)-epsilon*psiA1<<endl; // triangle 1 balance
		//cout<<"T3 balance "<<SA*du/mup-2*(psiOutR-psiInB)-epsilon*psiA3<<endl; // triangle 3 balance
		//cout<<"cell angular balance "<<SA*LT*LR-mu[m]*LR*(psiOutR-psiInL)-eta[m]*LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
		//cout<<"cell ang bal "<<SA*LT*LR-LR*(psiOutR-psiInL)-LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
		//cout<<SA*du-mup*(psiOutR-psiInL)-mup*(psiOutT-psiInB)-sigma*du*psiA<<endl;
	}
	else if ( muInPlane<muc ) { // ray splits the top and bottom of the cell
		double psiOut1, psiOut2;
		double Lout1, Lout2;
		Lout1=muInPlane*LR/sqrt(1.0-muInPlane*muInPlane);
		du=Lout1/muInPlane;
		A1=Lout1*LR*0.5; // Triangle 1 Area
		Lout2=LT-Lout1;
		A2=LT*LR-2*A1; // Parallelogram 2 Area
		if ( sigma<1e-10 ) {
			// triangle A
			psiOut1=psiInL+SA*du*imup*0.5; // find out going angular flux
			psiA1  =psiInL+SA*du*imup/3.0; // find cell angular flux
			
			// parallelogram B
			psiOut2=psiInB+SA*du*imup;   // find out going angular flux
			psiA2  =psiInB+SA*du*imup*0.5; // find cell angular flux
			
			// triangle C
			psiOutR=psiA2;              // find out going angular flux
			psiA3  =psiInB+SA*du*imup/3.0; // find cell angular flux
		}
		else {
			epsilon=sigma*du*imup; // optical thickness
			SA/=sigma;
			exp_epsilon=exp(-epsilon); // exponent of epsilon
			epsilon_2=epsilon*epsilon; // square of epsilon
			// triangle A
			psiOut1=(psiInL*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1))/epsilon; // find out going angular flux
			psiA1=2*(psiInL*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon))/epsilon_2; // find cell angular flux
			
			// parallelogram B
			psiOut2=psiInB*exp_epsilon+SA*(1-exp_epsilon);                       // find out going angular flux
			psiA2  =(psiInB*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1))/epsilon; // find cell angular flux
			
			// triangle C
			psiOutR=psiA2;                                                                                       // find out going angular flux
			psiA3  =2*(psiInB*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon))/epsilon_2; // find cell angular flux
			
		}
		
		// total
		psiOutT=(Lout1*psiOut1+Lout2*psiOut2)/LT;
		psiA=(A1*psiA1+A2*psiA2+A1*psiA3)/(LT*LR);
		
		//cout<<"balance 2"<<endl;
		//cout<<"T1 balance "<<SA*du/mup-2*(psiOut1-psiInL)-epsilon*psiA1<<endl; // triangle 1 balance
		//cout<<"P2 balance "<<SA*du/mup-(psiOut2-psiInB)-epsilon*psiA2<<endl; // parallelogram 2 balance
		//cout<<"T3 balance "<<SA*du/mup-2*(psiOutR-psiInB)-epsilon*psiA3<<endl; // triangle 3 balance
		//cout<<"cell angular balance "<<SA*LT*LR-mu[m]*LR*(psiOutR-psiInL)-eta[m]*LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
	}
	else { // ray splits the right and left side of the cell
		double psiOut2, psiOut3;
		double Lout2, Lout3;
		du=LT/muInPlane;
		A1=sqrt(du*du-LT*LT)*LT*0.5; // Triangle 1 Area
		Lout3=sqrt(du*du-LT*LT); // Triangle 3 Length
		Lout2=LR-Lout3;
		A2=LT*LR-2*A1; // Parallelogram 2 Area
		
		if ( sigma<1e-10 ) {
			// triangle A
			psiOutT=psiInL+SA*du*imup*0.5; // find out going angular flux
			psiA1  =psiInL+SA*du*imup/3.0; // find cell angular flux
			
			// parallelogram B
			psiOut2=psiInL+SA*du*imup; // find out going angular flux
			psiA2  =psiOutT;          // find cell angular flux
			
			// triangle C
			psiOut3=psiInB+SA*du*imup*0.5; // find out going angular flux
			psiA3  =psiInB+SA*du*imup*0.5; // find cell angular flux
			
		}
		else {
			epsilon=sigma*du*imup; // optical thickness
			SA/=sigma;
			exp_epsilon=exp(-epsilon); // exponent of epsilon
			epsilon_2=epsilon*epsilon; // square of epsilon
			// triangle A
			psiOutT=(psiInL*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1))/epsilon; // find out going angular flux
			psiA1=2*(psiInL*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon))/epsilon_2; // find cell angular flux
			
			// parallelogram B
			psiOut2=psiInL*exp_epsilon+SA*(1-exp_epsilon); // find out going angular flux
			psiA2=psiOutT;                                       // find cell angular flux
			
			// triangle C
			psiOut3=(psiInB*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1))/epsilon; // find out going angular flux
			psiA3=2*(psiInB*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon))/epsilon_2; // find cell angular flux
		}
		
		// total
		psiOutR=(Lout3*psiOut3+Lout2*psiOut2)/LR;
		psiA=(A1*psiA1+A2*psiA2+A1*psiA3)/(LT*LR);
		
		//cout<<"balance 3"<<endl;
		//cout<<"T1 balance "<<SA*du/mup-2*(psiOutT-psiInL)-epsilon*psiA1<<endl; // triangle 1 balance
		//cout<<"P2 balance "<<SA*du/mup-(psiOut2-psiInL)-epsilon*psiA2<<endl; // parallelogram 2 balance
		//cout<<"T3 balance "<<SA*du/mup-2*(psiOut3-psiInB)-epsilon*psiA3<<endl; // triangle 3 balance
		//cout<<"cell angular balance "<<SA*LT*LR-mu[m]*LR*(psiOutR-psiInL)-eta[m]*LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
	}
	//cout<<"cell angular balance "<<SA*LT*LR-mu[m]*LR*(psiOutR-psiInL)-eta[m]*LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
}
//======================================================================================//

