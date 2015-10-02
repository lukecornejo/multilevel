#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <time.h>  /* clock_t, clock, CLOCKS_PER_SEC */
#include <vector>
#include <stdlib.h>
#include "HO_1.6.h"
#include "LO_1.6.h"
#include "IO_1.6.h"
#include <omp.h>
//#include "H5Cpp.h"

using namespace std;
typedef ::Triplet T;

extern bool reflectiveB, reflectiveT, reflectiveL, reflectiveR;
extern bool KE_problem; // option variables
extern double k_eff;
extern int kbc;

//**************************************************************************************//
//**************************************************************************************//
//======================================================================================//
//++++++++++++++++++++++++++++++++ Low Order Solution ++++++++++++++++++++++++++++++++++//
//======================================================================================//
//**************************************************************************************//
//**************************************************************************************//

std::string LOSolution::writeLOSolverType() {
		if ( NDASolution ) return "Using Nonlinear Diffusion Acceleration for Low Order Problem (NDA) \n";
		else               return "Using Quasi Diffusion for Low Order Problem (QD) \n";
}

//++ Initialize Low Order Flux Solution ++++++++++++++++++++++++++++++++
void LOSolution::initializeSolution() {
	
	double pi=3.14159265358979323846;
	
	// Initialize Array Memory
	vector< vector<double> > grid_d(Nx,vector<double>(Ny, 0.0));
	vector< vector<double> > xgrid_d(Nx+1,vector<double>(Ny, 0.0)); // cell edge values on x grid
	vector< vector<double> > ygrid_d(Nx,vector<double>(Ny+1, 0.0)); // cell edge flux on y grid
	
	sourceScatter.resize(eta_star);
	phi.resize(eta_star);
	j_x.resize(eta_star);
	j_y.resize(eta_star);
	for (int k=0; k<eta_star; k++) { // Energy Grids
		sourceScatter[k]=grid_d;
		phi[k].resize(Ng[k]);
		j_x[k].resize(Ng[k]);
		j_y[k].resize(Ng[k]);
		for (int g=0; g<Ng[k]; g++) {
			phi[k][g]=grid_d;
			j_x[k][g]=xgrid_d;
			j_y[k][g]=ygrid_d;
		}
	}
	
	// Boundary Values
	FB.resize(eta_star);
	FL.resize(eta_star);
	for (int k=0; k<eta_star; k++) { // Energy Grids
		FB[k].resize(Ng[k]);
		FL[k].resize(Ng[k]);

		for (int g=0; g<Ng[k]; g++) { // Energy Groups
			FB[k][g].resize(Nx);
			FL[k][g].resize(Ny);
		}
	}
	
	sourceFission=sourceScatter;
	kappaLast=sourceScatter;
	phiLast=phi;
	
	FT=FB;
	FR=FL;
	jInB=FB;
	jInT=FT;
	jInL=FL;
	jInR=FR;
	phiInB=FB;
	phiInT=FT;
	phiInL=FL;
	phiInR=FR;
	
	if ( NDASolution ) {
		
		phiL=FL;
		phiR=FR;
		phiB=FB;
		phiT=FT;
		
		D_xP=j_x;
		D_xN=j_x;

		D_yP=j_y;
		D_yN=j_y;
		
		// Initialize Values
		for (int g=0; g<Ng[0]; g++) { // Energy Groups
			for (int i=0; i<Nx; i++) {
				phiB[0][g][i]=4*pi/Ng[0];
				phiT[0][g][i]=4*pi/Ng[0];
			}
			for (int j=0; j<Ny; j++) {
				phiL[0][g][j]=4*pi/Ng[0];
				phiR[0][g][j]=4*pi/Ng[0];
			}
		}
		
		// Coarse Group flux
		for (int k=1; k<eta_star; k++) {
			//# pragma omp parallel for
			for (int g=0; g<Ng[k]; g++) {
				for (int i=0; i<Nx; i++) {
					phiB[k][g][i]=0;
					phiT[k][g][i]=0;
					for (int gg=omegaP[k][g]; gg<omegaP[k][g+1]; gg++) {
						phiB[k][g][i]+=phiB[k-1][gg][i];
						phiT[k][g][i]+=phiT[k-1][gg][i];
					}
				}
				for (int j=0; j<Ny; j++) {
					phiL[k][g][j]=0;
					phiR[k][g][j]=0;
					for (int gg=omegaP[k][g]; gg<omegaP[k][g+1]; gg++) {
						phiL[k][g][j]+=phiL[k-1][g][j];
						phiR[k][g][j]+=phiR[k-1][g][j];
					}
				}
			}
		}
	}
	else {
		// QD Solution and Factors
		phi_x=j_x;
		phi_y=j_y;
		
		D_xxC=D_yyC=phi;
		D_xxL=D_yyL=D_xyL=phi;
		D_xxR=D_yyR=D_xyR=phi;
		D_xxB=D_yyB=D_xyB=phi;
		D_xxT=D_yyT=D_xyT=phi;
		
		// Set initial values for flux
		for (int g=0; g<Ng[0]; g++) { // Energy Groups
			for (int i=0; i<Nx+1; i++) for (int j=0; j<Ny; j++)   phi_x[0][g][i][j]=4*pi/Ng[0];
			for (int i=0; i<Nx; i++)   for (int j=0; j<Ny+1; j++) phi_y[0][g][i][j]=4*pi/Ng[0];
		}
		
		// Coarse Group flux
		for (int k=1; k<eta_star; k++) {
			//# pragma omp parallel for
			for (int g=0; g<Ng[k]; g++) {
				for (int i=0; i<Nx+1; i++) {
					for (int j=0; j<Ny; j++) {
						for (int gg=omegaP[k][g]; gg<omegaP[k][g+1]; gg++) phi_x[k][g][i][j]+=phi_x[k-1][gg][i][j];
					}
				}
				for (int i=0; i<Nx; i++) {
					for (int j=0; j<Ny+1; j++) {
						for (int gg=omegaP[k][g]; gg<omegaP[k][g+1]; gg++) phi_y[k][g][i][j]+=phi_y[k-1][gg][i][j];
					}
				}
			}
		}
	}
	
	// Initialize Values
	for (int g=0; g<Ng[0]; g++) { // Energy Groups
		for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) phi[0][g][i][j]=4*pi/Ng[0];
	}
	
	// Coarse Group flux
	for (int k=1; k<eta_star; k++) {
		//# pragma omp parallel for
		for (int g=0; g<Ng[k]; g++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					phi[k][g][i][j]=0;
					for (int gg=omegaP[k][g]; gg<omegaP[k][g+1]; gg++) phi[k][g][i][j]+=phi[k-1][gg][i][j];
				}
			}
		}
	}
}

//======================================================================================//
//++ Calculate Low Order Factor From HO Factors and Solution +++++++++++++++++++++++++++//
//======================================================================================//
void LOSolution::LOFromHOFactors(HOSolver &ho) {
	double pi=3.14159265358979323846;
	
	for (int g=0; g<Ng[0]; g++) {
		for (int j=0; j<Ny; j++) { // Left and Right
			if ( reflectiveL ) { // Left
				FL[0][g][j]=0.0;
				jInL[0][g][j]=0.0;
				phiInL[0][g][j]=0.0;
			}
			else {
				phiInL[0][g][j]=2.0*pi*ho.bcL[g][j];
				  jInL[0][g][j]=   -pi*ho.bcL[g][j];
					FL[0][g][j]=(-ho.j_x[g][0][j]-jInL[0][g][j])/(ho.phi_x[g][0][j]-phiInL[0][g][j]); // Left F
			}
			if ( reflectiveR ) { // Right
				FR[0][g][j]=0.0;
				jInR[0][g][j]=0.0;
				phiInR[0][g][j]=0.0;
			}
			else {
				phiInR[0][g][j]=2.0*pi*ho.bcR[g][j];
				  jInR[0][g][j]=   -pi*ho.bcR[g][j];
					FR[0][g][j]=(ho.j_x[g][Nx][j]-jInR[0][g][j])/(ho.phi_x[g][Nx][j]-phiInR[0][g][j]); // Right F
			}
		}
		
		for (int i=0; i<Nx; i++ ) { // Bottom and Top
			if ( reflectiveB ) { // Bottom
				FB[0][g][i]=0.0;
				jInB[0][g][i]=0.0;
				phiInB[0][g][i]=0.0;
			}
			else {
				phiInB[0][g][i]=2.0*pi*ho.bcB[g][i];
				  jInB[0][g][i]=   -pi*ho.bcB[g][i];
					FB[0][g][i]=(-ho.j_y[g][i][0]-jInB[0][g][i])/(ho.phi_y[g][i][0]-phiInB[0][g][i]); // Bottom F
			}
			if ( reflectiveT ) { // Top
				FT[0][g][i]=0.0;
				jInT[0][g][i]=0.0;
				phiInT[0][g][i]=0.0;
			}
			else {
				phiInT[0][g][i]=2.0*pi*ho.bcT[g][i];
				  jInT[0][g][i]=   -pi*ho.bcT[g][i];
					FT[0][g][i]=(ho.j_y[g][i][Ny]-jInT[0][g][i])/(ho.phi_y[g][i][Ny]-phiInT[0][g][i]); // Top F

			}
		}
	}
	
	if ( NDASolution ) {
		//# pragma omp parallel for
		for (int g=0; g<Ng[0]; g++) {
			
			for (int i=0; i<Nx+1; i++) for (int j=0; j<Ny; j++) D_xP[0][g][i][j]=ho.D_x[g][i][j]-0.5*ho.D_xT[g][i][j]*ho.xe[i]; // D_xP X Grid
			for (int i=0; i<Nx+1; i++) for (int j=0; j<Ny; j++) D_xN[0][g][i][j]=ho.D_x[g][i][j]+0.5*ho.D_xT[g][i][j]*ho.xe[i]; // D_xN X Grid
			for (int i=0; i<Nx; i++) for (int j=0; j<Ny+1; j++) D_yP[0][g][i][j]=ho.D_y[g][i][j]-0.5*ho.D_yT[g][i][j]*ho.ye[j]; // D_yP Y Grid
			for (int i=0; i<Nx; i++) for (int j=0; j<Ny+1; j++) D_yN[0][g][i][j]=ho.D_y[g][i][j]+0.5*ho.D_yT[g][i][j]*ho.ye[j]; // D_yN Y Grid
			
		}
	}
	else {
		//# pragma omp parallel for
		for (int g=0; g<Ng[0]; g++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					double sigmaT=ho.Total(g,i,j);
					D_xxC[0][g][i][j]=ho.E_xx[g][i][j]/sigmaT;
					D_yyC[0][g][i][j]=ho.E_yy[g][i][j]/sigmaT;
					
					D_xxL[0][g][i][j]=ho.E_xx_x[g][i][j]/sigmaT;
					D_yyL[0][g][i][j]=ho.E_yy_x[g][i][j]/sigmaT;
					D_xyL[0][g][i][j]=ho.E_xy_x[g][i][j]/sigmaT;
					
					D_xxR[0][g][i][j]=ho.E_xx_x[g][i+1][j]/sigmaT;
					D_yyR[0][g][i][j]=ho.E_yy_x[g][i+1][j]/sigmaT;
					D_xyR[0][g][i][j]=ho.E_xy_x[g][i+1][j]/sigmaT;
					
					D_xxB[0][g][i][j]=ho.E_xx_y[g][i][j]/sigmaT;
					D_yyB[0][g][i][j]=ho.E_yy_y[g][i][j]/sigmaT;
					D_xyB[0][g][i][j]=ho.E_xy_y[g][i][j]/sigmaT;
					
					D_xxT[0][g][i][j]=ho.E_xx_y[g][i][j+1]/sigmaT;
					D_yyT[0][g][i][j]=ho.E_yy_y[g][i][j+1]/sigmaT;
					D_xyT[0][g][i][j]=ho.E_xy_y[g][i][j+1]/sigmaT;
					
				}
			}
		}
	}
	
}

//======================================================================================//
//++ Collapse Grid Flux ++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolution::collapseSolution(int etaL) {
	
	//# pragma omp parallel for
	for (int g=0; g<Ng[etaL]; g++) {
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				phi[etaL][g][i][j]=0;
				for (int gg=omegaP[etaL][g]; gg<omegaP[etaL][g+1]; gg++) phi[etaL][g][i][j]+=phi[etaL-1][gg][i][j];
			}
		}
		for (int i=0; i<Nx+1; i++) {
			for (int j=0; j<Ny; j++) {
				j_x[etaL][g][i][j]=0;
				for (int gg=omegaP[etaL][g]; gg<omegaP[etaL][g+1]; gg++) j_x[etaL][g][i][j]+=j_x[etaL-1][gg][i][j];
			}
		}
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny+1; j++) {
				j_y[etaL][g][i][j]=0;
				for (int gg=omegaP[etaL][g]; gg<omegaP[etaL][g+1]; gg++) j_y[etaL][g][i][j]+=j_y[etaL-1][gg][i][j];
			}
		}
	}
	
	if ( NDASolution ) {
		//# pragma omp parallel for
		for (int g=0; g<Ng[etaL]; g++) {
			for (int i=0; i<Nx; i++) {
				phiB[etaL][g][i]=0;
				phiT[etaL][g][i]=0;
				for (int gg=omegaP[etaL][g]; gg<omegaP[etaL][g+1]; gg++) {
					phiB[etaL][g][i]+=phiB[etaL-1][gg][i];
					phiT[etaL][g][i]+=phiT[etaL-1][gg][i];
				}
			}
			for (int j=0; j<Ny; j++) {
				phiL[etaL][g][j]=0;
				phiR[etaL][g][j]=0;
				for (int gg=omegaP[etaL][g]; gg<omegaP[etaL][g+1]; gg++) {
					phiL[etaL][g][j]+=phiL[etaL-1][g][j];
					phiR[etaL][g][j]+=phiR[etaL-1][g][j];
				}
			}
		}
	}
	else {
		//# pragma omp parallel for
		for (int g=0; g<Ng[etaL]; g++) {
			for (int i=0; i<Nx+1; i++) {
				for (int j=0; j<Ny; j++) {
					phi_x[etaL][g][i][j]=0;
					for (int gg=omegaP[etaL][g]; gg<omegaP[etaL][g+1]; gg++) phi_x[etaL][g][i][j]+=phi_x[etaL-1][gg][i][j];
				}
			}
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny+1; j++) {
					phi_y[etaL][g][i][j]=0;
					for (int gg=omegaP[etaL][g]; gg<omegaP[etaL][g+1]; gg++) phi_y[etaL][g][i][j]+=phi_y[etaL-1][gg][i][j];
				}
			}
		}
	}
}


//======================================================================================//
//++ Find group average Coefficient ++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolution::averageFactors(int etaL) {
	vector< vector< vector<double> > > &phiG=phi[etaL];
	double Ssum, phiS, phiP, phiN;
	// Calculate Group Averaged values ////////////////////////////////////////////////////////////
	if ( NDASolution ) {
		// X Grid Diffusion consistency terms
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int i=1; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					phiP=0.0; phiN=0.0;
					D_xP[etaL+1][p][i][j]=0.0;
					D_xN[etaL+1][p][i][j]=0.0;
					for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
						phiP+=phiG[g][i][j];
						phiN+=phiG[g][i-1][j];
						D_xP[etaL+1][p][i][j]+=D_xP[etaL][g][i][j]*phiG[g][i][j];
						D_xN[etaL+1][p][i][j]+=D_xN[etaL][g][i][j]*phiG[g][i-1][j];
					}
					D_xP[etaL+1][p][i][j]/=phiP;
					D_xN[etaL+1][p][i][j]/=phiN;
				}
			}
		}
		// Y Grid Diffusion consistency terms
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int i=0; i<Nx; i++) {
				for (int j=1; j<Ny; j++) {
					phiP=0.0; phiN=0.0;
					D_yP[etaL+1][p][i][j]=0.0;
					D_yN[etaL+1][p][i][j]=0.0;
					for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
						phiP+=phiG[g][i][j];
						phiN+=phiG[g][i][j-1];
						D_yP[etaL+1][p][i][j]+=D_yP[etaL][g][i][j]*phiG[g][i][j];
						D_yN[etaL+1][p][i][j]+=D_yN[etaL][g][i][j]*phiG[g][i][j-1];
					}
					D_yP[etaL+1][p][i][j]/=phiP;
					D_yN[etaL+1][p][i][j]/=phiN;
				}
			}
		}
		// Left and Right Boundary Conditions
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int j=0; j<Ny+1; j++) {
				phiP=0.0; phiN=0.0;
				FL[etaL+1][p][j]=0.0;
				FR[etaL+1][p][j]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiN+=phiL[etaL][g][j];
					phiP+=phiR[etaL][g][j];
					FL[etaL+1][p][j]+=FL[etaL][g][j]*phiL[etaL][g][j];
					FR[etaL+1][p][j]+=FR[etaL][g][j]*phiR[etaL][g][j];
				}
				FL[etaL+1][p][j]/=phiN;
				FR[etaL+1][p][j]/=phiP;
				
				phiP=0.0; phiN=0.0;
				D_xP[etaL+1][p][0][j]=0.0;
				D_xN[etaL+1][p][0][j]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiP+=phiG[g][0][j];
					phiN+=phiL[etaL][g][j];
					D_xP[etaL+1][p][0][j]+=D_xP[etaL][g][0][j]*phiG[g][0][j];
					D_xN[etaL+1][p][0][j]+=D_xN[etaL][g][0][j]*phiL[etaL][g][j];
				}
				D_xP[etaL+1][p][0][j]/=phiP;
				D_xN[etaL+1][p][0][j]/=phiN;
				
				phiP=0.0; phiN=0.0;
				D_xP[etaL+1][p][Nx][j]=0.0;
				D_xN[etaL+1][p][Nx][j]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiP+=phiR[etaL][g][j];
					phiN+=phiG[g][Nx-1][j];
					D_xP[etaL+1][p][Nx][j]+=D_xP[etaL][g][Nx][j]*phiR[etaL][g][j];
					D_xN[etaL+1][p][Nx][j]+=D_xN[etaL][g][Nx][j]*phiG[g][Nx-1][j];
				}
				D_xP[etaL+1][p][Nx][j]/=phiP;
				D_xN[etaL+1][p][Nx][j]/=phiN;
			}
		}
		// Bottom and Top Boundary Conditions
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int i=0; i<Nx; i++) {
				phiP=0.0; phiN=0.0;
				FB[etaL+1][p][i]=0.0;
				FT[etaL+1][p][i]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiN+=phiB[etaL][g][i];
					phiP+=phiT[etaL][g][i];
					FB[etaL+1][p][i]+=FB[etaL][g][i]*phiB[etaL][g][i];
					FT[etaL+1][p][i]+=FT[etaL][g][i]*phiT[etaL][g][i];
				}
				FB[etaL+1][p][i]/=phiN;
				FT[etaL+1][p][i]/=phiP;
				
				phiP=0.0; phiN=0.0;
				D_yP[etaL+1][p][i][0]=0.0;
				D_yN[etaL+1][p][i][0]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiP+=phiG[g][i][0];
					phiN+=phiB[etaL][g][i];
					D_yP[etaL+1][p][i][0]+=D_yP[etaL][g][i][0]*phiG[g][i][0];
					D_yN[etaL+1][p][i][0]+=D_yN[etaL][g][i][0]*phiB[etaL][g][i];
				}
				D_yP[etaL+1][p][i][0]/=phiP;
				D_yN[etaL+1][p][i][0]/=phiN;
				
				phiP=0.0; phiN=0.0;
				D_yP[etaL+1][p][i][Ny]=0.0;
				D_yN[etaL+1][p][i][Ny]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiP+=phiT[etaL][g][i];
					phiN+=phiG[g][i][Ny-1];
					D_yP[etaL+1][p][i][Ny]+=D_yP[etaL][g][i][Ny]*phiT[etaL][g][i];
					D_yN[etaL+1][p][i][Ny]+=D_yN[etaL][g][i][Ny]*phiG[g][i][Ny-1];
				}
				D_yP[etaL+1][p][i][Ny]/=phiP;
				D_yN[etaL+1][p][i][Ny]/=phiN;
			}
		}
	}
	else {
		vector< vector< vector<double> > > &phi_xG=phi_x[etaL], &phi_yG=phi_y[etaL];
		// QD Factors
		// Cell Center Diffusion consistency terms
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					phiP=0.0;
					D_xxC[etaL+1][p][i][j]=0.0;
					D_yyC[etaL+1][p][i][j]=0.0;
					for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
						phiP+=phiG[g][i][j];
						D_xxC[etaL+1][p][i][j]+=D_xxC[etaL][g][i][j]*phiG[g][i][j];
						D_yyC[etaL+1][p][i][j]+=D_yyC[etaL][g][i][j]*phiG[g][i][j];
					}
					D_xxC[etaL+1][p][i][j]/=phiP;
					D_yyC[etaL+1][p][i][j]/=phiP;
				}
			}
		}
		// X Grid Diffusion consistency terms
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					phiP=0.0; phiN=0.0;
					D_xxL[etaL+1][p][i][j]=0.0;
					D_yyL[etaL+1][p][i][j]=0.0;
					D_xyL[etaL+1][p][i][j]=0.0;
					D_xxR[etaL+1][p][i][j]=0.0;
					D_yyR[etaL+1][p][i][j]=0.0;
					D_xyR[etaL+1][p][i][j]=0.0;
					for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
						phiN+=phi_xG[g][i][j];
						phiP+=phi_xG[g][i+1][j];
						D_xxL[etaL+1][p][i][j]+=D_xxL[etaL][g][i][j]*phi_xG[g][i][j];
						D_yyL[etaL+1][p][i][j]+=D_yyL[etaL][g][i][j]*phi_xG[g][i][j];
						D_xyL[etaL+1][p][i][j]+=D_xyL[etaL][g][i][j]*phi_xG[g][i][j];
						D_xxR[etaL+1][p][i][j]+=D_xxR[etaL][g][i][j]*phi_xG[g][i+1][j];
						D_yyR[etaL+1][p][i][j]+=D_yyR[etaL][g][i][j]*phi_xG[g][i+1][j];
						D_xyR[etaL+1][p][i][j]+=D_xyR[etaL][g][i][j]*phi_xG[g][i+1][j];
					}
					D_xxL[etaL+1][p][i][j]/=phiN;
					D_yyL[etaL+1][p][i][j]/=phiN;
					D_xyL[etaL+1][p][i][j]/=phiN;
					D_xxR[etaL+1][p][i][j]/=phiP;
					D_yyR[etaL+1][p][i][j]/=phiP;
					D_xyR[etaL+1][p][i][j]/=phiP;
				}
			}
		}
		// Y Grid Diffusion consistency terms
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					phiP=0.0; phiN=0.0;
					D_xxB[etaL+1][p][i][j]=0.0;
					D_yyB[etaL+1][p][i][j]=0.0;
					D_xyB[etaL+1][p][i][j]=0.0;
					D_xxT[etaL+1][p][i][j]=0.0;
					D_yyT[etaL+1][p][i][j]=0.0;
					D_xyT[etaL+1][p][i][j]=0.0;
					for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
						phiN+=phi_yG[g][i][j];
						phiP+=phi_yG[g][i][j+1];
						D_xxB[etaL+1][p][i][j]+=D_xxB[etaL][g][i][j]*phi_yG[g][i][j];
						D_yyB[etaL+1][p][i][j]+=D_yyB[etaL][g][i][j]*phi_yG[g][i][j];
						D_xyB[etaL+1][p][i][j]+=D_xyB[etaL][g][i][j]*phi_yG[g][i][j];
						D_xxT[etaL+1][p][i][j]+=D_xxT[etaL][g][i][j]*phi_yG[g][i][j+1];
						D_yyT[etaL+1][p][i][j]+=D_yyT[etaL][g][i][j]*phi_yG[g][i][j+1];
						D_xyT[etaL+1][p][i][j]+=D_xyT[etaL][g][i][j]*phi_yG[g][i][j+1];
					}
					
					D_xxB[etaL+1][p][i][j]/=phiN;
					D_yyB[etaL+1][p][i][j]/=phiN;
					D_xyB[etaL+1][p][i][j]/=phiN;
					D_xxT[etaL+1][p][i][j]/=phiP;
					D_yyT[etaL+1][p][i][j]/=phiP;
					D_xyT[etaL+1][p][i][j]/=phiP;
				}
			}
		}
		// Left and Right Boundary Conditions
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int j=0; j<Ny+1; j++) {
				phiP=0.0; phiN=0.0;
				FL[etaL+1][p][j]=0.0;
				FR[etaL+1][p][j]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiN+=phi_xG[g][0][j];
					phiP+=phi_xG[g][Nx][j];
					FL[etaL+1][p][j]+=FL[etaL][g][j]*phi_xG[g][0][j];
					FR[etaL+1][p][j]+=FR[etaL][g][j]*phi_xG[g][Nx][j];
				}
				FL[etaL+1][p][j]/=phiN;
				FR[etaL+1][p][j]/=phiP;
				
				
				jInL[etaL+1][p][j]=0.0;
				jInR[etaL+1][p][j]=0.0;
				phiInL[etaL+1][p][j]=0.0;
				phiInR[etaL+1][p][j]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					jInL[etaL+1][p][j]+=jInL[etaL][g][j];
					jInR[etaL+1][p][j]+=jInR[etaL][g][j];
					phiInL[etaL+1][p][j]+=phiInL[etaL][g][j];
					phiInR[etaL+1][p][j]+=phiInR[etaL][g][j];
				}
			}
		}
		// Bottom and Top Boundary Conditions
		//# pragma omp parallel for
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int i=0; i<Nx; i++) {
				phiP=0.0; phiN=0.0;
				FB[etaL+1][p][i]=0.0;
				FT[etaL+1][p][i]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiN+=phi_yG[g][i][0];
					phiP+=phi_yG[g][i][Ny];
					FB[etaL+1][p][i]+=FB[etaL][g][i]*phi_yG[g][i][0];
					FT[etaL+1][p][i]+=FT[etaL][g][i]*phi_yG[g][i][Ny];
				}
				FB[etaL+1][p][i]/=phiN;
				FT[etaL+1][p][i]/=phiP;
				
				
				jInB[etaL+1][p][i]=0.0;
				jInT[etaL+1][p][i]=0.0;
				phiInB[etaL+1][p][i]=0.0;
				phiInT[etaL+1][p][i]=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					jInB[etaL+1][p][i]+=jInB[etaL][g][i];
					jInT[etaL+1][p][i]+=jInT[etaL][g][i];
					phiInB[etaL+1][p][i]+=phiInB[etaL][g][i];
					phiInT[etaL+1][p][i]+=phiInT[etaL][g][i];
				}
			}
		}
	}
}

void LOSolution::normalizeEigen(int etaL) {
	double sum=0.0;
	vector< vector< vector<double> > > &phiG=phi[etaL];
	vector< vector< vector<double> > > &j_xG=j_x[etaL], &j_yG=j_y[etaL];
	
	// Find coefficient
	for (int g=0; g<Ng[etaL]; g++) {
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) sum+=hx[i]*hy[j]*phiG[g][i][j];
		}
	}
	sum/=(x[Nx]-x[0])*(y[Ny]-y[0]);
	// Normalize
	for (int g=0; g<Ng[etaL]; g++) {
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) phiG[g][i][j]/=sum;
		}
		//# pragma omp parallel for
		for (int i=0; i<Nx+1; i++) {
			for (int j=0; j<Ny; j++) j_xG[g][i][j]/=sum;
		}
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny+1; j++) j_yG[g][i][j]/=sum;
		}
	}
	
	if ( NDASolution ) {
		vector< vector<double> > &phiBG=phiB[etaL], &phiTG=phiT[etaL], &phiLG=phiL[etaL], &phiRG=phiR[etaL];
		// Normalize NDA
		for (int g=0; g<Ng[etaL]; g++) {
			//# pragma omp parallel for
			for (int i=0; i<Nx; i++) {
				phiBG[g][i]/=sum;
				phiTG[g][i]/=sum;
			}
			//# pragma omp parallel for
			for (int j=0; j<Ny; j++) {
				phiLG[g][j]/=sum;
				phiRG[g][j]/=sum;
			}
		}
	}
	else {
		vector< vector< vector<double> > > &phi_xG=phi_x[etaL], &phi_yG=phi_y[etaL];
		// Normalize QD
		for (int g=0; g<Ng[etaL]; g++) {
			//# pragma omp parallel for
			for (int i=0; i<Nx+1; i++) {
				for (int j=0; j<Ny; j++) phi_xG[g][i][j]/=sum;
			}
			//# pragma omp parallel for
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny+1; j++) phi_yG[g][i][j]/=sum;
			}
		}
	}
}

void LOSolution::calculateGroupCurrents(int etaL, int g) {
	// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
	vector< vector<double> > &phiG=phi[etaL][g], &j_xG=j_x[etaL][g], &j_yG=j_y[etaL][g];
	vector<double>     &FLG=    FL[etaL][g],     &FRG=    FR[etaL][g],     &FBG=    FB[etaL][g],     &FTG=    FT[etaL][g];
	vector<double>   &jInLG=  jInL[etaL][g],   &jInRG=  jInR[etaL][g],   &jInBG=  jInB[etaL][g],   &jInTG=  jInT[etaL][g];
	vector<double> &phiInLG=phiInL[etaL][g], &phiInRG=phiInR[etaL][g], &phiInBG=phiInB[etaL][g], &phiInTG=phiInT[etaL][g];
	
	if ( NDASolution ) {
		vector<double> &phiLG=phiL[etaL][g], &phiRG=phiR[etaL][g], &phiBG=phiB[etaL][g], &phiTG=phiT[etaL][g];
		vector< vector<double> > &D_xPG=D_xP[etaL][g], &D_xNG=D_xN[etaL][g], &D_yPG=D_yP[etaL][g], &D_yNG=D_yN[etaL][g];
		// Boundary currents 
		//# pragma omp parallel for
		for (int j=0; j<Ny; j++) j_xG[0][j]=-FLG[j]*phiLG[j]; // Left boundary current J_x
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) j_yG[i][0]=-FBG[i]*phiBG[i]; // Bottom boundary current J_y
		// Inner currents J_x J_y
		//# pragma omp parallel for
		for (int j=0; j<Ny; j++) for (int i=1; i<Nx; i++) j_xG[i][j]=-(D_xPG[i][j]*phiG[i][j]-D_xNG[i][j]*phiG[i-1][j])/xe[i];
		//# pragma omp parallel for
		for (int j=1; j<Ny; j++) for (int i=0; i<Nx; i++) j_yG[i][j]=-(D_yPG[i][j]*phiG[i][j]-D_yNG[i][j]*phiG[i][j-1])/ye[j];
		// Boundary currents
		//# pragma omp parallel for
		for (int j=0; j<Ny; j++) j_xG[Nx][j]=FRG[j]*phiRG[j]; // Right boundary current
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) j_yG[i][Ny]=FTG[i]*phiTG[i]; // Top boundary current
	}
	else {
		// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
		vector< vector<double> > &phi_xG=phi_x[etaL][g], &phi_yG=phi_y[etaL][g];
		vector< vector<double> > &D_xxCG=D_xxC[etaL][g], &D_yyCG=D_yyC[etaL][g];
		vector< vector<double> > &D_xxLG=D_xxL[etaL][g], &D_yyLG=D_yyL[etaL][g], &D_xyLG=D_xyL[etaL][g];
		vector< vector<double> > &D_xxRG=D_xxR[etaL][g], &D_yyRG=D_yyR[etaL][g], &D_xyRG=D_xyR[etaL][g];
		vector< vector<double> > &D_xxBG=D_xxB[etaL][g], &D_yyBG=D_yyB[etaL][g], &D_xyBG=D_xyB[etaL][g];
		vector< vector<double> > &D_xxTG=D_xxT[etaL][g], &D_yyTG=D_yyT[etaL][g], &D_xyTG=D_xyT[etaL][g];
		
		//# pragma omp parallel for
		for (int j=0; j<Ny; j++) j_xG[0][j]=-2.0*(D_xxCG[0][j]*phiG[0][j]-D_xxLG[0][j]*phi_xG[0][j])/hx[0]-(D_xyTG[0][j]*phi_yG[0][j+1]-D_xyBG[0][j]*phi_yG[0][j])/hy[j]; // Left boundary current J_x
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) j_yG[i][0]=-2.0*(D_yyCG[i][0]*phiG[i][0]-D_yyBG[i][0]*phi_yG[i][0])/hy[0]-(D_xyRG[i][0]*phi_xG[i+1][0]-D_xyLG[i][0]*phi_xG[i][0])/hx[i]; // Bottom boundary current J_y
		// Inner currents J_x J_y
		//# pragma omp parallel for
		for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) j_xG[i+1][j]=-2.0*(D_xxRG[i][j]*phi_xG[i+1][j]-D_xxCG[i][j]*phiG[i][j])/hx[i]-(D_xyTG[i][j]*phi_yG[i][j+1]-D_xyBG[i][j]*phi_yG[i][j])/hy[j];
		//# pragma omp parallel for
		for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) j_yG[i][j+1]=-2.0*(D_yyTG[i][j]*phi_yG[i][j+1]-D_yyCG[i][j]*phiG[i][j])/hy[j]-(D_xyRG[i][j]*phi_xG[i+1][j]-D_xyLG[i][j]*phi_xG[i][j])/hx[i];
	}
}
//======================================================================================//
//++ Infinity Norm of Newton Equations +++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
double LOSolution::NewtonLNorm(int L) {
	double res=0;
	int etaL=eta_star-1, g=0;
	vector< vector<double> > &phiG=phi[etaL][g], &j_xG=j_x[etaL][g], &j_yG=j_y[etaL][g];
	vector< vector<double> > &sigmaTG=sigmaT[etaL][g], &sigmaSG=sigmaS[etaL][g][g], &nuSigmaFG=nuSigmaF[etaL][g];
	vector<double>     &FLG=    FL[etaL][g],     &FRG=    FR[etaL][g],     &FBG=    FB[etaL][g],     &FTG=    FT[etaL][g];
	vector<double>   &jInLG=  jInL[etaL][g],   &jInRG=  jInR[etaL][g],   &jInBG=  jInB[etaL][g],   &jInTG=  jInT[etaL][g];
	vector<double> &phiInLG=phiInL[etaL][g], &phiInRG=phiInR[etaL][g], &phiInBG=phiInB[etaL][g], &phiInTG=phiInT[etaL][g];
	double L2_norm=0.0, LI_norm=0.0;
	
	if ( NDASolution ) {
		LI_norm=-(x[Nx]-x[0])*(y[Ny]-y[0]);
		for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) LI_norm+=hx[i]*hy[j]*phiG[i][j];
		LI_norm=abs(LI_norm);
		L2_norm=LI_norm*LI_norm;
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) { 
				res=abs((j_xG[i+1][j]-j_xG[i][j])/hx[i]+(j_yG[i][j+1]-j_yG[i][j])/hy[j]+
					(sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/k_eff)*phiG[i][j]);
				if ( res>LI_norm ) LI_norm=res;
				L2_norm+=res*res;
			}
		}
	}
	else {
		vector< vector<double> > &phi_xG=phi_x[etaL][g], &phi_yG=phi_y[etaL][g];
		vector< vector<double> > &D_xxCG=D_xxC[etaL][g], &D_yyCG=D_yyC[etaL][g];
		vector< vector<double> > &D_xxLG=D_xxL[etaL][g], &D_yyLG=D_yyL[etaL][g], &D_xyLG=D_xyL[etaL][g];
		vector< vector<double> > &D_xxRG=D_xxR[etaL][g], &D_yyRG=D_yyR[etaL][g], &D_xyRG=D_xyR[etaL][g];
		vector< vector<double> > &D_xxBG=D_xxB[etaL][g], &D_yyBG=D_yyB[etaL][g], &D_xyBG=D_xyB[etaL][g];
		vector< vector<double> > &D_xxTG=D_xxT[etaL][g], &D_yyTG=D_yyT[etaL][g], &D_xyTG=D_xyT[etaL][g];
		
		LI_norm=-(x[Nx]-x[0])*(y[Ny]-y[0]);
		for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) LI_norm+=hx[i]*hy[j]*phiG[i][j];
		LI_norm=abs(LI_norm);
		L2_norm=LI_norm*LI_norm;
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) { 
				res=abs(2.0*(-D_xxRG[i][j]*phi_xG[i+1][j]+2.0*D_xxCG[i][j]*phiG[i][j]-D_xxLG[i][j]*phi_xG[i][j])/hx[i]/hx[i]
					   +2.0*(-D_yyTG[i][j]*phi_yG[i][j+1]+2.0*D_yyCG[i][j]*phiG[i][j]-D_yyBG[i][j]*phi_yG[i][j])/hy[j]/hy[j]+
					(sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/k_eff)*phiG[i][j]);
				if ( res>LI_norm ) LI_norm=res;
				L2_norm+=res*res;
			}
		}
		for (int j=0; j<Ny; j++) {
			for (int i=1; i<Nx; i++) {
				res=abs(2.0*(D_xxRG[i-1][j]*phi_xG[i][j]-D_xxCG[i-1][j]*phiG[i-1][j])/hx[i-1]+(D_xyTG[i-1][j]*phi_yG[i-1][j+1]-D_xyBG[i-1][j]*phi_yG[i-1][j])/hy[j]
					   -2.0*(D_xxCG[i][j]  *phiG[i][j]  -D_xxLG[i][j]  *phi_xG[i][j])/hx[i]  -(D_xyTG[i][j]  *phi_yG[i][j+1]  -D_xyBG[i][j]  *phi_yG[i][j]  )/hy[j]);
				if ( res>LI_norm ) LI_norm=res;
				L2_norm+=res*res;
			}
		}
		for (int j=1; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				res=abs((-D_xyRG[i][j]/hx[i])*phi_xG[i+1][j] + (-2.0*D_yyCG[i][j]/hy[j])*phiG[i][j] + (D_xyLG[i][j]/hx[i])*phi_xG[i][j]
					+ (D_xyRG[i][j-1]/hx[i])*phi_xG[i+1][j-1] + (-2.0*D_yyCG[i][j-1]/hy[j-1])*phiG[i][j-1] + (-D_xyLG[i][j-1]/hx[i])*phi_xG[i][j-1]
					+ 2.0*(D_yyTG[i][j-1]/hy[j-1]+D_yyBG[i][j]/hy[j])*phi_yG[i][j]);
				if ( res>LI_norm ) LI_norm=res;
				L2_norm+=res*res;
			}
		}
		for (int j=0; j<Ny; j++) {
			res=abs((-2.0*D_xxCG[0][j]/hx[0])*phiG[0][j] + (2.0*D_xxLG[0][j]/hx[0]+FLG[j])*phi_xG[0][j] 
				+ (D_xyBG[0][j]/hy[j])*phi_yG[0][j] + (-D_xyTG[0][j]/hy[j])*phi_yG[0][j+1] - FLG[j]*phiInLG[j] + jInLG[j]);
			if ( res>LI_norm ) LI_norm=res;
			L2_norm+=res*res;
			res=abs((-2.0*D_xxCG[Nx-1][j]/hx[Nx-1])*phiG[Nx-1][j] + (2.0*D_xxRG[Nx-1][j]/hx[Nx-1]+FRG[j])*phi_xG[Nx][j] 
				+ (-D_xyBG[Nx-1][j]/hy[j])*phi_yG[Nx-1][j] + (D_xyTG[Nx-1][j]/hy[j])*phi_yG[Nx-1][j+1] - FRG[j]*phiInRG[j] + jInRG[j]);
			if ( res>LI_norm ) LI_norm=res;
			L2_norm+=res*res;
		}
		for (int i=0; i<Nx; i++) {
			res=abs((-2.0*D_yyCG[i][0]/hy[0])*phiG[i][0] + (D_xyLG[i][0]/hx[i])*phi_xG[i][0] + 
			(-D_xyRG[i][0]/hx[i])*phi_xG[i+1][0] + (2.0*D_yyBG[i][0]/hy[0]+FBG[i])*phi_yG[i][0] - FBG[i]*phiInBG[i] + jInBG[i]);
			if ( res>LI_norm ) LI_norm=res;
			L2_norm+=res*res;
			res=abs((-2.0*D_yyCG[i][Ny-1]/hy[Ny-1])*phiG[i][Ny-1] + (-D_xyLG[i][Ny-1]/hx[i])*phi_xG[i][Ny-1] + 
			(D_xyRG[i][Ny-1]/hx[i])*phi_xG[i+1][Ny-1] + (2.0*D_yyTG[i][Ny-1]/hy[Ny-1]+FTG[i])*phi_yG[i][Ny] - FTG[i]*phiInTG[i] + jInTG[i]);
			if ( res>LI_norm ) LI_norm=res;
			L2_norm+=res*res;
		}
	}
	
	if ( L==2 ) return sqrt(L2_norm);
	else        return LI_norm;
}

void LOSolution::writeResiduals(int etaL, ofstream& outfile) {
	vector< vector< vector<double> > > &phiG=phi[etaL];
	vector< vector< vector<double> > > &j_xG=j_x[etaL], &j_yG=j_y[etaL];
	vector< vector<double> >     &FLG=    FL[etaL],     &FRG=    FR[etaL],     &FBG=    FB[etaL],     &FTG=    FT[etaL];
	vector< vector<double> >   &jInLG=  jInL[etaL],   &jInRG=  jInR[etaL],   &jInBG=  jInB[etaL],   &jInTG=  jInT[etaL];
	vector< vector<double> > &phiInLG=phiInL[etaL], &phiInRG=phiInR[etaL], &phiInBG=phiInB[etaL], &phiInTG=phiInT[etaL];
	double res;
	double res_lbc=0.0, res_rbc=0.0, res_bbc=0.0, res_tbc=0.0;
	int i_bbc=101010, i_tbc=101010, j_lbc=101010, j_rbc=101010;
	int g_bbc=101010, g_tbc=101010, g_lbc=101010, g_rbc=101010;
	
	if ( NDASolution ) {
		vector< vector<double> >           &phiLG=phiL[etaL], &phiRG=phiR[etaL], &phiBG=phiB[etaL], &phiTG=phiT[etaL];
		vector< vector< vector<double> > > &D_xPG=D_xP[etaL], &D_xNG=D_xN[etaL], &D_yPG=D_yP[etaL], &D_yNG=D_yN[etaL];
		double res_x=0.0, res_y=0.0;
		int i_xx=101010, i_yy=101010, j_xx=101010, j_yy=101010;
		int g_xx=101010, g_yy=101010;

		for (int g=0; g<Ng[etaL]; g++) {
			for (int i=1; i<Nx-1; i++) { // First Moment X Grid
				for (int j=0; j<Ny; j++) {
					res=abs(j_xG[g][i][j]+(D_xPG[g][i][j]*phiG[g][i][j]-D_xNG[g][i][j]*phiG[g][i-1][j])/xe[i]); 
					if ( res>res_x ) { res_x=res; i_xx=i; j_xx=j; g_xx=g; }
				}
			}
			for (int i=0; i<Nx; i++) { // First Moment Y Grid
				for (int j=1; j<Ny-1; j++) {
					res=abs(j_yG[g][i][j]+(D_yPG[g][i][j]*phiG[g][i][j]-D_yNG[g][i][j]*phiG[g][i][j-1])/ye[j]); 
					if ( res>res_y ) { res_y=res; i_yy=i; j_yy=j; g_yy=g; }
				}
			}
			// Boundary Conditions
			for (int j=0; j<Ny; j++) { // Left and Right
				res=abs(-FLG[g][j]*phiLG[g][j]-j_xG[g][0][j]);  
				if ( res>res_lbc ) { res_lbc=res; j_lbc=j; g_lbc=g; } // Left BC residual
				res=abs(FRG[g][j]*phiRG[g][j]-j_xG[g][Nx][j]); 
				if ( res>res_rbc ) { res_rbc=res; j_rbc=j; g_rbc=g; } // Right BC residual
			}
			for (int i=0; i<Nx; i++) { // Bottom and Top
				res=abs(-FBG[g][i]*phiBG[g][i]-j_yG[g][i][0]);  
				if ( res>res_bbc ) { res_bbc=res; i_bbc=i; g_bbc=g; } // Bottom BC residual
				res=abs( FTG[g][i]*phiTG[g][i]-j_yG[g][i][Ny]); 
				if ( res>res_tbc ) { res_tbc=res; i_tbc=i; g_tbc=g; } // Top BC residual
			}
		}
		
		outfile<<"X       Grid Residual:"<<print_out(res_x)<<" in group "<<g_xx<<" at "<<i_xx<<" , "<<j_xx<<endl;
		outfile<<"Y       Grid Residual:"<<print_out(res_y)<<" in group "<<g_yy<<" at "<<i_yy<<" , "<<j_yy<<endl;
	}
	else {
		vector< vector< vector<double> > > &phi_xG=phi_x[etaL], &phi_yG=phi_y[etaL];
		vector< vector< vector<double> > > &D_xxCG=D_xxC[etaL], &D_yyCG=D_yyC[etaL];
		vector< vector< vector<double> > > &D_xxLG=D_xxL[etaL], &D_yyLG=D_yyL[etaL], &D_xyLG=D_xyL[etaL];
		vector< vector< vector<double> > > &D_xxRG=D_xxR[etaL], &D_yyRG=D_yyR[etaL], &D_xyRG=D_xyR[etaL];
		vector< vector< vector<double> > > &D_xxBG=D_xxB[etaL], &D_yyBG=D_yyB[etaL], &D_xyBG=D_xyB[etaL];
		vector< vector< vector<double> > > &D_xxTG=D_xxT[etaL], &D_yyTG=D_yyT[etaL], &D_xyTG=D_xyT[etaL];
		
		double res_l=0.0, res_r=0.0, res_b=0.0, res_t=0.0;
		int i_l=101010, i_r=101010, i_b=101010, i_t=101010;
		int j_l=101010, j_r=101010, j_b=101010, j_t=101010;
		int g_l=101010, g_r=101010, g_b=101010, g_t=101010;

		for (int g=0; g<Ng[etaL]; g++) {
			for (int i=0; i<Nx; i++) { // First Moment X Grid
				for (int j=0; j<Ny; j++) {
					res=abs(j_xG[g][i][j]   + 2.0*(D_xxCG[g][i][j]*phiG[g][i][j]    -D_xxLG[g][i][j]*phi_xG[g][i][j])/hx[i] + (D_xyTG[g][i][j]*phi_yG[g][i][j+1]-D_xyBG[g][i][j]*phi_yG[g][i][j])/hy[j] ); 
					if ( res>res_l ) { res_l=res; i_l=i; j_l=j; g_l=g; }
					res=abs(j_xG[g][i+1][j] + 2.0*(D_xxRG[g][i][j]*phi_xG[g][i+1][j]-D_xxCG[g][i][j]*phiG[g][i][j]  )/hx[i] + (D_xyTG[g][i][j]*phi_yG[g][i][j+1]-D_xyBG[g][i][j]*phi_yG[g][i][j])/hy[j] ); 
					if ( res>res_r ) { res_r=res; i_r=i; j_r=j; g_r=g; }
					res=abs(j_yG[g][i][j] + 2.0*(D_yyCG[g][i][j]*phiG[g][i][j]-D_yyBG[g][i][j]*phi_yG[g][i][j])/hy[i] + (D_xyRG[g][i][j]*phi_xG[g][i+1][j]-D_xyLG[g][i][j]*phi_xG[g][i][j])/hx[i] ); 
					if ( res>res_b ) { res_b=res; i_b=i; j_b=j; g_b=g; }
					res=abs(j_yG[g][i][j+1] + 2.0*(D_yyTG[g][i][j]*phi_yG[g][i][j+1]-D_yyCG[g][i][j]*phiG[g][i][j])/hy[i] + (D_xyRG[g][i][j]*phi_xG[g][i+1][j]-D_xyLG[g][i][j]*phi_xG[g][i][j])/hx[i] ); 
					if ( res>res_t ) { res_t=res; i_t=i; j_t=j; g_t=g; }
				}
			}
			
			// Boundary Conditions
			for (int j=0; j<Ny; j++) { // Left and Right
				res=abs(FLG[g][j]*(phi_xG[g][0][j]-phiInLG[g][j])+jInLG[g][j]+j_xG[g][0][j]);  
				if ( res>res_lbc ) { res_lbc=res; j_lbc=j; g_lbc=g; } // Left BC residual
				res=abs(FRG[g][j]*(phi_xG[g][Nx][j]-phiInRG[g][j])+jInRG[g][j]-j_xG[g][Nx][j]); 
				if ( res>res_rbc ) { res_rbc=res; j_rbc=j; g_rbc=g; } // Right BC residual
			}
			for (int i=0; i<Nx; i++) { // Bottom and Top
				res=abs(FBG[g][i]*(phi_yG[g][i][0]-phiInBG[g][i])+jInBG[g][i]+j_yG[g][i][0]);  
				if ( res>res_bbc ) { res_bbc=res; i_bbc=i; g_bbc=g; } // Bottom BC residual
				res=abs(FTG[g][i]*(phi_yG[g][i][Ny]-phiInTG[g][i])+jInTG[g][i]-j_yG[g][i][Ny]); 
				if ( res>res_tbc ) { res_tbc=res; i_tbc=i; g_tbc=g; } // Top BC residual
			}
			
		}
		
		outfile<<"Left    Grid Residual:"<<print_out(res_l)<<" in group "<<g_l<<" at "<<i_l<<" , "<<j_l<<endl;
		outfile<<"Right   Grid Residual:"<<print_out(res_r)<<" in group "<<g_r<<" at "<<i_r<<" , "<<j_r<<endl;
		outfile<<"Bottom  Grid Residual:"<<print_out(res_b)<<" in group "<<g_b<<" at "<<i_b<<" , "<<j_b<<endl;
		outfile<<"Top     Grid Residual:"<<print_out(res_t)<<" in group "<<g_t<<" at "<<i_t<<" , "<<j_t<<endl;
	}
	
	outfile<<"Left    BC   Residual:"<<print_out(res_lbc)<<" in group "<<g_lbc<<" at "<<j_lbc<<endl;
	outfile<<"Right   BC   Residual:"<<print_out(res_rbc)<<" in group "<<g_rbc<<" at "<<j_rbc<<endl;
	outfile<<"Bottom  BC   Residual:"<<print_out(res_bbc)<<" in group "<<g_bbc<<" at "<<i_bbc<<endl;
	outfile<<"Top     BC   Residual:"<<print_out(res_tbc)<<" in group "<<g_tbc<<" at "<<i_tbc<<endl;
}


void LOSolution::writeConsistency(ofstream& outfile) {
	double eps=1e-30;
	outfile<<"\n -- Consistency Between Low-order Solutions on Successive Grids -- \n";
	
	for (int k=1; k<eta_star; k++) {
		double conv_p=0.0, conv_px=0.0, conv_py=0.0, conv_jx=0.0, conv_jy=0.0;
		double conv_pl=0.0, conv_pr=0.0, conv_pb=0.0, conv_pt=0.0;
		double L2_p=0.0, L2_px=0.0, L2_py=0.0, L2_jx=0.0, L2_jy=0.0;
		double L2_pl=0.0, L2_pr=0.0, L2_pb=0.0, L2_pt=0.0;
		for (int p=0; p<Ng[k]; p++) {
			for (int i=0; i<Nx; i++ ) {
				for (int j=0; j<Ny; j++ ) {
					double phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=phi[k-1][g][i][j];
					double conv=abs(1-phi[k][p][i][j]/phi_sum); if (conv>conv_p) conv_p=conv;
					L2_p+=conv*conv;
				}
			}
			for (int i=0; i<Nx+1; i++ ) {
				for (int j=0; j<Ny; j++ ) {
					double phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=j_x[k-1][g][i][j];
					double conv=abs(1-(j_x[k][p][i][j]+eps)/(phi_sum+eps)); if (conv>conv_jx) conv_jx=conv;
					L2_jx+=conv*conv;
				}
			}
			for (int i=0; i<Nx; i++ ) {
				for (int j=0; j<Ny+1; j++ ) {
					double phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=j_y[k-1][g][i][j];
					double conv=abs(1-(j_y[k][p][i][j]+eps)/(phi_sum+eps)); if (conv>conv_jy) conv_jy=conv;
					L2_jy+=conv*conv;
				}
			}
		}
		
		outfile<<"Relative Difference Between Grid # "<<k-1<<" and Grid # "<<k<<endl;
		outfile<<"                            "<<"L-infinity Norm"<<"    L2 Norm     "<<endl;
		outfile<<"Cell      Averaged Flux : "<<print_out(conv_p)<<print_out(sqrt(L2_p))<<endl;
		
		if ( NDASolution ) {
			// NDA Solution
			for (int p=0; p<Ng[k]; p++) {
				for (int j=0; j<Ny; j++ ) {
					double phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=phiL[k-1][g][j];
					double conv=abs(1-phiL[k][p][j]/phi_sum); if (conv>conv_pl) conv_pl=conv;
					L2_pl+=conv*conv;
					phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=phiR[k-1][g][j];
					conv=abs(1-phiR[k][p][j]/phi_sum); if (conv>conv_pr) conv_pr=conv;
					L2_pr+=conv*conv;
				}
				for (int i=0; i<Nx; i++ ) {
					double phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=phiB[k-1][g][i];
					double conv=abs(1-phiB[k][p][i]/phi_sum); if (conv>conv_pb) conv_pb=conv;
					L2_pb+=conv*conv;
					phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=phiT[k-1][g][i];
					conv=abs(1-phiT[k][p][i]/phi_sum); if (conv>conv_pt) conv_pt=conv;
					L2_pt+=conv*conv;
				}
			}
			
			outfile<<" L        Boundary Flux : "<<print_out(conv_pl)<<print_out(sqrt(L2_pl))<<endl;
			outfile<<" R        Boundary Flux : "<<print_out(conv_pr)<<print_out(sqrt(L2_pr))<<endl;
			outfile<<" B        Boundary Flux : "<<print_out(conv_pb)<<print_out(sqrt(L2_pb))<<endl;
			outfile<<" T        Boundary Flux : "<<print_out(conv_pt)<<print_out(sqrt(L2_pt))<<endl;
		}
		else {
			// QD
			for (int p=0; p<Ng[k]; p++) {
				for (int i=0; i<Nx; i++ ) {
					for (int j=0; j<Ny+1; j++ ) {
						double phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=phi_y[k-1][g][i][j];
						double conv=abs(1-phi_y[k][p][i][j]/phi_sum); if (conv>conv_py) conv_py=conv;
						L2_py+=conv*conv;
						phi_sum=0.0; for (int g=omegaP[k][p]; g<omegaP[k][p+1]; g++) phi_sum+=j_y[k-1][g][i][j];
						conv=abs(1-(j_y[k][p][i][j]+eps)/(phi_sum+eps)); if (conv>conv_jy) conv_jy=conv;
						L2_jy+=conv*conv;
					}
				}
			}
			
			outfile<<"X Grid Face Avg    Flux : "<<print_out(conv_px)<<print_out(sqrt(L2_px))<<endl;
			outfile<<"Y Grid Face Avg    Flux : "<<print_out(conv_py)<<print_out(sqrt(L2_py))<<endl;
		}
		outfile<<"X Grid Face Avg Current : "<<print_out(conv_jx)<<print_out(sqrt(L2_jx))<<endl;
		outfile<<"Y Grid Face Avg Current : "<<print_out(conv_jy)<<print_out(sqrt(L2_jy))<<endl;
	}
	
}

void LOSolution::consistencyBetweenLOAndHO(HOSolver &ho, ofstream& outfile) {
	double eps=1e-30;
	// Find difference between Transport solution and NDA solution
	double conv, conv_p=0.0, conv_pl=0.0, conv_pr=0.0, conv_pb=0.0, conv_pt=0.0, conv_jx=0.0, conv_jy=0.0, conv_px=0.0, conv_py=0.0;
	
	
	for (int g=0; g<Ng[0]; g++) {
		for (int i=0; i<Nx; i++ ) {
			for (int j=0; j<Ny; j++ ) { double conv=abs((phi[0][g][i][j]-ho.phi[g][i][j])/ho.phi[g][i][j]); if ( conv>conv_p ) conv_p=conv; }
		}
		for (int i=0; i<Nx+1; i++ ) {
			for (int j=0; j<Ny; j++ ) { double conv=abs(1-(j_x[0][g][i][j]+eps)/(ho.j_x[g][i][j]+eps)); if ( conv>conv_jx ) conv_jx=conv; }
		}

		for (int i=0; i<Nx; i++ ) {
			for (int j=0; j<Ny+1; j++ ) { double conv=abs(1-(j_y[0][g][i][j]+eps)/(ho.j_y[g][i][j]+eps)); if ( conv>conv_jy ) conv_jy=conv; }
		}
	}
	outfile<<"\n -- L-Infinity Norm of Relative Difference Between Transport and NDA Solution -- \n";
	outfile<<"Cell      Averaged Flux : "<<print_out(conv_p)<<endl;
	
	if ( NDASolution ) {
		// NDA
		for (int g=0; g<Ng[0]; g++) {
			for (int i=0; i<Nx; i++ ) {
				for (int j=0; j<Ny; j++ ) { double conv=abs((phi[0][g][i][j]-ho.phi[g][i][j])/ho.phi[g][i][j]); if ( conv>conv_p ) conv_p=conv; }
			}
			for (int j=0; j<Ny; j++ ) {
				double conv=abs(1-phiL[0][g][j]/ho.phi_x[g][0][j]);  if ( conv>conv_pl ) conv_pl=conv;
					   conv=abs(1-phiR[0][g][j]/ho.phi_x[g][Nx][j]); if ( conv>conv_pr ) conv_pr=conv;
			}
			for (int i=0; i<Nx; i++ ) {
				double conv=abs(1-phiB[0][g][i]/ho.phi_y[g][i][0]);  if ( conv>conv_pb ) conv_pb=conv;
					   conv=abs(1-phiT[0][g][i]/ho.phi_y[g][i][Ny]); if ( conv>conv_pt ) conv_pt=conv;
			}
		}
		outfile<<" L        Boundary Flux : "<<print_out(conv_pl)<<endl;
		outfile<<" R        Boundary Flux : "<<print_out(conv_pr)<<endl;
		outfile<<" B        Boundary Flux : "<<print_out(conv_pb)<<endl;
		outfile<<" T        Boundary Flux : "<<print_out(conv_pt)<<endl;
	
	}
	else {
		// QD
		for (int g=0; g<Ng[0]; g++) {
			for (int i=0; i<Nx+1; i++ ) {
				for (int j=0; j<Ny; j++ ) { 
					conv=abs((phi_x[0][g][i][j]-ho.phi_x[g][i][j])/ho.phi_x[g][i][j]); if ( conv>conv_px ) conv_px=conv; 
				}
			}
			for (int i=0; i<Nx; i++ ) {
				for (int j=0; j<Ny+1; j++ ) { 
					conv=abs((phi_y[0][g][i][j]-ho.phi_y[g][i][j])/ho.phi_y[g][i][j]); if ( conv>conv_py ) conv_py=conv; 
				}
			}
		}
		outfile<<"\n -- L-Infinity Norm of Relative Difference Between Transport and NDA Solution -- \n";
		outfile<<"Cell      Averaged Flux : "<<print_out(conv_p)<<endl;
		outfile<<"X Grid Face Avg    Flux : "<<print_out(conv_px)<<endl;
		outfile<<"Y Grid Face Avg    Flux : "<<print_out(conv_py)<<endl;
	}
	
	outfile<<"X Grid Face Avg Current : "<<print_out(conv_jx)<<endl;
	outfile<<"Y Grid Face Avg Current : "<<print_out(conv_jy)<<endl;
}


void LOSolution::writeSolutionDat(std::string case_name) {
	int outw=16;
	
	if ( writeOutput ) {
		string lo_file=case_name+".lo.csv";
		#pragma omp critical
		cout<<lo_file<<endl;
		
		ofstream datfile (lo_file.c_str()); // open output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		datfile<<" # of x cells , # of y cells ,\n";
		datfile<<Nx<<" , "<<Ny<<", \n";
		datfile<<"x edge grid , "; for (int i=0; i<Nx+1; i++) datfile<<print_csv(x[i]); datfile<<endl;
		datfile<<"x center grid , "; for (int i=0; i<Nx; i++) datfile<<print_csv((x[i]+x[i+1])/2); datfile<<endl;
		datfile<<"y edge grid , "; for (int j=0; j<Ny+1; j++) datfile<<print_csv(y[j]); datfile<<endl;
		datfile<<"y center grid , "; for (int j=0; j<Ny; j++) datfile<<print_csv((y[j]+y[j+1])/2); datfile<<endl;
		datfile<<"number of energy grids, "<<eta_star<<",\n";
		datfile<<"number of groups in grid, ";
		for (int k=0; k<eta_star; k++) datfile<<Ng[k]<<" , ";
		datfile<<endl;
		
		datfile<<endl;
		datfile<<" ---------------------------- \n";
		if ( NDASolution ) datfile<<" -- Low Order NDA Solution -- \n"; // Write NDA solution
		else               datfile<<" -- Low Order QD  Solution -- \n"; // Write NDA solution
		datfile<<" ---------------------------- \n";
		
		datfile<<"\n -- Cell Averaged Scalar Flux -- \n";
		write_grid_dat(phi, Ng, x, y, outw, datfile);
		
		if ( NDASolution ) {
			datfile<<"\n -- Left and Right Boundary Flux -- \n";
			for (int k=0; k<eta_star; k++) {
				datfile<<" -- Energy Grid # "<<k<<endl;
				for (int g=0; g<Ng[k]; g++) {
					datfile<<"Grid # "<<k<<" Energy Group # "<<g<<endl;
					datfile<<"  index    sol. grid     Left   Flux     Right  Flux  \n";
					for (int j=Ny-1; j>=0; j-- ) datfile<<setw(6)<<j+1<<","<<print_csv((y[j]+y[j+1])/2)<<print_csv(phiL[k][g][j])<<print_csv(phiR[k][g][j])<<endl;
				}
			}
			
			datfile<<"\n -- Bottom and Top Boundary Flux -- \n";
			for (int k=0; k<eta_star; k++) {
				datfile<<" -- Energy Grid # "<<k<<endl;
				for (int g=0; g<Ng[k]; g++) {
					datfile<<"Grid # "<<k<<" Energy Group # "<<g<<endl;
					datfile<<"  index    sol. grid     Bottom  Flux     Top   Flux  \n";
					for (int i=Nx-1; i>=0; i-- ) datfile<<setw(6)<<i+1<<","<<print_csv((x[i]+x[i+1])/2)<<print_csv(phiB[k][g][i])<<print_csv(phiT[k][g][i])<<endl;
				}
			}
		}
		else {
			datfile<<"\n -- X Face Averaged Scalar Flux -- \n";
			write_grid_dat(phi_x, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- Y Face Averaged Scalar Flux -- \n";
			write_grid_dat(phi_y, Ng, x, y, outw, datfile);
		}
		
		datfile<<"\n -- X Face Average Normal Current J_x -- \n";
		write_grid_dat(j_x, Ng, x, y, outw, datfile); // call function to write out cell edge current on x grid
		
		datfile<<"\n -- Y Face Average Normal Current J_y -- \n";
		write_grid_dat(j_y, Ng, x, y, outw, datfile); // call function to write out cell edge current on y grid
		
		if ( NDASolution ) {
			// NDA Factors
			datfile<<endl;
			datfile<<" -------------------------------------- \n";
			datfile<<" -- Corrected Diffusion Coefficients -- \n";
			datfile<<" -------------------------------------- \n";
			
			datfile<<"\n -- X Grid Positive Diffusion Coefficient D^+_x -- \n";
			write_grid_dat(D_xP, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- Y Grid Positive Diffusion Coefficient D^+_y -- \n";
			write_grid_dat(D_yP, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- X Grid Negative Diffusion Coefficient D^-_x -- \n";
			write_grid_dat(D_xN, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- Y Grid Negative Diffusion Coefficient D^-_y -- \n";
			write_grid_dat(D_yN, Ng, x, y, outw, datfile);
			
			// Boundary conditions
			datfile<<"\n -- Bottom and Top Boundary Factors -- \n";
			for (int k=0; k<eta_star; k++) {
				datfile<<" -- Energy Grid # "<<k<<endl;
				for (int g=0; g<Ng[k]; g++) {
					datfile<<"Grid # "<<k<<" Energy Group # "<<g<<endl;
					datfile<<" index "<<"     x center   ,"<<"   F  Bottom   ,"<<"  J In  Bottom ,"<<" Phi In  Bottom,"<<"     F  Top    ,"<<"   J In  Top   ,"<<"  Phi In  Top  ,"<<"\n";
					for (int i=0; i<Nx; i++) datfile<<setw(6)<<i+1<<","<<print_csv((x[i]+x[i+1])/2)<<print_csv(FB[k][g][i])<<print_csv(jInB[k][g][i])<<print_csv(phiInB[k][g][i])
						<<print_csv(FT[k][g][i])<<print_csv(jInT[k][g][i])<<print_csv(phiInT[k][g][i])<<endl;
				}
			}
			
			datfile<<"\n -- Left and Right Boundary Factors -- \n";
			for (int k=0; k<eta_star; k++) {
				datfile<<" -- Energy Grid # "<<k<<endl;
				for (int g=0; g<Ng[k]; g++) {
					datfile<<"Grid # "<<k<<" Energy Group # "<<g<<endl;
					datfile<<" index "<<"     y center   ,"<<"    F  Left    ,"<<"   J In  Left  ,"<<"  Phi In  Left ,"<<"    F  Right   ,"<<"  J In  Right  ,"<<" Phi In  Right ,"<<"\n";
					for (int j=0; j<Ny; j++) datfile<<setw(6)<<j+1<<","<<print_csv((y[j]+y[j+1])/2)<<print_csv(FL[k][g][j])<<print_csv(jInL[k][g][j])<<print_csv(phiInL[k][g][j])
						<<print_csv(FR[k][g][j])<<print_csv(jInR[k][g][j])<<print_csv(phiInR[k][g][j])<<endl;
				}
			}
		}
		else {
			// Low Order QD Factors
			datfile<<endl;
			datfile<<" -------------------------------- \n";
			datfile<<" -- Modified Eddington Factors -- \n";
			datfile<<" -------------------------------- \n";
			
			datfile<<"\n -- D_xxC Cell Center Modified Eddington Factors -- \n";
			write_grid_dat(D_xxC, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_yyC Cell Center Modified Eddington Factors -- \n";
			write_grid_dat(D_yyC, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_xxL Left Side Modified Eddington Factors -- \n";
			write_grid_dat(D_xxL, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_yyL Left Side Modified Eddington Factors -- \n";
			write_grid_dat(D_yyL, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_xyL Left Side Modified Eddington Factors -- \n";
			write_grid_dat(D_xyL, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_xxR Right Side Modified Eddington Factors -- \n";
			write_grid_dat(D_xxR, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_yyR Right Side Modified Eddington Factors -- \n";
			write_grid_dat(D_yyR, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_xyR Right Side Modified Eddington Factors -- \n";
			write_grid_dat(D_xyR, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_xxB Bottom Modified Eddington Factors -- \n";
			write_grid_dat(D_xxB, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_yyB Bottom Modified Eddington Factors -- \n";
			write_grid_dat(D_yyB, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_xyB Bottom Modified Eddington Factors -- \n";
			write_grid_dat(D_xyB, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_xxT Top Modified Eddington Factors -- \n";
			write_grid_dat(D_xxT, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_yyT Top Modified Eddington Factors -- \n";
			write_grid_dat(D_yyT, Ng, x, y, outw, datfile);
			
			datfile<<"\n -- D_xyT Top Modified Eddington Factors -- \n";
			write_grid_dat(D_xyT, Ng, x, y, outw, datfile);
		}
		
		// Boundary conditions
		datfile<<"\n -- Bottom and Top Boundary Factors -- \n";
		for (int k=0; k<eta_star; k++) {
			datfile<<" -- Energy Grid # "<<k<<endl;
			for (int g=0; g<Ng[k]; g++) {
				datfile<<"Grid # "<<k<<" Energy Group # "<<g<<endl;
				datfile<<" index "<<"     x center   ,"<<"   F  Bottom   ,"<<"  J In  Bottom ,"<<" Phi In  Bottom,"<<"     F  Top    ,"<<"   J In  Top   ,"<<"  Phi In  Top  ,"<<"\n";
				for (int i=0; i<Nx; i++) datfile<<setw(6)<<i+1<<","<<print_csv((x[i]+x[i+1])/2)<<print_csv(FB[k][g][i])<<print_csv(jInB[k][g][i])<<print_csv(phiInB[k][g][i])
					<<print_csv(FT[k][g][i])<<print_csv(jInT[k][g][i])<<print_csv(phiInT[k][g][i])<<endl;
			}
		}
		
		datfile<<"\n -- Left and Right Boundary Factors -- \n";
		for (int k=0; k<eta_star; k++) {
			datfile<<" -- Energy Grid # "<<k<<endl;
			for (int g=0; g<Ng[k]; g++) {
				datfile<<"Grid # "<<k<<" Energy Group # "<<g<<endl;
				datfile<<" index "<<"     y center   ,"<<"    F  Left    ,"<<"   J In  Left  ,"<<"  Phi In  Left ,"<<"    F  Right   ,"<<"  J In  Right  ,"<<" Phi In  Right ,"<<"\n";
				for (int j=0; j<Ny; j++) datfile<<setw(6)<<j+1<<","<<print_csv((y[j]+y[j+1])/2)<<print_csv(FL[k][g][j])<<print_csv(jInL[k][g][j])<<print_csv(phiInL[k][g][j])
					<<print_csv(FR[k][g][j])<<print_csv(jInR[k][g][j])<<print_csv(phiInR[k][g][j])<<endl;
			}
		}
		
		datfile.close(); // close output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	}
}


void LOSolution::writeSolutionOut(ofstream& outfile) {
	int outw=16;
	outfile<<endl;
	outfile<<" ---------------------------- \n";
	outfile<<" -- Low Order NDA Solution -- \n"; // Write NDA solution
	outfile<<" ---------------------------- \n";
	
	outfile<<"\n -- Cell Averaged Scalar Flux -- \n";
	write_grid_out(phi, Ng, x, y, outw, outfile); // call function to write out cell average scalar flux
	
	outfile<<"\n -- X Face Average Normal Current J_x -- \n";
	write_grid_out(j_x, Ng, x, y, outw, outfile); // call function to write out cell edge current on x grid
	
	outfile<<"\n -- Y Face Average Normal Current J_y -- \n";
	write_grid_out(j_y, Ng, x, y, outw, outfile); // call function to write out cell edge scalar flux on y grid
	
}


//**************************************************************************************//
//**************************************************************************************//
//======================================================================================//
//++++++++++++++++++++++++++++++++ Low Order Solver ++++++++++++++++++++++++++++++++++++//
//======================================================================================//
//**************************************************************************************//
//**************************************************************************************//

void LOSolver::edgeFlux(int etaL, vector< vector<double> > &phiEL, vector< vector<double> > &phiER, vector< vector<double> > &phiEB, vector< vector<double> > &phiET) {
	if ( NDASolution ) {
		if (eta_star!=1) {
			for (int k=eta_star-2; k>=0; k--) correctFlux(k,phiL[k],phiL[k+1]);   // Correct Flux using correction factors <<<<<<<<<<<	
			for (int k=eta_star-2; k>=0; k--) correctFlux(k,phiR[k],phiR[k+1]);   // Correct Flux using correction factors <<<<<<<<<<<	
			for (int k=eta_star-2; k>=0; k--) correctFlux(k,phiB[k],phiB[k+1]);   // Correct Flux using correction factors <<<<<<<<<<<	
			for (int k=eta_star-2; k>=0; k--) correctFlux(k,phiT[k],phiT[k+1]);   // Correct Flux using correction factors <<<<<<<<<<<	
		}
		
		phiEL=phiL[etaL];
		phiER=phiR[etaL];
		phiEB=phiB[etaL];
		phiET=phiT[etaL];
	}
	else {
		if (eta_star!=1) {
			for (int k=eta_star-2; k>=0; k--) correctFlux(k,phi_x[k],phi_x[k+1]);   // Correct Flux using correction factors <<<<<<<<<<<	
			for (int k=eta_star-2; k>=0; k--) correctFlux(k,phi_y[k],phi_y[k+1]);   // Correct Flux using correction factors <<<<<<<<<<<
		}
		
		for(int g=0; g<Ng[etaL]; g++) {
			for(int j=0; j<Ny; j++) {
				phiEL[g][j]=phi_x[etaL][g][0][j];
				phiER[g][j]=phi_x[etaL][g][Nx][j];
			}
			for(int i=0; i<Nx; i++) {
				phiEB[g][i]=phi_y[etaL][g][i][0];
				phiET[g][i]=phi_y[etaL][g][i][Ny];
			}
		}
	}
	
}

void LOSolver::calculateGridCurrents(int etaL) {
	// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
	for (int g=0; g<Ng[etaL]; g++) calculateGroupCurrents(etaL,g);
}

//======================================================================================//
//++ Check to See if Grey Solution is Positive +++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
bool LOSolver::greySolutionPositive() {
	vector< vector<double> > &phiG=phi[eta_star-1][0];
	// Check that the solution is positive
	if (k_eff<0) return false;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) { 
			if ( phiG[i][j]<0 ) return false;
		}
	}
	return true;
}

//======================================================================================//
//++ Correct the flux on grid etaL using grid etaL+1 +++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolver::correctFlux(int etaL, std::vector< std::vector< std::vector<double> > > &phiG, std::vector< std::vector< std::vector<double> > > &phiP) {
	// etaL is grid number of higher order flux that is being corrected
	// phiG is the higher order flux and phiP is the lower order flux
	// This method only works if the lower order grids have already been corrected
	int nx=phiG[0].size(), ny=phiG[0][0].size();
	//# pragma omp parallel for
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) { 
			for (int p=0; p<Ng[etaL+1]; p++) {
				double phi_sum=0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) phi_sum+=phiG[g][i][j];
				double fp=phiP[p][i][j]/phi_sum;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) phiG[g][i][j]*=fp;
			}
		}
	}
}
//======================================================================================//
//++ Correct the flux on grid etaL using grid etaL+1 +++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolver::correctFlux(int etaL, std::vector< std::vector<double> > &phiG, std::vector< std::vector<double> > &phiP) {
	// etaL is grid number of higher order flux that is being corrected
	// phiG is the higher order flux and phiP is the lower order flux
	// This method only works if the lower order grids have already been corrected
	int nx=phiG[0].size();
	//# pragma omp parallel for
	for (int i=0; i<nx; i++) {
		for (int p=0; p<Ng[etaL+1]; p++) {
			double phi_sum=0.0;
			for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) phi_sum+=phiG[g][i];
			double fp=phiP[p][i]/phi_sum;
			for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) phiG[g][i]*=fp;
		}
	}
}


int LOSolver::initializeLO(HOSolver &ho) {
	cout<<"Initialize LO Data: ";
	
	initializeSolution();
	
	initializeLOXS(ho); //
	
	sigmaWS.resize(Nx);
	for (int i=0; i<Nx; i++) sigmaWS[i].resize(Ny);
	
	// initialize memory space
	res_bal.resize(eta_star);
	i_bal.resize(eta_star); j_bal.resize(eta_star); g_bal.resize(eta_star);
	res_mbal.resize(eta_star); i_mbal.resize(eta_star); j_mbal.resize(eta_star); g_mbal.resize(eta_star);
	res_Rmatrix.resize(eta_star); g_Rmatrix.resize(eta_star);
	res_Amatrix.resize(eta_star); g_Amatrix.resize(eta_star);
	
	rho_phi.resize(eta_star);
	rho_phiH.resize(eta_star);
	rho_keff.resize(eta_star);
	rho_kappa.resize(eta_star);
	norm_phi.resize(eta_star);
	norm_phiH.resize(eta_star);
	norm_keff.resize(eta_star);
	norm_kappa.resize(eta_star);
	k_keff.resize(eta_star);
	num_losi.resize(eta_star);
	for (int k=0; k<eta_star; k++) {
		rho_phi[k].push_back(0.5);
		rho_keff[k].push_back(0.5);
		norm_phi[k].push_back(1);
		norm_phiH[k].push_back(1);
		norm_keff[k].push_back(1);
		norm_kappa[k].push_back(1);
		num_losi[k].push_back(0);
	}
	err_lo.resize(eta_star);
	dt.resize(eta_star);
	dt_pc.resize(eta_star);
	num_logm.resize(eta_star);
	num_grid.resize(eta_star);
	cout<<"Complete\n";
	return 0;
}

//======================================================================================//
//++ Find group average cross sections +++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolver::averageXS(int etaL) {
	// Calculate Group Averaged values ////////////////////////////////////////////////////////////
	vector< vector< vector<double> > > &phiG=phi[etaL];
	// Cross sections
	# pragma omp parallel for
	for (int p=0; p<Ng[etaL+1]; p++) {
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				double phiS=0.0;
				sigmaT[etaL+1][p][i][j]  =0.0;
				nuSigmaF[etaL+1][p][i][j]=0.0;
				chi[etaL+1][p][i][j]     =0.0;
				s_ext[etaL+1][p][i][j]   =0.0;
				for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) {
					phiS+=phiG[g][i][j];
					sigmaT[etaL+1][p][i][j]  +=phiG[g][i][j]*Total(etaL,g,i,j);
					nuSigmaF[etaL+1][p][i][j]+=phiG[g][i][j]*Fission(etaL,g,i,j);
					chi[etaL+1][p][i][j]     +=Chi(etaL,g,i,j);
					s_ext[etaL+1][p][i][j]   +=Source(etaL,g,i,j);
				}
				sigmaT[etaL+1][p][i][j]  /=phiS;
				nuSigmaF[etaL+1][p][i][j]/=phiS;
			}
		}
	}
	# pragma omp parallel for
	for (int pp=0; pp<Ng[etaL+1]; pp++) {
		for (int p=0; p<Ng[etaL+1]; p++) {
			for (int i=0; i<Nx; i++) {
				for (int j=0; j<Ny; j++) {
					double phiS=0.0;
					sigmaS[etaL+1][pp][p][i][j]=0.0;
					for (int gg=omegaP[etaL+1][pp]; gg<omegaP[etaL+1][pp+1]; gg++) {
						double Ssum=0.0;
						for (int g=omegaP[etaL+1][p]; g<omegaP[etaL+1][p+1]; g++) Ssum+=Scatter(etaL,gg,g,i,j);
						sigmaS[etaL+1][pp][p][i][j]+=Ssum*phiG[gg][i][j];
						phiS+=phiG[gg][i][j];
					}
					sigmaS[etaL+1][pp][p][i][j]/=phiS;
				}
			}
		}
	}
}
//======================================================================================//

//======================================================================================//
//++ Use Grey Equation to calculate Eigenvalue +++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
double LOSolver::greyEigenvalue() {
	int etaL, g=0;
	etaL=eta_star-1; 
	vector< vector<double> > &phiG=phi[etaL][g], &j_xG=j_x[etaL][g], &j_yG=j_y[etaL][g];
	vector< vector<double> > &sigmaTG=sigmaT[etaL][g], &sigmaSG=sigmaS[etaL][g][g], &nuSigmaFG=nuSigmaF[etaL][g];
	
	double int_fission=0.0; double int_abs=0.0; double int_jx=0.0; double int_jy=0.0;
	for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) int_fission+=nuSigmaFG[i][j]*phiG[i][j]*hx[i]*hy[j];
	for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) int_abs+=(sigmaTG[i][j]-sigmaSG[i][j])*phiG[i][j]*hx[i]*hy[j];
	for (int j=0; j<Ny; j++) int_jx+=(j_xG[Nx][j]-j_xG[0][j])*hy[j];
	for (int i=0; i<Nx; i++) int_jy+=(j_yG[i][Ny]-j_yG[i][0])*hx[i];
	
	return int_fission/(int_jx+int_jy+int_abs);
}
//======================================================================================//
//++ Use MutiGroup Equation to calculate Eigenvalue +++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
double LOSolver::multigroupEigenvalue(int etaL) {
	vector< vector< vector<double> > > &phiG=phi[etaL], &j_xG=j_x[etaL], &j_yG=j_y[etaL];
	
	double int_fission=0.0, int_abs=0.0, int_jx=0.0, int_jy=0.0;
	for (int g=0; g<Ng[etaL]; g++) for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) int_fission+=Fission(etaL,g,i,j)*phiG[g][i][j]*hx[i]*hy[j];
	for (int g=0; g<Ng[etaL]; g++) {
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				double sigmaA=Total(etaL,g,i,j);
				for (int gg=0; gg<Ng[etaL]; gg++) sigmaA-=Scatter(etaL,g,gg,i,j);
				int_abs+=sigmaA*phiG[g][i][j]*hx[i]*hy[j];
			}
		}
	}
	for (int g=0; g<Ng[etaL]; g++) for (int j=0; j<Ny; j++) int_jx+=(j_xG[g][Nx][j]-j_xG[g][0][j])*hy[j];
	for (int g=0; g<Ng[etaL]; g++) for (int i=0; i<Nx; i++) int_jy+=(j_yG[g][i][Ny]-j_yG[g][i][0])*hx[i];
	
	return int_fission/(int_jx+int_jy+int_abs);
}
//======================================================================================//

//======================================================================================//
double LOSolver::residualIterative(int k, int &i_it, int &j_it, int &g_it) {
	// Calculate Iterative Residuals // Residuals of the actual NDA Equaitons
	double res_it=0.0; // Iterative Residuals
	for (int i=0; i<Nx; i++) { 
		for (int j=0; j<Ny; j++) {
			double sFission=0.0;
			for (int g=0; g<Ng[k]; g++) sFission+=Fission(k,g,i,j)*phi[k][g][i][j];
			for (int g=0; g<Ng[k]; g++) {
				double sScatter=0.0;
				for (int gg=0; gg<Ng[k]; gg++) sScatter+=Scatter(k,gg,g,i,j)*phi[k][gg][i][j];
				double res=abs((j_x[k][g][i+1][j]-j_x[k][g][i][j])/hx[i]+(j_y[k][g][i][j+1]-j_y[k][g][i][j])/hy[j]
				+Total(k,g,i,j)*phi[k][g][i][j]-sScatter-Chi(k,g,i,j)*sFission/k_eff-Source(k,g,i,j));
				if ( res>res_it ) { res_it=res; i_it=i; j_it=j; g_it=g; } // residual
			}
		}
	}
	
	return res_it;
}
//======================================================================================//
double LOSolver::findNormKappa(std::vector< std::vector<double> > &kappaL) {
	double norm_kap=0.0;
	std::vector< std::vector<double> > &phiH=phi[eta_star-1][0], &j_xH=j_x[eta_star-1][0], &j_yH=j_y[eta_star-1][0];
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) {
			double kappa=nuSigmaF[eta_star-1][0][i][j]*phiH[i][j]/
			((j_xH[i+1][j]-j_xH[i][j])/hx[i]+(j_yH[i][j+1]-j_yH[i][j])/hy[j]+
			(sigmaT[eta_star-1][0][i][j]-sigmaS[eta_star-1][0][0][i][i])*phiH[i][j]);
			if ( abs(kappa-kappaL[i][j])>norm_kap ) norm_kap=abs(kappa-kappaL[i][j]);
			kappaL[i][j]=kappa;
		}
	}
	return norm_kap;
}
//======================================================================================//
void LOSolver::logIteration(int etaL, double norm_p, double rho_p, double norm_pH, double rho_pH, double k, double norm_k, double rho_k, double norm_kap, double rho_kap) {
	norm_phi[etaL].push_back(norm_p);
	rho_phi[etaL].push_back(rho_p);
	norm_phiH[etaL].push_back(norm_pH);
	rho_phiH[etaL].push_back(rho_pH);
	k_keff[etaL].push_back(k_eff);
	norm_keff[etaL].push_back(norm_k);
	rho_keff[etaL].push_back(rho_k);
	norm_kappa[etaL].push_back(norm_kap);
	rho_kappa[etaL].push_back(rho_kap);
}
//======================================================================================//



//======================================================================================//
//++ LOSolver Constructor ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int LOSolver::readLO(std::vector<std::string> &line, HOSolver &ho) {
	vector<double> ho_epsilon=ho.epsilon_phi;
	
	Nx=ho.Nx; Ny=ho.Ny;
	x=ho.x; y=ho.y; hx=ho.hx; hy=ho.hy; xe=ho.xe; ye=ho.ye;
	
	// initialize default values
	cout<<"Read LO Data: ";
	vector<double> num_temp;
	vector<string> str_temp;
	try {
		num_temp=parseNumber(line[0]); // Read Line 0
		eta_star=int(num_temp[0]+0.5); // Get number of energy grids
		
		
		Ng.resize(eta_star); //
		Ng[0]=ho.Ng; // Set the number of energy groups for the original grid
		
		omegaP.resize(eta_star); // Define sets for coarse energy grids
		omegaP[0].resize(Ng[0]+1);
		
		omegaP[0][0]=0;
		for (int g=0; g<Ng[0]; g++) omegaP[0][g+1]=omegaP[0][g]+1;
		
		for (int k=1; k<eta_star-1; k++) {
			num_temp=parseNumber(line[k]);
			Ng[k]=int(num_temp[0]+0.5);
			omegaP[k].push_back(0);
			for (int i=1; i<num_temp.size(); i++) omegaP[k].push_back(omegaP[k].back() + int(num_temp[i]+0.5));
		}
		
		
		if (eta_star!=1) {
			Ng[eta_star-1]=1;
			omegaP[eta_star-1].push_back(0);
			omegaP[eta_star-1].push_back(Ng[eta_star-2]);
		}
		
		
		for (int k=1; k<eta_star; k++) if (omegaP[k][Ng[k]]!=Ng[k-1]) throw "Fatal Error: Inconsistency between groups";
		
		
		
		int start_read=eta_star-1;
		if (eta_star==1) start_read=1;
		for (int w=start_read; w<line.size(); w++) {
			str_temp=parseWord(line[w]);
			num_temp=parseNumber(line[w]);
			//cout<<str_temp[0]<<endl;
			if      (str_temp[0]=="LO_phi_epsilon") {
				if ( int(num_temp[0]+0.5) > 0 ) {
					epsilon_phi.resize(eta_star);
					for (int i=0; i<num_temp.size()-1; i++) epsilon_phi[i].push_back(num_temp[i+1]); // Read LO Flux NDA stopping criteria
				}
			}
			else if (str_temp[0]=="LO_keff_epsilon") {
				if ( int(num_temp[0]+0.5) > 0 ) {
					epsilon_keff.resize(eta_star);
					for (int i=0; i<num_temp.size()-1; i++) epsilon_keff[i].push_back(num_temp[i+1]); // Read LO Keff NDA stopping criteria
				}
			}
			else if (str_temp[0]=="LO_phi_truncation") {
				if ( int(num_temp[0]+0.5) > 0 ) {
					epsilon_phi.resize(eta_star);
					for (int i=0; i<num_temp.size()-1; i++) epsilon_phi[i].push_back(num_temp[i+1]); // Read LO Flux NDA stopping criteria
				}
			}
			else if (str_temp[0]=="LO_keff_truncation") {
				if ( int(num_temp[0]+0.5) > 0 ) {
					epsilon_keff.resize(eta_star);
					for (int i=0; i<num_temp.size()-1; i++) epsilon_keff[i].push_back(num_temp[i+1]); // Read LO Keff NDA stopping criteria
				}
			}
			else if (str_temp[0]=="LO_stopping")     {
				if ( int(num_temp[0]+0.5) > 0 ) for (int i=1; i<num_temp.size(); i++) stop_phi.push_back(int(num_temp[i]+0.5)); // Read Iteration # NDA stopping criteria
			}
			else if (str_temp[0]=="Relaxation")     {
				if ( int(num_temp[0]+0.5) > 0 ) for (int i=1; i<num_temp.size(); i++) relaxation.push_back(int(num_temp[i]+0.5)); // Read Iteration # NDA stopping criteria
			}
			else if (str_temp[0]=="Preconditioner")  {
				if ( int(num_temp[0]+0.5) > 0 ) for (int i=1; i<num_temp.size(); i++) preconditioner_type.push_back(int(num_temp[i]+0.5)); // Read preconditioner type
			}
			else if (str_temp[0]=="solver_epsilon")  epsilon_solver=num_temp[0];
			else if (str_temp[0]=="WS_delta")        delta=num_temp[0];
			else if (str_temp[0]=="NDA")             NDASolution=true;
			else if (str_temp[0]=="QD")              NDASolution=false;
			else if (str_temp[0]=="W_solve")         {
				if ( int(num_temp[0]+0.5) > 0 ) for (int i=1; i<num_temp.size(); i++) wSolve.push_back(int(num_temp[i]+0.5));
			}
			else if (str_temp[0]=="Initial_K")       initialK=num_temp[0];
			else if (str_temp[0]=="Grey_Newton")     greyNewton=int(num_temp[0]+0.5);
			else if (str_temp[0]=="Newton_Tol_RA") {
				if      ( num_temp.size()==2 ) { epsilon_phi.back()[0]=num_temp[0]; epsilon_phi.back()[1]=num_temp[1]; }
				else if ( num_temp.size()==1 ) { epsilon_phi.back()[0]=num_temp[0]; epsilon_phi.back()[1]=1e-15; }
				greyNewton=int(num_temp[0]+0.5);
			}
			else if (str_temp[0]=="Fix_K")           fixK=int(num_temp[0]+0.5);
			else if (str_temp[0]=="Track_Factor_Convergence") trackFactorConvergence=true;
			else if (str_temp[0]=="Relative_Convergence") relativeConvergence=true;
			else if (str_temp[0]=="Short_Output") writeOutput=false;
			else if (str_temp[0]=="Jacobi")    gaussSeidel=false;
			else cout<<">> Error: Unknown variable in LO_Data block <"<<str_temp[0]<<">\n";
		}
		if ( not KE_problem ) delta=0.0;
		
		// Initialize Low-order convergence criteria
		if ( epsilon_phi.size()!=eta_star ) {
			epsilon_phi.push_back(ho_epsilon);
			epsilon_phi[0][0]=epsilon_phi[0][0]/10.0;
			for (int k=1; k<eta_star; k++) {
				epsilon_phi.push_back(epsilon_phi[k-1]);
				epsilon_phi[k][0]=epsilon_phi[k][0]/10.0;
			}
		}
		if ( epsilon_keff.size()!=eta_star ) epsilon_keff=epsilon_phi; // default
		if (     stop_phi.size()!=eta_star ) stop_phi=vector<int> (eta_star,1000);
		else for (int k=0; k<eta_star; k++) if (stop_phi[k]<1) stop_phi[k]=1000;
		if (   relaxation.size()!=eta_star-1 ) relaxation=vector<int> (eta_star-1,1);
		else for (int k=0; k<eta_star-1; k++) if (relaxation[k]<1) relaxation[k]=1;
		if ( preconditioner_type.size()!=eta_star ) for (int k=0; k<eta_star; k++) preconditioner_type.push_back(3);
		
		if ( epsilon_solver<1e-30 ) epsilon_solver=epsilon_phi.back()[0]/10.0;
		if ( epsilon_solver>epsilon_phi.back()[0] ) cout<<">> Warning in Solver convergence criteria input!"<<epsilon_solver<<"\n";
		
		// Check Solver Preconditioner
		for (int k=0; k<eta_star; k++) {
			if ( preconditioner_type[k]>3 and preconditioner_type[k]<2 ) {
				cout<<">>Error in preconditioner Type input in grid "<<k<<" use default preconditioner 3"<<endl;
				preconditioner_type[k]=3;
				return 5;
			}
		}
		
		if ( eta_star==1 ) greyNewton=false;
		//if ( greyNewton and epsilon_phi.back().size()!=2 ) epsilon_phi.back().push_back(1e-15);
		
		
		if ( wSolve.size()==0 ) wSolve=(vector<bool> (eta_star-1,false));
		if ( wSolve.size()!=eta_star-1 ) { cout<<">>Error in W solve input \n"; return 6; }
		
		if ( epsilon_phi.back().size()>1 ) if ( epsilon_phi.back()[1]<1e-30 ) epsilon_phi.back()[1]=1e-15;
		if ( greyNewton ) {
			if ( epsilon_phi.back().size()!=2 ) epsilon_phi.back().push_back(epsilon_solver);
			tolNewtonR=epsilon_phi.back()[0];
			tolNewtonA=epsilon_phi.back()[1];
		}
		
		solveK=KE_problem;
		
		vector< vector< vector<double> > > scatterMatrix;
		scatterMatrix.resize(eta_star);
		
		int Nd;
		Ndown.resize(eta_star);
		for (int etaL=0; etaL<eta_star; etaL++) Ndown[etaL]=101010;
		
		for (int k=0; k<ho.sigma_gS.size(); k++) {
			scatterMatrix[0]=ho.sigma_gS[k];
			for (int etaL=1; etaL<eta_star; etaL++) {
				scatterMatrix[etaL].resize(Ng[etaL]);
				for (int pp=0; pp<Ng[etaL]; pp++) {
					scatterMatrix[etaL][pp].resize(Ng[etaL]);
					for (int p=0; p<Ng[etaL]; p++) {
						scatterMatrix[etaL][pp][p]=0.0;
						for (int gg=omegaP[etaL][pp]; gg<omegaP[etaL][pp+1]; gg++) for (int g=omegaP[etaL][p]; g<omegaP[etaL][p+1]; g++) scatterMatrix[etaL][pp][p]+=scatterMatrix[etaL-1][gg][g];
					}
				}
			}
			for (int etaL=0; etaL<eta_star; etaL++) {
				Nd=Ng[etaL]-1;
				for (int gg=Ng[etaL]-1; gg>=0; gg--) {
					for (int g=0; g<gg; g++) if ( abs(scatterMatrix[etaL][gg][g])>1e-30 ) Nd=gg-1;
				}
				if (Nd<Ndown[etaL]) Ndown[etaL]=Nd;
			}
		}

		ho.NDASolution=NDASolution;
	}
	catch (std::string error) {
		cout<<"Low-order Input Error >> "<<error;
		return 1;
	}
	cout<<"Complete\n";
	return 0;
}
//======================================================================================//

//======================================================================================//
//++ HOSolver Writer +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolver::writeLO(ofstream& outfile) {
	
	outfile<<" -- Low-order Solution Options -- \n";
	outfile<<"+-----------------------------------------------+\n";
	outfile<<" -- Energy Grid -- \n";
	outfile<<"Number of Low-order Grids -- "<<eta_star<<" -- \n";
	outfile<<"Number of Groups on Grid "<<0<<" is "<<Ng[0]<<endl;
	for (int k=1; k<eta_star; k++) {
		outfile<<"Number of Groups on Grid "<<k<<" is "<<Ng[k]<<" with group combinations";
		for (int g=0; g<Ng[k]; g++) outfile<<"  "<<omegaP[k][g]+1<<" to "<<omegaP[k][g+1];
		outfile<<endl;
	}
	outfile<<writeLinearSolverType(); 
	outfile<<writeLOSolverType(); 
	outfile<<"BiCGStab Solver Tolerance: "<<epsilon_solver<<endl;
	outfile<<  "          Grid #                : ";
	for (int k=0; k<eta_star; k++ ) outfile<<setw(10)<<k<<setw(6)<<"  ";
	outfile<<"\nLow-order  Convergence  Criteria: ";
	for (int k=0; k<eta_star; k++ ) outfile<<print_out(epsilon_phi[k][0]);
	outfile<<"\n        Absolute     Criteria   : ";
	for (int k=0; k<eta_star; k++ ) {
		if ( epsilon_phi[k].size()>1 ) outfile<<print_out(epsilon_phi[k][1]);
		else                           outfile<<print_out(0.0/0.0);
	}
	outfile<<"\nLO K_eff   Convergence  Criteria: ";
	for (int k=0; k<eta_star; k++ ) outfile<<print_out(epsilon_keff[k][0]);
	outfile<<"\n        Absolute     Criteria   : ";
	for (int k=0; k<eta_star; k++ ) {
		if ( epsilon_keff[k].size()>1 ) outfile<<print_out(epsilon_keff[k][1]);
		else                           outfile<<print_out(0.0/0.0);
	}
	outfile<<"\nMax Num. of Low-order Iterations: ";
	for (int k=0; k<eta_star; k++ ) outfile<<setw(10)<<stop_phi[k]<<setw(6)<<"  ";
	outfile<<"\nNumber of Low-order  Relaxations: ";
	for (int k=0; k<eta_star-1; k++ ) outfile<<setw(10)<<relaxation[k]<<setw(6)<<"  ";
	outfile<<"\nLow-order    Preconditioner Type: ";
	for (int k=0; k<eta_star; k++ ) outfile<<setw(10)<<preconditioner_type[k]<<setw(6)<<"  ";
	outfile<<endl;
	outfile<<" -- Options -- \n";
	outfile<<"Initial K = "<<print_out(initialK)<<" \n";
	outfile<<"Fix K after it Converges        : "<<setw(10)<<fixK<<setw(6)<<" \n";
	if ( trackFactorConvergence ) outfile<<"Track Convergence of Consistency Factors \n";
	if ( relativeConvergence )    outfile<<"Use Relative Convergence Criteria \n";
	if ( not writeOutput )        outfile<<"No output of LO Solution \n";
	if ( gaussSeidel )            outfile<<"Block Gauss Seidel Iterations in Energy \n";
	else                          outfile<<"Block Jacobi Iterations in Energy \n";
	outfile<<"W Cycle Solve on  Grids         : ";
	for (int k=0; k<eta_star-1; k++ ) outfile<<setw(10)<<wSolve[k]<<setw(6)<<"  ";
	outfile<<endl;
	if ( KE_problem ) {
		if ( greyNewton ) outfile<<"Grey Newton Iterations with Relative Tolerance "<<print_out(tolNewtonR)<<" and Absolute Tolerance "<<print_out(tolNewtonA)<<endl;
		else outfile<<"Grey Weilandt-Shift Iterations with Perameter Delta "<<print_out(delta)<<endl;
	}
	// Energy Solution Grid
	outfile<<" -- Energy Grid -- \n";
	outfile<<"Number of Low-order Grids -- "<<eta_star<<" -- \n";
	outfile<<"Number of Groups on Grid "<<0<<" is "<<Ng[0]<<endl;
	for (int k=1; k<eta_star; k++) {
		outfile<<"Number of Groups on Grid "<<k<<" is "<<Ng[k]<<" with group combinations";
		for (int g=0; g<Ng[k]; g++) outfile<<"  "<<omegaP[k][g]+1<<" to "<<omegaP[k][g+1];
		outfile<<endl;
	}
	
	//outfile<<"One Group Grid "<<" with group combination "<<omegaP[eta_star-1][1]-omegaP[eta_star-1][0]<<endl;
	outfile<<"Number of groups with only Down Scatter: ";
	for (int k=0; k<Ndown.size(); k++) outfile<<Ndown[k]<<"  ";
	outfile<<endl;
	
	
	outfile<<"+-----------------------------------------------+\n";
	//return 0;
}
//======================================================================================//

//======================================================================================//
//++ Function to Recursively Write Iteration data ++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolver::rec_write_iteration_out(int it, int etaL, ofstream& outfile) {
	int i, k, m;
	
	k=Ng[etaL]*num_losi[etaL][it];
	outfile<<" # of Grid "<<etaL<<" Iterations "<<num_losi[etaL][it+1]-num_losi[etaL][it]<<endl;
	
	int num_start=num_losi[etaL][it];
	int num_total=num_losi[etaL][it+1]-num_start;
	
	for (m=0; m<int(num_total/10); m++) {
		outfile<<string((etaL+1)*5,' ')<<"Grid "<<etaL<<"  Iteration  #  :";
		for (i=num_start+m*10; i<num_start+(m+1)*10; i++) outfile<<setw(10)<<i-num_losi[etaL][it]+1<<setw(6)<<" ";
		outfile<<endl;
		outfile<<string((etaL+1)*5,' ')<<"---- Flux      rho    :";
		for (i=num_start+m*10; i<num_start+(m+1)*10; i++) outfile<<print_out(rho_phi[etaL][i]);
		outfile<<endl;
		outfile<<string((etaL+1)*5,' ')<<"Flux Diff. L-inf Norm :";
		for (i=num_start+m*10; i<num_start+(m+1)*10; i++) outfile<<print_out(norm_phi[etaL][i]);
		outfile<<endl;
		outfile<<string((etaL+1)*5,' ')<<"---- K_eff    rho     :";
		for (i=num_start+m*10; i<num_start+(m+1)*10; i++) outfile<<print_out(rho_keff[etaL][i]);
		outfile<<endl;
		outfile<<string((etaL+1)*5,' ')<<"K_eff Diff. L-inf Norm:";
		for (i=num_start+m*10; i<num_start+(m+1)*10; i++) outfile<<print_out(norm_keff[etaL][i]);
		outfile<<endl;
		if ( greyNewton and etaL+1==eta_star ) {
			if ( m==0 ) outfile<<string((etaL+1)*5,' ')<<"Newton Initial  Res.  :"<<print_out(resInitialNewton[it])<<endl;
			outfile<<string((etaL+1)*5,' ')<<"Newton Iter. Residuals:";
			for (i=num_start+m*10; i<num_start+(m+1)*10; i++) outfile<<print_out(resNewton[i]);
			outfile<<endl;
		}
	}
	if ( m*10<num_total ) {
		outfile<<string((etaL+1)*5,' ')<<"Grid "<<etaL<<"  Iteration  #  :";
		for (i=num_start+m*10; i<num_losi[etaL][it+1]; i++) outfile<<setw(10)<<i-num_losi[etaL][it]+1<<setw(6)<<" ";
		outfile<<endl;
		outfile<<string((etaL+1)*5,' ')<<"---- Flux      rho    :";
		for (i=num_start+m*10; i<num_losi[etaL][it+1]; i++) outfile<<print_out(rho_phi[etaL][i]);
		outfile<<endl;
		outfile<<string((etaL+1)*5,' ')<<"Flux Diff. L-inf Norm :";
		for (i=num_start+m*10; i<num_losi[etaL][it+1]; i++) outfile<<print_out(norm_phi[etaL][i]);
		outfile<<endl;
		outfile<<string((etaL+1)*5,' ')<<"---- K_eff    rho     :";
		for (i=num_start+m*10; i<num_losi[etaL][it+1]; i++) outfile<<print_out(rho_keff[etaL][i]);
		outfile<<endl;
		outfile<<string((etaL+1)*5,' ')<<"K_eff Diff. L-inf Norm:";
		for (i=num_start+m*10; i<num_losi[etaL][it+1]; i++) outfile<<print_out(norm_keff[etaL][i]);
		outfile<<endl;
		if ( greyNewton and KE_problem and etaL+1==eta_star ) {
			if ( m==0 ) outfile<<string((etaL+1)*5,' ')<<"Newton Initial  Res.  :"<<print_out(resInitialNewton[it])<<endl;
			outfile<<string((etaL+1)*5,' ')<<"Newton Iter. Residuals:";
			for (i=num_start+m*10; i<num_losi[etaL][it+1]; i++) outfile<<print_out(resNewton[i]);
			outfile<<endl;
		}
	}
	
	if (etaL+1!=eta_star) {
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) {
			outfile<<string((etaL+2)*5,' ')<<"Grid "<<etaL<<" Iteration # "<<i-num_losi[etaL][it]+1;
			rec_write_iteration_out(i, etaL+1, outfile);
		}
	}
}
//======================================================================================//
void LOSolver::rec_write_iteration_dat(int it, int etaL, ofstream& datfile) {
	int i, k;
	if ( num_losi[etaL][it+1]-num_losi[etaL][it]>0 ) {
		k=Ng[etaL]*num_losi[etaL][it];
		datfile<<" # of Grid "<<etaL<<" Iterations ,"<<num_losi[etaL][it+1]-num_losi[etaL][it]<<","<<endl;
		datfile<<string((etaL+1)*5,' ')<<"Grid "<<etaL<<"  Iteration  #  :,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<setw(10)<<i-num_losi[etaL][it]+1<<setw(6)<<",";
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"---- Flux      rho    :,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(rho_phi[etaL][i]);
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"Flux Diff. L-inf Norm :,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(norm_phi[etaL][i]);
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"-- Grey Flux   rho    :,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(rho_phiH[etaL][i]);
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"Grey Flux Diff.  Norm :,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(norm_phiH[etaL][i]);
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"------- K_eff         :,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(k_keff[etaL][i]);
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"---- K_eff    rho     :,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(rho_keff[etaL][i]);
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"K_eff Diff. L-inf Norm:,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(norm_keff[etaL][i]);
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"---- Kappa    rho     :,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(rho_kappa[etaL][i]);
		datfile<<endl;
		datfile<<string((etaL+1)*5,' ')<<"Kappa Diff. L-inf Norm:,";
		for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(norm_kappa[etaL][i]);
		datfile<<endl;
		if (etaL+1!=eta_star) {
			for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) {
				datfile<<string((etaL+2)*5,' ')<<"Grid "<<etaL<<" Iteration # "<<i-num_losi[etaL][it]+1<<",";
				rec_write_iteration_dat(i, etaL+1, datfile);
			}
		}
		else if ( greyNewton and KE_problem ) {
			datfile<<string((etaL+1)*5,' ')<<"Newton Initial  Res.  :,"<<print_csv(resInitialNewton[it])<<endl;
			datfile<<string((etaL+1)*5,' ')<<"Newton Iter. Residuals:,";
			for (i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) datfile<<print_csv(resNewton[i]);
			datfile<<endl;
		}
	}
}
//======================================================================================//
void LOSolver::rec_write_iteration_long_dat(int it, int etaL, ofstream& datfile) {
	int k=Ng[etaL]*num_losi[etaL][it];
	for (int i=num_losi[etaL][it]; i<num_losi[etaL][it+1]; i++) {
		for (int g=0; g<Ng[etaL]; g++) {
			//datfile<<string((etaL+1)*5,' ')<<"Pre cond. time ,"<<print_csv(dt_pc[etaL][k])<<" # of BiCGStab , "
			//<<num_logm[etaL][k]<<" , Err. in LO Sol. ,"<<print_csv(err_lo[etaL][k])<<endl;
			datfile<<string((etaL+1)*5,' ')<<" # of BiCGStab , "
			<<num_logm[etaL][k]<<" , Err. in LO Sol. ,"<<print_csv(err_lo[etaL][k])<<endl;
			k++;
		}
		datfile<<string((etaL+1)*5,' ')<<"Convergence Rate, "<<print_csv(rho_phi[etaL][i])<<endl;
		if (etaL+1==eta_star) continue;
		else rec_write_iteration_long_dat(i, etaL+1, datfile);
	}
}
//======================================================================================//


void LOSolver::writeSpatialGrid(ofstream& outfile) {
	
	// output grid
	outfile<<"\n -- Low Order Solution Grid -- \n";
	outfile<<" X grid \n"<<" Index     Cell Edge       Width Avg     Cell center      Cell Width   ";
	if ( kbc==2 ) outfile<<"  Quad 2 BC In  "<<"  Quad 4 BC In  "<<endl; // Write Bottom and Top BC
	else {
		if ( reflectiveB ) outfile<<" Bottom BC REFL ";
		else outfile<<"  Bottom BC In  ";
		if ( reflectiveT ) outfile<<" Top    BC REFL "<<endl;
		else outfile<<"  Top    BC In  "<<endl;
	}
	
	for (int i=0; i<Nx; i++) {
		outfile<<setw(6)<<i+1<<print_out(x[i])<<print_out(xe[i])<<print_out((x[i]+x[i+1])/2)<<print_out(hx[i])<<endl;
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
		outfile<<setw(6)<<j+1<<print_out(y[j])<<print_out(ye[j])<<print_out((y[j]+y[j+1])/2)<<print_out(hy[j])<<endl;
	}
	outfile<<setw(6)<<Ny+1<<print_out(y[Ny])<<print_out(ye[Ny])<<endl;
	
}


//======================================================================================//
//++ Solve One-group eigenvalue problem with Newtons Iterations ++++++++++++++++++++++++//
//======================================================================================//
bool LOSolver::NDAgreyNewtonSolution(int preconditioner, vector< vector<double> > &phiLast) {
	double res=0;
	int etaL=eta_star-1, g=0;
	vector< vector<double> > &phiG=phi[etaL][g], &j_xG=j_x[etaL][g], &j_yG=j_y[etaL][g];
	vector<double> &phiLG=phiL[etaL][g], &phiRG=phiR[etaL][g], &phiBG=phiB[etaL][g], &phiTG=phiT[etaL][g];
	vector< vector<double> > &D_xPG=D_xP[etaL][g], &D_xNG=D_xN[etaL][g], &D_yPG=D_yP[etaL][g], &D_yNG=D_yN[etaL][g];
	vector<double> &FLG=FL[etaL][g], &FRG=FR[etaL][g], &FBG=FB[etaL][g], &FTG=FT[etaL][g];
	vector< vector<double> > &sigmaTG=sigmaT[etaL][g], &sigmaSG=sigmaS[etaL][g][g], &nuSigmaFG=nuSigmaF[etaL][g];
	
	double kL=k_eff;
	//vector< vector<double> > phiLast=phi[etaL][g];
	//vector< vector<double> > j_xL=j_x[etaL][g], j_yL=j_y[etaL][g];
	
	//temfile<<"NDA Solution Grid "<<etaL<<" group "<<g<<endl;
	
	vector<double> fL(Ny,0.0), fB(Nx,0.0), fR(Ny,0.0), fT(Nx,0.0);
	
	//# pragma omp parallel for
	for (int j=0; j<Ny; j++) fL[j]=D_xPG[0][j]/(D_xNG[0][j]+xe[0]*FLG[j]); // Left   BC Flux
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) fB[i]=D_yPG[i][0]/(D_yNG[i][0]+ye[0]*FBG[i]); // Bottom BC Flux
	//# pragma omp parallel for
	for (int j=0; j<Ny; j++) fR[j]=D_xNG[Nx][j]/(D_xPG[Nx][j]+xe[Nx]*FRG[j]); // Right BC Flux
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) fT[i]=D_yNG[i][Ny]/(D_yPG[i][Ny]+ye[Ny]*FTG[i]); // Top   BC Flux
	
	int N_ukn=Nx*Ny;
	
	std::vector<double> d(N_ukn+1), b(N_ukn+1);
	
	// Find Infinity Norm of Newton Residual
	double res_initial=NewtonLNorm(0);
	
	int p=0;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) { 
			b[p]=-(j_xG[i+1][j]-j_xG[i][j])/hx[i]-(j_yG[i][j+1]-j_yG[i][j])/hy[j]-
				(sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/k_eff)*phiG[i][j];
			d[p]=0;
			p++;
		}
	}
	b[N_ukn]=(x[Nx]-x[0])*(y[Ny]-y[0]);
	for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) b[N_ukn]-=hx[i]*hy[j]*phiG[i][j];
	d[N_ukn]=0;
	
	//vector<int> reserveSize(N_ukn+1,7);
	//reserveSize[N_ukn]=N_ukn;
	
	//A.reserve(VectorXi::Constant(N_ukn+1,7));
	//A.reserve(reserveSize);
	std::vector<T> triplets(7*N_ukn);
	
	// Assign matrix A and vector b
	p=0; int k=0; // ++++++++ Central Cell ++++++++
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			triplets[k++] = T(p,p,D_xNG[i+1][j]/xe[i+1]/hx[i] + D_xPG[i][j]/xe[i]/hx[i] + D_yNG[i][j+1]/ye[j+1]/hy[j] + D_yPG[i][j]/ye[j]/hy[j]+
			sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/k_eff);
			triplets[k++] = T(p,N_ukn,-nuSigmaFG[i][j]*phiG[i][j]);
			p++;
		}
	}
	{
	int i, j;
	p=0; // ++++++++ Periphery Cells ++++++++
	if ( Nx==1 and Ny==1 ) { // Single Cell
		i=0; j=0;
		triplets[k++] = T(0,0,-(D_xNG[0][0]/xe[0]/hx[0])*fL[j]-(D_yNG[0][0]/ye[0]/hy[0])*fB[i]
		-(D_xPG[1][0]/xe[1]/hx[0])*fR[j]-(hx[0]*D_yPG[0][1]/ye[1]/hy[0])*fT[i]);
	}
	else if ( Nx==1 ) { // One Cell Wide
		i=0; int j=0;
		triplets[k++] = T(p,p,-(D_xNG[0][0]/xe[0]/hx[0])*fL[j]);   triplets[k++] = T(p,p,-(D_yNG[0][0]/ye[0]/hy[0])*fB[i]);
		triplets[k++] = T(p,p,-(D_xPG[1][0]/xe[1]/hx[0])*fR[j]);   triplets[k++] = T(p,1,-D_yPG[0][1]/ye[1]/hy[0]);
		p++; // Bottom Cell
		for (j=1; j<Ny-1; j++ ) { // Middle Cells
			triplets[k++] = T(p,p,-(D_xNG[0][j]/xe[0]/hx[0])*fL[j]);   triplets[k++] = T(p,j-1,-hx[0]*D_yNG[0][j]  /ye[j]  /hy[j]);
			triplets[k++] = T(p,p,-(D_xPG[1][j]/xe[1]/hx[0])*fR[j]);   triplets[k++] = T(p,j+1,-hx[0]*D_yPG[0][j+1]/ye[j+1]/hy[j]);
			p++;
		}
		j=Ny-1;
		triplets[k++] = T(p,p,-(D_xNG[0][Ny-1]/xe[0]/hx[0])*fL[j]);   triplets[k++] = T(p,Ny-2,-D_yNG[0][Ny-1]/ye[Ny-1]/hy[Ny-1]);
		triplets[k++] = T(p,p,-(D_xPG[1][Ny-1]/xe[1]/hx[0])*fR[j]);   triplets[k++] = T(p,p,  -(D_yPG[0][Ny]  /ye[Ny]  /hy[Ny-1])*fT[i]);
		p++; // Top Cell	
	}
	else if ( Ny==1 ) { // One Cell Tall
		i=0; j=0;
		triplets[k++] = T(p,p,-(D_xNG[0][0]/xe[0]/hx[0])*fL[j]);   triplets[k++] = T(p,p,-(D_yNG[0][0]/ye[0]/hy[0])*fB[i]);
		triplets[k++] = T(p,1, -D_xPG[1][0]/xe[1]/hx[0]);          triplets[k++] = T(p,p,-(D_yPG[0][1]/ye[1]/hy[0])*fT[i]);
		p++; // Left Cell
		for (i=1; i<Nx-1; i++ ) { // Middle Cell
			triplets[k++] = T(p,i-1,-D_xNG[i][0]  /xe[i]  /hx[i]);  triplets[k++] = T(p,p,-(D_yNG[i][0]/ye[0]/hy[0])*fB[i]);
			triplets[k++] = T(p,i+1,-D_xPG[i+1][0]/xe[i+1]/hx[i]);  triplets[k++] = T(p,p,-(D_yPG[i][1]/ye[1]/hy[0])*fT[i]);
			p++;
		}
		i=Nx-1;
		triplets[k++] = T(p,Nx-2,-D_xNG[Nx-1][0]/xe[Nx-1]/hx[Nx-1]);         triplets[k++] = T(p,p,-(D_yNG[Nx-1][0]/ye[0]/hy[0])*fB[i]);
		triplets[k++] = T(p,p,  -(D_xPG[Nx][0]  /xe[Nx]  /hx[Nx-1])*fR[j]);  triplets[k++] = T(p,p,-(D_yPG[Nx-1][1]/ye[1]/hy[0])*fT[i]);
		p++; // Right Cell
	}
	else {
		i=0; j=0;
		triplets[k++] = T(p,p,-(D_xNG[0][0]/xe[0]/hx[0])*fL[j]);   triplets[k++] = T(p,p,-(D_yNG[0][0]/ye[0]/hy[0])*fB[i]);
		triplets[k++] = T(p,1, -D_xPG[1][0]/xe[1]/hx[0]);          triplets[k++] = T(p,Nx,-D_yPG[0][1]/ye[1]/hy[0]);
		p++; // Bottom Left Corner
		for (i=1; i<Nx-1; i++ ) { // Bottom
			triplets[k++] = T(p,p-1,-D_xNG[i][0]  /xe[i]  /hx[i]);  triplets[k++] = T(p,p,  -(D_yNG[i][0]/ye[0]/hy[0])*fB[i]);
			triplets[k++] = T(p,p+1,-D_xPG[i+1][0]/xe[i+1]/hx[i]);  triplets[k++] = T(p,Nx+i,-D_yPG[i][1]/ye[1]/hy[0]);
			p++;
		}
		i=Nx-1;
		triplets[k++] = T(p,Nx-2,-D_xNG[Nx-1][0]/xe[Nx-1]/hx[Nx-1]);        triplets[k++] = T(p,p,    -(D_yNG[Nx-1][0]/ye[0]/hy[0])*fB[i]);
		triplets[k++] = T(p,p,  -(D_xPG[Nx][0]  /xe[Nx]  /hx[Nx-1])*fR[j]); triplets[k++] = T(p,2*Nx-1,-D_yPG[Nx-1][1]/ye[1]/hy[0]);
		p++; // Bottom Right Corner
		for (j=1; j<Ny-1; j++ ) { // Middle
			i=0; // Left Side
			triplets[k++] = T(p,p, -(D_xNG[i][j]  /xe[i]  /hx[i])*fL[j]); triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
			triplets[k++] = T(p,p+1,-D_xPG[i+1][j]/xe[i+1]/hx[i]);        triplets[k++] = T(p,p+Nx,-D_yPG[i][j+1]/ye[j+1]/hy[j]);
			p++;
			for (i=1; i<Nx-1; i++ ) { // Centre Cells
				triplets[k++] = T(p,p-1,-D_xNG[i][j]  /xe[i]  /hx[i]);  triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
				triplets[k++] = T(p,p+1,-D_xPG[i+1][j]/xe[i+1]/hx[i]);  triplets[k++] = T(p,p+Nx,-D_yPG[i][j+1]/ye[j+1]/hy[j]);
				p++;
			}
			i=Nx-1; // Right Side
			triplets[k++] = T(p,p-1,-D_xNG[i][j]  /xe[i]  /hx[i]);         triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
			triplets[k++] = T(p,p, -(D_xPG[i+1][j]/xe[i+1]/hx[i])*fR[j]);  triplets[k++] = T(p,p+Nx,-D_yPG[i][j+1]/ye[j+1]/hy[j]);
			p++;
		}
		i=0; j=Ny-1; // Top Left Corner
		triplets[k++] = T(p,p, -(D_xNG[0][j]/xe[0]/hx[0])*fL[j]);  triplets[k++] = T(p,p-Nx,-D_yNG[0][j]  /ye[j]  /hy[j]);
		triplets[k++] = T(p,p+1,-D_xPG[1][j]/xe[1]/hx[0]);         triplets[k++] = T(p,p,  -(D_yPG[0][j+1]/ye[j+1]/hy[j])*fT[i]);
		p++;
		for (i=1; i<Nx-1; i++ ) { // Top
			triplets[k++] = T(p,p-1,-D_xNG[i][j]  /xe[i]  /hx[i]);  triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
			triplets[k++] = T(p,p+1,-D_xPG[i+1][j]/xe[i+1]/hx[i]);  triplets[k++] = T(p,p,  -(D_yPG[i][j+1]/ye[j+1]/hy[j])*fT[i]);
			p++;
		}
		i=Nx-1; // Top Right Corner
		triplets[k++] = T(p,p-1,-D_xNG[i][j]  /xe[i]  /hx[i]);         triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
		triplets[k++] = T(p,p, -(D_xPG[i+1][j]/xe[i+1]/hx[i])*fR[j]);  triplets[k++] = T(p,p,  -(D_yPG[i][j+1]/ye[j+1]/hy[j])*fT[i]);
		p++;
	}
	}
	
	p=0; // Normalization Row
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			triplets[k++] = T(N_ukn,p,hx[i]*hy[j]);
			p++;
		}
	}
	
	//
	constructAndSolve(preconditioner, etaL, g, d, b, triplets);
	
	
	// set solution back to problem values
	p=0;
	for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) phiG[i][j]+=d[p++]; // Cell Centre Flux
	k_eff=1/(1/k_eff + d[N_ukn]);
	//# pragma omp parallel for
	for (int j=0; j<Ny; j++) phiLG[j]=phiG[0][j]*fL[j]; // Left   BC Flux
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) phiBG[i]=phiG[i][0]*fB[i]; // Bottom BC Flux
	//# pragma omp parallel for
	for (int j=0; j<Ny; j++) phiRG[j]=phiG[Nx-1][j]*fR[j]; // Right BC Flux
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) phiTG[i]=phiG[i][Ny-1]*fT[i]; // Top   BC Flux
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
	calculateGroupCurrents(etaL, g);
	
	// Find Infinity Norm of Newton Residual
	double res_norm=NewtonLNorm(0);
	
	//Armijo Line Search
	bool newtonWorked=true;
	double alphaA=1e-4, lambda=-1.0;
	double etaA=0, sigmaA=0.5;
	//if ( res_norm>res_initial ) cout<<"Armijo \n";
	int jA=0, Amax=10;
	bool fix_negative=false;
	if ( stop_phi[etaL]==1 and !greySolutionPositive() ) fix_negative=true;
	
	while ( ( res_norm >= (1-alphaA*lambda)*res_initial or fix_negative ) and jA<Amax ) {
		jA++;
		//cout<<"Armijo Step "<<jA<<endl;
		lambda*=0.5;
		// set solution back to problem values
		p=0;
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) { phiG[i][j]+=lambda*d[p]; p++; } // Cell Centre Flux
		}
		k_eff=1/(1/k_eff + lambda*d[N_ukn]);
		//# pragma omp parallel for
		for (int j=0; j<Ny; j++) phiLG[j]=phiG[0][j]*fL[j]; // Left   BC Flux
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) phiBG[i]=phiG[i][0]*fB[i]; // Bottom BC Flux
		//# pragma omp parallel for
		for (int j=0; j<Ny; j++) phiRG[j]=phiG[Nx-1][j]*fR[j]; // Right BC Flux
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) phiTG[i]=phiG[i][Ny-1]*fT[i]; // Top   BC Flux
		
		
		////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
		calculateGroupCurrents(etaL, g);
		
		// Find Infinity Norm of Newton Residual
		res_norm=NewtonLNorm(0);
		fix_negative=!greySolutionPositive();
	}
	if (jA==Amax) {
		newtonWorked=false;
		cout<<">> Line search failed in Newton method\n";
	}

	// Check that the solution is positive
	
	// Matrix Residual
	double norm=-(x[Nx]-x[0])*(y[Ny]-y[0]);
	for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) norm+=hx[i]*hy[j]*phiG[i][j];
	res_mbal[etaL]=abs(norm);
	p=0;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			res=abs((j_xG[i+1][j]-j_xG[i][j])/hx[i]+(j_yG[i][j+1]-j_yG[i][j])/hy[j]+
				(sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/kL)*phiG[i][j] - nuSigmaFG[i][j]*phiLast[i][j]*d[N_ukn]);
			if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
			p++;
		}
	}
	return newtonWorked;
}
//======================================================================================//
//++ function to solve NDA problem ++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolver::NDAfixedSourceSolution(int preconditioner, int etaL, int g, std::vector< std::vector<double> > &source, std::vector< std::vector<double> > &loss) {
	double res=0;
	vector< vector<double> > &phiG=phi[etaL][g];
	vector<double> &phiLG=phiL[etaL][g], &phiRG=phiR[etaL][g], &phiBG=phiB[etaL][g], &phiTG=phiT[etaL][g];
	vector< vector<double> > &D_xPG=D_xP[etaL][g], &D_xNG=D_xN[etaL][g], &D_yPG=D_yP[etaL][g], &D_yNG=D_yN[etaL][g];
	vector<double>     &FLG=    FL[etaL][g],     &FRG=    FR[etaL][g],     &FBG=    FB[etaL][g],     &FTG=    FT[etaL][g];
	vector<double>   &jInLG=  jInL[etaL][g],   &jInRG=  jInR[etaL][g],   &jInBG=  jInB[etaL][g],   &jInTG=  jInT[etaL][g];
	vector<double> &phiInLG=phiInL[etaL][g], &phiInRG=phiInR[etaL][g], &phiInBG=phiInB[etaL][g], &phiInTG=phiInT[etaL][g];
	
	//temfile<<"NDA Solution Grid "<<etaL<<" group "<<g<<endl;
	
	vector<double> aL(Ny,0.0), aB(Nx,0.0), aR(Ny,0.0), aT(Nx,0.0);
	vector<double> bL(Ny,0.0), bB(Nx,0.0), bR(Ny,0.0), bT(Nx,0.0);
	// \phi_e = a*\phi_c + b
	//# pragma omp parallel for
	for (int j=0; j<Ny; j++) {
		aL[j]=D_xPG[0][j]/(D_xNG[0][j]+xe[0]*FLG[j]); // Left   BC Flux
		bL[j]=xe[0]*(jInLG[j]+FLG[j]*phiInLG[j])/(D_xNG[0][j]+xe[0]*FLG[j]); // Left   BC Flux
	}
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		aB[i]=D_yPG[i][0]/(D_yNG[i][0]+ye[0]*FBG[i]); // Bottom BC Flux
		bB[i]=ye[0]*(jInBG[i]+FBG[i]*phiInBG[i])/(D_yNG[i][0]+ye[0]*FBG[i]); // Bottom BC Flux
	}
	//# pragma omp parallel for
	for (int j=0; j<Ny; j++) {
		aR[j]=D_xNG[Nx][j]/(D_xPG[Nx][j]+xe[Nx]*FRG[j]); // Right BC Flux
		bR[j]=xe[Nx]*(jInRG[j]-FRG[j]*phiInRG[j])/(D_xPG[Nx][j]+xe[Nx]*FRG[j]); // Right BC Flux
	}
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		aT[i]=D_yNG[i][Ny]/(D_yPG[i][Ny]+ye[Ny]*FTG[i]); // Top   BC Flux
		bT[i]=ye[Ny]*(jInTG[i]-FTG[i]*phiInTG[i])/(D_yPG[i][Ny]+ye[Ny]*FTG[i]); // Top   BC Flux
	}
	
	int N_ukn=Nx*Ny;
	
	std::vector<double> d(N_ukn), b(N_ukn);
	
	int p=0; // initialize solution vector guess to transport solution
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) { 
			d[p++]=phiG[i][j]; // Initial guess

		}
	}
	
	
	std::vector<T> triplets(5*N_ukn);
	// Assign matrix A and vector b
	p=0; int k=0; // ++++++++ Central Cell ++++++++
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			triplets[k++] = T(p,p,D_xNG[i+1][j]/xe[i+1]/hx[i] + D_xPG[i][j]/xe[i]/hx[i] + D_yNG[i][j+1]/ye[j+1]/hy[j] + D_yPG[i][j]/ye[j]/hy[j]+loss[i][j]);
			b[p++]=source[i][j]; // Right hand side
		}
	}
	{
	int i, j;
	p=0; // ++++++++ Periphery Cells ++++++++
	if ( Nx==1 and Ny==1 ) { // Single Cell
		i=0; j=0;
		triplets[k++] = T(0,0,-(D_xNG[0][0]/xe[0]/hx[0])*aL[j]-(D_yNG[0][0]/ye[0]/hy[0])*aB[i]
		-(D_xPG[1][0]/xe[1]/hx[0])*aR[j]-(hx[0]*D_yPG[0][1]/ye[1]/hy[0])*aT[i]);
		b[p++]+=(D_xNG[0][0]/xe[0]/hx[0])*bL[j]+(D_yNG[0][0]/ye[0]/hy[0])*bB[i]+(D_xPG[1][0]/xe[1]/hx[0])*bR[j]+(hx[0]*D_yPG[0][1]/ye[1]/hy[0])*bT[i];
	}
	else if ( Nx==1 ) { // One Cell Wide
		i=0; int j=0; // Bottom Cell
		triplets[k++] = T(p,p,-(D_xNG[0][0]/xe[0]/hx[0])*aL[j]);   triplets[k++] = T(p,p,-(D_yNG[0][0]/ye[0]/hy[0])*aB[i]);
		triplets[k++] = T(p,p,-(D_xPG[1][0]/xe[1]/hx[0])*aR[j]);   triplets[k++] = T(p,1,-D_yPG[0][1]/ye[1]/hy[0]);
		b[p++]+=(D_xNG[0][0]/xe[0]/hx[0])*bL[j]+(D_yNG[0][0]/ye[0]/hy[0])*bB[i]+(D_xPG[1][0]/xe[1]/hx[0])*bR[j];
		for (j=1; j<Ny-1; j++ ) { // Middle Cells
			triplets[k++] = T(p,p,-(D_xNG[0][j]/xe[0]/hx[0])*aL[j]);   triplets[k++] = T(p,j-1,-hx[0]*D_yNG[0][j]  /ye[j]  /hy[j]);
			triplets[k++] = T(p,p,-(D_xPG[1][j]/xe[1]/hx[0])*aR[j]);   triplets[k++] = T(p,j+1,-hx[0]*D_yPG[0][j+1]/ye[j+1]/hy[j]);
			b[p++]+=(D_xNG[0][j]/xe[0]/hx[0])*bL[j]+(D_xPG[1][j]/xe[1]/hx[0])*bR[j];
		}
		j=Ny-1; // Top Cell
		triplets[k++] = T(p,p,-(D_xNG[0][Ny-1]/xe[0]/hx[0])*aL[j]);   triplets[k++] = T(p,Ny-2,-D_yNG[0][Ny-1]/ye[Ny-1]/hy[Ny-1]);
		triplets[k++] = T(p,p,-(D_xPG[1][Ny-1]/xe[1]/hx[0])*aR[j]);   triplets[k++] = T(p,p,  -(D_yPG[0][Ny]  /ye[Ny]  /hy[Ny-1])*aT[i]);
		b[p++]+=(D_xNG[0][Ny-1]/xe[0]/hx[0])*bL[j] + (D_xPG[1][Ny-1]/xe[1]/hx[0])*bR[j] + (D_yPG[0][Ny]  /ye[Ny]  /hy[Ny-1])*bT[i];
	}
	else if ( Ny==1 ) { // One Cell Tall
		i=0; j=0; // Left Cell
		triplets[k++] = T(p,p,-(D_xNG[0][0]/xe[0]/hx[0])*aL[j]);   triplets[k++] = T(p,p,-(D_yNG[0][0]/ye[0]/hy[0])*aB[i]);
		triplets[k++] = T(p,1, -D_xPG[1][0]/xe[1]/hx[0]);          triplets[k++] = T(p,p,-(D_yPG[0][1]/ye[1]/hy[0])*aT[i]);
		b[p++]+=(D_xNG[0][0]/xe[0]/hx[0])*bL[j] + (D_yNG[0][0]/ye[0]/hy[0])*bB[i] + (D_yPG[0][1]/ye[1]/hy[0])*bT[i];
		for (i=1; i<Nx-1; i++ ) { // Middle Cell
			triplets[k++] = T(p,i-1,-D_xNG[i][0]  /xe[i]  /hx[i]);  triplets[k++] = T(p,p,-(D_yNG[i][0]/ye[0]/hy[0])*aB[i]);
			triplets[k++] = T(p,i+1,-D_xPG[i+1][0]/xe[i+1]/hx[i]);  triplets[k++] = T(p,p,-(D_yPG[i][1]/ye[1]/hy[0])*aT[i]);
			b[p++]+=(D_yNG[i][0]/ye[0]/hy[0])*bB[i] + (D_yPG[i][1]/ye[1]/hy[0])*bT[i];
		}
		i=Nx-1; // Right Cell
		triplets[k++] = T(p,Nx-2,-D_xNG[Nx-1][0]/xe[Nx-1]/hx[Nx-1]);         triplets[k++] = T(p,p,-(D_yNG[Nx-1][0]/ye[0]/hy[0])*aB[i]);
		triplets[k++] = T(p,p,  -(D_xPG[Nx][0]  /xe[Nx]  /hx[Nx-1])*aR[j]);  triplets[k++] = T(p,p,-(D_yPG[Nx-1][1]/ye[1]/hy[0])*aT[i]);
		b[p++]+=(D_yNG[Nx-1][0]/ye[0]/hy[0])*bB[i] + (D_xPG[Nx][0]  /xe[Nx]  /hx[Nx-1])*bR[j] + (D_yPG[Nx-1][1]/ye[1]/hy[0])*bT[i];
	}
	else {
		i=0; j=0; // Bottom Left Corner
		triplets[k++] = T(p,p,-(D_xNG[0][0]/xe[0]/hx[0])*aL[j]);   triplets[k++] = T(p,p,-(D_yNG[0][0]/ye[0]/hy[0])*aB[i]);
		triplets[k++] = T(p,1, -D_xPG[1][0]/xe[1]/hx[0]);          triplets[k++] = T(p,Nx,-D_yPG[0][1]/ye[1]/hy[0]);
		b[p++]+=(D_xNG[0][0]/xe[0]/hx[0])*bL[j] + (D_yNG[0][0]/ye[0]/hy[0])*bB[i];
		for (i=1; i<Nx-1; i++ ) { // Bottom
			triplets[k++] = T(p,p-1,-D_xNG[i][0]  /xe[i]  /hx[i]);  triplets[k++] = T(p,p,  -(D_yNG[i][0]/ye[0]/hy[0])*aB[i]);
			triplets[k++] = T(p,p+1,-D_xPG[i+1][0]/xe[i+1]/hx[i]);  triplets[k++] = T(p,Nx+i,-D_yPG[i][1]/ye[1]/hy[0]);
			b[p++]+=(D_yNG[i][0]/ye[0]/hy[0])*bB[i];
		}
		i=Nx-1; // Bottom Right Corner
		triplets[k++] = T(p,Nx-2,-D_xNG[Nx-1][0]/xe[Nx-1]/hx[Nx-1]);        triplets[k++] = T(p,p,    -(D_yNG[Nx-1][0]/ye[0]/hy[0])*aB[i]);
		triplets[k++] = T(p,p,  -(D_xPG[Nx][0]  /xe[Nx]  /hx[Nx-1])*aR[j]); triplets[k++] = T(p,2*Nx-1,-D_yPG[Nx-1][1]/ye[1]/hy[0]);
		b[p++]+=(D_yNG[Nx-1][0]/ye[0]/hy[0])*bB[i] + (D_xPG[Nx][0]  /xe[Nx]  /hx[Nx-1])*bR[j];
		for (j=1; j<Ny-1; j++ ) { // Middle
			i=0; // Left Side
			triplets[k++] = T(p,p, -(D_xNG[i][j]  /xe[i]  /hx[i])*aL[j]); triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
			triplets[k++] = T(p,p+1,-D_xPG[i+1][j]/xe[i+1]/hx[i]);        triplets[k++] = T(p,p+Nx,-D_yPG[i][j+1]/ye[j+1]/hy[j]);
			b[p++]+=(D_xNG[i][j]  /xe[i]  /hx[i])*bL[j];
			for (i=1; i<Nx-1; i++ ) { // Centre Cells
				triplets[k++] = T(p,p-1,-D_xNG[i][j]  /xe[i]  /hx[i]);  triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
				triplets[k++] = T(p,p+1,-D_xPG[i+1][j]/xe[i+1]/hx[i]);  triplets[k++] = T(p,p+Nx,-D_yPG[i][j+1]/ye[j+1]/hy[j]);
				p++;
			}
			i=Nx-1; // Right Side
			triplets[k++] = T(p,p-1,-D_xNG[i][j]  /xe[i]  /hx[i]);         triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
			triplets[k++] = T(p,p, -(D_xPG[i+1][j]/xe[i+1]/hx[i])*aR[j]);  triplets[k++] = T(p,p+Nx,-D_yPG[i][j+1]/ye[j+1]/hy[j]);
			b[p++]+=(D_xPG[i+1][j]/xe[i+1]/hx[i])*bR[j];
		}
		i=0; j=Ny-1; // Top Left Corner
		triplets[k++] = T(p,p, -(D_xNG[0][j]/xe[0]/hx[0])*aL[j]);  triplets[k++] = T(p,p-Nx,-D_yNG[0][j]  /ye[j]  /hy[j]);
		triplets[k++] = T(p,p+1,-D_xPG[1][j]/xe[1]/hx[0]);         triplets[k++] = T(p,p,  -(D_yPG[0][j+1]/ye[j+1]/hy[j])*aT[i]);
		b[p++]+=(D_xNG[0][j]/xe[0]/hx[0])*bL[j] + (D_yPG[0][j+1]/ye[j+1]/hy[j])*bT[i];
		for (i=1; i<Nx-1; i++ ) { // Top
			triplets[k++] = T(p,p-1,-D_xNG[i][j]  /xe[i]  /hx[i]);  triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
			triplets[k++] = T(p,p+1,-D_xPG[i+1][j]/xe[i+1]/hx[i]);  triplets[k++] = T(p,p,  -(D_yPG[i][j+1]/ye[j+1]/hy[j])*aT[i]);
			b[p++]+=(D_yPG[i][j+1]/ye[j+1]/hy[j])*bT[i];
		}
		i=Nx-1; // Top Right Corner
		triplets[k++] = T(p,p-1,-D_xNG[i][j]  /xe[i]  /hx[i]);         triplets[k++] = T(p,p-Nx,-D_yNG[i][j]  /ye[j]  /hy[j]);
		triplets[k++] = T(p,p, -(D_xPG[i+1][j]/xe[i+1]/hx[i])*aR[j]);  triplets[k++] = T(p,p,  -(D_yPG[i][j+1]/ye[j+1]/hy[j])*aT[i]);
		b[p++]+=(D_xPG[i+1][j]/xe[i+1]/hx[i])*bR[j] + (D_yPG[i][j+1]/ye[j+1]/hy[j])*bT[i];
	}
	}
	
	//
	constructAndSolve(preconditioner, etaL, g, d, b, triplets);
	
	// set solution back to problem values
	p=0;
	for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) phiG[i][j]=d[p++]; // Cell Centre Flux
	//# pragma omp parallel for
	for (int j=0; j<Ny; j++) phiLG[j]=phiG[0][j]*aL[j]+bL[j]; // Left   BC Flux
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) phiBG[i]=phiG[i][0]*aB[i]+bB[i]; // Bottom BC Flux
	//# pragma omp parallel for
	for (int j=0; j<Ny; j++) phiRG[j]=phiG[Nx-1][j]*aR[j]+bR[j]; // Right BC Flux
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) phiTG[i]=phiG[i][Ny-1]*aT[i]+bT[i]; // Top   BC Flux
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
	calculateGroupCurrents(etaL, g);
	{
	int i=0; int j=0; // Calculate residuals of matrix equations
	res=abs(-D_xPG[i+1][j]*phiG[i+1][j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiG[i][j+1]/ye[j+1]/hy[j]
	+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
	+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiLG[j]/xe[i]/hx[i]-D_yNG[i][j]*phiBG[i]/ye[j]/hy[j]-source[i][j]);
	if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
	for (int i=1; i<Nx-1; i++) {
		res=abs(-D_xPG[i+1][j]*phiG[i+1][j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiG[i][j+1]/ye[j+1]/hy[j]
		+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
		+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiG[i-1][j]/xe[i]/hx[i]-D_yNG[i][j]*phiBG[i]/ye[j]/hy[j]-source[i][j]);
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
	}
	i=Nx-1;
	res=abs(-D_xPG[i+1][j]*phiRG[j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiG[i][j+1]/ye[j+1]/hy[j]
	+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
	+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiG[i-1][j]/xe[i]/hx[i]-D_yNG[i][j]*phiBG[i]/ye[j]/hy[j]-source[i][j]);
	if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
	for (int j=1; j<Ny-1; j++) {
		int i=0;
		res=abs(-D_xPG[i+1][j]*phiG[i+1][j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiG[i][j+1]/ye[j+1]/hy[j]
		+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
		+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiLG[j]/xe[i]/hx[i]-D_yNG[i][j]*phiG[i][j-1]/ye[j]/hy[j]-source[i][j]);
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
		for (int i=1; i<Nx-1; i++) {
			res=abs(-D_xPG[i+1][j]*phiG[i+1][j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiG[i][j+1]/ye[j+1]/hy[j]
			+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
			+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiG[i-1][j]/xe[i]/hx[i]-D_yNG[i][j]*phiG[i][j-1]/ye[j]/hy[j]-source[i][j]);
			if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
		}
		i=Nx-1;
		res=abs(-D_xPG[i+1][j]*phiRG[j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiG[i][j+1]/ye[j+1]/hy[j]
		+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
		+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiG[i-1][j]/xe[i]/hx[i]-D_yNG[i][j]*phiG[i][j-1]/ye[j]/hy[j]-source[i][j]);
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
	}
	i=0; j=Ny-1;
	res=abs(-D_xPG[i+1][j]*phiG[i+1][j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiTG[i]/ye[j+1]/hy[j]
	+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
	+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiLG[j]/xe[i]/hx[i]-D_yNG[i][j]*phiG[i][j-1]/ye[j]/hy[j]-source[i][j]);
	if (res>res_mbal[etaL]) { res_mbal[etaL]=res; g_mbal[etaL]=g; i_mbal[etaL]=i; j_mbal[etaL]=j; }
	for (int i=1; i<Nx-1; i++) {
		res=abs(-D_xPG[i+1][j]*phiG[i+1][j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiTG[i]/ye[j+1]/hy[j]
		+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
		+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiG[i-1][j]/xe[i]/hx[i]-D_yNG[i][j]*phiG[i][j-1]/ye[j]/hy[j]-source[i][j]);
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; g_mbal[etaL]=g; i_mbal[etaL]=i; j_mbal[etaL]=j; }
	}
	i=Nx-1;
	res=abs(-D_xPG[i+1][j]*phiRG[j]/xe[i+1]/hx[i]-D_yPG[i][j+1]*phiTG[i]/ye[j+1]/hy[j]
	+(D_xNG[i+1][j]/xe[i+1]/hx[i]+D_xPG[i][j]/xe[i]/hx[i]+D_yNG[i][j+1]/ye[j+1]/hy[j]+D_yPG[i][j]/ye[j]/hy[j]
	+loss[i][j])*phiG[i][j]-D_xNG[i][j]*phiG[i-1][j]/xe[i]/hx[i]-D_yNG[i][j]*phiG[i][j-1]/ye[j]/hy[j]-source[i][j]);
	if (res>res_mbal[etaL]) { res_mbal[etaL]=res; g_mbal[etaL]=g; i_mbal[etaL]=i; j_mbal[etaL]=j; }
	}
	
}
//======================================================================================//

//======================================================================================//
//++ Solve One-group eigenvalue problem with Newtons Iterations ++++++++++++++++++++++++//
//======================================================================================//
bool LOSolver::QDgreyNewtonSolution(int preconditioner, vector< vector<double> > &phiLast) {
	double res=0;
	int etaL=eta_star-1, g=0;
	vector< vector<double> > &phiG=phi[etaL][g], &phi_xG=phi_x[etaL][g], &phi_yG=phi_y[etaL][g], &j_xG=j_x[etaL][g], &j_yG=j_y[etaL][g];
	vector< vector<double> > &D_xxCG=D_xxC[etaL][g], &D_yyCG=D_yyC[etaL][g];
	vector< vector<double> > &D_xxLG=D_xxL[etaL][g], &D_yyLG=D_yyL[etaL][g], &D_xyLG=D_xyL[etaL][g];
	vector< vector<double> > &D_xxRG=D_xxR[etaL][g], &D_yyRG=D_yyR[etaL][g], &D_xyRG=D_xyR[etaL][g];
	vector< vector<double> > &D_xxBG=D_xxB[etaL][g], &D_yyBG=D_yyB[etaL][g], &D_xyBG=D_xyB[etaL][g];
	vector< vector<double> > &D_xxTG=D_xxT[etaL][g], &D_yyTG=D_yyT[etaL][g], &D_xyTG=D_xyT[etaL][g];
	vector<double>     &FLG=    FL[etaL][g],     &FRG=    FR[etaL][g],     &FBG=    FB[etaL][g],     &FTG=    FT[etaL][g];
	vector<double>   &jInLG=  jInL[etaL][g],   &jInRG=  jInR[etaL][g],   &jInBG=  jInB[etaL][g],   &jInTG=  jInT[etaL][g];
	vector<double> &phiInLG=phiInL[etaL][g], &phiInRG=phiInR[etaL][g], &phiInBG=phiInB[etaL][g], &phiInTG=phiInT[etaL][g];
	vector< vector<double> > &sigmaTG=sigmaT[etaL][g], &sigmaSG=sigmaS[etaL][g][g], &nuSigmaFG=nuSigmaF[etaL][g];
	
	double kL=k_eff;
	
	//temfile<<"NDA Solution Grid "<<etaL<<" group "<<g<<endl;
	
	int N_c=Nx*Ny;
	int N_xf=Nx*Ny+Ny;
	int N_yf=Nx*Ny+Nx;
	int N_ukn=3*Nx*Ny+Nx+Ny+1;
	
	std::vector<double> d(N_ukn), b(N_ukn);
	
	//write_cell_dat(j_xG, x, y, 16, temfile);
	//write_cell_dat(j_yG, x, y, 16, temfile);
	
	// Find Infinity Norm of Newton Residual
	double res_initial=NewtonLNorm(0);
	
	
	
	int p=0;
	for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) 
		b[p++]=-2.0*(-D_xxRG[i][j]*phi_xG[i+1][j]+2.0*D_xxCG[i][j]*phiG[i][j]-D_xxLG[i][j]*phi_xG[i][j])/hx[i]/hx[i]
			   -2.0*(-D_yyTG[i][j]*phi_yG[i][j+1]+2.0*D_yyCG[i][j]*phiG[i][j]-D_yyBG[i][j]*phi_yG[i][j])/hy[j]/hy[j]
				-(sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/k_eff)*phiG[i][j];
	// X grid equations ++++++++++++++++++++++++++++++++++++++++
	for (int j=0; j<Ny; j++) {
		// Left BC
		b[p++]=-((-2.0*D_xxCG[0][j]/hx[0])*phiG[0][j] + (2.0*D_xxLG[0][j]/hx[0]+FLG[j])*phi_xG[0][j] 
			+ (D_xyBG[0][j]/hy[j])*phi_yG[0][j] + (-D_xyTG[0][j]/hy[j])*phi_yG[0][j+1] - FLG[j]*phiInLG[j] + jInLG[j]); // right hand side
		for (int i=1; i<Nx; i++) {
			b[p++]=-(2.0*(D_xxRG[i-1][j]*phi_xG[i][j]-D_xxCG[i-1][j]*phiG[i-1][j])/hx[i-1]+(D_xyTG[i-1][j]*phi_yG[i-1][j+1]-D_xyBG[i-1][j]*phi_yG[i-1][j])/hy[j]
			       -2.0*(D_xxCG[i][j]  *phiG[i][j]  -D_xxLG[i][j]  *phi_xG[i][j])/hx[i]  -(D_xyTG[i][j]  *phi_yG[i][j+1]  -D_xyBG[i][j]  *phi_yG[i][j]  )/hy[j]);
		}
		// Right BC
		b[p++]=-((-2.0*D_xxCG[Nx-1][j]/hx[Nx-1])*phiG[Nx-1][j] + (2.0*D_xxRG[Nx-1][j]/hx[Nx-1]+FRG[j])*phi_xG[Nx][j] 
			+ (-D_xyBG[Nx-1][j]/hy[j])*phi_yG[Nx-1][j] + (D_xyTG[Nx-1][j]/hy[j])*phi_yG[Nx-1][j+1] - FRG[j]*phiInRG[j] + jInRG[j]); // right hand side
	}
	// Y grid equations ++++++++++++++++++++++++++++++++++++++++
	// Bottom BC
	for (int i=0; i<Nx; i++) {
		b[p++] =-((-2.0*D_yyCG[i][0]/hy[0])*phiG[i][0] + (D_xyLG[i][0]/hx[i])*phi_xG[i][0] + 
		(-D_xyRG[i][0]/hx[i])*phi_xG[i+1][0] + (2.0*D_yyBG[i][0]/hy[0]+FBG[i])*phi_yG[i][0] - FBG[i]*phiInBG[i] + jInBG[i]); // Bottom BC
	}
	for (int j=1; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			b[p++] = -((-2.0*D_yyCG[i][j-1]/hy[j-1])*phiG[i][j-1] + (-2.0*D_yyCG[i][j]/hy[j])*phiG[i][j] + (-D_xyLG[i][j-1]/hx[i])*phi_xG[i][j-1] + (D_xyRG[i][j-1]/hx[i])*phi_xG[i+1][j-1] +
			(D_xyLG[i][j]/hx[i])*phi_xG[i][j] + (-D_xyRG[i][j]/hx[i])*phi_xG[i+1][j] + (2.0*(D_yyTG[i][j-1]/hy[j-1]+D_yyBG[i][j]/hy[j]))*phi_yG[i][j]);
		}
	}
	// Top BC
	for (int i=0; i<Nx; i++) {
		b[p++] =-((-2.0*D_yyCG[i][Ny-1]/hy[Ny-1])*phiG[i][Ny-1] + (-D_xyLG[i][Ny-1]/hx[i])*phi_xG[i][Ny-1] + 
		(D_xyRG[i][Ny-1]/hx[i])*phi_xG[i+1][Ny-1] + (2.0*D_yyTG[i][Ny-1]/hy[Ny-1]+FTG[i])*phi_yG[i][Ny] - FTG[i]*phiInTG[i] + jInTG[i]); // Top BC
	}
	b.back()=(x[Nx]-x[0])*(y[Ny]-y[0]);
	for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) b.back()-=hx[i]*hy[j]*phiG[i][j];
	
	
	std::vector<T> triplets(21*Nx*Ny+Nx+Ny);
	
	// Assign matrix A and vector b
	// Balance equations ++++++++++++++++++++++++++++++++++++++++
	int k=0; p=0;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			triplets[k++] = T(p,p         , 4.0*(D_xxCG[i][j]/hx[i]/hx[i]+D_yyCG[i][j]/hy[j]/hy[j])+sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/k_eff); // cell centre flux
			triplets[k++] = T(p,N_c+p+j   , -2.0*D_xxLG[i][j]/hx[i]/hx[i]);  triplets[k++] = T(p,N_c+p+j+1,     -2.0*D_xxRG[i][j]/hx[i]/hx[i]); // x grid flux
			triplets[k++] = T(p,N_c+N_xf+p, -2.0*D_yyBG[i][j]/hy[j]/hy[j]);  triplets[k++] = T(p,N_c+N_xf+p+Nx, -2.0*D_yyTG[i][j]/hy[j]/hy[j]); // y grid flux
			triplets[k++] = T(p,N_ukn-1   ,  -nuSigmaFG[i][j]*phiG[i][j] );   
			p++;
		}
	}
	// X grid equations ++++++++++++++++++++++++++++++++++++++++
	int c=0;
	for (int j=0; j<Ny; j++) {
		// Left BC
		triplets[k++] = T(p,c,           -2.0*D_xxCG[0][j]/hx[0]); // cell centre flux
		triplets[k++] = T(p,N_c+c+j,      2.0*D_xxLG[0][j]/hx[0]+FLG[j]); // x grid flux
		triplets[k++] = T(p,N_c+N_xf+c,       D_xyBG[0][j]/hy[j]);       triplets[k++] = T(p,N_c+N_xf+c+Nx,  -D_xyTG[0][j]/hy[j]); // y grid flux
		p++; c++;
		for (int i=1; i<Nx; i++) {
			triplets[k++] = T(p,c-1         , -2.0*D_xxCG[i-1][j]/hx[i-1]);  triplets[k++] = T(p,c,          -2.0*D_xxCG[i][j]  /hx[i]); // centre flux
			triplets[k++] = T(p,N_c+c+j     , 2.0*(D_xxRG[i-1][j]/hx[i-1]+D_xxLG[i][j]/hx[i]));     // x grid flux
			triplets[k++] = T(p,N_c+N_xf+c-1,     -D_xyBG[i-1][j]/hy[j]  );  triplets[k++] = T(p,N_c+N_xf+c-1+Nx, D_xyTG[i-1][j]/hy[j]);  // y grid flux left
			triplets[k++] = T(p,N_c+N_xf+c  ,      D_xyBG[i][j]  /hy[j]  );  triplets[k++] = T(p,N_c+N_xf+c+Nx  ,-D_xyTG[i][j]  /hy[j]); // y grid flux right
			p++; c++;
		}
		// Right BC
		triplets[k++] = T(p,c-1,           -2.0*D_xxCG[Nx-1][j]/hx[Nx-1]); // cell centre flux
		triplets[k++] = T(p,N_c+c+j,        2.0*D_xxRG[Nx-1][j]/hx[Nx-1]+FRG[j]); // x grid flux
		triplets[k++] = T(p,N_c+N_xf+c-1,      -D_xyBG[Nx-1][j]/hy[j]);           triplets[k++] = T(p,N_c+N_xf+c+Nx-1,  D_xyTG[Nx-1][j]/hy[j]); // y grid flux
		p++;
	}
	// Y grid equations ++++++++++++++++++++++++++++++++++++++++
	c=0;
	for (int i=0; i<Nx; i++) {
		// Bottom BC
		triplets[k++] = T(p,c,            -2.0*D_yyCG[i][0]/hy[0]); // cell centre flux
		triplets[k++] = T(p,N_c+c,             D_xyLG[i][0]/hx[i]);   triplets[k++] = T(p,N_c+c+1,  -D_xyRG[i][0]/hx[i]); // x grid flux
		triplets[k++] = T(p,N_c+N_xf+c,    2.0*D_yyBG[i][0]/hy[0]+FBG[i]); // y grid flux
		p++; c++;
	}
	for (int j=1; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			triplets[k++] = T(p,c-Nx        ,  -2.0*D_yyCG[i][j-1]/hy[j-1]);  triplets[k++] = T(p,c         , -2.0*D_yyCG[i][j]  /hy[j]); // centre flux
			triplets[k++] = T(p,N_c+c+j-Nx-1,      -D_xyLG[i][j-1]/hx[i]  );  triplets[k++] = T(p,N_c+c+j-Nx,      D_xyRG[i][j-1]/hx[i]);  // x grid flux bottom
			triplets[k++] = T(p,N_c+c+j     ,       D_xyLG[i][j]  /hx[i]  );  triplets[k++] = T(p,N_c+c+j+1 ,     -D_xyRG[i][j]  /hx[i]); // x grid flux top
			triplets[k++] = T(p,N_c+N_xf+c  ,  2.0*(D_yyTG[i][j-1]/hy[j-1]+D_yyBG[i][j]/hy[j]));  // y grid flux
			p++; c++;
		}
	}
	for (int i=0; i<Nx; i++) {
		// Top BC
		triplets[k++] = T(p,c-Nx,           -2.0*D_yyCG[i][Ny-1]/hy[Ny-1]); // cell centre flux
		triplets[k++] = T(p,N_c+c-Nx+Ny-1,      -D_xyLG[i][Ny-1]/hx[i]);     triplets[k++] = T(p,N_c+c-Nx+Ny,  D_xyRG[i][Ny-1]/hx[i]); // x grid flux
		triplets[k++] = T(p,N_c+N_xf+c,      2.0*D_yyTG[i][Ny-1]/hy[Ny-1]+FTG[i]); // y grid flux
		p++; c++;
	}
	
	p=0; // Normalization Row
	for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) triplets[k++] = T(N_ukn-1,p++,hx[i]*hy[j]);
	
	// Initial guess for Newton Step
	for (int i=0; i<N_ukn; i++) d[i]=0.0;
	
	//
	constructAndSolve(preconditioner, etaL, g, d, b, triplets);
	
	// set solution back to problem values
	p=0;
	for (int j=0; j<Ny; j++)   for (int i=0; i<Nx; i++)   phiG[i][j]  +=d[p++]; // cell centre flux
	for (int j=0; j<Ny; j++)   for (int i=0; i<Nx+1; i++) phi_xG[i][j]+=d[p++]; // x grid flux
	for (int j=0; j<Ny+1; j++) for (int i=0; i<Nx; i++)   phi_yG[i][j]+=d[p++]; // y grid flux
	k_eff=1/(1/k_eff + d.back());
	
	
	
	// Find Infinity Norm of Newton Residual
	double res_norm=NewtonLNorm(0);
	
	//Armijo Line Search
	bool newtonWorked=true;
	double alphaA=1e-4, lambda=-1.0;
	double etaA=0, sigmaA=0.5;
	//if ( res_norm>res_initial ) cout<<"Armijo \n";
	int jA=0, Amax=10;
	bool fix_negative=false;
	if ( stop_phi[etaL]==1 and ( !greySolutionPositive() or k_eff<0.0 ) ) fix_negative=true;
	
	while ( ( res_norm >= (1-alphaA*lambda)*res_initial or fix_negative ) and jA<Amax ) {
		jA++;
		//cout<<"Armijo Step "<<jA<<endl;
		lambda*=0.5;
		// set solution back to problem values
		p=0;
		for (int j=0; j<Ny; j++)   for (int i=0; i<Nx; i++)   phiG[i][j]  +=lambda*d[p++]; // cell centre flux
		for (int j=0; j<Ny; j++)   for (int i=0; i<Nx+1; i++) phi_xG[i][j]+=lambda*d[p++]; // x grid flux
		for (int j=0; j<Ny+1; j++) for (int i=0; i<Nx; i++)   phi_yG[i][j]+=lambda*d[p++]; // y grid flux
		k_eff=1/(1/k_eff + lambda*d.back());
		
		// Find Infinity Norm of Newton Residual
		res_norm=NewtonLNorm(0);
		fix_negative=( !greySolutionPositive() or k_eff<0.0 );
	}
	if (jA==Amax) {
		newtonWorked=false;
		cout<<">> Line search failed in Newton method\n";
	}
	
	// Check that the solution is positive
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
	calculateGroupCurrents(etaL, g);
	
	// Matrix Residual
	double norm=-(x[Nx]-x[0])*(y[Ny]-y[0]);
	for (int j=0; j<Ny; j++) for (int i=0; i<Nx; i++) norm+=hx[i]*hy[j]*phiG[i][j];
	res_mbal[etaL]=abs(norm);
	//cout<<"Normal "<<res_mbal[etaL]<<endl;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			res=abs((j_xG[i+1][j]-j_xG[i][j])/hx[i]+(j_yG[i][j+1]-j_yG[i][j])/hy[j]+
				(sigmaTG[i][j]-sigmaSG[i][j]-nuSigmaFG[i][j]/kL)*phiG[i][j] - nuSigmaFG[i][j]*phiLast[i][j]*d.back());
			//cout<<"mbal res "<<res<<endl;
			if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
		}
	}
	for (int j=0; j<Ny; j++) {
		for (int i=1; i<Nx; i++) {
			res=abs((-D_xyTG[i][j]/hy[j])*phi_yG[i][j+1] + (-2.0*D_xxCG[i][j]/hx[i])*phiG[i][j] + (D_xyBG[i][j]/hy[j])*phi_yG[i][j]
				+ (D_xyTG[i-1][j]/hy[j])*phi_xG[i-1][j+1] + (-2.0*D_xxCG[i-1][j]/hx[i-1])*phiG[i-1][j] + (-D_xyBG[i-1][j]/hy[j])*phi_yG[i-1][j]
				+ 2.0*(D_xxRG[i-1][j]/hx[i-1]+D_xxLG[i][j]/hx[i])*phi_xG[i][j]);
			res=abs(2.0*(D_xxRG[i-1][j]*phi_xG[i][j]-D_xxCG[i-1][j]*phiG[i-1][j])/hx[i-1]+(D_xyTG[i-1][j]*phi_yG[i-1][j+1]-D_xyBG[i-1][j]*phi_yG[i-1][j])/hy[j]
			       -2.0*(D_xxCG[i][j]  *phiG[i][j]  -D_xxLG[i][j]  *phi_xG[i][j])/hx[i]  -(D_xyTG[i][j]  *phi_yG[i][j+1]  -D_xyBG[i][j]  *phi_yG[i][j]  )/hy[j]);
			//cout<<"mbal res x "<<res<<" i "<<i<<" j "<<j<<endl;
			if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
		}
	}
	for (int j=1; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			res=abs((-D_xyRG[i][j]/hx[i])*phi_xG[i+1][j] + (-2.0*D_yyCG[i][j]/hy[j])*phiG[i][j] + (D_xyLG[i][j]/hx[i])*phi_xG[i][j]
				+ (D_xyRG[i][j-1]/hx[i])*phi_xG[i+1][j-1] + (-2.0*D_yyCG[i][j-1]/hy[j-1])*phiG[i][j-1] + (-D_xyLG[i][j-1]/hx[i])*phi_xG[i][j-1]
				+ 2.0*(D_yyTG[i][j-1]/hy[j-1]+D_yyBG[i][j]/hy[j])*phi_yG[i][j]);
			//cout<<"mbal res y "<<res<<endl;
			if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
		}
	}
	for (int j=0; j<Ny; j++) {
		res=abs((-2.0*D_xxCG[0][j]/hx[0])*phiG[0][j] + (2.0*D_xxLG[0][j]/hx[0]+FLG[j])*phi_xG[0][j] 
			+ (D_xyBG[0][j]/hy[j])*phi_yG[0][j] + (-D_xyTG[0][j]/hy[j])*phi_yG[0][j+1] - FLG[j]*phiInLG[j] + jInLG[j]);
		//cout<<"mbal res l "<<res<<endl;
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=0; j_mbal[etaL]=j; }
		res=abs((-2.0*D_xxCG[Nx-1][j]/hx[Nx-1])*phiG[Nx-1][j] + (2.0*D_xxRG[Nx-1][j]/hx[Nx-1]+FRG[j])*phi_xG[Nx][j] 
			+ (-D_xyBG[Nx-1][j]/hy[j])*phi_yG[Nx-1][j] + (D_xyTG[Nx-1][j]/hy[j])*phi_yG[Nx-1][j+1] - FRG[j]*phiInRG[j] + jInRG[j]);
		//cout<<"mbal res r "<<res<<endl;
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=Nx; j_mbal[etaL]=j; }
	}
	for (int i=0; i<Nx; i++) {
		res=abs((-2.0*D_yyCG[i][0]/hy[0])*phiG[i][0] + (D_xyLG[i][0]/hx[i])*phi_xG[i][0] + 
		(-D_xyRG[i][0]/hx[i])*phi_xG[i+1][0] + (2.0*D_yyBG[i][0]/hy[0]+FBG[i])*phi_yG[i][0] - FBG[i]*phiInBG[i] + jInBG[i]);
		//cout<<"mbal res b "<<res<<endl;
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=0; }
		res=abs((-2.0*D_yyCG[i][Ny-1]/hy[Ny-1])*phiG[i][Ny-1] + (-D_xyLG[i][Ny-1]/hx[i])*phi_xG[i][Ny-1] + 
		(D_xyRG[i][Ny-1]/hx[i])*phi_xG[i+1][Ny-1] + (2.0*D_yyTG[i][Ny-1]/hy[Ny-1]+FTG[i])*phi_yG[i][Ny] - FTG[i]*phiInTG[i] + jInTG[i]);
		//cout<<"mbal res t "<<res<<endl;
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=Ny; }
	}
	return newtonWorked;
}
//======================================================================================//
//++ function to solve NDA problem ++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOSolver::QDfixedSourceSolution(int preconditioner, int etaL, int g, std::vector< std::vector<double> > &source, std::vector< std::vector<double> > &loss) {
	double res=0;
	vector< vector<double> > &phiG=phi[etaL][g], &phi_xG=phi_x[etaL][g], &phi_yG=phi_y[etaL][g];
	vector< vector<double> > &D_xxCG=D_xxC[etaL][g], &D_yyCG=D_yyC[etaL][g];
	vector< vector<double> > &D_xxLG=D_xxL[etaL][g], &D_yyLG=D_yyL[etaL][g], &D_xyLG=D_xyL[etaL][g];
	vector< vector<double> > &D_xxRG=D_xxR[etaL][g], &D_yyRG=D_yyR[etaL][g], &D_xyRG=D_xyR[etaL][g];
	vector< vector<double> > &D_xxBG=D_xxB[etaL][g], &D_yyBG=D_yyB[etaL][g], &D_xyBG=D_xyB[etaL][g];
	vector< vector<double> > &D_xxTG=D_xxT[etaL][g], &D_yyTG=D_yyT[etaL][g], &D_xyTG=D_xyT[etaL][g];
	vector<double>     &FLG=    FL[etaL][g],     &FRG=    FR[etaL][g],     &FBG=    FB[etaL][g],     &FTG=    FT[etaL][g];
	vector<double>   &jInLG=  jInL[etaL][g],   &jInRG=  jInR[etaL][g],   &jInBG=  jInB[etaL][g],   &jInTG=  jInT[etaL][g];
	vector<double> &phiInLG=phiInL[etaL][g], &phiInRG=phiInR[etaL][g], &phiInBG=phiInB[etaL][g], &phiInTG=phiInT[etaL][g];
	
	//temfile<<"NDA Solution Grid "<<etaL<<" group "<<g<<endl;
	
	int N_c=Nx*Ny;
	int N_xf=Nx*Ny+Ny;
	int N_yf=Nx*Ny+Nx;
	int N_ukn=3*Nx*Ny+Nx+Ny;
	
	std::vector<double> d(N_ukn), b(N_ukn);
	
	
	std::vector<T> triplets(19*Nx*Ny+Nx+Ny);
	
	//temfile<<"-- D_xxL --\n";
	//write_cell_dat(D_xxLG, x, y, 16, temfile);
	//temfile<<"-- D_yyL --\n";
	//write_cell_dat(D_yyLG, x, y, 16, temfile);
	
	// Assign matrix A and vector b
	// Balance equations ++++++++++++++++++++++++++++++++++++++++
	int p=0, k=0;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			triplets[k++] = T(p,p,4.0*(D_xxCG[i][j]/hx[i]/hx[i]+D_yyCG[i][j]/hy[j]/hy[j])+loss[i][j]); // cell centre flux
			triplets[k++] = T(p,N_c+p+j,    -2*D_xxLG[i][j]/hx[i]/hx[i]);  triplets[k++] = T(p,N_c+p+j+1,     -2*D_xxRG[i][j]/hx[i]/hx[i]); // x grid flux
			triplets[k++] = T(p,N_c+N_xf+p, -2*D_yyBG[i][j]/hy[j]/hy[j]);  triplets[k++] = T(p,N_c+N_xf+p+Nx, -2*D_yyTG[i][j]/hy[j]/hy[j]); // y grid flux
			b[p++] = source[i][j]; // right hand side
		}
	}
	// X grid equations ++++++++++++++++++++++++++++++++++++++++
	int c=0;
	for (int j=0; j<Ny; j++) {
		// Left BC
		triplets[k++] = T(p,c,           -2.0*D_xxCG[0][j]/hx[0]); // cell centre flux
		triplets[k++] = T(p,N_c+c+j,      2.0*D_xxLG[0][j]/hx[0]+FLG[j]); // x grid flux
		triplets[k++] = T(p,N_c+N_xf+c,       D_xyBG[0][j]/hy[j]);       triplets[k++] = T(p,N_c+N_xf+c+Nx,  -D_xyTG[0][j]/hy[j]); // y grid flux
		b[p++] = FLG[j]*phiInLG[j]-jInLG[j]; // right hand side
		c++;
		for (int i=1; i<Nx; i++) {
			triplets[k++] = T(p,c-1         , -2.0*D_xxCG[i-1][j]/hx[i-1]);  triplets[k++] = T(p,c,          -2.0*D_xxCG[i][j]  /hx[i]); // centre flux
			triplets[k++] = T(p,N_c+c+j     , 2.0*(D_xxRG[i-1][j]/hx[i-1]+D_xxLG[i][j]/hx[i]));     // x grid flux
			triplets[k++] = T(p,N_c+N_xf+c-1,     -D_xyBG[i-1][j]/hy[j]  );  triplets[k++] = T(p,N_c+N_xf+c-1+Nx, D_xyTG[i-1][j]/hy[j]);  // y grid flux left
			triplets[k++] = T(p,N_c+N_xf+c  ,      D_xyBG[i][j]  /hy[j]  );  triplets[k++] = T(p,N_c+N_xf+c+Nx  ,-D_xyTG[i][j]  /hy[j]); // y grid flux right
			b[p++] = 0.0; // right hand side
			c++;
		}
		// Right BC
		triplets[k++] = T(p,c-1,           -2.0*D_xxCG[Nx-1][j]/hx[Nx-1]); // cell centre flux
		triplets[k++] = T(p,N_c+c+j,        2.0*D_xxRG[Nx-1][j]/hx[Nx-1]+FRG[j]); // x grid flux
		triplets[k++] = T(p,N_c+N_xf+c-1,      -D_xyBG[Nx-1][j]/hy[j]);           triplets[k++] = T(p,N_c+N_xf+c+Nx-1,  D_xyTG[Nx-1][j]/hy[j]); // y grid flux
		b[p++] = FRG[j]*phiInRG[j]-jInRG[j]; // right hand side
	}
	// Y grid equations ++++++++++++++++++++++++++++++++++++++++
	c=0;
	for (int i=0; i<Nx; i++) {
		// Bottom BC
		triplets[k++] = T(p,c,            -2.0*D_yyCG[i][0]/hy[0]); // cell centre flux
		triplets[k++] = T(p,N_c+c,             D_xyLG[i][0]/hx[i]);   triplets[k++] = T(p,N_c+c+1,  -D_xyRG[i][0]/hx[i]); // x grid flux
		triplets[k++] = T(p,N_c+N_xf+c,    2.0*D_yyBG[i][0]/hy[0]+FBG[i]); // y grid flux
		b[p++] = FBG[i]*phiInBG[i]-jInBG[i]; // right hand side
		c++;
	}
	for (int j=1; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			triplets[k++] = T(p,c-Nx        ,  -2.0*D_yyCG[i][j-1]/hy[j-1]);  triplets[k++] = T(p,c         , -2.0*D_yyCG[i][j]  /hy[j]); // centre flux
			triplets[k++] = T(p,N_c+c+j-Nx-1,      -D_xyLG[i][j-1]/hx[i]  );  triplets[k++] = T(p,N_c+c+j-Nx,      D_xyRG[i][j-1]/hx[i]);  // x grid flux bottom
			triplets[k++] = T(p,N_c+c+j     ,       D_xyLG[i][j]  /hx[i]  );  triplets[k++] = T(p,N_c+c+j+1 ,     -D_xyRG[i][j]  /hx[i]); // x grid flux top
			triplets[k++] = T(p,N_c+N_xf+c  ,  2.0*(D_yyTG[i][j-1]/hy[j-1]+D_yyBG[i][j]/hy[j]));  // y grid flux
			b[p++] = 0.0; // right hand side
			c++;
		}
	}
	for (int i=0; i<Nx; i++) {
		// Top BC
		triplets[k++] = T(p,c-Nx,           -2.0*D_yyCG[i][Ny-1]/hy[Ny-1]); // cell centre flux
		triplets[k++] = T(p,N_c+c-Nx+Ny-1,      -D_xyLG[i][Ny-1]/hx[i]);     triplets[k++] = T(p,N_c+c-Nx+Ny,  D_xyRG[i][Ny-1]/hx[i]); // x grid flux
		triplets[k++] = T(p,N_c+N_xf+c,      2.0*D_yyTG[i][Ny-1]/hy[Ny-1]+FTG[i]); // y grid flux
		b[p++] = FTG[i]*phiInTG[i]-jInTG[i]; // right hand side
		c++;
	}
	
	p=0;
	for (int j=0; j<Ny; j++)   for (int i=0; i<Nx; i++)   d[p++]=phiG[i][j]; // cell centre flux
	for (int j=0; j<Ny; j++)   for (int i=0; i<Nx+1; i++) d[p++]=phi_xG[i][j]; // x grid flux
	for (int j=0; j<Ny+1; j++) for (int i=0; i<Nx; i++)   d[p++]=phi_yG[i][j]; // y grid flux
	
	//
	constructAndSolve(preconditioner, etaL, g, d, b, triplets);
	
	// set solution back to problem values
	p=0;
	for (int j=0; j<Ny; j++)   for (int i=0; i<Nx; i++)   phiG[i][j]  =d[p++]; // cell centre flux
	for (int j=0; j<Ny; j++)   for (int i=0; i<Nx+1; i++) phi_xG[i][j]=d[p++]; // x grid flux
	for (int j=0; j<Ny+1; j++) for (int i=0; i<Nx; i++)   phi_yG[i][j]=d[p++]; // y grid flux
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
	calculateGroupCurrents(etaL, g);
	
	
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			res=abs((2.0/hx[i]/hx[i])*( - D_xxRG[i][j]*phi_xG[i+1][j] + 2.0*D_xxCG[i][j]*phiG[i][j] - D_xxLG[i][j]*phi_xG[i][j]) 
			             + (2.0/hy[j]/hy[j])*( - D_yyTG[i][j]*phi_yG[i][j+1] + 2.0*D_yyCG[i][j]*phiG[i][j] - D_yyBG[i][j]*phi_yG[i][j]) + loss[i][j]*phiG[i][j] - source[i][j]);
			//cout<<"res bal "<<res<<endl;
			if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
		}
	}
	
	for (int j=0; j<Ny; j++) {
		for (int i=1; i<Nx; i++) {
			res=abs((-D_xyTG[i][j]/hy[j])*phi_yG[i][j+1] + (-2.0*D_xxCG[i][j]/hx[i])*phiG[i][j] + (D_xyBG[i][j]/hy[j])*phi_yG[i][j]
				+ (D_xyTG[i-1][j]/hy[j])*phi_yG[i-1][j+1] + (-2.0*D_xxCG[i-1][j]/hx[i-1])*phiG[i-1][j] + (-D_xyBG[i-1][j]/hy[j])*phi_yG[i-1][j]
				+ 2.0*(D_xxRG[i-1][j]/hx[i-1]+D_xxLG[i][j]/hx[i])*phi_xG[i][j]);
			//cout<<"res x "<<res<<endl;
			if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
		}
	}
	for (int j=1; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			res=abs((-D_xyRG[i][j]/hx[i])*phi_xG[i+1][j] + (-2.0*D_yyCG[i][j]/hy[j])*phiG[i][j] + (D_xyLG[i][j]/hx[i])*phi_xG[i][j]
				+ (D_xyRG[i][j-1]/hx[i])*phi_xG[i+1][j-1] + (-2.0*D_yyCG[i][j-1]/hy[j-1])*phiG[i][j-1] + (-D_xyLG[i][j-1]/hx[i])*phi_xG[i][j-1]
				+ 2.0*(D_yyTG[i][j-1]/hy[j-1]+D_yyBG[i][j]/hy[j])*phi_yG[i][j]);
			//cout<<"res y "<<res<<endl;
			if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=j; }
		}
	}
	/*
	for (int j=0; j<Ny; j++) {
		res=abs((-2.0*D_xxCG[0][j]/hx[0])*phiG[0][j]+(2.0*D_xxLG[0][j]+FLG[j])*phi_xG[0][j]+(-D_xyTG[0][j]/hy[j])*phi_yG[0][j+1]+(D_xyBG[0][j]/hy[j])*phi_yG[0][j]-FLG[j]*phiInLG[j]+jInLG[j]);
		
		cout<<"res l "<<res<<endl;
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=0; j_mbal[etaL]=j; }
		res=abs((-2.0*D_xxCG[Nx-1][j]/hx[Nx-1])*phiG[Nx-1][j]+(2.0*D_xxRG[Nx-1][j]/hx[Nx-1]+FRG[j])*phi_xG[Nx][j]+ (D_xyTG[Nx-1][j]/hy[j])*phi_yG[Nx-1][j+1]+ (-D_xyBG[Nx-1][j]/hy[j])*phi_yG[Nx-1][j]-FRG[j]*phiInRG[j]+jInRG[j]);
		//cout<<"res r "<<res<<endl;
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=Nx; j_mbal[etaL]=j; }
	}
	for (int i=0; i<Nx; i++) {
		res=abs((-2.0*D_yyCG[i][0]/hy[0])*phiG[i][0] + (D_xyRG[i][0]/hx[i])*phi_xG[i+1][0] + (-D_xyLG[i][0]/hx[i])*phi_xG[i][0] + (2.0*D_yyTG[i][0]/hy[0]+FBG[i])*phi_yG[i][0] - FBG[i]*phiInBG[i] + jInBG[i]);
		cout<<"res b "<<res<<endl;
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=0; }
		res=abs((-2.0*D_yyCG[i][Ny-1]/hy[Ny-1])*phiG[i][Ny-1] + (D_xyRG[i][Ny-1]/hx[i])*phi_xG[i+1][Ny-1] + (-D_xyLG[i][Ny-1]/hx[i])*phi_xG[i][Ny-1] + (2.0*D_yyTG[i][Ny-1]/hy[Ny-1]+FTG[i])*phi_yG[i][Ny] - FTG[i]*phiInTG[i] + jInTG[i]);
		//cout<<"res t "<<res<<endl;
		if (res>res_mbal[etaL]) { res_mbal[etaL]=res; i_mbal[etaL]=i; j_mbal[etaL]=Ny; }
	}
	*/
}
//======================================================================================//

bool LOSolver::greyNewtonSolution(int preconditioner, vector< vector<double> > &phiLast) {
	bool worked;
	if ( NDASolution ) worked=NDAgreyNewtonSolution(preconditioner, phiLast);
	else               worked= QDgreyNewtonSolution(preconditioner, phiLast);
	return worked;
}

void LOSolver::fixedSourceSolution(int preconditioner, int etaL, int g, std::vector< std::vector<double> > &source, std::vector< std::vector<double> > &loss) {
	if ( NDASolution ) NDAfixedSourceSolution(preconditioner, etaL, g, source, loss);
	else                QDfixedSourceSolution(preconditioner, etaL, g, source, loss);
}

//**************************************************************************************//
//**************************************************************************************//
//======================================================================================//
//++++++++++++++++++++++++++++ Cross Section Methods +++++++++++++++++++++++++++++++++++//
//======================================================================================//
//**************************************************************************************//
//**************************************************************************************//


double LOXS::Total(int etaL, int g, int i, int j) {
	if ( etaL>0 ) return sigmaT[etaL][g][i][j];
	else           return sigma_gT[mIndex[i][j]][g];
	return 0.0;
}
double LOXS::Fission(int etaL, int g, int i, int j) {
	if ( etaL>0 ) return nuSigmaF[etaL][g][i][j];
	else           return nuSigma_gF[mIndex[i][j]][g];
	return 0.0;
}
double LOXS::Chi(int etaL, int g, int i, int j) {
	if ( etaL>0 ) return chi[etaL][g][i][j];
	else           return chi_g[mIndex[i][j]][g];
	return 0.0;
}
double LOXS::Source(int etaL, int g, int i, int j) {
	if ( etaL>0 ) return s_ext[etaL][g][i][j];
	else           return s_gext[mIndex[i][j]][g];
	return 0.0;
}
double LOXS::Scatter(int etaL, int g, int gg, int i, int j) {
	if ( etaL>0 ) return sigmaS[etaL][g][gg][i][j];
	else           return sigma_gS[mIndex[i][j]][g][gg];
	return 0.0;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
vector< vector<double> > LOXS::Total(int etaL, int g) {
	if ( etaL==0 ) {
		vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) sigma[i][j]=Total(0,g,i,j);
		}
		return sigma;
	}
	else {
		vector< vector<double> > sigma=sigmaT[etaL][g];
		return sigma;
	}
}
vector< vector<double> > LOXS::Fission(int etaL, int g) {
	if ( etaL==0 ) {
		vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) sigma[i][j]=Fission(0,g,i,j);
		}
		return sigma;
	}
	else {
		vector< vector<double> > sigma=nuSigmaF[etaL][g];
		return sigma;
	}
}
vector< vector<double> > LOXS::Chi(int etaL, int g) {
	if ( etaL==0 ) {
		vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) sigma[i][j]=Chi(0,g,i,j);
		}
		return sigma;
	}
	else {
		vector< vector<double> > sigma=chi[etaL][g];
		return sigma;
	}
}
vector< vector<double> > LOXS::Source(int etaL, int g) {
	if ( etaL==0 ) {
		vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) sigma[i][j]=Source(0,g,i,j);
		}
		return sigma;
	}
	else {
		vector< vector<double> > sigma=s_ext[etaL][g];
		return sigma;
	}
}
vector< vector<double> > LOXS::Scatter(int etaL, int g, int gg) {
	if ( etaL==0 ) {
		vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
		//# pragma omp parallel for
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) sigma[i][j]=Scatter(0,g,gg,i,j);
		}
		return sigma;
	}
	else {
		vector< vector<double> > sigma=sigmaS[etaL][g][gg];
		return sigma;
	}
}
vector< vector<double> > LOXS::Diffusion(int g) {
	vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) sigma[i][j]=Diffusion(g,i,j);
	}
	return sigma;
}
//======================================================================================//


//======================================================================================//
//++ LOXS Initialize XS Memory ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int LOXS::initializeLOXS(HOSolver &ho) {
	
	x=ho.x; y=ho.y; Nx=ho.Nx; Ny=ho.Ny;
	
	mIndex    =ho.mIndex;
	material  =ho.material;
	sigma_gT  =ho.sigma_gT;
	nuSigma_gF=ho.nuSigma_gF;
	chi_g     =ho.chi_g;
	s_gext    =ho.s_gext;
	D_g       =ho.D_g;
	sigma_gS  =ho.sigma_gS;
	
	matnum=ho.matnum;
	xsname=ho.xsname;
	
	cout<<"<< Initialize XS Data: ";
	if (eta_star!=1) {
		// Initialize XS Memory
		vector< vector<double> > grid_d(Nx,vector<double>(Ny, 0.0));
		// cross section data
		sigmaT.resize(eta_star);
		sigmaS.resize(eta_star);
		for (int k=1; k<eta_star; k++) {
			sigmaT[k].resize(Ng[k]);
			sigmaS[k].resize(Ng[k]);
			for (int g=0; g<Ng[k]; g++) {
				sigmaT[k][g]=grid_d;
				sigmaS[k][g].resize(Ng[k]);
				for (int gg=0; gg<Ng[k]; gg++) sigmaS[k][g][gg]=grid_d;
			}
		}
		nuSigmaF=sigmaT;
		chi     =sigmaT;
		s_ext   =sigmaT;
		
		
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				sigmaT[eta_star-1][0][i][j]=0;
				nuSigmaF[eta_star-1][0][i][j]=0;
				for (int g=0; g<Ng[0]; g++) {
					sigmaT[eta_star-1][0][i][j]+=Total(0,g,i,j);
					nuSigmaF[eta_star-1][0][i][j]+=Fission(0,g,i,j);
				}
			}
		}
	}
	
	
	cout<<"Complete >>";
	return 0;
}
//======================================================================================//

//======================================================================================//
//++ LOXS Writer ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOXS::writeLOMatXS(ofstream& outfile) {
	outfile<<" -- Material and Cross Sections -- \n";
	outfile<<"+-----------------------------------------------+\n";
	outfile<<"Write Cross Sections : "<<write_xs<<endl;
	outfile<<"\n -- Material Properties --\n";
	for (int k=0; k<matnum.size(); k++) {
		outfile<<"Material # "<<matnum[k]<<"  Cross-section name: "<<xsname[k]<<endl;
		outfile<<"Group# |    Total XS   |   Fission XS  |      nuF      |      chi      | Ext Source \n";
		outfile.precision(6);
		for (int g=0; g<Ng[0]; g++) outfile<<setw(6)<<g+1<<print_out(sigma_gT[k][g])<<print_out(sigma_gF[k][g])
			<<print_out(nu_gF[k][g])<<print_out(chi_g[k][g])<<print_out(s_gext[k][g])<<endl;
		outfile<<" Scattering Matrix \n";
		outfile<<"  g \\ g'  ";
		for (int gg=0; gg<Ng[0]; gg++) outfile<<setw(6)<<gg+1<<setw(10)<<" ";
		outfile<<endl;
		for (int g=0; g<Ng[0]; g++) {
			outfile<<setw(6)<<g+1;
			for (int gg=0; gg<Ng[0]; gg++) outfile<<print_out(sigma_gS[k][g][gg]);
			outfile<<endl;
		}
	}
	
	outfile<<"\n -- Material Map -- \n";
	for (int j=Ny-1; j>=0; j--) {
		for (int i=0; i<Nx; i++) outfile<<setw(3)<<material[i][j];
		outfile<<endl;
	}
	outfile<<"+-----------------------------------------------+\n";
	//return 0;
}
//======================================================================================//


void LOXS::writeLOXSFile (std::string case_name) {
	if ( write_xs ) {
		int outw=16;
		vector< vector<double> > outp;
		string xs_file=case_name+".xs.csv";
		#pragma omp critical
		cout<<xs_file<<endl;
		
		ofstream datfile (xs_file.c_str()); // open output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		datfile<<" # of x cells , # of y cells ,\n";
		datfile<<Nx<<" , "<<Ny<<", \n";
		datfile<<"number of energy grids, "<<eta_star<<",\n";
		datfile<<"number of groups in grid, ";
		for (int k=0; k<eta_star; k++) datfile<<Ng[k]<<" , ";
		datfile<<endl;
		
		datfile<<" -------------------- \n";
		datfile<<" -- Cross-Sections -- \n"; // Write NDA solution
		datfile<<" -------------------- \n";
		
		datfile<<" -- Total XS -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int g=0; g<Ng[0]; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Total(0,g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		for (int k=1; k<eta_star; k++) {
			if (k==eta_star-1) datfile<<" -- Grey Energy Grid --\n";
			else datfile<<" -- Energy Grid # "<<k<<" --\n";
			write_group_dat(sigmaT[k], Ng[k], x, y, k, outw, datfile);
		}
		
		datfile<<" -- NuF x Fission XS -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int g=0; g<Ng[0]; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Fission(0,g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		for (int k=1; k<eta_star; k++) {
			if (k==eta_star-1) datfile<<" -- Grey Energy Grid --\n";
			else datfile<<" -- Energy Grid # "<<k<<" --\n";
			write_group_dat(nuSigmaF[k], Ng[k], x, y, k, outw, datfile);
		}
		
		datfile<<" -- chi -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int g=0; g<Ng[0]; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Chi(0,g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		for (int k=1; k<eta_star; k++) {
			if (k==eta_star-1) datfile<<" -- Grey Energy Grid --\n";
			else datfile<<" -- Energy Grid # "<<k<<" --\n";
			write_group_dat(chi[k], Ng[k], x, y, k, outw, datfile);
		}
		
		datfile<<" -- Scattering XS -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int gg=0; gg<Ng[0]; gg++) {
			for (int g=0; g<Ng[0]; g++) {
				datfile<<" scattering from group "<<gg+1<<" to group "<<g+1<<endl;
				outp=Scatter(0,gg,g);
				write_cell_dat(outp, x, y, outw, datfile);
			}
		}
		for (int k=1; k<eta_star; k++) {
			datfile<<"Energy Grid # "<<k<<endl;
			for (int gg=0; gg<Ng[k]; gg++) {
				for (int g=0; g<Ng[k]; g++) {
					datfile<<" scattering from group "<<gg+1<<" to group "<<g+1<<endl;
					write_cell_dat(sigmaS[k][gg][g], x, y, outw, datfile);
				}
			}
		}
		
		datfile<<" -- External Source -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int g=0; g<Ng[0]; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Source(0,g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		for (int k=1; k<eta_star; k++) {
			if (k==eta_star-1) datfile<<" -- Grey Energy Grid --\n";
			else datfile<<" -- Energy Grid # "<<k<<" --\n";
			write_group_dat(s_ext[k], Ng[k], x, y, k, outw, datfile);
		}
		
		datfile<<" -- Cell Average Diffusion Coefficient -- \n";
		for (int g=0; g<Ng[0]; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Diffusion(g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		
		datfile.close(); // close output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	}
}


