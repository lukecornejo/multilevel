#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <time.h>  /* clock_t, clock, CLOCKS_PER_SEC */
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Eigen/IterativeLinearSolvers"
#include <vector>
#include "LO_1.6.h"
#include "IO_1.6.h"
#include <omp.h>
//#include "H5Cpp.h"

using namespace Eigen;
using namespace std;

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
//typedef Eigen::Triplet<double> T;
typedef ::Triplet T;

extern bool KE_problem; // option variables
extern bool reflectiveB, reflectiveT, reflectiveL, reflectiveR;

extern int kbc;
extern double k_eff; // K-effective

extern ofstream temfile;

void LOSolver::checkPreconditioner() {
	for (int k=0; k<eta_star; k++) {
		if ( preconditioner_type[k]>3 and preconditioner_type[k]<2 ) {
			cout<<">>Error in preconditioner Type input in grid "<<k<<" use default preconditioner 3"<<endl;
			preconditioner_type[k]=3;
		}
	}
}

std::string LOSolver::writeLinearSolverType() {
	return "Using Eigen Linear Algebra Package \n";
}


Eigen::Triplet<double> setTriplet(T triplet) {
	return Eigen::Triplet<double> (triplet.Row(), triplet.Column(), triplet.Value());
}

void LOSolver::constructAndSolve(int preconditioner, int etaL, int g, std::vector<double> &d, std::vector<double> &b, std::vector<T> &triplets) {
	int N_ukn=d.size();
	
	
	VectorXd D(N_ukn);
	VectorXd B(N_ukn);
	SpMat A(N_ukn,N_ukn);
	
	for (int i=0; i<N_ukn; i++) {
		D[i]=d[i];
		B[i]=b[i];
	}
	b.resize(0); // Remove all elements from vector b
	std::vector< Eigen::Triplet<double> > elements(triplets.size());
	
	for (int i=0; i<elements.size(); i++) elements[i]=setTriplet(triplets[i]);
	triplets.resize(0); // Remove all elements from vector triplets
	
	A.setFromTriplets(elements.begin(), elements.end());
	A.prune(1e-17, 10);
	A.makeCompressed();
	
	//cout<<"norm A "<<A.norm()<<" "<<A.squaredNorm()<<endl;
	//cout<<"norm B "<<B.lpNorm<1>()<<" "<<B.lpNorm<2>()<<" "<<B.lpNorm<Infinity>()<<endl;
	double normI=1/B.lpNorm<2>();
	A=normI*A;
	B=normI*B;
	
	clock_t solveT=clock();
	// Solve Ax=b iteratively with BiCGSTAB
	if ( preconditioner==2 ) {
		clock_t t_pc=clock(); // start low-order timer
		BiCGSTAB<SpMat> solver;
		solver.setTolerance(epsilon_solver);     // set convergence criteria 
		solver.setMaxIterations(stop_solver); // set the max number of lo iterations
		solver.compute(A);
		
		t_pc=clock()-t_pc; // stop low-order timer
		dt_pc[etaL].push_back(((double)t_pc)/CLOCKS_PER_SEC); // add high-order solution time to vector
		
		D = solver.solveWithGuess(B,D);
		//D = solver.solveWithGuess(B,D);
		err_lo[etaL].push_back(solver.error());      // error in lo solution
		num_logm[etaL].push_back(solver.iterations()); // number of lo iterations
	}
	else if ( preconditioner==3 ) {
		
		clock_t t_pc=clock(); // start low-order timer
		BiCGSTAB<SpMat,IncompleteLUT<double> > solver;
		solver.preconditioner().setFillfactor(15);
		solver.setTolerance(epsilon_solver);     // set convergence criteria 
		solver.setMaxIterations(stop_solver); // set the max number of lo iterations
		solver.compute(A);
		
		t_pc=clock()-t_pc; // stop low-order timer
		dt_pc[etaL].push_back(((double)t_pc)/CLOCKS_PER_SEC); // add high-order solution time to vector
		//cout<<"A "<<A<<endl;
		D = solver.solveWithGuess(B,D);
		
		err_lo[etaL].push_back(solver.error());      // error in lo solution
		num_logm[etaL].push_back(solver.iterations()); // number of lo iterations
	}
	else if ( preconditioner==4 ) {
		
		clock_t t_pc=clock(); // start low-order timer
		BiCGSTAB<SpMat,IncompleteLUT<double> > solver;
		solver.preconditioner().setFillfactor(5);
		solver.setTolerance(epsilon_solver);     // set convergence criteria 
		solver.setMaxIterations(stop_solver); // set the max number of lo iterations
		solver.compute(A);
		
		t_pc=clock()-t_pc; // stop low-order timer
		dt_pc[etaL].push_back(((double)t_pc)/CLOCKS_PER_SEC); // add high-order solution time to vector
		//cout<<"A "<<A<<endl;
		D = solver.solveWithGuess(B,D);
		
		err_lo[etaL].push_back(solver.error());      // error in lo solution
		num_logm[etaL].push_back(solver.iterations()); // number of lo iterations
	}
	else cout<<">>Preconditioner Type Error \n";
	num_sol++;
	solveT=clock()-solveT;
	
	VectorXd R(N_ukn);
	R=A*D-B;
	
	double matrix_res=R.lpNorm<Infinity>();
	//cout<<matrix_res<<endl;
	if ( matrix_res>res_Amatrix[etaL] ) { res_Amatrix[etaL]=matrix_res; g_Amatrix[etaL]=g; }
	matrix_res=matrix_res/B.lpNorm<Infinity>();
	if ( matrix_res>res_Rmatrix[etaL] ) { res_Rmatrix[etaL]=matrix_res; g_Rmatrix[etaL]=g; }
	
	for (int i=0; i<N_ukn; i++) d[i]=D[i];
	
}



