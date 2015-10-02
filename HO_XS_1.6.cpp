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

extern bool reflectiveB, reflectiveT, reflectiveL, reflectiveR;
extern double k_eff;
extern bool KE_problem; // option variables
extern int    kbc; // Kind of BC


//**************************************************************************************//
//**************************************************************************************//
//======================================================================================//
//++++++++++++++++++++++++++++ Cross Section Methods +++++++++++++++++++++++++++++++++++//
//======================================================================================//
//**************************************************************************************//
//**************************************************************************************//

//======================================================================================//
vector< vector<double> > HOXS::Total(int g) {
	vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) sigma[i][j]=Total(g,i,j);
	}
	return sigma;
}
vector< vector<double> > HOXS::Fission(int g) {
	vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) sigma[i][j]=Fission(g,i,j);
	}
	return sigma;
}
vector< vector<double> > HOXS::Chi(int g) {
	vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) sigma[i][j]=Chi(g,i,j);
	}
	return sigma;
}
vector< vector<double> > HOXS::Source(int g) {
	vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) sigma[i][j]=Source(g,i,j);
	}
	return sigma;
}
vector< vector<double> > HOXS::Scatter(int g, int gg) {
	vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) sigma[i][j]=Scatter(g,gg,i,j);
	}
	return sigma;
}
vector< vector<double> > HOXS::Diffusion(int g) {
	vector< vector<double> > sigma(Nx,vector<double>(Ny, 0.0));
	//# pragma omp parallel for
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) sigma[i][j]=Diffusion(g,i,j);
	}
	return sigma;
}
//======================================================================================//

//++ Find Material Index ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
int HOXS::materialIndex(vector<int> &materialNumber, vector<int> &type, vector<int> &regionNumber, int i, int j, vector< vector<double> > &shape) {
	int k, m, materialIndex;
	double xc=(x[i+1]+x[i])/2.0;
	double yc=(y[j+1]+y[j])/2.0;
	for (k=0; k<regionNumber.size(); k++) {
		if ( type[k]==1 ) {
			if ( abs(xc-shape[k][0])<=0.5*shape[k][2] && abs(yc-shape[k][1])<=0.5*shape[k][3] ) {
				materialIndex=findIndex(regionNumber[k], materialNumber);
				return materialIndex;
			}
		}
		else if (type[k]==2) {
			if ( sqrt(pow(xc-shape[k][0],2)+pow(yc-shape[k][1],2))<=shape[k][2] ) {
				materialIndex=findIndex(regionNumber[k], materialNumber);
				return materialIndex;
			}
		}
		else cout<<">>Error in function materialRegion\n";
	}
	cout<<">> Error: no material region found for cell "<<i<<","<<j<<endl;
	return 0;
}

//++ Read XS Values ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
int HOXS::XSinput(int k, int ng) {
	vector<double> num_temp;
	int i, G, m;
	string line;
	
	string file_name=xsname[k]+".xs";
	// open file +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	ifstream xsfile (file_name.c_str()); // open XS input file
	
	try {
		if ( !xsfile.is_open() ) throw "File "+file_name+" could not be opened";
		// read in data
		getline(xsfile, line); // read in name of the input
		
		getline(xsfile, line); // get # of groups
		num_temp=parseNumber(line);
		G=int(num_temp[0]+0.5);
		if ( G!=ng ) throw "Material "+file_name+" has mismatched # of Groups";
		
		getline(xsfile, line); // space
		
		for (int g=0; g<G; g++) {
			getline(xsfile, line); // Get line of XS
			num_temp=parseNumber(line);
			sigma_gT[k][g]=num_temp[0];
			sigma_gF[k][g]=num_temp[1];
			nu_gF[k][g]=num_temp[2];
			chi_g[k][g]=num_temp[3];
		}
		
		getline(xsfile, line); // space
		
		for (int g=0; g<G; g++) {
			getline(xsfile, line); // Get line of XS
			num_temp=parseNumber(line);
			for (int p=0; p<G; p++) sigma_gS[k][g][p]=num_temp[p];
		}
		xsfile.close(); // close XS input file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	}
	catch (std::string error) {
		cout<<"Error in Cross-Section Input >> "<<error<<endl;
		return 1;
	}
	return 0;
}
//======================================================================================//

//======================================================================================//
//++ HOXS Constructor ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int HOXS::readHOXS(std::vector<std::string> &line, std::vector<double> &x1, std::vector<double> &y1, int Ng1) {
	
	x=x1; y=y1; Ng=Ng1;
	Nx=x.size()-1; Ny=y.size()-1;
	
	int p;
	write_xs=false;
	cout<<"Read XS Data: ";
	vector<double> num_temp;
	vector<string> str_temp;
	
	// Shapes
	vector<int> shapeNumber, shapeType, shapeMaterial;
	vector< vector<double> > shape;
	// Lattice
	vector<int> latticeNumber;
	vector< vector< vector<int> > > latticeShapes;
	vector< vector<double> > lattice;
	// Domain
	vector<int> domainMaterial;
	vector< vector<double> > domain;
	try {
		for (int w=0; w<line.size(); w++) {
			str_temp=parseWord(line[w]);
			num_temp=parseNumber(line[w]);
			
			if (str_temp[0]=="mat" or str_temp[0]=="material") {
				//
				xsname.push_back(str_temp[3]);
				
				matnum.push_back(int(num_temp[0]+0.5));
				string str_source="  ";
				for (int i=0; i<str_temp.size(); i++) {
					if ( str_temp[i]=="source" ) {
						for (int j=i; j<str_temp.size(); j++) str_source=str_source+str_temp[j]+"  ";
						break;
					}
				}
				num_temp=parseNumber(str_source);
				
				vector<double> rowd(Ng,0.0);
				s_gext.push_back(rowd);
				
				for (int i=0; i<num_temp.size(); i=i+2) s_gext.back()[int(num_temp[i]-0.5)]=num_temp[i+1];
				//
			}
			else if (str_temp[0]=="square" or str_temp[0]=="rectangle") {
				// Shapes
				if (num_temp.size()!=6) throw "Fatal Error: wrong number of arguments for 'square'";
				shapeNumber.push_back(int(num_temp[0]+0.5)); // Identification Number of the Shape (int)
				shapeType.push_back(1); // Shape type (int) 1 Rectangle, 2 Circle
				shapeMaterial.push_back(int(num_temp[1]+0.5)); // Material Number in Shape (int)
				shape.push_back(vector<double>());
				for (int i=2; i<num_temp.size(); i++) shape.back().push_back(num_temp[i]); // Left   Boundary of Region (double)
				if ( shape.back().size()!=4 ) cout<<">>Error in shape type\n";
				num_temp=parseNumber(line[w]);
			}
			else if (str_temp[0]=="circle") {
				// Shapes
				if (num_temp.size()!=5) throw "Fatal Error: wrong number of arguments for 'circle'";
				shapeNumber.push_back(int(num_temp[0]+0.5)); // Identification Number of the Shape (int)
				shapeType.push_back(2); // Shape type (int) 1 Rectangle, 2 Circle
				shapeMaterial.push_back(int(num_temp[1]+0.5)); // Material Number in Shape (int)
				shape.push_back(vector<double>());
				for (int i=2; i<num_temp.size(); i++) shape.back().push_back(num_temp[i]); // Left   Boundary of Region (double)
				if ( shape.back().size()!=3 ) cout<<">> Error in shape type\n";
				num_temp=parseNumber(line[w]);
			}
			else if (str_temp[0]=="lattice") {
				// Lattice
				if (num_temp.size()!=7) throw "Fatal Error: wrong number of arguments for 'lattice'";
				latticeNumber.push_back(int(num_temp[0]+0.5)); // Identification Number of lattice (int)
				lattice.push_back(vector<double> ());
				for (int i=3; i<num_temp.size(); i++) lattice.back().push_back(num_temp[i]); // Location and Size of the lattice (double)
				latticeShapes.push_back(vector< vector<int> > () );
				int ny=int(num_temp[1]+0.5);
				int nx=int(num_temp[2]+0.5);
				latticeShapes.back().resize(ny);
				for (int i=0; i<ny; i++) latticeShapes.back()[i].resize(nx);
				for (int j=ny-1; j>=0; j--) {
					w++;
					num_temp=parseNumber(line[w]);
					if (num_temp.size()!=nx) throw "Fatal Error: wrong number of cells in 'lattice'";
					for (int i=0; i<latticeShapes.back().size(); i++) latticeShapes.back()[i][j]=int(num_temp[i]+0.5);
				}
			}
			else if (str_temp[0]=="domain") {
				if ( num_temp.size()!=5 ) throw "Fatal Error: wrong number of arguments in 'domain' input";
				domainMaterial.push_back(int(num_temp[0]+0.5)); // Material Number in Region (int)
				domain.push_back(vector<double> (4,0.0));
				for (int i=0; i<4; i++) domain.back()[i]=num_temp[i+1]; // Location and size of domain (double)
			}
			else if (str_temp[0]=="Write_XS") {
				if (num_temp.size()!=1) throw "Fatal Error: wrong number of arguments in 'Write_XS' input";
				write_xs=int(num_temp[0]+0.5); // Write Cross Sections (bool)
			}
			else cout<<">> Error: Unknown variable in XS block\n";
		}
		
		
		// Region
		vector<int> regionMaterial, regionType;
		vector< vector<double> > region;
		
		for (int m=0; m<latticeNumber.size(); m++) {
			for (int i=0; i<latticeShapes[m].size(); i++) {
				for (int j=0; j<latticeShapes[m][i].size(); j++) {
					int k=findIndex(latticeShapes[m][i][j], shapeNumber);
					regionMaterial.push_back(shapeMaterial[k]);
					regionType.push_back(shapeType[k]);
					region.push_back(shape[k]);
					region.back()[0]+=lattice[m][0]-lattice[m][2]*(0.5-(i+0.5)/double(latticeShapes[m].size()));
					region.back()[1]+=lattice[m][1]-lattice[m][3]*(0.5-(j+0.5)/double(latticeShapes[m][i].size()));
					//cout<<"xo "<<region[p][0]<<"yo "<<region[p][1]<<"x "<<region[p][2]<<"y "<<region[p][3]<<endl;
				}
			}
		}
		
		for (int m=0; m<domainMaterial.size(); m++) {
			if ( domain[m][0]+0.5*domain[m][2]-1e-10>x[Nx] or domain[m][0]-0.5*domain[m][2]+1e-10<x[0] or domain[m][1]+0.5*domain[m][3]-1e-10>y[Ny] or domain[m][1]-0.5*domain[m][3]+1e-10<y[0] )
				cout<<">> Warning: material domain "<<m<<" is outside the problem domain \n";
			regionMaterial.push_back(domainMaterial[m]);
			regionType.push_back(1);
			region.push_back(domain[m]);
		}
		
		// cross sections
		sigma_gT=s_gext;
		sigma_gS.resize(matnum.size());
		for (int k=0; k<matnum.size(); k++) {
			sigma_gS[k].resize(Ng);
			for (int g=0; g<Ng; g++) sigma_gS[k][g].resize(Ng);
		}
		sigma_gF=s_gext;
		nu_gF=s_gext;
		chi_g=s_gext;
		D_g=s_gext;
		nuSigma_gF=s_gext;
		
		for (int k=0; k<matnum.size(); k++) if ( XSinput(k, Ng)!=0 ) throw "Fatal Error in Cross-Section File"; // Get Cross-Sections for each Material
		
		// Ensure there are no errors in cross sections
		for (int k=0; k<matnum.size(); k++) {
			double dtemp=0.0; for (int g=0; g<Ng; g++) dtemp+=chi_g[k][g]; // Make sure chi sums to 1
			if ( dtemp>1e-14 ) for (int g=0; g<Ng; g++) chi_g[k][g]/=dtemp;
			for (int g=0; g<Ng; g++) if ( sigma_gT[k][g]==0.0 ) sigma_gT[k][g]=1e-23;
			for (int g=0; g<Ng; g++) D_g[k][g]=1.0/3.0/sigma_gT[k][g]; // Diffusion Coefficient. Isotropic scattering so sigma_tr = sigma_t
			for (int g=0; g<Ng; g++) nuSigma_gF[k][g]=nu_gF[k][g]*sigma_gF[k][g];
		}
		
		if ( KE_problem )
			for (int k=0; k<matnum.size(); k++) for (int g=0; g<Ng; g++) s_gext[k][g]=0; // Set source to zero if it is a k-eigenvalue problem
		
		// Initialize XS Memory
		material.resize(Nx);
		for (int i=0; i<Nx; i++) material[i].resize(Ny);
		mIndex=material;
		//cout<<"Size "<<mIndex.size()<<endl;
		
		for (int i=0; i<Nx; i++) {
			for (int j=0; j<Ny; j++) {
				int k=materialIndex(matnum, regionType, regionMaterial, i, j, region);
				mIndex[i][j]=k;
				material[i][j]=matnum[k];
			}
		}
	}
	catch (std::string error) {
		cout<<"Error in Reading High-order Materials >> "<<error<<endl;
		return 1;
	}
	cout<<"Complete\n";
	return 0;
}
//======================================================================================//


//======================================================================================//
//++ HOXS Writer ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void HOXS::writeHOXS(ofstream& outfile) {
	outfile<<" -- Material and Cross Sections -- \n";
	outfile<<"+-----------------------------------------------+\n";
	outfile<<"Write Cross Sections : "<<write_xs<<endl;
	outfile<<"\n -- Material Properties --\n";
	for (int k=0; k<matnum.size(); k++) {
		outfile<<"Material # "<<matnum[k]<<"  Cross-section name: "<<xsname[k]<<endl;
		outfile<<"Group# |    Total XS   |   Fission XS  |      nuF      |      chi      | Ext Source \n";
		outfile.precision(6);
		for (int g=0; g<Ng; g++) outfile<<setw(6)<<g+1<<print_out(sigma_gT[k][g])<<print_out(sigma_gF[k][g])
			<<print_out(nu_gF[k][g])<<print_out(chi_g[k][g])<<print_out(s_gext[k][g])<<endl;
		outfile<<" Scattering Matrix \n";
		outfile<<"  g \\ g'  ";
		for (int gg=0; gg<Ng; gg++) outfile<<setw(6)<<gg+1<<setw(10)<<" ";
		outfile<<endl;
		for (int g=0; g<Ng; g++) {
			outfile<<setw(6)<<g+1;
			for (int gg=0; gg<Ng; gg++) outfile<<print_out(sigma_gS[k][g][gg]);
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

//======================================================================================//
//++ HOXS Writer ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void HOXS::writeMatFile(string case_name) {
	string material_file=case_name+".mat.csv";
	cout<<material_file<<endl;
	ofstream datfile (material_file.c_str()); // open output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	//write_column_cell_data(material, x, y, 9, datfile);
	
	datfile.close(); // close output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
}
//======================================================================================//


void HOXS::writeHOXSFile (std::string case_name) {
	if ( write_xs ) {
		int outw=16;
		vector< vector<double> > outp;
		string xs_file=case_name+".hoxs.csv";
		#pragma omp critical
		cout<<xs_file<<endl;
		
		ofstream datfile (xs_file.c_str()); // open output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		datfile<<" # of x cells , # of y cells ,\n";
		datfile<<Nx<<" , "<<Ny<<", \n";
		datfile<<"number of groups, "<<Ng<<",\n";
		datfile<<endl;
		
		datfile<<" -------------------- \n";
		datfile<<" -- Cross-Sections -- \n"; // Write NDA solution
		datfile<<" -------------------- \n";
		
		datfile<<" -- Total XS -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int g=0; g<Ng; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Total(g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		
		datfile<<" -- NuF x Fission XS -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int g=0; g<Ng; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Fission(g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		
		datfile<<" -- chi -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int g=0; g<Ng; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Chi(g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		
		datfile<<" -- Scattering XS -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int gg=0; gg<Ng; gg++) {
			for (int g=0; g<Ng; g++) {
				datfile<<" scattering from group "<<gg+1<<" to group "<<g+1<<endl;
				outp=Scatter(gg,g);
				write_cell_dat(outp, x, y, outw, datfile);
			}
		}
		
		datfile<<" -- External Source -- \n";
		datfile<<" -- Fine Energy Grid -- \n";
		for (int g=0; g<Ng; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Source(g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		
		datfile<<" -- Cell Average Diffusion Coefficient -- \n";
		for (int g=0; g<Ng; g++) {
			datfile<<"Grid # "<<0<<" Energy Group # "<<g+1<<endl;
			outp=Diffusion(g);
			write_cell_dat(outp, x, y, outw, datfile);
		}
		
		datfile.close(); // close output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	}
}




