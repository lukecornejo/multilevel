#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include "HO_1.6.h"
#include "LO_1.6.h"
#include "IO_1.6.h"
#include <omp.h>
#include <time.h>  /* clock_t, clock, CLOCKS_PER_SEC */
//#include "H5Cpp.h"

using namespace std;


extern string test_name;

extern bool KE_problem; // option variables
extern double k_eff;

extern int kbc;
extern bool reflectiveB, reflectiveT, reflectiveL, reflectiveR;

extern HOSolver ho;
extern LOSolver lo;

// Function to parse Integer input
int iparse(std::string str, int& i) {
	int num=0;
	while ( isdigit(str[i+1]) ) {
		num+=str[i]-'0';
		num*=10;
		i++;
	}
	num+=str[i]-'0';
	return num;
}

// Function to parse Decimal input
double dparse(std::string str, int& i) {
	double num=0.0, sgn=1.0;
	int p, t, power;
	if ( i!=0 ) if ( str[i-1]=='-' ) sgn=-1.0;
	
	while ( isdigit(str[i+1]) ) {
		t=str[i]-'0';
		num+=t;
		num*=10.0;
		i++;
	}
	t=str[i]-'0';
	num+=t;
	if ( str[i+1]=='.' ) {
		i++; p=1;
		while ( isdigit(str[i+1]) ) {
			i++;
			t=str[i]-'0';
			num+=t/pow(10.0,double(p));
			p++;
		}
	}
	if ( str[i+1]=='e' or str[i+1]=='E' ) {
		i++;
		if ( str[i+1]=='-' ) {
			i+=2;
			if ( isdigit(str[i]) ) power=iparse(str,i);
			num/=pow(10.0,double(power));
		}
		else if ( str[i+1]=='+' ) {
			i+=2;
			if ( isdigit(str[i]) ) power=iparse(str,i);
			num*=pow(10.0,double(power));
		}
		else if ( isdigit(str[i+1]) ) {
			i++;
			power=iparse(str,i);
			num*=pow(10.0,double(power));
		}
	}
	return sgn*num;
}

string print_out(double value) {
	string dest;
	char buffer[50];
    sprintf(buffer, "%.8e", value); // First print out using scientific notation with 0 mantissa digits
	if ( buffer[0]=='n' and buffer[1]=='a' )      dest="      nan      ";
	else if ( buffer[0]=='i' and buffer[1]=='n' ) dest="      inf      ";
	else dest=buffer;
	if ( value>=0.0 ) dest=" " + dest;
	dest=" " + dest;
	return dest;
}
string print_outH(double value) {
	string dest;
	char buffer[50];
    sprintf(buffer, "%.15e", value); // First print out using scientific notation with 0 mantissa digits
	if ( buffer[0]=='n' and buffer[1]=='a' )      dest="      nan      ";
	else if ( buffer[0]=='i' and buffer[1]=='n' ) dest="      inf      ";
	else dest=buffer;
	if ( value>=0.0 ) dest=" " + dest;
	dest=" " + dest;
	return dest;
}
string print_out(int value) {
	string dest;
	char buffer[50];
    sprintf(buffer, "%4i    ", value); // First print out using scientific notation with 0 mantissa digits
	if ( buffer[0]=='n' and buffer[1]=='a' )      dest="      nan      ";
	else if ( buffer[0]=='i' and buffer[1]=='n' ) dest="      inf      ";
	else dest=buffer;
	if ( value>=0.0 ) dest=" " + dest;
	return dest;
}

string print_csv(double value) {
	string dest;
	char buffer[50];
    sprintf(buffer, "%.8e", value); // First print out using scientific notation with 0 mantissa digits
	if ( buffer[0]=='n' and buffer[1]=='a' )      dest="      nan      ";
	else if ( buffer[0]=='i' and buffer[1]=='n' ) dest="      inf      ";
	else dest=buffer;
	if ( value>=0.0 ) dest=" " + dest;
	dest=dest+",";
	return dest;
}
string print_csv(int value) {
	string dest;
	char buffer[50];
    sprintf(buffer, "%4i   ,", value); // First print out using scientific notation with 0 mantissa digits
	if ( buffer[0]=='n' and buffer[1]=='a' )      dest="      nan     ,";
	else if ( buffer[0]=='i' and buffer[1]=='n' ) dest="      inf     ,";
	else dest=buffer;
	if ( value>=0.0 ) dest=" " + dest;
	return dest;
}
string print_csvS(double value) {
	string dest;
	char buffer[50];
    sprintf(buffer, "%.2e", value); // First print out using scientific notation with 0 mantissa digits
	if ( buffer[0]=='n' and buffer[1]=='a' )      dest="      nan      ";
	else if ( buffer[0]=='i' and buffer[1]=='n' ) dest="      inf      ";
	else dest=buffer;
	if ( value>=0.0 ) dest=" " + dest;
	dest=dest+",";
	return dest;
}

// Function to find the material region
int findIndex(int ident, vector<int> values) {
	int _index;
	for (int m=0; m<values.size(); m++) {
		if ( ident==ho.matnum[m] ) {
			_index=m;
			break;
		}
	}
	return _index;
}
// Parse a line of input and return a vector of the numbers in that line
std::vector<double> parseNumber(std::string input_string) {
	vector<double> output_data;
	int string_length=input_string.size();
	for (int i=0; i<string_length; i++) {
		if ( isdigit(input_string[i]) ) output_data.push_back(dparse(input_string,i));
		if ( input_string[i]==';' ) break;
	}
	return output_data;
}
// Parse a line of input and return a vector of the numbers in that line
std::vector<std::string> parseWord(std::string input_string) {
	vector<string> output_data;
	int string_length=input_string.size();
	for (int i=0; i<string_length; i++) {
		if ( input_string[i]==' ' ) continue;
		else if ( input_string[i]==';' ) break;
		else {
			string word(1,input_string[i]);
			for (int j=i+1; j<string_length; j++) {
				if ( input_string[j]==' ' ) {
					i=j-1;
					break;
				}
				else if ( input_string[j]==';' ) {
					i=j-1;
					break;
				}
				else word=word+input_string[j];
			}
			output_data.push_back(word);
		}
	}
	return output_data;
}
//======================================================================================//

//======================================================================================//
//++ Read Input ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int input(string case_name) //
{
	string line;
	vector<double> num_temp;
	vector<string> str_temp;
	int i, j, g, p;
	
	// input file name
	string input_name=case_name+".inp";
	cout<<input_name<<endl;
	
	// open file +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	ifstream infile (input_name.c_str()); // open input file
	try {
		if ( !infile.is_open() ) throw "Input File "+input_name+" could not be opened";
		ho.readGrid(infile);
		
		vector<string> HOblock, LOblock, XSblock;
		int ho_err=1, lo_err=1, xs_err=1;
		while ( getline (infile,line) ) {
			str_temp=parseWord(line);
			if (str_temp.size()>1) {
				if (str_temp[0]=="begin") {
					string block_name=str_temp[1];
					
					vector<string> block;
					while ( getline (infile,line) ) {
						str_temp=parseWord(line);
						if (str_temp.size()>0) {
							if (str_temp[0]=="end") break;
							block.push_back(line);
						}
					}
					if (str_temp[1]!=block_name) throw "Fatal Error: Mismatched block names "+str_temp[1]+" "+block_name;
					
					if      ( block_name=="HO_Data" ) HOblock=block;
					else if ( block_name=="LO_Data" ) LOblock=block;
					else if ( block_name=="XS"      ) XSblock=block;
				}
			}
		}
		
		if (ho.readHO(HOblock)                     !=0) throw "Fatal Error: 'HO_Data' block read failed";
		if (ho.readHOXS(XSblock, ho.x, ho.y, ho.Ng)!=0) throw "Fatal Error: 'XS' block read failed";
		if (lo.readLO(LOblock, ho)                 !=0) throw "Fatal Error: 'LO_Data' block read failed";
		
		infile.close(); // close input file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if ( not KE_problem ) k_eff=1.0;
	
	
	}
	catch (std::string error) {
		cout<<"Error in Input Function >> "<<error<<endl;
		return 1;
	}
	return 0;
}
//======================================================================================//

//======================================================================================//
//++ function to write cell average values in viewer friendly format +++++++++++++++++++//
//======================================================================================//
void write_cell_out(vector< vector<int> > &outp, vector<double> &x, vector<double> &y, int outw, ofstream& file) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	int i, j, m, Nx=x.size()-1, Ny=y.size()-1;
	// cell average values
	for (m=0; m<int(Nx/10); m++) {
		file<<setw(outw)<<"sol. grid "<<setw(6)<<" ";
		for (i=m*10; i<(m+1)*10; i++) file<<print_out((x[i]+x[i+1])/2);
		file<<endl;
		file<<setw(outw)<<" "<<setw(6)<<"index";
		for (i=m*10; i<(m+1)*10; i++) file<<setw(outw-6)<<i+1<<setw(6)<<" ";
		file<<endl;
		for (j=Ny-1; j>=0; j--) {
			file<<print_out((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ";
			for (i=m*10; i<(m+1)*10; i++) file<<print_out(outp[i][j]);
			file<<endl;
		}
	}
	if ( m*10<Nx ) {
		file<<setw(outw)<<"sol. grid "<<setw(6)<<" ";
		for (i=m*10; i<Nx; i++) file<<print_out((x[i]+x[i+1])/2);
		file<<endl;
		file<<setw(outw)<<" "<<setw(6)<<"index";
		for (i=m*10; i<Nx; i++) file<<setw(outw-6)<<i+1<<setw(6)<<" ";
		file<<endl;
		for (j=Ny-1; j>=0; j--) {
			file<<print_out((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ";
			for (i=m*10; i<Nx; i++) file<<print_out(outp[i][j]);
			file<<endl;
		}
	}
}
//======================================================================================//
//++ function to write cell values in viewer friendly format +++++++++++++++++++++++++++//
//======================================================================================//
void write_cell_out(vector< vector<double> > &outp, vector<double> &x, vector<double> &y, int outw, ofstream& file) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	int i, j, m, Nx=x.size()-1, Ny=y.size()-1;
	int nx=outp.size();
	int ny=outp[0].size();
	if ( nx==Nx and ny==Ny ) {
		// cell average values
		for (m=0; m<int(Nx/10); m++) {
			file<<setw(outw)<<"sol. grid "<<setw(6)<<" ";
			for (i=m*10; i<(m+1)*10; i++) file<<print_out((x[i]+x[i+1])/2);
			file<<endl;
			file<<setw(outw)<<" "<<setw(6)<<"index";
			for (i=m*10; i<(m+1)*10; i++) file<<setw(outw-6)<<i+1<<setw(6)<<" ";
			file<<endl;
			for (j=Ny-1; j>=0; j--) {
				file<<print_out((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ";
				for (i=m*10; i<(m+1)*10; i++) file<<print_out(outp[i][j]);
				file<<endl;
			}
		}
		if ( m*10<Nx ) {
			file<<setw(outw)<<"sol. grid "<<setw(6)<<" ";
			for (i=m*10; i<Nx; i++) file<<print_out((x[i]+x[i+1])/2);
			file<<endl;
			file<<setw(outw)<<" "<<setw(6)<<"index";
			for (i=m*10; i<Nx; i++) file<<setw(outw-6)<<i+1<<setw(6)<<" ";
			file<<endl;
			for (j=Ny-1; j>=0; j--) {
				file<<print_out((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ";
				for (i=m*10; i<Nx; i++) file<<print_out(outp[i][j]);
				file<<endl;
			}
		}
	}
	else if ( nx==Nx+1 and ny==Ny ) {
		// cell edge values on x grid
		for (m=0; m<int(Nx/10); m++) {
			file<<setw(outw)<<"sol. grid "<<setw(6)<<" ";
			for (i=m*10; i<(m+1)*10; i++) file<<print_out(x[i]); file<<endl;
			file<<setw(outw)<<" "<<setw(6)<<"index";
			for (i=m*10; i<(m+1)*10; i++) file<<setw(outw-6)<<i+1<<setw(6)<<" "; file<<endl;
			for (j=Ny-1; j>=0; j--) {
				file<<print_out((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ";
				for (i=m*10; i<(m+1)*10; i++) file<<print_out(outp[i][j]);
				file<<endl;
			}
		}
		if ( m*10<Nx+1 ) {
			file<<setw(outw)<<"sol. grid "<<setw(6)<<" ";
			for (i=m*10; i<Nx+1; i++) file<<print_out(x[i]); file<<endl;
			file<<setw(outw)<<" "<<setw(6)<<"index";
			for (i=m*10; i<Nx+1; i++) file<<setw(outw-6)<<i+1<<setw(6)<<" "; file<<endl;
			for (j=Ny-1; j>=0; j--) {
				file<<print_out((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ";
				for (i=m*10; i<Nx+1; i++) file<<print_out(outp[i][j]); file<<endl;
			}
		}
	}
	else if ( nx==Nx and ny==Ny+1 ) {
		// cell edge values on y grid
		for (m=0; m<int(Nx/10); m++) {
			file<<setw(outw)<<"sol. grid "<<setw(6)<<" ";
			for (i=m*10; i<(m+1)*10; i++) file<<print_out((x[i]+x[i+1])/2); file<<endl;
			file<<setw(outw)<<" "<<setw(6)<<"index";
			for (i=m*10; i<(m+1)*10; i++) file<<setw(outw-6)<<i+1<<setw(6)<<" "; file<<endl;
			for (j=Ny; j>=0; j--) {
				file<<print_out(y[j])<<setw(5)<<j+1<<" ";
				for (i=m*10; i<(m+1)*10; i++) file<<print_out(outp[i][j]); file<<endl;
			}
		}
		if ( m*10<Nx ) {
			file<<setw(outw)<<"sol. grid "<<setw(6)<<" ";
			for (i=m*10; i<Nx; i++) file<<print_out((x[i]+x[i+1])/2); file<<endl;
			file<<setw(outw)<<" "<<setw(6)<<"index";
			for (i=m*10; i<Nx; i++) file<<setw(outw-6)<<i+1<<setw(6)<<" "; file<<endl;
			for (j=Ny; j>=0; j--) {
				file<<print_out(y[j])<<setw(5)<<j+1<<" ";
				for (i=m*10; i<Nx; i++) file<<print_out(outp[i][j]); file<<endl;
			}
		}
	}
	else cout<<">> Error in output\n";
}
//======================================================================================//
//++ function to write cell average values in data format ++++++++++++++++++++++++++++++//
//======================================================================================//
void write_cell_dat(vector< vector<int> > &outp, vector<double> &x, vector<double> &y, int outw, ofstream& file) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	int i, j, Nx=x.size()-1, Ny=y.size()-1;
	// cell average values
	file<<setw(outw)<<" sol. grid ,"<<setw(6)<<" "<<",";
	for (i=0; i<Nx; i++) file<<print_csv((x[i]+x[i+1])/2);
	file<<endl;
	file<<setw(outw)<<","<<setw(6)<<"index"<<",";
	for (i=0; i<Nx; i++) file<<setw(outw-6)<<i+1<<setw(6)<<",";
	file<<endl;
	for (j=0; j<Ny; j++) {
		file<<print_csv((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ,";
		for (i=0; i<Nx; i++) file<<print_csv(outp[i][j]);
		file<<endl;
	}
}
//======================================================================================//
//++ function to write cell values in data format ++++++++++++++++++++++++++++++//
//======================================================================================//
void write_cell_dat(vector< vector<double> > &outp, vector<double> &x, vector<double> &y, int outw, ofstream& file) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	int i, j, Nx=x.size()-1, Ny=y.size()-1;
	int nx=outp.size();
	int ny=outp[0].size();
	if ( nx==Nx and ny==Ny ) {
		// cell average values
		file<<setw(outw)<<" sol. grid ,"<<setw(6)<<" "<<",";
		for (i=0; i<Nx; i++) file<<print_csv((x[i]+x[i+1])/2);
		file<<endl;
		file<<setw(outw)<<","<<setw(6)<<"index"<<",";
		for (i=0; i<Nx; i++) file<<setw(outw-6)<<i+1<<setw(6)<<",";
		file<<endl;
		for (j=0; j<Ny; j++) {
			file<<print_csv((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ,";
			for (i=0; i<Nx; i++) file<<print_csv(outp[i][j]);
			file<<endl;
		}
	}
	else if ( nx==Nx+1 and ny==Ny ) {
		// cell edge values on x grid
		file<<setw(outw)<<" sol. grid ,"<<setw(6)<<" "<<",";
		for (i=0; i<Nx+1; i++) file<<print_csv(x[i]);
		file<<endl;
		file<<setw(outw)<<","<<setw(6)<<"index"<<",";
		for (i=0; i<Nx+1; i++) file<<setw(outw-6)<<i+1<<setw(6)<<",";
		file<<endl;
		for (j=0; j<Ny; j++) {
			file<<print_csv((y[j]+y[j+1])/2)<<setw(5)<<j+1<<" ,";
			for (i=0; i<Nx+1; i++) file<<print_csv(outp[i][j]);
			file<<endl;
		}
	}
	else if ( nx==Nx and ny==Ny+1 ) {
		// cell edge values on y grid
		file<<setw(outw)<<" sol. grid ,"<<setw(6)<<" "<<",";
		for (i=0; i<Nx; i++) file<<print_csv((x[i]+x[i+1])/2);
		file<<endl;
		file<<setw(outw)<<","<<setw(6)<<"index"<<",";
		for (i=0; i<Nx; i++) file<<setw(outw-6)<<i+1<<setw(6)<<",";
		file<<endl;
		for (j=0; j<Ny+1; j++) {
			file<<print_csv(y[j])<<setw(5)<<j+1<<" ,";
			for (i=0; i<Nx; i++) file<<print_csv(outp[i][j]);
			file<<endl;
		}
	}
	else cout<<">> Error in 'Data' output\n";
}

//======================================================================================//
//++ function to write multi group cell values in output format ++++++++++++++++++++++++//
//======================================================================================//
void write_group_out(vector< vector< vector<double> > > &outp, int Ng, vector<double> &x, vector<double> &y, int etaL, int outw, ofstream& file) { //<<<<<<<<<<<<
	for (int g=0; g<Ng; g++) {
		file<<"Grid # "<<etaL<<" Energy Group # "<<g+1<<endl;
		write_cell_out(outp[g], x, y, outw, file);
	}
}
//======================================================================================//
//++ function to write multi group values in data format +++++++++++++++++++++++++++++++//
//======================================================================================//
void write_group_dat(vector< vector< vector<double> > > &outp, int Ng, vector<double> &x, vector<double> &y, int etaL, int outw, ofstream& file) { //<<<<<<<<<<<
	for (int g=0; g<Ng; g++) {
		file<<"Grid # "<<etaL<<" Energy Group # "<<g+1<<endl;
		write_cell_dat(outp[g], x, y, outw, file);
	}
}
//======================================================================================//
//++ function to write multi grid cell edge avg values in output format ++++++++++++++++//
//======================================================================================//
void write_grid_out(vector< vector< vector< vector<double> > > > &outp, vector<int> &Ng, vector<double> &x, vector<double> &y, int outw, ofstream& file) { //<<<<<<<<<<<<
	int eta_star=Ng.size();
	for (int k=0; k<eta_star; k++) {
		if (k==0 ) file<<" -- Fine Energy Grid -- \n";
		else if (k==eta_star-1) file<<" -- Grey Energy Grid -- \n";
		else file<<" -- Energy Grid # "<<k<<" -- \n";
		write_group_out(outp[k], Ng[k], x, y, k, outw, file);
	}
}
//======================================================================================//
//++ function to write multi grid cell average values in data format +++++++++++++++++++//
//======================================================================================//
void write_grid_dat(vector< vector< vector< vector<double> > > > &outp, vector<int> &Ng, vector<double> &x, vector<double> &y, int outw, ofstream& file) { //<<<<<<<<<<<<
	int eta_star=Ng.size();
	for (int k=0; k<eta_star; k++) {
		if (k==0 ) file<<" -- Fine Energy Grid -- \n";
		else if (k==eta_star-1) file<<" -- Grey Energy Grid -- \n";
		else file<<" -- Energy Grid # "<<k<<" -- \n";
		write_group_dat(outp[k], Ng[k], x, y, k, outw, file);
	}
}
//======================================================================================//

//======================================================================================//
//++ function to write cell average values in column format +++++++++++++++++//
//======================================================================================//
void write_column_cell_data(vector< vector<double> > &outp, vector<double> &x, vector<double> &y, int outw, ofstream& file) {
	int Nx=x.size()-1, Ny=y.size()-1;
	file<<" x index , x grid , y index , y grid ,"<<" scalar ,"<<endl;
	for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) file<<i+1<<","<<print_csv((x[i]+x[i+1])/2)<<j+1<<","<<print_csv((y[j]+y[j+1])/2)<<print_csv(outp[i][j])<<endl;
}
void write_column_cell_data(vector< vector<int> > &outp, vector<double> &x, vector<double> &y, int outw, ofstream& file) {
	int Nx=x.size()-1, Ny=y.size()-1;
	file<<" x index , x grid , y index , y grid ,"<<" scalar ,"<<endl;
	for (int i=0; i<Nx; i++) for (int j=0; j<Ny; j++) file<<i+1<<","<<print_csv((x[i]+x[i+1])/2)<<j+1<<","<<print_csv((y[j]+y[j+1])/2)<<print_csv(outp[i][j])<<endl;
}
//======================================================================================//
//++ function to write multi grid cell average values in column format +++++++++++++++++//
//======================================================================================//
void write_column_grid_data(vector< vector< vector<double> > > &outp, vector<int> &Ng, vector<double> &x, vector<double> &y, int etaL, int outw, ofstream& file) {
	int Nx=x.size()-1, Ny=y.size()-1;
	file<<" x index , x grid , y index , y grid ,";
	for (int g=0; g<Ng[etaL]; g++) file<<" group "<<g<<",";
	file<<endl;
	for (int i=0; i<Nx; i++) {
		for (int j=0; j<Ny; j++) {
			file<<i+1<<","<<print_csv((x[i]+x[i+1])/2)<<j+1<<","<<print_csv((y[j]+y[j+1])/2);
			for (int g=0; g<Ng[etaL]; g++) file<<print_csv(outp[g][i][j]);
			file<<endl;
		}
	}
}
//======================================================================================//

//======================================================================================//
//++ function to output data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void preOutput(std::string case_name) {
	int p, g, gg, i, j, m, k, outw=16;

	cout<<"Begin Output\n";
	string output_file=case_name+".out";
	cout<<output_file<<endl;
	ofstream outfile (output_file.c_str()); // open output file. closed in output function
	// file output ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	outfile<<"Output of File : "<<case_name+".inp"<<endl;
	outfile<<"2D Multi-level Method with Step Characteristics Transport\n";
	outfile<<"Version: 1.6\n";
	outfile<<"Compliled on : "<<__DATE__<<" at "<<__TIME__<<endl;
	outfile<<"Programer : Luke Cornejo\n";
	outfile<<"Case Name : "<<test_name<<endl;
	// current date/time based on current system
	time_t now = time(0);
	// convert now to string form
	char* dt = ctime(&now);
	
	outfile<<"Program Ran On: "<<dt<<endl;
	
	outfile<<" -- Problem -- \n";
	outfile<<"+-----------------------------------------------+\n";
	if ( KE_problem ) outfile<<" K-eigenvalue Problem \n";
	else outfile<<" Fixed Source Problem \n";
	outfile<<"X Domain from "<<ho.x[0]<<" cm to "<<ho.x.back()<<" cm \n";
	outfile<<"Y Domain from "<<ho.y[0]<<" cm to "<<ho.y.back()<<" cm \n";
	outfile<<"Spatial Grid "<<ho.Nx<<" x "<<ho.Ny<<" cells \n";
	// Energy Solution Grid
	outfile<<" -- Energy Grid -- \n";
	outfile<<"Number of Low-order Grids -- "<<lo.eta_star<<" -- \n";
	outfile<<"Number of Groups on Grid "<<0<<" is "<<lo.Ng[0]<<endl;
	for (k=1; k<lo.eta_star; k++) {
		outfile<<"Number of Groups on Grid "<<k<<" is "<<lo.Ng[k]<<" with group combinations";
		for (g=0; g<lo.Ng[k]; g++) outfile<<"  "<<lo.omegaP[k][g]+1<<" to "<<lo.omegaP[k][g+1];
		outfile<<endl;
	}
	//outfile<<"One Group Grid "<<" with group combination "<<omegaP[eta_star-1][1]-omegaP[eta_star-1][0]<<endl;
	
	// boundary conditions
	outfile<<"\n -- Boundary Conditions -- \n";
	outfile<<"Type of BC: "<<kbc;   // Write type of BC
	if (kbc==1) outfile<<" Incoming Flux by Side\n";
	else if (kbc==2) outfile<<" Incoming Flux by Angle\n";
	else {
		outfile<<" Reflective on ";
		if ( reflectiveL ) outfile<<"| Left ";
		if ( reflectiveR ) outfile<<"| Right ";
		if ( reflectiveB ) outfile<<"| Bottom ";
		if ( reflectiveT ) outfile<<"| Top ";
		outfile<<"|\n";
	}
	
	outfile<<"+-----------------------------------------------+\n";
	
	ho.writeHO(outfile);
	lo.writeLO(outfile);
	ho.writeHOXS(outfile);
	// Output High Order Grid
	ho.writeSpatialGrid(outfile);
	
	outfile.close(); // close output file opened in input file +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	
}
//======================================================================================//

//======================================================================================//
//++ function to output data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void output(string case_name, double run_time) {
	double eps=1e-30;
	int outw=16;
	int eta_star=lo.eta_star;
	string output_file=case_name+".out";
	cout<<output_file<<endl;
	ofstream outfile (output_file.c_str(),ios::app); // open output file. closed in output function
	
	ho.outputQuadrature(outfile); // output quadrature
	// calculate residuals ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Equation residuals ///////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	// file output ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	outfile<<endl;
	outfile<<" -------------- \n";
	outfile<<" -- Solution -- \n";
	outfile<<" -------------- \n";
	outfile<<"Program run time : "<<run_time<<" seconds"<<endl;
	outfile<<"\nNumber of iterations "<<ho.n_iterations<<endl;
	
	for (int k=0; k<eta_star; k++) {
		lo.rho_phi[k].erase(lo.rho_phi[k].begin());
		lo.rho_keff[k].erase(lo.rho_keff[k].begin());
		lo.norm_phi[k].erase(lo.norm_phi[k].begin());
		lo.norm_phiH[k].erase(lo.norm_phiH[k].begin());
		lo.norm_keff[k].erase(lo.norm_keff[k].begin());
		lo.norm_kappa[k].erase(lo.norm_kappa[k].begin());
	}
	
	
	outfile<<"\n -- Residuals -- \n";
	outfile<<"High-order Residual \n";
	outfile<<"Balance Residual:"<<print_out(ho.res_ho.back())<<" at "<<ho.i_ho.back()<<" , "<<ho.j_ho.back()<<" , "<<ho.g_ho.back()<<endl;
	
	outfile<<"Low-order Residuals \n";
	for (int k=0; k<eta_star; k++) {
		outfile<<"Energy Grid # "<<k<<endl;
		outfile<<"Residual of Equations Solved In Matrix \n";
		outfile<<"Absolute Matrix Residual:"<<print_out(lo.res_Amatrix[k])<<" in group "<<lo.g_Amatrix[k]<<endl;
		outfile<<"Relative Matrix Residual:"<<print_out(lo.res_Rmatrix[k])<<" in group "<<lo.g_Rmatrix[k]<<endl;
		outfile<<"Cell Balance Residual:"<<print_out(lo.res_mbal[k])<<" in group "<<lo.g_mbal[k]<<" at "<<lo.i_mbal[k]<<" , "<<lo.j_mbal[k]<<endl; 
		
		outfile<<"Residual of General Equations \n";
		outfile<<"Cell Balance Residual:"<<print_out(lo.res_bal[k])<<" in group "<<lo.g_bal[k]<<" at "<<lo.i_bal[k]<<" , "<<lo.j_bal[k]<<endl; 
		lo.writeResiduals(k,outfile);
	}
	
	// Calculate Iterative Residuals for High-Order Problem // Residuals of the actual Equaitons
	int i_it=101010, j_it=101010, g_it=101010; // Iterative Residuals Location
	outfile<<" -- Iterative Residuals -- \n";
	outfile<<"High-order Iterative Residual "<<print_out(ho.residualIterative(i_it, j_it, g_it));
	outfile<<" in group "<<g_it<<" at "<<i_it<<" , "<<j_it<<endl;
	outfile<<"Low-order Iterative Residuals \n";
	for (int k=0; k<eta_star; k++) {
		i_it=101010; j_it=101010; g_it=101010; // Iterative Residuals Location
		outfile<<"Grid "<<k<<" Residual "<<print_out(lo.residualIterative(k, i_it, j_it, g_it));
		outfile<<" in group "<<g_it<<" at "<<i_it<<" , "<<j_it<<endl;
	}
	
	// ++ Evaluate Consistency between solutions +++++++++++++++++++++++++++++
	lo.consistencyBetweenLOAndHO(ho, outfile);
	
	lo.writeConsistency(outfile);
	
	if (KE_problem) {
		outfile<<"\n -- K-eigenvalue -- \n";
		outfile<<"K_eff = "<<print_outH(k_eff)<<endl;
	}
	
	outfile<<"\n -- Iteration Data -- \n";
	outfile<<"  Iter."<<"    Estimated   "<<"    High-order  "<<"   k_effetive   "<<"     rho of     "<<"   k_effetive   "<<"  High-order    "<<"   Low-order   ";
	for (int k=0; k<eta_star; k++) outfile<<" Grid # "<<setw(2)<<k<<" ";
	outfile<<"   Number of   \n";
	outfile<<"    #  "<<"  Spectral Rad. "<<" Diff L-inf Norm"<<"                "<<"  k_effective   "<<" Diff L-inf Norm"<<" Sol. Time [s]  "<<" Sol. Time [s] ";
	for (int k=0; k<eta_star; k++) outfile<<"iterations ";
	
	outfile<<"   Matrix Sol.  \n";
	for (int i=0; i<ho.n_iterations; i++) {
		outfile<<setw(4)<<i<<setw(2)<<" "<<ho.writeIteration(i)<<print_out(lo.dt[0][i]);
		for (int k=0; k<eta_star; k++) outfile<<setw(7)<<lo.num_grid[k][i]<<setw(4)<<" ";
		outfile<<setw(10)<<lo.num_mtot[i]<<setw(6)<<endl;
	}
	// Summary
	for (int i=0; i<1+7*16; i++) outfile<<" ";
	for (int k=0; k<eta_star; k++) {
		int summary=0;
		for (int i=0; i<ho.n_iterations; i++) summary+=lo.num_grid[k][i];
		outfile<<setw(7)<<summary<<setw(4)<<" ";
	}
	{
		int summary=0;
		for (int i=0; i<ho.n_iterations; i++) summary+=lo.num_mtot[i];
		outfile<<setw(10)<<summary<<setw(6)<<endl;
	}
	// Output Detailed Convergence data
	for (int k=0; k<eta_star; k++) for (int i=0; i<lo.num_losi[k].size()-1; i++) lo.num_losi[k][i+1]+=lo.num_losi[k][i]; // Cumulative sum of all the iterations
	outfile<<" -- Nested Convergence Data -- \n";
	for (int i=0; i<ho.n_iterations; i++) {
		outfile<<"Transport Iteration # "<<i;
		lo.rec_write_iteration_out(i, 0, outfile);
	}
	
	if ( lo.Nx<20 and lo.Ny<20 ) {
		// output flux
		lo.writeSolutionOut(outfile);
		
		//////////////////////////////////////////////////////////////////////////////////////////////////
		outfile<<"\n ------------------------- ";
		outfile<<"\n -- High Order Solution -- "; // Write Step Characteristics solution
		outfile<<"\n ------------------------- \n";
		outfile<<"\n -- Cell Averaged Scalar Flux -- \n";
		write_group_out(ho.phi, ho.Ng, ho.x, ho.y, 0, outw, outfile); // call function to write out cell average scalar flux
		
		outfile<<"\n -- X Face Average Normal Current J_x -- \n";
		write_group_out(ho.j_x, ho.Ng, ho.x, ho.y, 0, outw, outfile); // call function to write out cell edge current on x grid
		
		outfile<<"\n -- Y Face Average Normal Current J_y -- \n";
		write_group_out(ho.j_y, ho.Ng, ho.x, ho.y, 0, outw, outfile); // call function to write out cell edge scalar flux on y grid
	}
	else outfile<<" -- See *.lo.csv and *.ho.csv files for solution\n";
	
	outfile.close(); // close output file opened in input file +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	
	///////////////////////////////////////////////
	string data_file=case_name+".csv";
	cout<<data_file<<endl;
	ofstream datfile (data_file.c_str()); // open output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// output flux
	datfile<<" Iter.,"<<"    HO Flux    ,"<<"    Estimated  ,"<<"    High-order ,"<<"     rho of    ,"<<"      Grey     ,"<<"   k_effetive  ,"<<"     rho of    ,"<<"   k_effetive  ,"<<"     rho of    ,"<<"     kappa     ,";
	for (int k=0; k<eta_star; k++) datfile<<" Grid # "<<setw(2)<<k<<",";
	datfile<<"   Number of   ,\n";
	datfile<<"    # ,"<<"    Inf Norm   ,"<<"  Spectral Rad.,"<<"Diff L-inf Norm,"<<" Grey Solution ,"<<"Diff L-inf Norm,"<<"               ,"<<"  k_effective  ,"<<"Diff L-inf Norm,"<<"     kappa     ,"<<"Diff L-inf Norm,";
	for (int k=0; k<eta_star; k++) datfile<<"iterations,";
	datfile<<"   Matrix Sol.  ,\n";
	for (int i=0; i<ho.n_iterations; i++) {
		datfile<<setw(4)<<i<<setw(2)<<","<<print_csv(ho.norm_inf[i])<<print_csv(ho.rho_phi[i])<<print_csv(ho.norm_phi[i])<<print_csv(ho.rho_phiH[i])<<print_csv(ho.norm_phiH[i])
		<<print_csv(ho.k_keff[i])<<print_csv(ho.rho_keff[i])<<print_csv(ho.norm_keff[i])<<print_csv(ho.rho_kappa[i])<<print_csv(ho.norm_kappa[i]);
		for (int k=0; k<eta_star; k++) datfile<<setw(7)<<lo.num_grid[k][i]<<setw(4)<<",";
		datfile<<setw(10)<<lo.num_mtot[i]<<setw(6)<<","<<endl;
	}
	// Summary
	datfile<<"     ,";
	for (int i=0; i<10; i++) datfile<<"               ,";
	for (int k=0; k<eta_star; k++) {
		int summary=0;
		for (int i=0; i<ho.n_iterations; i++) summary+=lo.num_grid[k][i];
		datfile<<setw(7)<<summary<<setw(4)<<",";
	}
	{
		int summary=0;
		for (int i=0; i<ho.n_iterations; i++) summary+=lo.num_mtot[i];
		datfile<<setw(10)<<summary<<setw(6)<<","<<endl;
	}
	
	
	datfile<<"# of iterations ,"<<ho.n_iterations<<endl;
	datfile<<" -- Iteration Data -- \n";
	datfile<<"  Iter.,   Convergence,  High-order   ,   Low-order   ,"<<"    High-order ,"<<"      ,"<<"      ,"<<"      ,"<<"    High-order ,"<<"      ,"<<"      ,"<<"      ,"<<"D Tilde Factors,"<<"D Tilde Factors,";
	for (int g=0; g<ho.Ng; g++) datfile<<" HO Diff. Norm ,"<<"     ,"<<"     ,";
	datfile<<endl;
	datfile<<"    #  ,      Rate    , Sol. Time [s] ,  Sol. Time [s],"<<"Diff L-inf Norm,"<<"   i  ,"<<"   j  ,"<<"   g  ,"<<" Res L-inf Norm,"<<"   i  ,"<<"   j  ,"<<"   g  ,"<<"Diff L-inf Norm,"<<" Diff L-2 Norm ,";
	for (int g=0; g<ho.Ng; g++) datfile<<"  in group "<<setw(2)<<g+1<<"  ,"<<"     ,"<<"     ,";
	datfile<<endl;
	for (int i=0; i<ho.n_iterations; i++) {
		datfile<<setw(4)<<i<<setw(2)<<" "<<","<<print_csv(ho.rho_phi[i])<<print_csv(ho.dt[i])<<print_csv(lo.dt[0][i])<<
		print_csv(ho.norm_phi[i])<<setw(5)<<ho.i_phi[i]<<setw(2)<<","<<setw(5)<<ho.j_phi[i]<<setw(2)<<","<<setw(5)<<ho.g_phi[i]<<setw(2)<<","<<
		print_csv(ho.bal_res[i]) <<setw(5)<<ho.i_res[i]<<setw(2)<<","<<setw(5)<<ho.j_res[i]<<setw(2)<<","<<setw(5)<<ho.g_res[i]<<setw(2)<<","<<
		print_csv(lo.norm_DTLI[i])<<print_csv(lo.norm_DTL2[i]);
		for (int g=0; g<ho.Ng; g++) datfile<<print_csv(ho.norm_gphi[i][g])<<setw(4)<<ho.i_gphi[i][g]<<setw(2)<<","<<setw(4)<<ho.j_gphi[i][g]<<setw(2)<<",";
		datfile<<endl;
	}
	datfile<<"Number of High-order Iterations, "<<ho.n_iterations<<endl;
	// Output Detailed Convergence data
	for (int i=0; i<ho.n_iterations; i++) {
		datfile<<"Transport Iteration # "<<i<<",";
		lo.rec_write_iteration_dat(i, 0, datfile);
	}
	datfile<<"Number of High-order Iterations, "<<ho.n_iterations<<endl;
	// Output Detailed Convergence data
	for (int i=0; i<ho.n_iterations; i++) {
		datfile<<"High-order Iteration #, "<<i<<endl;
		datfile<<"High-order Solution Time, "<<print_csv(ho.dt[i])<<endl;
		datfile<<"Low- order Solution Time, "<<print_csv(lo.dt[0][i])<<endl;
		lo.rec_write_iteration_long_dat(i, 0, datfile);
	}
	
	
	datfile<<" # of x cells , # of y cells ,\n";
	datfile<<ho.Nx<<" , "<<ho.Ny<<", \n";
	
	// output grid
	ho.writeSpatialGridDat(datfile);
	
	datfile.close(); // close output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	#pragma omp parallel sections
	{
	#pragma omp section
	lo.writeSolutionDat(case_name);
	#pragma omp section
	ho.writeSolutionDat(case_name);
	#pragma omp section
	lo.writeLOXSFile(case_name);
	}
	
	
}
//======================================================================================//



