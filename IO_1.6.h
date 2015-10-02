#ifndef IOMLNDA_H // Include guard
#define IOMLNDA_H

// ioMLNDA2.7.h
#include <fstream>
#include <vector>


double dparse(std::string, int&);

int iparse(std::string, int&);

int findIndex(int, std::vector<int>);

std::vector<double> parseNumber(std::string);

std::vector<std::string> parseWord(std::string);

std::string print_out(double);
std::string print_outH(double);

std::string print_csv(double);

void write_cell_dat(                          std::vector< std::vector<double> >     &, std::vector<double> &, std::vector<double> &, int, std::ofstream&);
void write_group_dat(            std::vector< std::vector< std::vector<double> > >   &, int, std::vector<double> &, std::vector<double> &, int, int, std::ofstream&);
void write_grid_dat(std::vector< std::vector< std::vector< std::vector<double> > > > &, std::vector<int> &, std::vector<double> &, std::vector<double> &, int, std::ofstream&);

void write_grid_out(std::vector< std::vector< std::vector< std::vector<double> > > > &, std::vector<int> &, std::vector<double> &, std::vector<double> &, int, std::ofstream&);

void write_column_cell_data(std::vector< std::vector<int> >, std::vector<double> &, std::vector<double> &, int, std::ofstream&);
int input(std::string);

void output(std::string, double);          // output code
void preOutput(std::string);

#endif // IOMLNDA_H

