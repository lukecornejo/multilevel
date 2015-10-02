# this target will compile all
CC=c++

CFLAGS=-O3 -fopenmp -pipe
VERSION=1.6
	
all:
	$(CC) $(CFLAGS) -I ./eigen LO_$(VERSION).cpp HO_$(VERSION).cpp IO_$(VERSION).cpp MultiGrid_$(VERSION).cpp LO_E_$(VERSION).cpp  HO_XS_$(VERSION).cpp -o SC_ML_$(VERSION).exe
	
	
clean:
	rm -f *.o *.d

