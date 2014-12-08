
debug		:= 0
include		../Makefile.inc

all: clean SNR
 
SNR: SNR.cpp
	$(CC) -o $(PROJ_BASE)/bin/SNRTest SNR.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/SNRTest

