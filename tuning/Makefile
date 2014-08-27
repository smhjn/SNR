
debug		:= 0
include		../Makefile.inc

all: clean SNR

SNR: SNR.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/SNRTuning SNR.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/SNRTuning

