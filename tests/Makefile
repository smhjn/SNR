
debug		:= 0
include		../Makefile.inc

all: clean SNR SNRAVX SNRPhi
 
SNR: SNR.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/SNRTest SNR.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

SNRAVX: SNRSIMD.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/SNRAVXTest SNRSIMD.cpp $(CPU_INCLUDES) $(CPU_LIBS) $(CPU_CFLAGS) $(CPU_LDFLAGS)

SNRPhi: SNRSIMD.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/SNRPhiTest SNRSIMD.cpp $(PHI_INCLUDES) $(PHI_LIBS) $(PHI_CFLAGS) $(PHI_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/SNRTest
	rm -f $(PROJ_BASE)/bin/SNRAVXTest
	rm -f $(PROJ_BASE)/bin/SNRPhiTest

