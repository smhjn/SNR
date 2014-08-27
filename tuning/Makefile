
debug		:= 0
include		../Makefile.inc

all: clean SNR SNRAVX SNRPhi

SNR: SNR.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/SNRTuning SNR.cpp $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

SNRAVX: SNRSIMD.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/SNRAVXTuning SNRSIMD.cpp $(CPU_INCLUDES) $(CPU_CFLAGS) $(CPU_LDFLAGS)

SNRPhi: SNRSIMD.cpp
	$(INTELCXX) -o $(PROJ_BASE)/bin/SNRPhiTuning SNRSIMD.cpp $(PHI_INCLUDES) $(PHI_CFLAGS) $(PHI_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/SNRTuning
	rm -f $(PROJ_BASE)/bin/SNRAVXTuning
	rm -f $(PROJ_BASE)/bin/SNRPhiTuning

