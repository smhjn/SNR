
debug		:= 0
include		../Makefile.inc

all: clean SNRAVX SNRPhi

SNRAVX: SNRSIMD.cpp
	$(CC) -o $(PROJ_BASE)/bin/SNRAVXTuning SNRSIMD.cpp $(CPU_INCLUDES) $(CPU_CFLAGS) $(CPU_LDFLAGS)

SNRPhi: SNRSIMD.cpp
	$(CC) -o $(PROJ_BASE)/bin/SNRPhiTuning SNRSIMD.cpp $(PHI_INCLUDES) $(PHI_CFLAGS) $(PHI_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/SNRAVXTuning
	rm -f $(PROJ_BASE)/bin/SNRPhiTuning

