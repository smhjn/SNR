
debug		:= 0
include		../Makefile.inc

all: clean SNRAVX SNRPhi
 
SNRAVX: SNRSIMD.cpp
	$(CC) -o $(PROJ_BASE)/bin/SNRAVXTest SNRSIMD.cpp $(CPU_INCLUDES) $(CPU_LIBS) $(CPU_CFLAGS) $(CPU_LDFLAGS)

SNRPhi: SNRSIMD.cpp
	$(CC) -o $(PROJ_BASE)/bin/SNRPhiTest SNRSIMD.cpp $(PHI_INCLUDES) $(PHI_LIBS) $(PHI_CFLAGS) $(PHI_LDFLAGS)

clean:
	rm -f $(PROJ_BASE)/bin/SNRAVXTest
	rm -f $(PROJ_BASE)/bin/SNRPhiTest

