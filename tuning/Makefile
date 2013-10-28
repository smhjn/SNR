
debug		:= 0
include		../Makefile.inc


all: clean SNR
 

SNR: SNR.cpp
	$(LINKER) -o $(PROJ_BASE)/bin/SNRTuning SNR.cpp $(UTILS_SRC) $(INCLUDES) $(LIBS) $(CFLAGS) $(LDFLAGS) 

clean:
	rm -f $(PROJ_BASE)/bin/SNRTuning
