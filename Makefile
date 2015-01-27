
# https://github.com/isazi/utils
UTILS := $(HOME)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(HOME)/src/OpenCL
# https://github.com/isazi/AstroData
ASTRODATA := $(HOME)/src/AstroData

INCLUDES := -I"include" -I"$(ASTRODATA)/include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL

CC := g++

# Dependencies
DEPS := $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o bin/SNR.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 


all: bin/SNR.o bin/SNRTest bin/SNRTuning bin/printCode

bin/SNR.o: $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/utils.o include/SNR.hpp src/SNR.cpp
	$(CC) -o bin/SNR.o -c src/SNR.cpp $(CL_INCLUDES) $(CFLAGS)

bin/SNRTest: $(CL_DEPS) src/SNRTest.cpp
	$(CC) -o bin/SNRTest src/SNRTest.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/SNRTuning: $(CL_DEPS) src/SNRTuning.cpp
	$(CC) -o bin/SNRTuning src/SNRTuning.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/printCode: $(DEPS) src/printCode.cpp
	$(CC) -o bin/printCode src/printCode.cpp $(DEPS) $(INCLUDES) $(LDFLAGS) $(CFLAGS)

clean:
	-@rm bin/*

