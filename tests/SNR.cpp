// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ctime>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <SNR.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char *argv[]) {
  bool print = false;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
  unsigned int nrDMsPerBlock = 0;
  unsigned int nrPeriodsPerBlock = 0;
  unsigned int nrDMsPerThread = 0;
  unsigned int nrPeriodsPerThread = 0;
	long long unsigned int wrongSamples = 0;
	AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);
    print = args.getSwitch("-print");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    nrDMsPerBlock = args.getSwitchArgument< unsigned int >("-db");
    nrPeriodsPerBlock = args.getSwitchArgument< unsigned int >("-pb");
    nrDMsPerThread = args.getSwitchArgument< unsigned int >("-dt");
    nrPeriodsPerThread = args.getSwitchArgument< unsigned int >("-pt");
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), 0, 0);
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-print] -opencl_platform ... -opencl_device ... -padding ... -db ... -pb ... -dt ... -pt ... -dms ... -periods ... -bins ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
  std::vector< dataType > foldedData = std::vector< dataType >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  cl::Buffer foldedData_d;
  std::vector< dataType > snrs = std::vector< dataType >(observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< dataType > snrs_c = std::vector< dataType >(observation.getNrPeriods() * observation.getNrPaddedDMs());
  cl::Buffer snrs_d;
  try {
    foldedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, foldedData.size() * sizeof(dataType), NULL, NULL);
    snrs_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrs.size() * sizeof(dataType), NULL, NULL);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(NULL));
  for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
        foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM] = static_cast< dataType >(rand() % 10);
      }
    }
	}
  std::fill(snrs.begin(), snrs.end(), static_cast< dataType >(0));
  std::fill(snrs_c.begin(), snrs_c.end(), static_cast< dataType >(0));

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(foldedData_d, CL_FALSE, 0, foldedData.size() * sizeof(dataType), reinterpret_cast< void * >(foldedData.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(snrs_d, CL_FALSE, 0, snrs.size() * sizeof(dataType), reinterpret_cast< void * >(snrs.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  cl::Kernel * kernel;
  std::string * code = PulsarSearch::getSNROpenCL(nrDMsPerBlock, nrPeriodsPerBlock, nrDMsPerThread, nrPeriodsPerThread, typeName, observation);
  if ( print ) {
    std::cout << *code << std::endl;
  }

  try {
    kernel = isa::OpenCL::compile("snr", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
  } catch ( isa::OpenCL::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global(observation.getNrPaddedDMs() / nrDMsPerThread, observation.getNrPeriods() / nrPeriodsPerThread);
    cl::NDRange local(nrDMsPerBlock, nrPeriodsPerBlock);

    kernel->setArg(0, foldedData_d);
    kernel->setArg(1, snrs_d);
    
    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, NULL, NULL);
    PulsarSearch::snrFoldedTS(observation, foldedData, snrs_c);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrs_d, CL_TRUE, 0, snrs.size() * sizeof(dataType), reinterpret_cast< void * >(snrs.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      if ( ! isa::utils::same(snrs_c[(period * observation.getNrPaddedDMs()) + dm], snrs[(period * observation.getNrPaddedDMs()) + dm]) ) {
        wrongSamples++;
      }
    }
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods()) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

