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
#include <algorithm>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <SNR.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

typedef float dataType;
string typeName("float");


int main(int argc, char * argv[]) {
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int minThreads = 0;
	unsigned int maxThreadsPerBlock = 0;
	unsigned int maxItemsPerThread = 0;
	unsigned int maxColumns = 0;
	unsigned int maxRows = 0;
  AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);

		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreadsPerBlock = args.getSwitchArgument< unsigned int >("-max_threads");
		maxItemsPerThread = args.getSwitchArgument< unsigned int >("-max_items");
		maxColumns = args.getSwitchArgument< unsigned int >("-max_columns");
		maxRows = args.getSwitchArgument< unsigned int >("-max_rows");
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), 0, 0);
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch ( isa::utils::EmptyCommandLine &err ) {
		std::cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -min_threads ... -max_threads ... -max_items ... -max_columns ... -max_rows ... -dms ... -periods ... -bins ... " << std::endl;
		return 1;
	} catch ( std::exception &err ) {
		std::cerr << err.what() << std::endl;
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
  cl::Buffer snrs_d;
  try {
    foldedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, foldedData.size() * sizeof(dataType), 0, 0);
    snrs_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrs.size() * sizeof(dataType), 0, 0);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(0));
  for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
        foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM] = static_cast< dataType >(rand() % 10);
      }
    }
	}
  std::fill(snrs.begin(), snrs.end(), static_cast< dataType >(0));

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(foldedData_d, CL_FALSE, 0, foldedData.size() * sizeof(dataType), reinterpret_cast< void * >(foldedData.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(snrs_d, CL_FALSE, 0, snrs.size() * sizeof(dataType), reinterpret_cast< void * >(snrs.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

	// Find the parameters
	std::vector< unsigned int > DMsPerBlock;
	for ( unsigned int DMs = minThreads; DMs <= maxColumns; DMs += minThreads ) {
		if ( (observation.getNrPaddedDMs() % DMs) == 0 ) {
			DMsPerBlock.push_back(DMs);
		}
	}
	std::vector< unsigned int > periodsPerBlock;
	for ( unsigned int periods = 1; periods <= maxRows; periods++ ) {
		if ( (observation.getNrPeriods() % periods) == 0 ) {
			periodsPerBlock.push_back(periods);
		}
	}

	std::cout << std::fixed << std::endl;
	std::cout << "# nrDMs nrPeriods nrBins DMsPerBlock periodsPerBlock DMsPerThread periodsPerThread GFLOP/s err time err" << std::endl << std::endl;

  for ( std::vector< unsigned int >::iterator DMs = DMsPerBlock.begin(); DMs != DMsPerBlock.end(); ++DMs ) {
    for ( std::vector< unsigned int >::iterator periods = periodsPerBlock.begin(); periods != periodsPerBlock.end(); ++periods ) {
      if ( (*DMs * *periods) > maxThreadsPerBlock ) {
        break;
      }
      for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
        if ( observation.getNrPaddedDMs() % (*DMs * DMsPerThread) != 0 ) {
          continue;
        }
        for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsPerThread; periodsPerThread++ ) {
          if ( observation.getNrPeriods() % (*periods * periodsPerThread) != 0 ) {
            continue;
          }
          if ( DMsPerThread + periodsPerThread > maxItemsPerThread ) {
            break;
          }
          // Generate kernel
          double flops = isa::utils::giga(static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins());
          isa::utils::Timer timer;
          isa::utils::Stats< double > stats;
          cl::Event event;
          cl::Kernel * kernel;
          std::string * code = PulsarSearch::getSNROpenCL(*DMs, *periods, DMsPerThread, periodsPerThread, typeName, observation);

          try {
            kernel = isa::OpenCL::compile("snr", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
          } catch ( isa::OpenCL::OpenCLError &err ) {
            std::cerr << err.what() << std::endl;
            continue;
          }

          cl::NDRange global(observation.getNrPaddedDMs() / DMsPerThread, observation.getNrPeriods() / periodsPerThread);
          cl::NDRange local(*DMs, *periods);

          kernel->setArg(0, foldedData_d);
          kernel->setArg(1, snrs_d);

          // Warm-up run
          try {
            clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
            event.wait();
          } catch ( cl::Error &err ) {
            std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
            continue;
          }
          // Tuning runs
          try {
            for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
              timer.start();
              clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
              event.wait();
              timer.stop();
              stats.addElement(flops / timer.getLastRunTime());
            }
          } catch ( cl::Error &err ) {
            std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
            continue;
          }

          std::cout << observation.getNrDMs() << " " << observation.getNrPeriods() << " " << observation.getNrBins() << " " << *DMs << " " << *periods << " " << DMsPerThread << " " << periodsPerThread << " " << std::setprecision(3) << stats.getAverage() << " " << stats.getStdDev() << " " << std::setprecision(6) << timer.getAverageTime() << " " << timer.getStdDev() << std::endl;
        }
      }
    }
  }

	std::cout << std::endl;

	return 0;
}

