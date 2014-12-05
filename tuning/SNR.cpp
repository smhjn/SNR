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
std::string typeName("float");


int main(int argc, char * argv[]) {
  bool dSNR = false;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
  unsigned int threadUnit = 0;
  unsigned int threadInc = 0;
	unsigned int minThreads = 0;
	unsigned int maxThreadsPerBlock = 0;
	unsigned int maxItemsPerThread = 0;
	unsigned int maxColumns = 0;
	unsigned int maxRows = 0;
  AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);

    dSNR = args.getSwitch("-dedispersed");
    bool fSNR = args.getSwitch("-folded");
    if ( (dSNR && fSNR) || (!dSNR && ! fSNR) ) {
      throw std::exception();
    }
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    threadUnit = args.getSwitchArgument< unsigned int >("-thread_unit");
    threadInc = args.getSwitchArgument< unsigned int >("-thread_inc");
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreadsPerBlock = args.getSwitchArgument< unsigned int >("-max_threads");
		maxItemsPerThread = args.getSwitchArgument< unsigned int >("-max_items");
		maxColumns = args.getSwitchArgument< unsigned int >("-max_columns");
		maxRows = args.getSwitchArgument< unsigned int >("-max_rows");
    if ( dSNR ) {
      observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    }
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), 0, 0);
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch ( isa::utils::EmptyCommandLine & err ) {
		std::cerr << argv[0] << " [-dedispersed | -folded] -iterations ... -opencl_platform ... -opencl_device ... -padding ... -thread_unit ... -thread_inc ... -min_threads ... -max_threads ... -max_items ... -max_columns ... -max_rows ... -dms ..." << std::endl;
    std::cerr << "\t -dedispersed -samples ..." << std::endl;
    std::cerr << "\t -folded -pb ... -pt ... -periods .... -bins ..." << std::endl;
		return 1;
	} catch ( std::exception & err ) {
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
  std::vector< dataType > foldedData, snrs;
  std::vector< dataType > transposedData, maxS, meanS, rmsS;
  cl::Buffer foldedData_d, snrs_d;
  cl::Buffer tranposedData_d, maxS_d, meanS_d, rmsS_d;
  if ( dSNR ) {
    foldedData.resize(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedBins());
    snrs.resize(observation.getNrPeriods() * observation.getNrPaddedDMs());
  } else {
    tranposedData.resize(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
    maxS.resize(observation.getNrPaddedDMs());
    meanS.resize(observation.getNrPaddedDMs());
    rmsS.resize(observation.getNrPaddedDMs());
  }
  try {
    if ( dSNR ) {
      transposedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, transposedData.size() * sizeof(dataType), 0, 0);
      maxS_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, maxS.size() * sizeof(dataType), 0, 0);
      meanS_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, meanS.size() * sizeof(dataType), 0, 0);
      rmsS_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, rmsS.size() * sizeof(dataType), 0, 0);
    } else {
      foldedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, foldedData.size() * sizeof(dataType), 0, 0);
      snrs_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrs.size() * sizeof(dataType), 0, 0);
    }
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(0));
  if ( dSNR ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
        transposedData[(sample * observation.getNrPaddedDMs()) + dm] = static_cast< dataType >(rand() % 10);
      }
    }
    std::fill(maxS.begin(), maxS.end(), static_cast< dataType >(0));
    std::fill(meanS.begin(), meanS.end(), static_cast< dataType >(0));
    std::fill(rmsS.begin(), rmsS.end(), static_cast< dataType >(0));
  } else {
    for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
      for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
        for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
          foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM] = static_cast< dataType >(rand() % 10);
        }
      }
    }
    std::fill(snrs.begin(), snrs.end(), static_cast< dataType >(0));
  }

  // Copy data structures to device
  try {
    if ( dSNR ) {
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(transposedData_d, CL_FALSE, 0, transposedData.size() * sizeof(dataType), reinterpret_cast< void * >(transposedData.data()));
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(maxS_d, CL_FALSE, 0, maxS.size() * sizeof(dataType), reinterpret_cast< void * >(maxS.data()));
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(meanS_d, CL_FALSE, 0, meanS.size() * sizeof(dataType), reinterpret_cast< void * >(meanS.data()));
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(rmsS_d, CL_FALSE, 0, rmsS.size() * sizeof(dataType), reinterpret_cast< void * >(rmsS.data()));
    } else {
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(foldedData_d, CL_FALSE, 0, foldedData.size() * sizeof(dataType), reinterpret_cast< void * >(foldedData.data()));
      clQueues->at(clDeviceID)[0].enqueueWriteBuffer(snrs_d, CL_FALSE, 0, snrs.size() * sizeof(dataType), reinterpret_cast< void * >(snrs.data()));
    }
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	// Find the parameters
	std::vector< unsigned int > DMsPerBlock;
	for ( unsigned int DMs = minThreads; DMs <= maxColumns; DMs += threadInc ) {
		if ( (observation.getNrPaddedDMs() % DMs) == 0 ) {
			DMsPerBlock.push_back(DMs);
		}
	}
	std::vector< unsigned int > periodsPerBlock;
  if ( !dSNR ) {
    for ( unsigned int periods = 1; periods <= maxRows; periods++ ) {
      if ( (observation.getNrPeriods() % periods) == 0 ) {
        periodsPerBlock.push_back(periods);
      }
    }
  }

	std::cout << std::fixed << std::endl;
  if ( dSNR ) {
    std::cout << "# nrDMs nrSamples DMsPerBlock DMsPerThread GFLOP/s GB/s time stdDeviation COV" << std::endl << std::endl;
  } else {
    std::cout << "# nrDMs nrPeriods nrBins DMsPerBlock periodsPerBlock DMsPerThread periodsPerThread GFLOP/s GB/s time stdDeviation COV" << std::endl << std::endl;
  }

  for ( std::vector< unsigned int >::iterator DMs = DMsPerBlock.begin(); DMs != DMsPerBlock.end(); ++DMs ) {
    if ( dSNR ) {
      if ( *DMs % threadUnit != 0 ) {
        continue;
      }
      for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
        if ( observation.getNrPaddedDMs() % (*DMs * DMsPerThread) != 0 ) {
          continue;
        }
        if ( (4 * DMsPerThread) > maxItemsPerThread ) {
          break;
        }

        // Generate kernel
        double flops = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrSamplesPerSecond() * 3) + (static_cast< long long unsigned int >(observation.getNrDMs()) * 12));
        double gbs = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrSamplesPerSecond() * sizeof(dataType)) + (static_cast< long long unsigned int >(observation.getNrDMs()) * 3 * sizeof(dataType)));
        cl::Event event;
        cl::Kernel * kernel;
        isa::utils::Timer timer;
        std::string * code = PulsarSearch::getSNRDedispersedOpenCL(*DMs, DMsPerThread, typeName, observation);

        try {
          kernel = isa::OpenCL::compile("snrDedispersed", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
        } catch ( isa::OpenCL::OpenCLError & err ) {
          std::cerr << err.what() << std::endl;
          continue;
        }
        delete code;

        cl::NDRange global(observation.getNrPaddedDMs() / DMsPerThread);
        cl::NDRange local(*DMs);

        kernel->setArg(0, 0);
        kernel->setArg(1, transposedData_d);
        kernel->setArg(2, maxS_d);
        kernel->setArg(3, meanS_d);
        kernel->setArg(4, rmsS_d);

        // Warm-up run
        try {
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
          event.wait();
        } catch ( cl::Error & err ) {
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
          }
        } catch ( cl::Error & err ) {
          std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
          continue;
        }

        std::cout << observation.getNrDMs() << " " << observation.getNrSamplesPerSecond() << " ";
        std::cout << *DMs << " " << DMsPerThread << " ";
        std::cout << std::setprecision(3);
        std::cout << flops / timer.getAverageTime() << " " << gbs / timer.getAverageTime() << " ";
        std::cout << std::setprecision(6);
        std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
      }
    } else {
      for ( std::vector< unsigned int >::iterator periods = periodsPerBlock.begin(); periods != periodsPerBlock.end(); ++periods ) {
        if ( (*DMs * *periods) > maxThreadsPerBlock ) {
          break;
        } else if ( (*DMs * *periods) % threadUnit != 0 ) {
          continue;
        }
        for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
          if ( observation.getNrPaddedDMs() % (*DMs * DMsPerThread) != 0 ) {
            continue;
          }
          for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsPerThread; periodsPerThread++ ) {
            if ( observation.getNrPeriods() % (*periods * periodsPerThread) != 0 ) {
              continue;
            }
            if ( DMsPerThread + periodsPerThread + (3 * DMsPerThread * periodsPerThread) > maxItemsPerThread ) {
              break;
            }

            // Generate kernel
            double flops = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins() * 3) + (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * 4));
            double gbs = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins() * sizeof(dataType)) + (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * sizeof(dataType)));
            cl::Event event;
            cl::Kernel * kernel;
            isa::utils::Timer timer;
            std::string * code = PulsarSearch::getSNRFoldedOpenCL(*DMs, *periods, DMsPerThread, periodsPerThread, typeName, observation);

            try {
              kernel = isa::OpenCL::compile("snrFolded", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
            } catch ( isa::OpenCL::OpenCLError & err ) {
              std::cerr << err.what() << std::endl;
              continue;
            }
            delete code;

            cl::NDRange global(observation.getNrPaddedDMs() / DMsPerThread, observation.getNrPeriods() / periodsPerThread);
            cl::NDRange local(*DMs, *periods);

            kernel->setArg(0, foldedData_d);
            kernel->setArg(1, snrs_d);

            // Warm-up run
            try {
              clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
              event.wait();
            } catch ( cl::Error & err ) {
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
              }
            } catch ( cl::Error & err ) {
              std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
              continue;
            }

            std::cout << observation.getNrDMs() << " " << observation.getNrPeriods() << " " << observation.getNrBins() << " ";
            std::cout << *DMs << " " << *periods << " " << DMsPerThread << " " << periodsPerThread << " ";
            std::cout << std::setprecision(3);
            std::cout << flops / timer.getAverageTime() << " " << gbs / timer.getAverageTime() << " ";
            std::cout << std::setprecision(6);
            std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
          }
        }
      }
    }
  }

	std::cout << std::endl;

	return 0;
}

