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


void initializeDeviceMemoryD(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< dataType > * transposedData, cl::Buffer * transposedData_d, std::vector< dataType > * maxS, cl::Buffer * maxS_d, std::vector< dataType > * meanS, cl::Buffer * meanS_d, std::vector< dataType > * rmsS, cl::Buffer * rmsS_d);
void initializeDeviceMemoryF(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< dataType > * foldedData, cl::Buffer * foldedData_d, std::vector< dataType > * snrs, cl::Buffer * snrs_d);

int main(int argc, char * argv[]) {
  bool reInit = true;
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
  PulsarSearch::snrDedispersedConf dConf;
  PulsarSearch::snrFoldedConf fConf;
  cl::Event event;

	try {
    isa::utils::ArgumentList args(argc, argv);

    dSNR = args.getSwitch("-dedispersed");
    bool fSNR = args.getSwitch("-folded");
    if ( (dSNR && fSNR) || (!dSNR && ! fSNR) ) {
      throw isa::utils::EmptyCommandLine();
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
    } else {
      observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), 0, 0);
      observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
    }
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
	} catch ( isa::utils::EmptyCommandLine & err ) {
		std::cerr << argv[0] << " [-dedispersed | -folded] -iterations ... -opencl_platform ... -opencl_device ... -padding ... -thread_unit ... -thread_inc ... -min_threads ... -max_threads ... -max_items ... -max_columns ... -max_rows ... -dms ..." << std::endl;
    std::cerr << "\t -dedispersed -samples ..." << std::endl;
    std::cerr << "\t -folded -periods .... -bins ..." << std::endl;
		return 1;
	} catch ( std::exception & err ) {
		std::cerr << err.what() << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context clContext;
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = 0;

	// Allocate memory
  std::vector< dataType > transposedData, maxS, meanS, rmsS;
  std::vector< dataType > foldedData, snrs;
  cl::Buffer transposedData_d, maxS_d, meanS_d, rmsS_d;
  cl::Buffer foldedData_d, snrs_d;
  if ( dSNR ) {
    transposedData.resize(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
    maxS.resize(observation.getNrPaddedDMs());
    meanS.resize(observation.getNrPaddedDMs());
    rmsS.resize(observation.getNrPaddedDMs());
  } else {
    foldedData.resize(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
    snrs.resize(observation.getNrPeriods() * observation.getNrPaddedDMs());
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

	// Find the parameters
	std::vector< unsigned int > DMsPerBlock;
	for ( unsigned int DMs = minThreads; DMs <= maxColumns; DMs += threadInc ) {
		if ( (observation.getNrPaddedDMs() % DMs) == 0 || (observation.getNrDMs() % DMs) == 0 ) {
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
      dConf.setNrDMsPerBlock(*DMs);
      for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
        if ( observation.getNrPaddedDMs() % (dConf.getNrDMsPerBlock() * DMsPerThread) != 0 ) {
          continue;
        }
        dConf.setNrDMsPerThread(DMsPerThread);
        if ( (4 * dConf.getNrDMsPerThread()) > maxItemsPerThread ) {
          break;
        }

        // Generate kernel
        unsigned int nrThreads = 0;
        double flops = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrSamplesPerSecond() * 3) + (static_cast< long long unsigned int >(observation.getNrDMs()) * 12));
        double gbs = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrSamplesPerSecond() * sizeof(dataType)) + (static_cast< long long unsigned int >(observation.getNrDMs()) * 3 * sizeof(dataType)));
        cl::Kernel * kernel;
        isa::utils::Timer timer;
        std::string * code = PulsarSearch::getSNRDedispersedOpenCL(dConf, typeName, observation);

        if ( reInit ) {
          delete clQueues;
          clQueues = new std::vector< std::vector< cl::CommandQueue > >();
          isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
          try {
            initializeDeviceMemoryD(clContext, &(clQueues->at(clDeviceID)[0]), &foldedData, &foldedData_d, &maxS, &maxS_d, &meanS, &meanS_d, &rmsS, &rmsS_d);
          } catch ( cl::Error & err ) {
            return -1;
          }
          reInit = false;
        }
        try {
          kernel = isa::OpenCL::compile("snrDedispersed", *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
        } catch ( isa::OpenCL::OpenCLError & err ) {
          std::cerr << err.what() << std::endl;
          delete code;
          break;
        }
        delete code;

        if ( (observation.getNrDMs() % (dConf.getNrDMsPerBlock() * dConf.getNrDMsPerThread())) == 0 ) {
          nrThreads = observation.getNrDMs() / dConf.getNrDMsPerThread();
        } else {
          nrThreads = observation.getNrPaddedDMs() / dConf.getNrDMsPerThread();
        }
        cl::NDRange global = cl::NDRange(nrThreads);
        cl::NDRange local = cl::NDRange(dConf.getNrDMsPerBlock());

        kernel->setArg(0, 0);
        kernel->setArg(1, transposedData_d);
        kernel->setArg(2, maxS_d);
        kernel->setArg(3, meanS_d);
        kernel->setArg(4, rmsS_d);

        try {
          // Warm-up run
          clQueues->at(clDeviceID)[0].finish();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
          event.wait();
          // Tuning runs
          for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
            timer.start();
            clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
            event.wait();
            timer.stop();
          }
        } catch ( cl::Error & err ) {
          std::cerr << "OpenCL error kernel execution (";
          std::cerr << dConf.print();
          std::cerr << "): " << isa::utils::toString(err.err()) << "." << std::endl;
          delete kernel;
          if ( err.err() == -4 || err.err() == -61 ) {
            return -1;
          }
          reInit = true;
          break;
        }
        delete kernel;

        std::cout << observation.getNrDMs() << " " << observation.getNrSamplesPerSecond() << " ";
        std::cout << dConf.print() << " ";
        std::cout << std::setprecision(3);
        std::cout << flops / timer.getAverageTime() << " " << gbs / timer.getAverageTime() << " ";
        std::cout << std::setprecision(6);
        std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
      }
    } else {
      fConf.setNrDMsPerBlock(*DMs);
      for ( std::vector< unsigned int >::iterator periods = periodsPerBlock.begin(); periods != periodsPerBlock.end(); ++periods ) {
        if ( (fConf.getNrDMsPerBlock() * *periods) > maxThreadsPerBlock ) {
          break;
        } else if ( (fConf.getNrDMsPerBlock() * *periods) % threadUnit != 0 ) {
          continue;
        }
        fConf.setNrPeriodsPerBlock(*periods);
        for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
          if ( observation.getNrPaddedDMs() % (fConf.getNrDMsPerBlock() * DMsPerThread) != 0 && observation.getNrDMs() % (fConf.getNrDMsPerBlock() * DMsPerThread) != 0 ) {
            continue;
          }
          fConf.setNrDMsPerThread(DMsPerThread);
          for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsPerThread; periodsPerThread++ ) {
            if ( observation.getNrPeriods() % (fConf.getNrPeriodsPerBlock() * periodsPerThread) != 0 ) {
              continue;
            }
            if ( fConf.getNrDMsPerThread() + periodsPerThread + (3 * fConf.getNrDMsPerThread() * periodsPerThread) > maxItemsPerThread ) {
              break;
            }
            fConf.setNrPeriodsPerThread(periodsPerThread);

            // Generate kernel
            unsigned int nrThreads;
            double flops = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins() * 3) + (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * 4));
            double gbs = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins() * sizeof(dataType)) + (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * sizeof(dataType)));
            cl::Kernel * kernel;
            isa::utils::Timer timer;
            std::string * code = PulsarSearch::getSNRFoldedOpenCL(fConf, typeName, observation);

            if ( reInit ) {
              delete clQueues;
              clQueues = new std::vector< std::vector< cl::CommandQueue > >();
              isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
              try {
                initializeDeviceMemoryF(clContext, &(clQueues->at(clDeviceID)[0]), &transposedData, &transposedData_d, &snrs, &snrs_d);
              } catch ( cl::Error & err ) {
                return -1;
              }
              reInit = false;
            }
            try {
              kernel = isa::OpenCL::compile("snrFolded", *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
            } catch ( isa::OpenCL::OpenCLError & err ) {
              std::cerr << err.what() << std::endl;
              delete code;
              break;
            }
            delete code;

            if ( (observation.getNrDMs() % (fConf.getNrDMsPerBlock() * fConf.getNrDMsPerThread())) == 0 ) {
              nrThreads = observation.getNrDMs() / fConf.getNrDMsPerThread();
            } else {
              nrThreads = observation.getNrPaddedDMs() / fConf.getNrDMsPerThread();
            }
            cl::NDRange global = cl::NDRange(nrThreads, observation.getNrPeriods() / fConf.getNrPeriodsPerThread());
            cl::NDRange local = cl::NDRange(fConf.getNrDMsPerBlock(), fConf.getNrPeriodsPerBlock());

            kernel->setArg(0, foldedData_d);
            kernel->setArg(1, snrs_d);

            try {
              // Warm-up run
              clQueues->at(clDeviceID)[0].finish();
              clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
              event.wait();
              // Tuning runs
              for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
                timer.start();
                clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
                event.wait();
                timer.stop();
              }
            } catch ( cl::Error & err ) {
              std::cerr << "OpenCL error kernel execution (";
              std::cerr << fConf.print();
              std::cerr << "): " << isa::utils::toString(err.err()) << "." << std::endl;
              delete kernel;
              if ( err.err() == -4 || err.err() == -61 ) {
                return -1;
              }
              reInit = true;
              break;
            }
            delete kernel;

            std::cout << observation.getNrDMs() << " " << observation.getNrPeriods() << " " << observation.getNrBins() << " ";
            std::cout << fConf.print() << " ";
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

void initializeDeviceMemoryD(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< dataType > * transposedData, cl::Buffer * transposedData_d, std::vector< dataType > * maxS, cl::Buffer * maxS_d, std::vector< dataType > * meanS, cl::Buffer * meanS_d, std::vector< dataType > * rmsS, cl::Buffer * rmsS_d) {
  try {
    *transposedData_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, transposedData->size() * sizeof(dataType), 0, 0);
    *maxS_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, maxS->size() * sizeof(dataType), 0, 0);
    *meanS_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, meanS->size() * sizeof(dataType), 0, 0);
    *rmsS_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, rmsS->size() * sizeof(dataType), 0, 0);
    clQueue->enqueueWriteBuffer(*transposedData_d, CL_FALSE, 0, transposedData->size() * sizeof(dataType), reinterpret_cast< void * >(transposedData->data()));
    clQueue->enqueueWriteBuffer(*maxS_d, CL_FALSE, 0, maxS->size() * sizeof(dataType), reinterpret_cast< void * >(maxS->data()));
    clQueue->enqueueWriteBuffer(*meanS_d, CL_FALSE, 0, meanS->size() * sizeof(dataType), reinterpret_cast< void * >(meanS->data()));
    clQueue->enqueueWriteBuffer(*rmsS_d, CL_FALSE, 0, rmsS->size() * sizeof(dataType), reinterpret_cast< void * >(rmsS->data()));
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    throw;
  }
}

void initializeDeviceMemoryF(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< dataType > * foldedData, cl::Buffer * foldedData_d, std::vector< dataType > * snrs, cl::Buffer * snrs_d) {
  try {
    *foldedData_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, foldedData->size() * sizeof(dataType), 0, 0);
    *snrs_d = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, snrs->size() * sizeof(dataType), 0, 0);
    clQueue->enqueueWriteBuffer(*foldedData_d, CL_FALSE, 0, foldedData->size() * sizeof(dataType), reinterpret_cast< void * >(foldedData->data()));
    clQueue->enqueueWriteBuffer(*snrs_d, CL_FALSE, 0, snrs->size() * sizeof(dataType), reinterpret_cast< void * >(snrs->data()));
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    throw;
  }
}

