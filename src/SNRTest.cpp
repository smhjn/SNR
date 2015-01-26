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
  bool printCode = false;
  bool printResults = false;
  bool dSNR = false;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	long long unsigned int wrongSamples = 0;
	AstroData::Observation observation;
  PulsarSearch::snrDedispersedConf dConf;
  PulsarSearch::snrFoldedConf fConf;

	try {
    isa::utils::ArgumentList args(argc, argv);
    printCode = args.getSwitch("-print_code");
    printResults = args.getSwitch("-print_res");
    dSNR = args.getSwitch("-dedispersed");
    bool fSNR = args.getSwitch("-folded");
    if ( (dSNR && fSNR) || (!dSNR && ! fSNR) ) {
      throw std::exception();
    }
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    if ( fSNR ) {
      fConf.setNrDMsPerBlock(args.getSwitchArgument< unsigned int >("-db"));
      fConf.setNrDMsPerThread(args.getSwitchArgument< unsigned int >("-dt"));
      fConf.setNrPeriodsPerBlock(args.getSwitchArgument< unsigned int >("-pb"));
      fConf.setNrPeriodsPerThread(args.getSwitchArgument< unsigned int >("-pt"));
    } else {
      dConf.setNrDMsPerBlock(args.getSwitchArgument< unsigned int >("-db"));
      dConf.setNrDMsPerThread(args.getSwitchArgument< unsigned int >("-dt"));
      observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
    }
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    if ( fSNR ) {
      observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), 0, 0);
      observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
    }
	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-dedispersed | -folded] [-print_code] [-print_res] -opencl_platform ... -opencl_device ... -padding ... -db ... -dt ... -dms ..." << std::endl;
    std::cerr << "\t -dedispersed -samples ..." << std::endl;
    std::cerr << "\t -folded -pb ... -pt ... -periods .... -bins ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
  std::vector< dataType > foldedData, snrs, snrs_c;
  std::vector< dataType > transposedData, maxS, meanS, rmsS, maxS_c, meanS_c, rmsS_c;
  cl::Buffer foldedData_d, snrs_d;
  cl::Buffer transposedData_d, maxS_d, meanS_d, rmsS_d;
  if ( dSNR ) {
    transposedData.resize(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs());
    maxS.resize(observation.getNrPaddedDMs());
    meanS.resize(observation.getNrPaddedDMs());
    rmsS.resize(observation.getNrPaddedDMs());
    maxS_c.resize(observation.getNrPaddedDMs());
    meanS_c.resize(observation.getNrPaddedDMs());
    rmsS_c.resize(observation.getNrPaddedDMs());
  } else {
    foldedData.resize(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
    snrs.resize(observation.getNrPeriods() * observation.getNrPaddedDMs());
    snrs_c.resize(observation.getNrPeriods() * observation.getNrPaddedDMs());
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
  } catch ( cl::Error &err ) {
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
    std::fill(maxS_c.begin(), maxS_c.end(), static_cast< dataType >(0));
    std::fill(meanS_c.begin(), meanS_c.end(), static_cast< dataType >(0));
    std::fill(rmsS_c.begin(), rmsS_c.end(), static_cast< dataType >(0));
  } else {
    for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
      for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
        for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
          foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM] = static_cast< dataType >(rand() % 10);
        }
      }
    }
    std::fill(snrs.begin(), snrs.end(), static_cast< dataType >(0));
    std::fill(snrs_c.begin(), snrs_c.end(), static_cast< dataType >(0));
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
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  cl::Kernel * kernel;
  std::string * code;
  if ( dSNR ) {
    code = PulsarSearch::getSNRDedispersedOpenCL(dConf, typeName, observation);
  } else {
    code = PulsarSearch::getSNRFoldedOpenCL(fConf, typeName, observation);
  }
  if ( printCode ) {
    std::cout << *code << std::endl;
  }

  try {
    if ( dSNR ) {
      kernel = isa::OpenCL::compile("snrDedispersed", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    } else {
      kernel = isa::OpenCL::compile("snrFolded", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    }
  } catch ( isa::OpenCL::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  try {
    unsigned int nrThreads = 0;
    cl::NDRange global;
    cl::NDRange local;

    if ( dSNR ) {
      if ( (observation.getNrDMs() % (dConf.getNrDMsPerBlock() * dConf.getNrDMsPerThread())) == 0 ) {
        nrThreads = observation.getNrDMs() / dConf.getNrDMsPerThread();
      } else {
        nrThreads = observation.getNrPaddedDMs() / dConf.getNrDMsPerThread();
      }
      global = cl::NDRange(nrThreads);
      local = cl::NDRange(dConf.getNrDMsPerBlock());

      kernel->setArg(0, 0);
      kernel->setArg(1, transposedData_d);
      kernel->setArg(2, maxS_d);
      kernel->setArg(3, meanS_d);
      kernel->setArg(4, rmsS_d);
    } else {
      if ( (observation.getNrDMs() % (fConf.getNrDMsPerBlock() * fConf.getNrDMsPerThread())) == 0 ) {
        nrThreads = observation.getNrDMs() / fConf.getNrDMsPerThread();
      } else {
        nrThreads = observation.getNrPaddedDMs() / fConf.getNrDMsPerThread();
      }
      global = cl::NDRange(nrThreads, observation.getNrPeriods() / fConf.getNrPeriodsPerThread());
      local = cl::NDRange(fConf.getNrDMsPerBlock(), fConf.getNrPeriodsPerBlock());

      kernel->setArg(0, foldedData_d);
      kernel->setArg(1, snrs_d);
    }
    
    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, 0);
    if ( dSNR ) {
      PulsarSearch::snrDedispersedTS(0, observation, transposedData, maxS_c, meanS_c, rmsS_c);
      clQueues->at(clDeviceID)[0].enqueueReadBuffer(maxS_d, CL_TRUE, 0, maxS.size() * sizeof(dataType), reinterpret_cast< void * >(maxS.data()));
      clQueues->at(clDeviceID)[0].enqueueReadBuffer(meanS_d, CL_TRUE, 0, meanS.size() * sizeof(dataType), reinterpret_cast< void * >(meanS.data()));
      clQueues->at(clDeviceID)[0].enqueueReadBuffer(rmsS_d, CL_TRUE, 0, rmsS.size() * sizeof(dataType), reinterpret_cast< void * >(rmsS.data()));
    } else {
      PulsarSearch::snrFoldedTS(observation, foldedData, snrs_c);
      clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrs_d, CL_TRUE, 0, snrs.size() * sizeof(dataType), reinterpret_cast< void * >(snrs.data()));
    }
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  if ( dSNR) {
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
      dataType snr = (maxS[dm] - meanS[dm]) / std::sqrt(rmsS[dm]);
      dataType snr_c = (maxS_c[dm] - meanS_c[dm]) / std::sqrt(rmsS_c[dm]);
      if ( ! isa::utils::same(snr, snr_c) ) {
        wrongSamples++;
        if ( printResults ) {
          std::cout << "**" << snr << " != " << snr_c << "** ";
        }
      } else if (printResults ) {
        std::cout << snr << " ";
      }
    }
    if ( printResults ) {
      std::cout << std::endl;
    }
  } else {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
        if ( ! isa::utils::same(snrs_c[(period * observation.getNrPaddedDMs()) + dm], snrs[(period * observation.getNrPaddedDMs()) + dm]) ) {
          wrongSamples++;
          if ( printResults ) {
            std::cout << "**" << snrs[(period * observation.getNrPaddedDMs()) + dm] << " != " << snrs_c[(period * observation.getNrPaddedDMs()) + dm] << "** ";
          }
        } else if ( printResults ) {
          std::cout << snrs[(period * observation.getNrPaddedDMs()) + dm] << "  ";
        }
      }
      if ( printResults ) {
        std::cout << std::endl;
      }
    }
  }

  if ( wrongSamples > 0 ) {
    if ( dSNR ) {
      std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast< long long unsigned int >(observation.getNrDMs()) << "%)." << std::endl;
    } else {
      std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / (static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods()) << "%)." << std::endl;
    }
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

