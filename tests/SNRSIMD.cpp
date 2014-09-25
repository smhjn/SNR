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
#include <Exceptions.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Shifts.hpp>
#include <SNR.hpp>


int main(int argc, char *argv[]) {
  bool avx = false;
  bool phi = false;
  unsigned int nrDMsPerThread = 0;
  unsigned int nrPeriodsPerThread = 0;
	long long unsigned int wrongSamples = 0;
	AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);
    avx = args.getSwitch("-avx");
    phi = args.getSwitch("-phi");
    if ( ! (avx || phi) ) {
      throw isa::Exceptions::EmptyCommandLine();
    }
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    nrDMsPerThread = args.getSwitchArgument< unsigned int >("-dt");
    nrPeriodsPerThread = args.getSwitchArgument< unsigned int >("-pt");
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), 0, 0);
	} catch  ( isa::Exceptions::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-avx] [-phi] ... -padding ... -dt ... -pt ... -dms ... -periods ... -bins ..." << std::endl;
		return 1;
	}

	// Allocate memory
  std::vector< float > foldedData = std::vector< float >(observation.getNrBins() * observation.getNrPeriods() * observation.getNrPaddedDMs());
  std::vector< dataType > snrs = std::vector< dataType >(observation.getNrPeriods() * observation.getNrPaddedDMs());

	srand(time(NULL));
  for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
      for ( unsigned int DM = 0; DM < observation.getNrDMs(); DM++ ) {
        foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + DM] = static_cast< dataType >(rand() % 10);
      }
    }
	}
  std::fill(snrs.begin(), snrs.end(), static_cast< dataType >(0));

  // Generate kernel
  PulsarSearch::snrFunc< float > snr = 0;

  if ( avx ) {
    snr = functionPointers->at("snrAVX" + isa::utils::toString< unsigned int >(nrDMsPerThread) + "x" + isa::utils::toString< unsigned int >(nrPeriodsPerThread));
  } else if ( phi ) {
    snr = functionPointers->at("snrPhi" + isa::utils::toString< unsigned int >(nrDMsPerThread) + "x" + isa::utils::toString< unsigned int >(nrPeriodsPerThread));
  }

  // Run SIMD kernel and CPU control
  snr(observation, foldedData.data(), snrs.data());
  PulsarSearch::snrFoldedTS(observation, foldedData, snrs_c);

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

