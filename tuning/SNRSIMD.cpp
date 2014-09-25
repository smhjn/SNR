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
#include <SNR.hpp>
#include <utils.hpp>
#include <Exceptions.hpp>
#include <Timer.hpp>
#include <Stats.hpp>


int main(int argc, char * argv[]) {
  bool avx = false;
  bool phi = false;
	unsigned int nrIterations = 0;
	unsigned int maxItemsPerThread = 0;
  AstroData::Observation observation;

	try {
    isa::utils::ArgumentList args(argc, argv);

    avx = args.getSwitch("=avx");
    phi = args.getSwitch("-phi");
    if ( !(avx || phi) ) {
      throw isa::Exceptions::EmptyCommandLine();
    }
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
		maxItemsPerThread = args.getSwitchArgument< unsigned int >("-max_items");
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
    observation.setPeriodRange(args.getSwitchArgument< unsigned int >("-periods"), 0, 0);
    observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch ( isa::Exceptions::EmptyCommandLine &err ) {
		std::cerr << argv[0] << " [-avx] [-phi] -iterations ... -padding ... -max_items ... -dms ... -periods ... -bins ... " << std::endl;
		return 1;
	} catch ( std::exception &err ) {
		std::cerr << err.what() << std::endl;
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

	std::cout << std::fixed << std::endl;
	std::cout << "# nrDMs nrPeriods nrBins DMsPerThread periodsPerThread GFLOP/s err time err" << std::endl << std::endl;

  for ( unsigned int DMsPerThread = 1; DMsPerThread <= maxItemsPerThread; DMsPerThread++ ) {
    if ( avx ){
      if ( observation.getNrPaddedDMs() % (DMsPerThread * 8) != 0 ) {
        continue;
      }
    } else ( phi ) {
      if ( observation.getNrPaddedDMs() % (DMsPerThread * 16) != 0 ) {
        continue;
      }
    }
    for ( unsigned int periodsPerThread = 1; periodsPerThread <= maxItemsPerThread; periodsPerThread++ ) {
      if ( observation.getNrPeriods() % periodsPerThread != 0 ) {
        continue;
      }
      if ( DMsPerThread + periodsPerThread > maxItemsPerThread ) {
        break;
      }

      // Tuning runs
      double flops = isa::utils::giga(static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrPeriods() * observation.getNrBins());
      isa::utils::Timer timer;
      isa::utils::Stats< double > stats;
      PulsarSearch::snrFunc< float > snr = 0;

      if ( avx ) {
        snr = functionPointers->at("snrAVX" + isa::utils::toString< unsigned int >(DMsPerThread) + "x" + isa::utils::toString< unsigned int >(periodsPerThread));
      } else if ( phi ) {
        snr = functionPointers->at("snrPhi" + isa::utils::toString< unsigned int >(DMsPerThread) + "x" + isa::utils::toString< unsigned int >(periodsPerThread));
      }
      for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
        std::memcpy(foldedData.data(), foldedData.data(), foldedData.size() * sizeof(float));
        timer.start();
        snr(observation, foldedData.data(), snrs.data());
        timer.stop();
        stats.addElement(flops / timer.getLastRunTime());
      }

      std::cout << observation.getNrDMs() << " " << observation.getNrPeriods() << " " << observation.getNrBins() << " " << DMsPerThread << " " << periodsPerThread << " " << std::setprecision(3) << stats.getAverage() << " " << stats.getStdDev() << " " << std::setprecision(6) << timer.getAverageTime() << " " << timer.getStdDev() << std::endl;
    }
  }

	std::cout << std::endl;

	return 0;
}

