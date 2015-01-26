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
#include <utils.hpp>
#include <SNR.hpp>


int main(int argc, char *argv[]) {
  bool dSNR = false;
  std::string typeName;
	AstroData::Observation observation;
  PulsarSearch::snrDedispersedConf dConf;
  PulsarSearch::snrFoldedConf fConf;

	try {
    isa::utils::ArgumentList args(argc, argv);
    dSNR = args.getSwitch("-dedispersed");
    bool fSNR = args.getSwitch("-folded");
    if ( (dSNR && fSNR) || (!dSNR && ! fSNR) ) {
      throw std::exception();
    }
    typeName = args.getSwitchArgument< std::string >("-type");
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
    std::cerr << "Usage: " << argv[0] << " [-dedispersed | -folded] -type ... -padding ... -db ... -dt ... -dms ..." << std::endl;
    std::cerr << "\t -dedispersed -samples ..." << std::endl;
    std::cerr << "\t -folded -pb ... -pt ... -periods .... -bins ..." << std::endl;
		return 1;
	}

  // Generate kernel
  std::string * code;
  if ( dSNR ) {
    code = PulsarSearch::getSNRDedispersedOpenCL(dConf, typeName, observation);
  } else {
    code = PulsarSearch::getSNRFoldedOpenCL(fConf, typeName, observation);
  }
  std::cout << *code << std::endl;

	return 0;
}

