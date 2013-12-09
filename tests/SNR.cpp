//
// Copyright (C) 2012
// Alessio Sclocco <a.sclocco@vu.nl>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <exception>
using std::exception;
#include <iomanip>
using std::fixed;
using std::setprecision;
#include <limits>
using std::numeric_limits;
#include <cmath>
#include <ctime>

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <Observation.hpp>
using AstroData::Observation;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::same;
#include <SNR.hpp>
using PulsarSearch::SNR;
#include <SNRCPU.hpp>
using PulsarSearch::pulsarSNR;

typedef float dataType;
const string typeName("float");


int main(int argc, char *argv[]) {
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int nrDMsPerBlock = 0;
	unsigned int nrPeriodsPerBlock = 0;
	long long unsigned int wrongValues = 0;
	Observation< dataType > observation("SNRTest", typeName);
	CLData< dataType > foldedData("FoldedData", true);
	CLData< dataType > SNRData("SNRData", true);
	CLData< dataType > SNRDataCPU("SNRDataCPU", true);

	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");

		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
		observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-periods"));
		observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));

		nrDMsPerBlock = args.getSwitchArgument< unsigned int >("-p1");
		nrPeriodsPerBlock = args.getSwitchArgument< unsigned int >("-p2");
	} catch ( exception &err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// OpenCL Initialization
	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();
	
	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);
	
	// Allocate memory
	foldedData.allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	foldedData.setCLContext(clContext);
	foldedData.setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	SNRData.allocateHostData(observation.getNrPeriods() * observation.getNrPaddedDMs());
	SNRData.setCLContext(clContext);
	SNRData.setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	SNRDataCPU.allocateHostData(observation.getNrPeriods() * observation.getNrPaddedDMs());

	try {
		foldedData.allocateDeviceData();
		SNRData.allocateDeviceData();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	srand(time(NULL));
	for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
		for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
			for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
				foldedData.setHostDataItem((bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + dm, rand() % 100);
			}
		}
	}

	// Generate kernel
	clSNR.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
	clSNR.setObservation(&observation);
	clSNR.setNrDMsPerBlock(nrDMsPerBlock);
	clSNR.setNrPeriodsPerBlock(nrPeriodsPerBlock);
	clSNR.setPulsarPipeline();
	clSNR.generateCode();

	// SNR test
	foldedData.copyHostToDevice();
	clSNR(&foldedData, &SNRData);
	SNRData.copyDeviceToHost();
	pulsarSNR(observation, foldedData.getDeviceData(), SNRDataCPU.getDeviceData());
	for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
		for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
			if ( !same(SNRDataCPU[(period * observation.getNrPaddedDMs()) + dm], SNRData[(period * observation.getNrPaddedDMs()) + dm]) ) {
				wrongValues++;
			}
		}
	}

	cout << "Wrong values: " << wrongValues << " (" << setprecision(2) << (wrongValues * 100.0f) / (observation.getNrDMs() * observation.getNrPeriods()) << "%)" << endl;
	cout << endl;

	return 0;
}

