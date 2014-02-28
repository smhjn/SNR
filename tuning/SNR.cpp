//
// Copyright (C) 2013
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
using std::fixed;
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <exception>
using std::exception;
#include <fstream>
using std::ofstream;
#include <iomanip>
using std::setprecision;
#include <limits>
using std::numeric_limits;
#include <cmath>

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <Observation.hpp>
using AstroData::Observation;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::toStringValue;
#include <SNR.hpp>
using PulsarSearch::SNR;
#include <Timer.hpp>
using isa::utils::Timer;

typedef float dataType;
const string typeName("float");


int main(int argc, char * argv[]) {
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	unsigned int minThreads = 0;
	unsigned int maxThreadsPerBlock = 0;
	unsigned int maxRows = 0;
	Observation< dataType > observation("SNRTuning", typeName);
	CLData< dataType > * foldedData = new CLData< dataType >("FoldedData", true);
	CLData< dataType > * SNRData = new CLData< dataType >("SNRData", true);


	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxThreadsPerBlock = args.getSwitchArgument< unsigned int >("-max_threads");
		maxRows = args.getSwitchArgument< unsigned int >("-max_rows");
		observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
		observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-periods"));
		observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
	} catch ( EmptyCommandLine err ) {
		cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -min_threads ... -max_threads ... -max_rows ... -dms ... -periods ... -bins ..." << endl;
		return 1;
	} catch ( exception & err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();
	
	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);
	
	cout << fixed << endl;
	cout << "# nrDMs nrPeriods nrBins nrDMsPerBlock nrPeriodsPerBlock GFLOP/s err time err GB/s err " << endl << endl;
	
	// Allocate memory
	foldedData->allocateHostData(observation.getNrPaddedDMs() * observation.getNrBins() * observation.getNrPeriods());
	SNRData->allocateHostData(observation.getNrPaddedDMs() * observation.getNrPeriods());

	foldedData->setCLContext(clContext);
	foldedData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));
	SNRData->setCLContext(clContext);
	SNRData->setCLQueue(&((clQueues->at(clDeviceID)).at(0)));

	try {
		foldedData->allocateDeviceData();
		foldedData->copyHostToDevice();
		SNRData->allocateDeviceData();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
	}

	// Find the parameters
	vector< unsigned int > DMsPerBlock;
	for ( unsigned int DMs = minThreads; DMs <= maxThreadsPerBlock; DMs += minThreads ) {
		if ( (observation.getNrPaddedDMs() % DMs) == 0 ) {
			DMsPerBlock.push_back(DMs);
		}
	}
	vector< unsigned int > periodsPerBlock;
	for ( unsigned int periods = 1; periods <= maxThreadsPerBlock; periods++ ) {
		if ( (observation.getNrPeriods() % periods) == 0 ) {
			periodsPerBlock.push_back(periods);
		}
	}

	for ( vector< unsigned int >::iterator DMs = DMsPerBlock.begin(); DMs != DMsPerBlock.end(); DMs++ ) {
		for ( vector< unsigned int >::iterator periods = periodsPerBlock.begin(); periods != periodsPerBlock.end(); periods++ ) {
			if ( (*DMs * *periods) > maxThreadsPerBlock ) {
				break;
			}
			try {
				// Generate kernel
				SNR< dataType > clSNR("clSNR", typeName);
				clSNR.bindOpenCL(clContext, &(clDevices->at(clDeviceID)), &((clQueues->at(clDeviceID)).at(0)));
				clSNR.setObservation(&observation);
				clSNR.setNrDMsPerBlock(*DMs);
				clSNR.setNrPeriodsPerBlock(*periods);
				clSNR.setPulsarPipeline();
				clSNR.generateCode();

				// Warming up
				clSNR(foldedData, SNRData);
				(clSNR.getTimer()).reset();

				// Measurements
				for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
					clSNR(foldedData, SNRData);
				}

				cout << observation.getNrDMs() << " " << observation.getNrPeriods() << " " << observation.getNrBins() << " " << *DMs << " " << *periods << " " << setprecision(3) << clSNR.getGFLOPs() << " " << clSNR.getGFLOPsErr() << " " << setprecision(6) << clSNR.getTimer().getAverageTime() << " " << clSNR.getTimer().getStdDev() << " " << setprecision(3) << clSNR.getGBs() << " " << clSNR.getGBsErr() << endl;
			} catch ( OpenCLError err ) {
				cerr << err.what() << endl;
				continue;
			}
		}
	}

	cout << endl;

	return 0;
}
