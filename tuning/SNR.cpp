/*
 * Copyright (C) 2013
 * Alessio Sclocco <a.sclocco@vu.nl>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cmath>
using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::vector;
using std::exception;
using std::ofstream;
using std::fixed;
using std::setprecision;
using std::numeric_limits;

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <CLData.hpp>
#include <utils.hpp>
#include <SNR.hpp>
#include <Timer.hpp>
using isa::utils::ArgumentList;
using isa::utils::toStringValue;
using isa::utils::Timer;
using AstroData::Observation;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::CLData;
using PulsarSearch::SNR;

typedef float dataType;
const string typeName("float");
const unsigned int maxThreadsPerBlock = 1024;
const unsigned int maxThreadMultiplier = 32;
const unsigned int padding = 32;

// Periods
const unsigned int nrBins = 128;


int main(int argc, char * argv[]) {
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	Observation< dataType > observation("SNRTuning", typeName);
	CLData< dataType > * foldedData = new CLData< dataType >("FoldedData", true);
	CLData< dataType > * SNRData = new CLData< dataType >("SNRData", true);


	try {
		ArgumentList args(argc, argv);

		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
	} catch ( exception & err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// Setup of the observation
	observation.setPadding(padding);
	observation.setNrBins(nrBins);
	
	cl::Context * clContext = new cl::Context();
	vector< cl::Platform > * clPlatforms = new vector< cl::Platform >();
	vector< cl::Device > * clDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * clQueues = new vector< vector < cl::CommandQueue > >();
	
	initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);
	
	cout << fixed << endl;
	cout << "# nrDMs nrPeriods nrBins nrDMsPerBlock nrPeriodsPerBlock GFLOP/s err time err GB/s err " << endl << endl;
	
	for ( unsigned int nrDMs = 32; nrDMs <= 4096; nrDMs *= 2 )	{
		observation.setNrDMs(nrDMs);
		
		for ( unsigned int nrPeriods = 2; nrPeriods <= 1024; nrPeriods *= 2 ) {
			observation.setNrPeriods(nrPeriods);

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
			for ( unsigned int DMs = 2; DMs <= maxThreadsPerBlock; DMs++ ) {
				if ( (observation.getNrDMs() % DMs) == 0 ) {
					DMsPerBlock.push_back(DMs);
				}
			}
			vector< unsigned int > periodsPerBlock;
			for ( unsigned int periods = 2; periods <= maxThreadMultiplier; periods++ ) {
				if ( (observation.getNrPeriods() % periods) == 0 ) {
					periodsPerBlock.push_back(periods);
				}
			}

			for ( vector< unsigned int >::iterator DMs = DMsPerBlock.begin(); DMs != DMsPerBlock.end(); DMs++ ) {
				for ( vector< unsigned int >::iterator periods = periodsPerBlock.begin(); periods != periodsPerBlock.end(); periods++ ) {
					if ( (*DMs * *periods) > maxThreadsPerBlock ) {
						break;
					}
					double Acur[2] = {0.0, 0.0};
					double Aold[2] = {0.0, 0.0};
					double Vcur[2] = {0.0, 0.0};
					double Vold[2] = {0.0, 0.0};

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
							
							if ( iteration == 0 ) {
								Acur[0] = clSNR.getGFLOP() / clSNR.getTimer().getLastRunTime();
								Acur[1] = clSNR.getGB() / clSNR.getTimer().getLastRunTime();
							} else {
								Aold[0] = Acur[0];
								Vold[0] = Vcur[0];

								Acur[0] = Aold[0] + (((clSNR.getGFLOP() / clSNR.getTimer().getLastRunTime()) - Aold[0]) / (iteration + 1));
								Vcur[0] = Vold[0] + (((clSNR.getGFLOP() / clSNR.getTimer().getLastRunTime()) - Aold[0]) * ((clSNR.getGFLOP() / clSNR.getTimer().getLastRunTime()) - Acur[0]));

								Aold[1] = Acur[1];
								Vold[1] = Vcur[1];

								Acur[1] = Aold[1] + (((clSNR.getGB() / clSNR.getTimer().getLastRunTime()) - Aold[1]) / (iteration + 1));
								Vcur[1] = Vold[1] + (((clSNR.getGB() / clSNR.getTimer().getLastRunTime()) - Aold[1]) * ((clSNR.getGB() / clSNR.getTimer().getLastRunTime()) - Acur[1]));
							}
						}
						Vcur[0] = sqrt(Vcur[0] / nrIterations);
						Vcur[1] = sqrt(Vcur[1] / nrIterations);

						cout << nrDMs << " " << nrPeriods << " " << observation.getNrBins() << " " << *DMs << " " << *periods << " " << setprecision(3) << Acur[0] << " " << Vcur[0] << " " << setprecision(6) << clSNR.getTimer().getAverageTime() << " " << clSNR.getTimer().getStdDev() << " " << setprecision(3) << Acur[1] << " " << Vcur[1] << endl;
					} catch ( OpenCLError err ) {
						cerr << err.what() << endl;
						continue;
					}
				}
			}
		}
		cout << endl << endl;
	}

	cout << endl;

	return 0;
}
