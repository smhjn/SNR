// Copyright 2013 Alessio Sclocco <a.sclocco@vu.nl>
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
    unsigned int maxColumns = 0;
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
        maxColumns = args.getSwitchArgument< unsigned int >("-max_columns");
        maxRows = args.getSwitchArgument< unsigned int >("-max_rows");
        observation.setNrDMs(args.getSwitchArgument< unsigned int >("-dms"));
        observation.setNrPeriods(args.getSwitchArgument< unsigned int >("-periods"));
        observation.setNrBins(args.getSwitchArgument< unsigned int >("-bins"));
    } catch ( EmptyCommandLine err ) {
        cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -min_threads ... -max_threads ... -max_columns ... -max_rows ... -dms ... -periods ... -bins ..." << endl;
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
    for ( unsigned int DMs = minThreads; DMs <= maxColumns; DMs += minThreads ) {
        if ( (observation.getNrPaddedDMs() % DMs) == 0 ) {
            DMsPerBlock.push_back(DMs);
        }
    }
    vector< unsigned int > periodsPerBlock;
    for ( unsigned int periods = 1; periods <= maxRows; periods++ ) {
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
                clSNR.resetStats();

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
