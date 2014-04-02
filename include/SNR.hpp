// Copyright 2012 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <string>
using std::string;

#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <utils.hpp>
using isa::utils::toStringValue;
using isa::utils::giga;
#include <Kernel.hpp>
using isa::OpenCL::Kernel;
#include <Observation.hpp>
using AstroData::Observation;


#ifndef SNR_HPP
#define SNR_HPP

namespace PulsarSearch {

// OpenCL SNR
template< typename T > class SNR : public Kernel< T > {
public:
	SNR(string name, string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(CLData< T > * input, CLData< T > * output) throw (OpenCLError);

	inline void setNrDMsPerBlock(unsigned int DMs);
	inline void setNrPeriodsPerBlock(unsigned int periods);

	inline void setTransientPipeline();
	inline void setPulsarPipeline();

	inline void setObservation(Observation< T > * obs);

private:
	cl::NDRange globalSize;
	cl::NDRange localSize;

	unsigned int nrDMsPerBlock;
	unsigned int nrPeriodsPerBlock;

	bool transientPipeline;
	bool pulsarPipeline;

	Observation< T > * observation;
};


// Implementation
template< typename T > SNR< T >::SNR(string name, string dataType) : Kernel< T >(name, dataType), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), nrDMsPerBlock(0), nrPeriodsPerBlock(0), transientPipeline(false), pulsarPipeline(false), observation(0) {}

template< typename T > void SNR< T >::generateCode() throw (OpenCLError) {
	delete this->code;
	this->code = new string();

	if ( transientPipeline ) {
		throw OpenCLError("Transient pipeline not implemented.");
	} else if ( pulsarPipeline ) {
		// Begin kernel's template
		string nrDMsPerBlock_s = toStringValue< unsigned int >(nrDMsPerBlock);
		string nrPeriodsPerBlock_s = toStringValue< unsigned int >(nrPeriodsPerBlock);
		string nrBins_s = toStringValue< unsigned int >(observation->getNrBins());
		string nrPeriods_s = toStringValue< unsigned int >(observation->getNrPeriods());
		string nrPaddedDMs_s = toStringValue< unsigned int >(observation->getNrPaddedDMs());
		string nrBinsInverse_s = toStringValue< float >(1.0f / observation->getNrBins());

		*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict foldedData, __global " + this->dataType + " * const restrict snrs) {\n"
			"const unsigned int dm = (get_group_id(0) * " + nrDMsPerBlock_s + ") + get_local_id(0);\n"
			"const unsigned int period = (get_group_id(1) * " + nrPeriodsPerBlock_s + ") + get_local_id(1);\n"
			+ this->dataType + " average = 0;\n"
			+ this->dataType + " rms = 0;\n"
			+ this->dataType + " max = 0;\n"
			"\n"
			"for ( unsigned int bin = 0; bin < " + nrBins_s + "; bin++ ) {\n"
			"const " + this->dataType + " globalItem = foldedData[(bin * " + nrPeriods_s + " * " + nrPaddedDMs_s + ") + (period * " + nrPaddedDMs_s + ") + dm];\n"
			"average += globalItem;\n"
			"rms += (globalItem * globalItem);\n"
			"max = fmax(max, globalItem);\n"
			"}\n"
			"average *= " + nrBinsInverse_s + "f;\n"
			"rms *= " + nrBinsInverse_s + "f;\n"
			"snrs[(period * " + nrPaddedDMs_s + ") + dm] = (max - average) / native_sqrt(rms);\n"
			"}\n";
		// End kernel's template

		globalSize = cl::NDRange(observation->getNrPaddedDMs(), observation->getNrPeriods());
		localSize = cl::NDRange(nrDMsPerBlock, nrPeriodsPerBlock);

		this->gflop = giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrPeriods() * ((observation->getNrBins() * 3) + 4));
		this->gb = giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrPeriods() * ((observation->getNrBins() * sizeof(T)) + sizeof(T)));
	} else {
		throw OpenCLError("Pipeline not implemented.");
	}

	this->compile();
}

template< typename T > void SNR< T >::operator()(CLData< T > * input, CLData< T > * output) throw (OpenCLError) {
	this->setArgument(0, *(input->getDeviceData()));
	this->setArgument(1, *(output->getDeviceData()));

	this->run(globalSize, localSize);
}

template< typename T > inline void SNR< T >::setNrDMsPerBlock(unsigned int DMs) {
	nrDMsPerBlock = DMs;
}

template< typename T > inline void SNR< T >::setNrPeriodsPerBlock(unsigned int periods) {
	nrPeriodsPerBlock = periods;
}

template< typename T > inline void SNR< T >::setTransientPipeline() {
	transientPipeline = true;
	pulsarPipeline = false;
}

template< typename T > inline void SNR< T >::setPulsarPipeline() {
	pulsarPipeline = true;
	transientPipeline = false;
}

template< typename T >inline void SNR< T >::setObservation(Observation< T > *obs) {
	observation = obs;
}

} // PulsarSearch

#endif // SNR_HPP
