/*
 * Copyright (C) 2012
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

#define __CL_set_EXCEPTIONS
#include <CL/cl.hpp>
#include <string>
#include <cmath>
using std::string;
using std::sqrt;

#include <Exceptions.hpp>
#include <CLData.hpp>
#include <utils.hpp>
#include <Kernel.hpp>
#include <Observation.hpp>
using isa::Exceptions::OpenCLError;
using isa::OpenCL::CLData;
using isa::utils::toStringValue;
using isa::utils::giga;
using isa::OpenCL::Kernel;
using AstroData::Observation;

#ifndef SNR_HPP
#define SNR_HPP

namespace TDM {

// OpenCL SNR
template< typename T > class SNR : public Kernel< T > {
public:
	SNR(string name, string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(CLData< T > * input, CLData< T > * output) throw (OpenCLError);

	inline void setNrPeriodsPerBlock(unsigned int periodsPerBlock);
	
	inline void setTransientPipeline();
	inline void setPulsarPipeline();

	inline void setObservation(Observation< T > * obs);
	
private:
	cl::NDRange globalSize;
	cl::NDRange localSize;

	unsigned int nrPeriodsPerBlock;

	bool transientPipeline;
	bool pulsarPipeline;

	Observation< T > * observation;
};


// Implementation
template< typename T > SNR< T >::SNR(string name, string dataType) : Kernel< T >(name, dataType), globalSize(cl::NDRange(1, 1, 1)), localSize(cl::NDRange(1, 1, 1)), nrPeriodsPerBlock(0), transientPipeline(false), pulsarPipeline(false), observation(0) {}

template< typename T > void SNR< T >::generateCode() throw (OpenCLError) {
	delete this->code;
	this->code = new string();

	if ( transientPipeline ) {
		throw OpenCLError("Transient pipeline not implemented.");
	}
	else if ( pulsarPipeline ) {
		// Begin kernel's template
		string nrPeriodsPerBlock_s = toStringValue< unsigned int >(nrPeriodsPerBlock);
		string nrBins_s = toStringValue< unsigned int >(observation->getNrBins());
		string nrPaddedPeriods_s = toStringValue< unsigned int >(observation->getNrPaddedPeriods());
		string nrBinsInverse_s = toStringValue< float >(1.0f / observation->getNrBins());

		*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict foldedData, __global " + this->dataType + " * const restrict snrs) {\n"
			"const unsigned int dm = get_group_id(1);\n"
			"const unsigned int period = ( get_group_id(0) * " + nrPeriodsPerBlock_s + " ) + get_local_id(0);\n"
			"float average = 0;\n"
			"float rms = 0;\n"
			+ this->dataType + " max = 0;\n"
			"\n"
			"for ( unsigned int bin = 0; bin < " + nrBins_s + "; bin++ ) {\n"
			"const " + this->dataType + " globalItem = foldedData[( ( ( dm * " + nrBins_s + " ) + bin ) * " + nrPaddedPeriods_s + " ) + period];\n"
			"average += globalItem;\n"
			"rms += ( globalItem * globalItem );\n"
			"max = fmax(max, globalItem);\n"
			"}\n"
			"average *= " + nrBinsInverse_s + ";\n"
			"rms *= " + nrBinsInverse_s + ";\n"
			"snrs[( dm * " + nrPaddedPeriods_s + ") + period] = ( max - average ) / native_sqrt(rms);\n"
			"}\n";
		// End kernel's template

		globalSize = cl::NDRange(observation->getNrPeriods(), observation->getNrDMs());
		localSize = cl::NDRange(nrPeriodsPerBlock, 1);

		this->gflop = giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrPeriods() * ( ( observation->getNrBins() * 3 ) + 4 ));
		this->gb = giga(static_cast< long long unsigned int >(observation->getNrDMs()) * observation->getNrPeriods() * ( ( observation->getNrBins() * sizeof(T) ) + sizeof(T) ));
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

template< typename T > inline void SNR< T >::setNrPeriodsPerBlock(unsigned int periodsPerBlock) {
	nrPeriodsPerBlock = periodsPerBlock;
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

} // TDM

#endif // SNR_HPP
