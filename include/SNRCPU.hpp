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

#include <cmath>
using std::sqrt;

#include <Observation.hpp>
using AstroData::Observation;


#ifndef SNR_CPU_HPP
#define SNR_CPU_HPP

namespace PulsarSearch {

// Sequential SNR
/*template< typename T > void transientSNR(Observation< T > & observation, CLData< T > * input, CLData< T > * output);*/
template< typename T > void pulsarSNR(const Observation< T > & observation, const T * const __restrict__ foldedData, T * const __restrict__ snrs);


// Implementation
template< typename T > void pulsarSNR(const Observation< T > & observation, const T * const __restrict__ foldedData, T * const __restrict__ snrs) {
	for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
		for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
			T max = 0;
			float average = 0.0f;
			float rms = 0.0f;

			for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
				T value = foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + dm];

				average += value;
				rms += (value * value);

				if ( value > max ) {
					max = value;
				}
			}
			average /= observation.getNrBins();
			rms = sqrt(rms / observation.getNrBins());

			snrs[(period * observation.getNrPaddedDMs()) + dm] = (max - average) / rms;
		}
	}
}

} // PulsarSearch

#endif // SNR_CPU_HPP
