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

#include <string>
#include <vector>
#include <cmath>
#include <x86intrin.h>
using std::string;
using std::vector;
using std::make_pair;
using std::pow;
using std::ceil;


#ifndef SNR_PHI_HPP
#define SNR_PHI_HPP

namespace PulsarSearch {

// OpenMP SNR
/*template< typename T > void transientSNR(Observation< T > & observation, CLData< T > * input, CLData< T > * output);*/
template< typename T > void pulsarSNR(const unsigned int nrDMs, const unsigned int nrPeriods, const unsigned int nrPaddedPeriods, const unsigned int nrBins, const unsigned int nrPaddedBins, const T * const __restrict__ foldedData, T * const __restrict__ snrs);


// Implementation
template< typename T > void pulsarSNR(const unsigned int nrDMs, const unsigned int nrPeriods, const unsigned int nrPaddedPeriods, const unsigned int nrBins, const unsigned int nrPaddedBins, const T * const __restrict__ foldedData, T * const __restrict__ snrs) {
	#pragma offload target(mic) nocopy(foldedData: alloc_if(0) free_if(0)) nocopy(snrs: alloc_if(0) free_if(0))
	{
		#pragma omp parallel for
		for ( unsigned int dm = 0; dm < nrDMs; dm++ ) {
			#pragma omp parallel for
			for ( unsigned int period = 0; period < nrPeriods; period++ ) {
				T max = 0;
				float average = 0.0f;
				float rms = 0.0f;

				for ( unsigned int bin = 0; bin < nrBins; bin++ ) {
					T value = foldedData[( ( ( dm * nrPeriods ) + period ) * nrPaddedBins ) + bin];

					average += value;
					rms += (value * value);

					if ( value > max ) {
						max = value;
					}
				}
				average /= nrBins;
				rms = sqrt(rms / nrBins);

				snrs[( dm * nrPaddedPeriods ) + period] = ( max - average) / rms;
			}
		}
	}
}

} // PulsarSearch

#endif // SNR_PHI_HPP
