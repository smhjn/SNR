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

namespace TDM {

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

} // TDM

#endif // SNR_PHI_HPP
