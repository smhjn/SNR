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