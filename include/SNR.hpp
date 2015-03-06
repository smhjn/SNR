// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <vector>
#include <string>
#include <cmath>

#include <utils.hpp>
#include <Observation.hpp>


#ifndef SNR_HPP
#define SNR_HPP

namespace PulsarSearch {

class snrDedispersedConf {
public:
  snrDedispersedConf();
  ~snrDedispersedConf();
  // Get
  unsigned int getNrDMsPerBlock() const;
  unsigned int getNrDMsPerThread() const;
  // Set
  void setNrDMsPerBlock(unsigned int dms);
  void setNrDMsPerThread(unsigned int dms);
  // utils
  std::string print() const;

private:
  unsigned int nrDMsPerBlock;
  unsigned int nrDMsPerThread;
};

class snrFoldedConf : public snrDedispersedConf {
public:
  snrFoldedConf();
  ~snrFoldedConf();
  // Get
  unsigned int getNrPeriodsPerBlock() const;
  unsigned int getNrPeriodsPerThread() const;
  // Set
  void setNrPeriodsPerBlock(unsigned int periods);
  void setNrPeriodsPerThread(unsigned int periods);
  // utils
  std::string print() const;

private:
  unsigned int nrPeriodsPerBlock;
  unsigned int nrPeriodsPerThread;
};

// Sequential SNR
template< typename T > void snrDedispersed(const unsigned int second, const AstroData::Observation & observation, const std::vector< T > & dedispersed, std::vector< T > & maxS, std::vector< float > & meanS, std::vector< float > & varianceS);
template< typename T > void snrFolded(const AstroData::Observation & observation, const std::vector< T > & folded, std::vector< T > & snrs);
// OpenCL SNR
std::string * getSNRDedispersedOpenCL(const snrDedispersedConf & conf, const std::string & dataType, const AstroData::Observation & observation);
std::string * getSNRFoldedOpenCL(const snrFoldedConf & conf, const std::string & dataType, const AstroData::Observation & observation);


// Implementations
inline unsigned int snrDedispersedConf::getNrDMsPerBlock() const {
  return nrDMsPerBlock;
}

inline unsigned int snrDedispersedConf::getNrDMsPerThread() const {
  return nrDMsPerThread;
}

inline void snrDedispersedConf::setNrDMsPerBlock(unsigned int dms) {
  nrDMsPerBlock = dms;
}

inline void snrDedispersedConf::setNrDMsPerThread(unsigned int dms) {
  nrDMsPerThread = dms;
}

inline unsigned int snrFoldedConf::getNrPeriodsPerBlock() const {
  return nrPeriodsPerBlock;
}

inline unsigned int snrFoldedConf::getNrPeriodsPerThread() const {
  return nrPeriodsPerThread;
}

inline void snrFoldedConf::setNrPeriodsPerBlock(unsigned int periods) {
  nrPeriodsPerBlock = periods;
}

inline void snrFoldedConf::setNrPeriodsPerThread(unsigned int periods) {
  nrPeriodsPerThread = periods;
}

template< typename T > void snrDedispersed(const unsigned int second, const AstroData::Observation & observation, const std::vector< T > & dedispersed, std::vector< T > & maxS, std::vector< float > & meanS, std::vector< float > & varianceS) {
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    unsigned int nrElements = (second * observation.getNrSamplesPerSecond()) + 1;
    T max = 0;
    float mean = 0.0f;
    float variance = 0.0f;

    if ( second == 0 ) {
      T value = dedispersed[dm * observation.getNrSamplesPerPaddedSecond()];
      max = value;
      mean = value;
    } else {
      T value = dedispersed[dm * observation.getNrSamplesPerPaddedSecond()];
      max = maxS[dm];
      mean = meanS[dm];
      variance = varianceS[dm];

      mean += (value - mean) / nrElements;
      variance += (value - meanS[dm]) * (value - mean);
      if ( value > max ) {
        max = value;
      }
    }

    for ( unsigned int sample = 1; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      T value = dedispersed[(dm * observation.getNrSamplesPerPaddedSecond()) + sample];
      float oldMean = mean;

      nrElements++;
      mean += (value - mean) / nrElements;
      variance += (value - oldMean) * (value - mean);
      if ( value > max ) {
        max = value;
      }
    }

    if ( max > maxS[dm] ) {
      maxS[dm] = max;
    }
    meanS[dm] = mean;
    varianceS[dm] = variance;
  }
}

template< typename T > void snrFolded(AstroData::Observation & observation, const std::vector< T > & folded, std::vector< T > & snrs) {
  for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
			T max = folded[(period * observation.getNrPaddedDMs()) + dm];
			float mean = folded[(period * observation.getNrPaddedDMs()) + dm];
			float variance = 0.0f;

			for ( unsigned int bin = 1; bin < observation.getNrBins(); bin++ ) {
				T value = folded[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + dm];
        float oldMean = mean;

        mean += (value - mean) / (bin + 1);
        variance += (value - oldMean) * (value - mean);
				if ( value > max ) {
					max = value;
				}
			}
      variance /= observation.getNrBins() - 1;

			snrs[(period * observation.getNrPaddedDMs()) + dm] = (max - mean) / std::sqrt(variance);
		}
	}
}

} // PulsarSearch

#endif // SNR_HPP

