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

#include <string>
#include <cmath>
#include <x86intrin.h>

#include <utils.hpp>
#include <Observation.hpp>


#ifndef SNR_HPP
#define SNR_HPP

namespace PulsarSearch {

template< typename T > using snrFunc = void (*)(const AstroData::Observation &, const float *, float *);

// Sequential SNR
template< typename T > void snrDedispersedTS(const unsigned int second, const AstroData::Observation & observation, const std::vector< T > & dedispersedTS, std::vector< T > & maxS, std::vector< float > & meanS, std::vector< float > & rmsS);
template< typename T > void snrFoldedTS(const AstroData::Observation & observation, const std::vector< T > & foldedTS, std::vector< T > & snrs);
// SIMD SNR
std::string * getSNRSIMD(const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, bool phi = false);


// Implementations
template< typename T > void snrDedispersedTS(const unsigned int second, const AstroData::Observation & observation, const std::vector< T > & dedispersedTS, std::vector< T > & maxS, std::vector< float > & meanS, std::vector< float > & rmsS) {
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    T max = 0;
    float mean = 0.0f;
    float rms = 0.0f;

    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      T value = dedispersedTS[(dm * observation.getNrSamplesPerPaddedSecond()) + sample];

      mean += value;
      rms += (value * value);

      if ( value > max ) {
        max = value;
      }
    }

    if ( max > maxS[dm] ) {
      maxS[dm] = max;
    }
    meanS[dm] = ((meanS[dm] * observation.getNrSamplesPerSecond() * second) + mean) / (observation.getNrSamplesPerSecond() * (second + 1));
    rmsS[dm] = ((rmsS[dm] * observation.getNrSamplesPerSecond() * second) + rms) / (observation.getNrSamplesPerSecond() * (second + 1));
  }
}

template< typename T > void snrFoldedTS(AstroData::Observation & observation, const std::vector< T > & foldedTS, std::vector< T > & snrs) {
  for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
			T max = 0;
			float average = 0.0f;
			float rms = 0.0f;

			for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
				T value = foldedTS[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + (period * observation.getNrPaddedDMs()) + dm];

				average += value;
				rms += (value * value);

				if ( value > max ) {
					max = value;
				}
			}
			average /= observation.getNrBins();
			rms = std::sqrt(rms / observation.getNrBins());

			snrs[(period * observation.getNrPaddedDMs()) + dm] = (max - average) / rms;
		}
	}
}

std::string * getSNRSIMD(const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, bool phi) {
  std::string * code = new std::string();
  std::string * computeTemplate = new std::string();

  // Begin kernel's template
  if ( !phi ) {
    *code = "namespace PulsarSearch {\n"
      "template< typename T > void snrAVX" + isa::utils::toString(nrDMsPerThread) + "x" + isa::utils::toString(nrPeriodsPerThread) + "(const AstroData::Observation & observation, const float * const __restrict__ foldedData, float * const __restrict__ snrs) {\n"
      "{\n"
      "__m256 max = _mm256_setzero_ps();\n"
      "__m256 average = _mm256_setzero_ps();\n"
      "__m256 rms = _mm256_setzero_ps();\n"
      "#pragma omp parallel for schedule(static)\n"
      "for ( unsigned int period = 0; period < observation.getNrPeriods(); period += " + isa::utils::toString(nrPeriodsPerThread) + ") {\n"
      "#pragma omp parallel for schedule(static)\n"
      "for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm += " + isa::utils::toString(nrPeriodsPerThread) + " * 8) {\n"
      "<%COMPUTE%>"
      "}\n"
      "}\n"
      "}\n"
      "}\n"
      "}\n";
    *computeTemplate = "max = _mm256_setzero_ps();\n"
      "average = _mm256_setzero_ps();\n"
      "rms = _mm256_setzero_ps();\n"
      "\n"
      "for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {\n"
      "__m256 value = _mm256_load_ps(&(foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + ((period + <%PERIOD_NUM%>) * observation.getNrPaddedDMs()) + dm + <%DM_NUM%>]));\n"
      "\n"
      "average = _mm256_add_ps(average, value);\n"
      "rms = _mm256_add_ps(rms, _mm256_mul_ps(value, value));\n"
      "max = _mm256_max_ps(max, value);\n"
      "}\n"
      "average = _mm256_div_ps(average, _mm256_set1_ps(observation.getNrBins()));\n"
      "rms = _mm256_sqrt_ps(_mm256_div_ps(rms, _mm256_set1_ps(observation.getNrBins())));\n"
      "_mm256_store_ps(&(snrs[((period + <%PERIOD_NUM%>) * observation.getNrPaddedDMs()) + dm + <%DM_NUM%>]),_mm256_div_ps(_mm256_sub_ps(max, average), rms));\n";
  } else {
    *code = "namespace PulsarSearch {\n"
      "template< typename T > void snrPhi" + isa::utils::toString(nrDMsPerThread) + "x" + isa::utils::toString(nrPeriodsPerThread) + "(const AstroData::Observation & observation, const float * const __restrict__ foldedData, float * const __restrict__ snrs) {\n"
      "{\n"
      "__m512 max = _mm512_setzero_ps();\n"
      "__m512 average = _mm512_setzero_ps();\n"
      "__m512 rms = _mm512_setzero_ps();\n"
      "#pragma omp parallel for schedule(static)\n"
      "for ( unsigned int period = 0; period < observation.getNrPeriods(); period += " + isa::utils::toString(nrPeriodsPerThread) + ") {\n"
      "#pragma omp parallel for schedule(static)\n"
      "for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm += " + isa::utils::toString(nrPeriodsPerThread) + " * 16) {\n"
      "<%COMPUTE%>"
      "}\n"
      "}\n"
      "}\n"
      "}\n"
      "}\n";
    *computeTemplate = "max = _mm512_setzero_ps();\n"
      "average = _mm512_setzero_ps();\n"
      "rms = _mm512_setzero_ps();\n"
      "\n"
      "for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {\n"
      "__m512 value = _mm512_load_ps(&(foldedData[(bin * observation.getNrPeriods() * observation.getNrPaddedDMs()) + ((period + <%PERIOD_NUM%>) * observation.getNrPaddedDMs()) + dm + <%DM_NUM%>]));\n"
      "\n"
      "average = _mm512_add_ps(average, value);\n"
      "rms = _mm512_add_ps(rms, _mm512_mul_ps(value, value));\n"
      "max = _mm512_max_ps(max, value);\n"
      "}\n"
      "average = _mm512_div_ps(average, _mm512_set1_ps(observation.getNrBins()));\n"
      "rms = _mm512_sqrt_ps(_mm512_div_ps(rms, _mm512_set1_ps(observation.getNrBins())));\n"
      "_mm512_store_ps(&(snrs[((period + <%PERIOD_NUM%>) * observation.getNrPaddedDMs()) + dm + <%DM_NUM%>]),_mm512_div_ps(_mm512_sub_ps(max, average), rms));\n";
  }
  // End kernel's template

  std::string * compute_s = new std::string();

  for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
    std::string period_s = isa::utils::toString< unsigned int >(period);

    for ( unsigned int dm = 0; dm < nrDMsPerThread; dm++ ) {
      std::string dm_s = isa::utils::toString< unsigned int >(dm);
      std::string * temp_s = 0;

      temp_s = isa::utils::replace(computeTemplate, "<%PERIOD_NUM%>", period_s);
      temp_s = isa::utils::replace(temp_s, "<%DM_NUM%>", dm_s, true);
      compute_s->append(*temp_s);
      delete temp_s;
    }
  }
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  delete compute_s;

  delete computeTemplate;
  return code;
}

} // PulsarSearch

#endif // SNR_HPP

