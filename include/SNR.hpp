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
template< typename T > void snrDedispersedTS(const AstroData::Observation & observation, const std::vector< T > & dedispersedTS, std::vector< T > & snrs);
template< typename T > void snrFoldedTS(const AstroData::Observation & observation, const std::vector< T > & foldedTS, std::vector< T > & snrs);
// OpenCL SNR
std::string * getSNRDedispersedOpenCL(const unsigned int nrDMsPerBlock, const unsigned int nrDMsPerThread, const std::string & dataType, const AstroData::Observation & observation);
std::string * getSNRFoldedOpenCL(const unsigned int nrDMsPerBlock, const unsigned int nrPeriodsPerBlock, const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, const std::string & dataType, const AstroData::Observation & observation);
// SIMD SNR
std::string * getSNRSIMD(const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, bool phi = false);


// Implementations
template< typename T > void snrDedispersedTS(const AstroData::Observation & observation, const std::vector< T > & dedispersedTS, std::vector< T > & snrs) {
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    T max = 0;
    float average = 0.0f;
    float rms = 0.0f;

    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      T value = dedispersedTS[(dm * observation.getNrSamplesPerPaddedSecond()) + sample];

      average += value;
      rms += (value * value);

      if ( value > max ) {
        max = value;
      }
    }
    average /= observation.getNrSamplesPerSecond();
    rms = std::sqrt(rms / observation.getNrSamplesPerSecond());

    if ( snrs[dm] < (max - average) / rms ) {
      snrs[dm] = (max - average) / rms;
    }
  }
}

template< typename T > void snrFoldedTS(AstroData::Observation & observation, const std::vector< T > & foldedTS, std::vector< T > & snrs) {
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int period = 0; period < observation.getNrPeriods(); period++ ) {
			T max = 0;
			float average = 0.0f;
			float rms = 0.0f;

			for ( unsigned int bin = 0; bin < observation.getNrBins(); bin++ ) {
				T value = foldedTS[(dm * observation.getNrPeriods() * observation.getNrPaddedBins()) + (period * observation.getNrPaddedBins()) + bin];

				average += value;
				rms += (value * value);

				if ( value > max ) {
					max = value;
				}
			}
			average /= observation.getNrBins();
			rms = std::sqrt(rms / observation.getNrBins());

			snrs[(dm * observation.getNrPaddedPeriods()) + period] = (max - average) / rms;
		}
	}
}

std::string * getSNRDedispersedOpenCL(const unsigned int nrDMsPerBlock, const unsigned int nrDMsPerThread, const std::string & dataType, const AstroData::Observation & observation) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void snrDedispersed(__global const " + dataType + " * const restrict dedispersedData, __global " + dataType + " * const restrict snrs) {\n"
    "<%DEF_DM%>"
    "<%LOAD_DM%>"
    + dataType + " globalItem = 0;\n"
    "\n"
    "for ( unsigned int sample = 0; sample < " + isa::utils::toString< unsigned int >(observation.getNrSamplesPerSecond()) + "; sample++ ) {\n"
    "<%COMPUTE_DM%>"
    "}\n"
    "<%STORE_DM%>"
    "}\n";
    std::string defDMTemplate = "const unsigned int dm<%DM_NUM%> = (get_group_id(0) * " + isa::utils::toString< unsigned int >(nrDMsPerBlock * nrDMsPerThread) + ") + get_local_id(0) + <%OFFSET%>;\n"
      + dataType + " averageDM<%DM_NUM%> = 0;\n"
      + dataType + " rmsDM<%DM_NUM%> = 0;\n"
      + dataType + " maxDM<%DM_NUM%> = 0;\n";
    std::string loadDMTemplate = dataType + " snrDM<%DM_NUM%> = snrs[dm<%DM_NUM%>];\n";
  std::string computeDMTemplate = "globalItem = dedispersedData[(sample * " + isa::utils::toString< unsigned int >(observation.getNrPaddedDMs()) + ") + dm<%DM_NUM%>];\n"
    "averageDM<%DM_NUM%> += globalItem;\n"
    "rmsDM<%DM_NUM%> += (globalItem * globalItem);\n"
    "maxDM<%DM_NUM%> = fmax(maxDM<%DM_NUM%>, globalItem);\n";
  std::string storeDMTemplate = "averageDM<%DM_NUM%> *= " + isa::utils::toString< float >(1.0f / observation.getNrSamplesPerSecond()) + "f;\n"
    "rmsDM<%DM_NUM%> *= " + isa::utils::toString< float >(1.0f / observation.getNrSamplesPerSecond()) + "f;\n"
    "globalItem = fmax(snrDM<%DM_NUM%>, (maxDM<%DM_NUM%> - averageDM<%DM_NUM%>) / native_sqrt(rmsDM<%DM_NUM%>));\n"
    "snrs[dm<%DM_NUM%>] = globalItem;\n";
  // End kernel's template

  std::string * defDM_s = new std::string();
  std::string * loadDM_s = new std::string();
  std::string * computeDM_s = new std::string();
  std::string * storeDM_s = new std::string();

  for ( unsigned int dm = 0; dm < nrDMsPerThread; dm++ ) {
    std::string dm_s = isa::utils::toString< unsigned int >(dm);
    std::string offset_s = isa::utils::toString< unsigned int >(dm * nrDMsPerBlock);
    std::string * temp_s = 0;

    temp_s = isa::utils::replace(&defDMTemplate, "<%DM_NUM%>", dm_s);
    temp_s = isa::utils::replace(temp_s, "<%OFFSET%>", offset_s, true);
    defDM_s->append(*temp_s);
    delete temp_s;
    temp_s = isa::utils::replace(&loadDMTemplate, "<%DM_NUM%>", dm_s);
    loadDM_s->append(*temp_s);
    delete temp_s;
    temp_s = isa::utils::replace(&computeDMTemplate, "<%DM_NUM%>", dm_s);
    computeDM_s->append(*temp_s);
    delete temp_s;
    temp_s = isa::utils::replace(&storeDMTemplate, "<%DM_NUM%>", dm_s);
    storeDM_s->append(*temp_s);
    delete temp_s;
  }

  code = isa::utils::replace(code, "<%DEF_DM%>", *defDM_s, true);
  code = isa::utils::replace(code, "<%LOAD_DM%>", *loadDM_s, true);
  code = isa::utils::replace(code, "<%COMPUTE_DM%>", *computeDM_s, true);
  code = isa::utils::replace(code, "<%STORE_DM%>", *storeDM_s, true);

  return code;
}

std::string * getSNRFoldedOpenCL(const unsigned int nrDMsPerBlock, const unsigned int nrPeriodsPerBlock, const unsigned int nrDMsPerThread, const unsigned int nrPeriodsPerThread, const std::string & dataType, const AstroData::Observation & observation) {
  std::string * code = new std::string();

  // Begin kernel's template
  std::string nrPaddedDMs_s = isa::utils::toString< unsigned int >(observation.getNrPaddedDMs());
  std::string nrBinsInverse_s = isa::utils::toString< float >(1.0f / observation.getNrBins());

  *code = "__kernel void snrFolded(__global const " + dataType + " * const restrict foldedData, __global " + dataType + " * const restrict snrs) {\n"
    "<%DEF_DM%>"
    "<%DEF_PERIOD%>"
    + dataType + " globalItem = 0;\n"
    + dataType + " average = 0;\n"
    + dataType + " rms = 0;\n"
    + dataType + " max = 0;\n"
    "\n"
    "<%COMPUTE%>"
    "}\n";
    std::string defDMsTemplate = "const unsigned int dm<%DM_NUM%> = (get_group_id(0) * " + isa::utils::toString< unsigned int >(nrDMsPerBlock * nrDMsPerThread) + ") + get_local_id(0) + <%DM_NUM%>;\n";
  std::string defPeriodsTemplate = "const unsigned int period<%PERIOD_NUM%> = (get_group_id(1) * " + isa::utils::toString< unsigned int >(nrPeriodsPerBlock * nrPeriodsPerThread) + ") + get_local_id(1) + <%PERIOD_NUM%>;\n";
  std::string computeTemplate = "average = 0;\n"
    "rms = 0;\n"
    "max = 0;\n"
    "for ( unsigned int bin = 0; bin < " + isa::utils::toString< unsigned int >(observation.getNrBins()) + "; bin++ ) {\n"
    "globalItem = foldedData[(bin * " + isa::utils::toString< unsigned int >(observation.getNrPeriods()) + " * " + nrPaddedDMs_s + ") + (period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ") + dm<%DM_NUM%>];\n"
    "average += globalItem;\n"
    "rms += (globalItem * globalItem);\n"
    "max = fmax(max, globalItem);\n"
    "}\n"
    "average *= " + nrBinsInverse_s + "f;\n"
    "rms *= " + nrBinsInverse_s + "f;\n"
    "snrs[(period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ") + dm<%DM_NUM%>] = (max - average) / native_sqrt(rms);\n";
  // End kernel's template

  std::string * defDM_s = new std::string();
  std::string * defPeriod_s = new std::string();
  std::string * compute_s = new std::string();

  for ( unsigned int dm = 0; dm < nrDMsPerThread; dm++ ) {
    std::string dm_s = isa::utils::toString< unsigned int >(dm);
    std::string * temp_s = 0;

    temp_s = isa::utils::replace(&defDMsTemplate, "<%DM_NUM%>", dm_s);
    defDM_s->append(*temp_s);
    delete temp_s;
  }
  for ( unsigned int period = 0; period < nrPeriodsPerThread; period++ ) {
    std::string period_s = isa::utils::toString< unsigned int >(period);
    std::string * temp_s = 0;

    temp_s = isa::utils::replace(&defPeriodsTemplate, "<%PERIOD_NUM%>", period_s);
    defPeriod_s->append(*temp_s);
    delete temp_s;

    for ( unsigned int dm = 0; dm < nrDMsPerThread; dm++ ) {
      std::string dm_s = isa::utils::toString< unsigned int >(dm);

      temp_s = isa::utils::replace(&computeTemplate, "<%PERIOD_NUM%>", period_s);
      temp_s = isa::utils::replace(temp_s, "<%DM_NUM%>", dm_s, true);
      compute_s->append(*temp_s);
      delete temp_s;
    }
  }

  code = isa::utils::replace(code, "<%DEF_DM%>", *defDM_s, true);
  code = isa::utils::replace(code, "<%DEF_PERIOD%>", *defPeriod_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);

  return code;
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

