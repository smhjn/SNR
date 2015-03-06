// Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <SNR.hpp>

namespace PulsarSearch {

snrDedispersedConf::snrDedispersedConf() {}

snrDedispersedConf::~snrDedispersedConf() {}

snrFoldedConf::snrFoldedConf() {}

snrFoldedConf::~snrFoldedConf() {}

std::string snrDedispersedConf::print() const {
  return std::string(isa::utils::toString(nrDMsPerBlock) + " " + isa::utils::toString(nrDMsPerThread));
}

std::string snrFoldedConf::print() const {
  return std::string(isa::utils::toString(this->getNrDMsPerBlock()) + " " + isa::utils::toString(nrPeriodsPerBlock) + " " + isa::utils::toString(this->getNrDMsPerThread()) + " " + isa::utils::toString(nrPeriodsPerThread));
}

std::string * getSNRDedispersedOpenCL(const snrDedispersedConf & conf, const std::string & dataType, const AstroData::Observation & observation) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void snrDedispersed(const float second, __global const " + dataType + " * const restrict dedispersedData, __global " + dataType + " * const restrict maxS, __global float * const restrict meanS, __global float * const restrict varianceS) {\n"
    "unsigned int nrElements = (second * " + isa::utils::toString(observation.getNrSamplesPerSecond()) + ") + 1;\n"
    "<%DEF_DM%>"
    "\n"
    "if ( second == 0 ) {\n"
    + dataType + " globalItem = 0;\n"
    "<%COMPUTE_DM_00%>"
    "} else {\n"
    + dataType + " globalItem = 0;\n"
    + dataType + " oldMean = 0;\n"
    "<%COMPUTE_DM_X0%>"
    "}\n"
    "for ( unsigned int sample = 1; sample < " + isa::utils::toString< unsigned int >(observation.getNrSamplesPerSecond()) + "; sample++ ) {\n"
    + dataType + " globalItem = 0;\n"
    + dataType + " oldMean = 0;\n"
    "nrElements++;\n"
    "<%COMPUTE_DM%>"
    "}\n"
    "<%STORE_DM%>"
    "}\n";
  std::string defDMTemplate = "const unsigned int dm<%DM_NUM%> = (get_group_id(0) * " + isa::utils::toString< unsigned int >(conf.getNrDMsPerBlock() * conf.getNrDMsPerThread()) + ") + get_local_id(0) + <%OFFSET%>;\n"
    + dataType + " meanDM<%DM_NUM%> = 0;\n"
    + dataType + " varianceDM<%DM_NUM%> = 0;\n"
    + dataType + " maxDM<%DM_NUM%> = 0;\n";
  std::string computeDM00Template = "globalItem = dedispersedData[dm<%DM_NUM%>];\n"
    "meanDM<%DM_NUM%> = globalItem;\n"
    "maxDM<%DM_NUM%> = globalItem;\n";
  std::string computeDMX0Template = "globalItem = dedispersedData[dm<%DM_NUM%>];\n"
    "meanDM<%DM_NUM%> = meanS[dm<%DM_NUM%>];\n"
    "varianceDM<%DM_NUM%> = varianceS[dm<%DM_NUM%>];\n"
    "maxDM<%DM_NUM%> = maxS[dm<%DM_NUM%>];\n"
    "oldMean = meanDM<%DM_NUM%>;\n"
    "meanDM<%DM_NUM%> += (globalItem - meanDM<%DM_NUM%>) / nrElements;\n"
    "variance<%DM_NUM%> += (globalItem - oldMean) * (globalItem - mean);\n"
    "maxDM<%DM_NUM%> = fmax(maxDM<%DM_NUM%>, globalItem);\n";
  std::string computeDMTemplate = "globalItem = dedispersedData[(sample * " + isa::utils::toString< unsigned int >(observation.getNrPaddedDMs()) + ") + dm<%DM_NUM%>];\n"
    "oldMean = meanDM<%DM_NUM%>;\n"
    "meanDM<%DM_NUM%> += (globalItem - meanDM<%DM_NUM%>) / nrElements;\n"
    "variance<%DM_NUM%> += (globalItem - oldMean) * (globalItem - mean);\n"
    "maxDM<%DM_NUM%> = fmax(maxDM<%DM_NUM%>, globalItem);\n";
  std::string storeDMTemplate = "maxS[dm<%DM_NUM%>] = maxDM<%DM_NUM%>;\n"
    "meanS[dm<%DM_NUM%>] = meanS[dm<%DM_NUM%>];\n"
    "varianceS[dm<%DM_NUM%>] = varianceS[dm<%DM_NUM%>];\n";
  // End kernel's template

  std::string * defDM_s = new std::string();
  std::string * computeDM00_s = new std::string();
  std::string * computeDMX0_s = new std::string();
  std::string * computeDM_s = new std::string();
  std::string * storeDM_s = new std::string();

  for ( unsigned int dm = 0; dm < conf.getNrDMsPerThread(); dm++ ) {
    std::string dm_s = isa::utils::toString< unsigned int >(dm);
    std::string offset_s = isa::utils::toString< unsigned int >(dm * conf.getNrDMsPerBlock());
    std::string * temp_s = 0;

    temp_s = isa::utils::replace(&defDMTemplate, "<%DM_NUM%>", dm_s);
    temp_s = isa::utils::replace(temp_s, "<%OFFSET%>", offset_s, true);
    defDM_s->append(*temp_s);
    delete temp_s;
    temp_s = isa::utils::replace(&computeDM00Template, "<%DM_NUM%>", dm_s);
    computeDM00_s->append(*temp_s);
    delete temp_s;
    temp_s = isa::utils::replace(&computeDMX0Template, "<%DM_NUM%>", dm_s);
    computeDMX0_s->append(*temp_s);
    delete temp_s;
    temp_s = isa::utils::replace(&computeDMTemplate, "<%DM_NUM%>", dm_s);
    computeDM_s->append(*temp_s);
    delete temp_s;
    temp_s = isa::utils::replace(&storeDMTemplate, "<%DM_NUM%>", dm_s);
    storeDM_s->append(*temp_s);
    delete temp_s;
  }

  code = isa::utils::replace(code, "<%DEF_DM%>", *defDM_s, true);
  delete defDM_s;
  code = isa::utils::replace(code, "<%COMPUTE_DM_00%>", *computeDM00_s, true);
  delete computeDM00_s;
  code = isa::utils::replace(code, "<%COMPUTE_DM_X0%>", *computeDMX0_s, true);
  delete computeDMX0_s;
  code = isa::utils::replace(code, "<%COMPUTE_DM%>", *computeDM_s, true);
  delete computeDM_s;
  code = isa::utils::replace(code, "<%STORE_DM%>", *storeDM_s, true);
  delete storeDM_s;

  return code;
}

std::string * getSNRFoldedOpenCL(const snrFoldedConf & conf, const std::string & dataType, const AstroData::Observation & observation) {
  std::string * code = new std::string();

  // Begin kernel's template
  std::string nrPaddedDMs_s = isa::utils::toString< unsigned int >(observation.getNrPaddedDMs());
  std::string nrBinsInverse_s = isa::utils::toString< float >(1.0f / (observation.getNrBins() - 1));

  *code = "__kernel void snrFolded(__global const " + dataType + " * const restrict foldedData, __global " + dataType + " * const restrict snrs) {\n"
    + dataType + " globalItem = 0;\n"
    + dataType + " oldMean = 0;\n"
    "<%DEF_DM%>"
    "<%DEF_PERIOD%>"
    "<%DEF_DM_PERIOD%>"
    "\n"
    "<%COMPUTE0%>"
    "for ( unsigned int bin = 1; bin < " + isa::utils::toString< unsigned int >(observation.getNrBins()) + "; bin++ ) {\n"
    "<%COMPUTE%>"
    "}\n"
    "<%STORE%>"
    "}\n";
    std::string defDMsTemplate = "const unsigned int dm<%DM_NUM%> = (get_group_id(0) * " + isa::utils::toString< unsigned int >(conf.getNrDMsPerBlock() * conf.getNrDMsPerThread()) + ") + get_local_id(0) + <%DM_OFFSET%>;\n";
  std::string defPeriodsTemplate = "const unsigned int period<%PERIOD_NUM%> = (get_group_id(1) * " + isa::utils::toString< unsigned int >(conf.getNrPeriodsPerBlock() * conf.getNrPeriodsPerThread()) + ") + get_local_id(1) + <%PERIOD_OFFSET%>;\n";
  std::string defDMsPeriodsTemplate = dataType + " meanDM<%DM_NUM%>p<%PERIOD_NUM%> = 0;\n"
    + dataType + " varianceDM<%DM_NUM%>p<%PERIOD_NUM%> = 0;\n"
    + dataType + " maxDM<%DM_NUM%>p<%PERIOD_NUM%> = 0;\n";
  std::string compute0Template = "globalItem = foldedData[(period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ") + dm<%DM_NUM%>];\n"
    "meanDM<%DM_NUM%>p<%PERIOD_NUM%> = globalItem;\n"
    "maxDM<%DM_NUM%>p<%PERIOD_NUM%> = globalItem;\n";
  std::string computeTemplate = "globalItem = foldedData[(bin * " + isa::utils::toString(observation.getNrPeriods() * observation.getNrPaddedDMs()) + ") + (period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ") + dm<%DM_NUM%>];\n"
    "oldMean = meanDM<%DM_NUM%>p<%PERIOD_NUM%>;\n"
    "meanDM<%DM_NUM%>p<%PERIOD_NUM%> += (globalItem - meanDM<%DM_NUM%>p<%PERIOD_NUM%>) / (bin + 1);\n"
    "varianceDM<%DM_NUM%>p<%PERIOD_NUM%> += (globalItem - oldMean) * (globalItem - meanDM<%DM_NUM%>p<%PERIOD_NUM%>);\n"
    "maxDM<%DM_NUM%>p<%PERIOD_NUM%> = fmax(maxDM<%DM_NUM%>p<%PERIOD_NUM%>, globalItem);\n";
  std::string storeTemplate = "varianceDM<%DM_NUM%>p<%PERIOD_NUM%> *= " + nrBinsInverse_s + "f;\n"
    "snrs[(period<%PERIOD_NUM%> * " + nrPaddedDMs_s + ") + dm<%DM_NUM%>] = (maxDM<%DM_NUM%>p<%PERIOD_NUM%> - meanDM<%DM_NUM%>p<%PERIOD_NUM%>) / native_sqrt(varianceDM<%DM_NUM%>p<%PERIOD_NUM%>);\n";
  // End kernel's template

  std::string * defDM_s = new std::string();
  std::string * defPeriod_s = new std::string();
  std::string * defDMPeriod_s = new std::string();
  std::string * compute0_s = new std::string();
  std::string * compute_s = new std::string();
  std::string * store_s = new std::string();

  for ( unsigned int dm = 0; dm < conf.getNrDMsPerThread(); dm++ ) {
    std::string dm_s = isa::utils::toString< unsigned int >(dm);
    std::string * temp_s = 0;

    temp_s = isa::utils::replace(&defDMsTemplate, "<%DM_NUM%>", dm_s);
    if ( dm == 0 ) {
      std::string empty_s;
      temp_s = isa::utils::replace(temp_s, " + <%DM_OFFSET%>", empty_s, true);
    } else {
      std::string offset_s = isa::utils::toString(dm * conf.getNrDMsPerBlock());
      temp_s = isa::utils::replace(temp_s, "<%DM_OFFSET%>", offset_s, true);
    }
    defDM_s->append(*temp_s);
    delete temp_s;
  }
  for ( unsigned int period = 0; period < conf.getNrPeriodsPerThread(); period++ ) {
    std::string period_s = isa::utils::toString< unsigned int >(period);
    std::string * temp_s = 0;

    temp_s = isa::utils::replace(&defPeriodsTemplate, "<%PERIOD_NUM%>", period_s);
    if ( period == 0 ) {
      std::string empty_s;
      temp_s = isa::utils::replace(temp_s, " + <%PERIOD_OFFSET%>", empty_s, true);
    } else {
      std::string offset_s = isa::utils::toString(period * conf.getNrPeriodsPerBlock());
      temp_s = isa::utils::replace(temp_s, "<%PERIOD_OFFSET%>", offset_s, true);
    }
    defPeriod_s->append(*temp_s);
    delete temp_s;

    for ( unsigned int dm = 0; dm < conf.getNrDMsPerThread(); dm++ ) {
      std::string dm_s = isa::utils::toString< unsigned int >(dm);

      temp_s = isa::utils::replace(&defDMsPeriodsTemplate, "<%DM_NUM%>", dm_s);
      temp_s = isa::utils::replace(temp_s, "<%PERIOD_NUM%>", period_s, true);
      defDMPeriod_s->append(*temp_s);
      delete temp_s;
      temp_s = isa::utils::replace(&compute0Template, "<%DM_NUM%>", dm_s);
      temp_s = isa::utils::replace(temp_s, "<%PERIOD_NUM%>", period_s, true);
      compute0_s->append(*temp_s);
      delete temp_s;
      temp_s = isa::utils::replace(&computeTemplate, "<%DM_NUM%>", dm_s);
      temp_s = isa::utils::replace(temp_s, "<%PERIOD_NUM%>", period_s, true);
      compute_s->append(*temp_s);
      delete temp_s;
      temp_s = isa::utils::replace(&storeTemplate, "<%DM_NUM%>", dm_s);
      temp_s = isa::utils::replace(temp_s, "<%PERIOD_NUM%>", period_s, true);
      store_s->append(*temp_s);
      delete temp_s;
    }
  }

  code = isa::utils::replace(code, "<%DEF_DM%>", *defDM_s, true);
  delete defDM_s;
  code = isa::utils::replace(code, "<%DEF_PERIOD%>", *defPeriod_s, true);
  delete defPeriod_s;
  code = isa::utils::replace(code, "<%DEF_DM_PERIOD%>", *defDMPeriod_s, true);
  delete defDMPeriod_s;
  code = isa::utils::replace(code, "<%COMPUTE0%>", *compute0_s, true);
  delete compute0_s;
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  delete compute_s;
  code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
  delete store_s;

  return code;
}

} // PulsarSearch

