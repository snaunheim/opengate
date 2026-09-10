/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "GateDigitizerOpticalGenerativeActor.h"

void init_GateDigitizerOpticalGenerativeActor(py::module &m) {

  py::class_<GateDigitizerOpticalGenerativeActor,
             std::unique_ptr<GateDigitizerOpticalGenerativeActor,
                             py::nodelete>,
             GateVDigitizerWithOutputActor>(
      m, "GateDigitizerOpticalGenerativeActor")
      .def(py::init<py::dict &>())
      .def("SetGeneratorFunction",
           &GateDigitizerOpticalGenerativeActor::SetGeneratorFunction)

      .def_readwrite("fScintillationYield",
                     &GateDigitizerOpticalGenerativeActor::fScintillationYield)

      // inputs set by C++ before each generator call
      .def_readwrite("fInputX", &GateDigitizerOpticalGenerativeActor::fInputX)
      .def_readwrite("fInputY", &GateDigitizerOpticalGenerativeActor::fInputY)
      .def_readwrite("fInputZ", &GateDigitizerOpticalGenerativeActor::fInputZ)
      .def_readwrite("fInputTime",
                     &GateDigitizerOpticalGenerativeActor::fInputTime)
      .def_readwrite("fInputN",
                     &GateDigitizerOpticalGenerativeActor::fInputN)

      // outputs filled by Python — one column per declared model output, each
      // of length fInputN, in the order of the manifest's 'outputs' list
      .def_readwrite("fOutputColumns",
                     &GateDigitizerOpticalGenerativeActor::fOutputColumns);
}
