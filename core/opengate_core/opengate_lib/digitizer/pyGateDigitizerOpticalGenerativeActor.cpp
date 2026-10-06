/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "GateDigitizerOpticalGenerativeActor.h"

// hands a pooled input to python as a numpy view, so the batch is not copied
static py::array_t<double>
ViewOf(GateDigitizerOpticalGenerativeActor &actor, std::vector<double> &v) {
  return py::array_t<double>({static_cast<py::ssize_t>(v.size())},
                             {sizeof(double)}, v.data(), py::cast(&actor));
}

// the generator runs on the thread that owns the hits, so every access goes
// to that thread's own set
#define IO(a) ((a).GetGeneratorIO())

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
      .def_property(
          "fInputX",
          [](GateDigitizerOpticalGenerativeActor &a) { return IO(a).x; },
          [](GateDigitizerOpticalGenerativeActor &a, double v) { IO(a).x = v; })
      .def_property(
          "fInputY",
          [](GateDigitizerOpticalGenerativeActor &a) { return IO(a).y; },
          [](GateDigitizerOpticalGenerativeActor &a, double v) { IO(a).y = v; })
      .def_property(
          "fInputZ",
          [](GateDigitizerOpticalGenerativeActor &a) { return IO(a).z; },
          [](GateDigitizerOpticalGenerativeActor &a, double v) { IO(a).z = v; })
      .def_property(
          "fInputTime",
          [](GateDigitizerOpticalGenerativeActor &a) { return IO(a).time; },
          [](GateDigitizerOpticalGenerativeActor &a, double v) {
            IO(a).time = v;
          })
      .def_property(
          "fInputN",
          [](GateDigitizerOpticalGenerativeActor &a) { return IO(a).n; },
          [](GateDigitizerOpticalGenerativeActor &a, long v) { IO(a).n = v; })

      // pooled inputs, one entry per photon, used when the actor pools
      .def_property_readonly(
          "fInputXs",
          [](GateDigitizerOpticalGenerativeActor &a) {
            return ViewOf(a, IO(a).xs);
          })
      .def_property_readonly(
          "fInputYs",
          [](GateDigitizerOpticalGenerativeActor &a) {
            return ViewOf(a, IO(a).ys);
          })
      .def_property_readonly(
          "fInputZs",
          [](GateDigitizerOpticalGenerativeActor &a) {
            return ViewOf(a, IO(a).zs);
          })
      .def_property_readonly(
          "fInputTimes",
          [](GateDigitizerOpticalGenerativeActor &a) {
            return ViewOf(a, IO(a).times);
          })

      // outputs filled by Python, one column per declared model output, each
      // of length fInputN, in the order of the manifest's 'outputs' list
      .def_property(
          "fOutputColumns",
          [](GateDigitizerOpticalGenerativeActor &a) { return IO(a).columns; },
          [](GateDigitizerOpticalGenerativeActor &a,
             std::vector<std::vector<double>> v) {
            IO(a).columns = std::move(v);
          });
}
