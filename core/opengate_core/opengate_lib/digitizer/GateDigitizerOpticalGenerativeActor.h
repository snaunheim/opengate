/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#ifndef GateDigitizerOpticalGenerativeActor_h
#define GateDigitizerOpticalGenerativeActor_h

#include "GateVDigitizerWithOutputActor.h"
#include <G4Cache.hh>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/*
 * For every input digi (one energy-deposition step), calls a Python
 * generative model to synthesize N optical-photon-like records
 * (X, Y, dX, dY, dZ, Ekine, LogTime), without running G4OpticalPhysics.
 * This lets a model such as OptiGAN bridge the optical simulation live,
 * inside a standard system simulation.
 */

class GateDigitizerOpticalGenerativeActor : public GateVDigitizerWithOutputActor {

public:
  // signature of the callback function in Python that generates,
  // for one input digi (position, energy, time), a batch of N synthetic
  // optical photon records by filling fOutput* below
  using GeneratorType =
      std::function<void(GateDigitizerOpticalGenerativeActor *)>;

  explicit GateDigitizerOpticalGenerativeActor(py::dict &user_info);

  ~GateDigitizerOpticalGenerativeActor() override;

  void InitializeUserInfo(py::dict &user_info) override;

  void StartSimulationAction() override;

  void EndOfEventAction(const G4Event *event) override;

  void SetGeneratorFunction(GeneratorType &f);

  // input to the generator (set before each call to fGenerator)
  double fInputX, fInputY, fInputZ;
  double fInputEdep;
  double fInputTime;

  // output from the generator (filled by Python, read back after each call)
  std::vector<double> fOutputX;
  std::vector<double> fOutputY;
  std::vector<double> fOutputDX;
  std::vector<double> fOutputDY;
  std::vector<double> fOutputDZ;
  std::vector<double> fOutputEkine;
  std::vector<double> fOutputLogTime;

protected:
  void DigitInitialize(
      const std::vector<std::string> &attributes_not_in_filler) override;

  GeneratorType fGenerator;

  GateVDigiAttribute *fOutputXAttribute{};
  GateVDigiAttribute *fOutputYAttribute{};
  GateVDigiAttribute *fOutputDXAttribute{};
  GateVDigiAttribute *fOutputDYAttribute{};
  GateVDigiAttribute *fOutputDZAttribute{};
  GateVDigiAttribute *fOutputEkineAttribute{};
  GateVDigiAttribute *fOutputTimeAttribute{};
  GateVDigiAttribute *fOutputSourceHitIndexAttribute{};

  struct threadLocalT {
    double *edep{};
    G4ThreeVector *pos{};
    double *time{};
  };
  G4Cache<threadLocalT> fThreadLocalData;
};

#endif // GateDigitizerOpticalGenerativeActor_h
