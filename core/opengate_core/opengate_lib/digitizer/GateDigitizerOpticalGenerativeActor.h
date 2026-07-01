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
 * For every input digi (one energy-deposition step), samples N from a Poisson
 * distribution (N = Poisson(edep * scintillation_yield)) and calls a Python
 * generative model N times to synthesize one optical-photon-like record per
 * call (X, Y, dX, dY, dZ, Ekine, LogTime), without running G4OpticalPhysics.
 * This lets a model such as OptiGAN bridge the optical simulation live,
 * inside a standard system simulation.
 */

class GateDigitizerOpticalGenerativeActor : public GateVDigitizerWithOutputActor {

public:
  // signature of the callback: called once per synthetic photon;
  // fills fOutputX/Y/DX/DY/DZ/Ekine/LogTime with a single photon record
  using GeneratorType =
      std::function<void(GateDigitizerOpticalGenerativeActor *)>;

  explicit GateDigitizerOpticalGenerativeActor(py::dict &user_info);

  ~GateDigitizerOpticalGenerativeActor() override;

  void InitializeUserInfo(py::dict &user_info) override;

  void StartSimulationAction() override;

  void EndOfEventAction(const G4Event *event) override;

  void SetGeneratorFunction(GeneratorType &f);

  // scintillation yield in photons/MeV; N ~ Poisson(edep * yield)
  double fScintillationYield;

  // input to the generator (set before each call to fGenerator)
  double fInputX, fInputY, fInputZ;
  double fInputTime;

  // output from the generator (filled by Python for ONE photon per call)
  double fOutputX;
  double fOutputY;
  double fOutputDX;
  double fOutputDY;
  double fOutputDZ;
  double fOutputEkine;
  double fOutputLogTime;

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
