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
 * generative model once per hit with the batch size N. The model returns N
 * synthetic optical-photon records (X, Y, dX, dY, dZ, Ekine, Time) in one
 * call, enabling efficient GPU inference without running G4OpticalPhysics.
 */

class GateDigitizerOpticalGenerativeActor : public GateVDigitizerWithOutputActor {

public:
  // Callback called once per hit; reads fInputX/Y/Z/Time/N and fills
  // fOutputX/Y/DX/DY/DZ/Ekine/Time as vectors of length fInputN.
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

  // input to the generator (set once per hit before calling fGenerator)
  double fInputX, fInputY, fInputZ;
  double fInputTime;
  long fInputN;  // number of photons to generate for this hit

  // output from the generator: vectors of length fInputN, one entry per photon
  std::vector<double> fOutputX;
  std::vector<double> fOutputY;
  std::vector<double> fOutputDX;
  std::vector<double> fOutputDY;
  std::vector<double> fOutputDZ;
  std::vector<double> fOutputEkine;
  std::vector<double> fOutputTime;

protected:
  void DigitInitialize(
      const std::vector<std::string> &attributes_not_in_filler) override;

  GeneratorType fGenerator;
  std::string fPositionAttributeName;

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
    // monotonic count of hits processed so far on this thread, across the
    // whole run (not reset per event); used as SourceHitIndex so that hits
    // from different events never alias onto the same index.
    long nHitsProcessed{0};
  };
  G4Cache<threadLocalT> fThreadLocalData;
};

#endif // GateDigitizerOpticalGenerativeActor_h
