/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateDigitizerOpticalGenerativeActor.h"
#include "../GateHelpersDict.h"
#include "GateDigiCollectionManager.h"
#include "GateHelpersDigitizer.h"
#include "GateTDigiAttribute.h"
#include <G4Poisson.hh>
#include <cmath>

GateDigitizerOpticalGenerativeActor::GateDigitizerOpticalGenerativeActor(
    py::dict &user_info)
    : GateVDigitizerWithOutputActor(user_info, true) {
  fActions.insert("EndOfEventAction");
}

GateDigitizerOpticalGenerativeActor::~GateDigitizerOpticalGenerativeActor() =
    default;

void GateDigitizerOpticalGenerativeActor::InitializeUserInfo(
    py::dict &user_info) {
  GateVDigitizerWithOutputActor::InitializeUserInfo(user_info);
}

void GateDigitizerOpticalGenerativeActor::SetGeneratorFunction(
    GeneratorType &f) {
  fGenerator = f;
}

void GateDigitizerOpticalGenerativeActor::StartSimulationAction() {
  // Output schema is fully custom (synthetic optical photons), it is not a
  // copy of the input attributes, so we do not call
  // InitDigiAttributesFromCopy. We build StartSimulationAction "by hand"
  // (instead of GateVDigitizerWithOutputActor::StartSimulationAction) to
  // avoid the input-copy step.
  auto *hcm = GateDigiCollectionManager::GetInstance();
  fInputDigiCollection = hcm->GetDigiCollection(fInputDigiCollectionName);

  fOutputDigiCollection = hcm->NewDigiCollection(fOutputDigiCollectionName);
  std::string outputPath;
  if (!GetWriteToDisk(fOutputNameRoot)) {
    outputPath = "";
  } else {
    outputPath = GetOutputPath(fOutputNameRoot);
  }
  fOutputDigiCollection->SetFilenameAndInitRoot(outputPath);

  fOutputDigiCollection->InitDigiAttribute(new GateTDigiAttribute<double>("X"));
  fOutputDigiCollection->InitDigiAttribute(new GateTDigiAttribute<double>("Y"));
  fOutputDigiCollection->InitDigiAttribute(
      new GateTDigiAttribute<double>("dX"));
  fOutputDigiCollection->InitDigiAttribute(
      new GateTDigiAttribute<double>("dY"));
  fOutputDigiCollection->InitDigiAttribute(
      new GateTDigiAttribute<double>("dZ"));
  fOutputDigiCollection->InitDigiAttribute(
      new GateTDigiAttribute<double>("Ekine"));
  fOutputDigiCollection->InitDigiAttribute(
      new GateTDigiAttribute<double>("Time"));
  fOutputDigiCollection->InitDigiAttribute(
      new GateTDigiAttribute<double>("SourceHitIndex"));

  fOutputDigiCollection->RootInitializeTupleForMaster();

  // check required attributes on the input (raw Hits) collection
  CheckRequiredAttribute(fInputDigiCollection, "TotalEnergyDeposit");
  CheckRequiredAttribute(fInputDigiCollection, "PostPositionLocal");
  CheckRequiredAttribute(fInputDigiCollection, "GlobalTime");
}

void GateDigitizerOpticalGenerativeActor::DigitInitialize(
    const std::vector<std::string> &attributes_not_in_filler) {
  // No filler is needed: the output schema does not mirror the input
  // schema (1 input row can produce N output rows), so we only set up the
  // input iterator and the output attribute pointers ourselves.
  (void)attributes_not_in_filler;

  // Must call this here since we bypass GateVDigitizerWithOutputActor::DigitInitialize
  fOutputDigiCollection->RootInitializeTupleForWorker();

  fOutputXAttribute = fOutputDigiCollection->GetDigiAttribute("X");
  fOutputYAttribute = fOutputDigiCollection->GetDigiAttribute("Y");
  fOutputDXAttribute = fOutputDigiCollection->GetDigiAttribute("dX");
  fOutputDYAttribute = fOutputDigiCollection->GetDigiAttribute("dY");
  fOutputDZAttribute = fOutputDigiCollection->GetDigiAttribute("dZ");
  fOutputEkineAttribute = fOutputDigiCollection->GetDigiAttribute("Ekine");
  fOutputTimeAttribute = fOutputDigiCollection->GetDigiAttribute("Time");
  fOutputSourceHitIndexAttribute =
      fOutputDigiCollection->GetDigiAttribute("SourceHitIndex");

  auto &lr = fThreadLocalVDigitizerData.Get();
  lr.fInputIter = fInputDigiCollection->NewIterator();
  auto &l = fThreadLocalData.Get();
  lr.fInputIter.TrackAttribute("TotalEnergyDeposit", &l.edep);
  lr.fInputIter.TrackAttribute("PostPositionLocal", &l.pos);
  lr.fInputIter.TrackAttribute("GlobalTime", &l.time);
}

void GateDigitizerOpticalGenerativeActor::EndOfEventAction(
    const G4Event * /*event*/) {
  auto &lr = fThreadLocalVDigitizerData.Get();
  auto &iter = lr.fInputIter;
  auto &l = fThreadLocalData.Get();
  iter.GoToBegin();

  while (!iter.IsAtEnd()) {
    if (*l.edep > 0) {
      // sample N from Poisson(edep * scintillation_yield)
      const long N =
          G4Poisson((*l.edep) * fScintillationYield);

      if (N > 0) {
        // set position and time inputs (same for all N photons of this hit)
        fInputX = l.pos->x();
        fInputY = l.pos->y();
        fInputZ = l.pos->z();
        fInputTime = *l.time;

        const auto sourceHitIndex = static_cast<double>(iter.fIndex);
        for (long i = 0; i < N; ++i) {
          // generator fills the scalar fOutput* fields for one photon
          fGenerator(this);
          fOutputXAttribute->FillDValue(fOutputX);
          fOutputYAttribute->FillDValue(fOutputY);
          fOutputDXAttribute->FillDValue(fOutputDX);
          fOutputDYAttribute->FillDValue(fOutputDY);
          fOutputDZAttribute->FillDValue(fOutputDZ);
          fOutputEkineAttribute->FillDValue(fOutputEkine);
          // LogTime from the model is converted to linear time before storage
          fOutputTimeAttribute->FillDValue(std::exp(fOutputLogTime));
          fOutputSourceHitIndexAttribute->FillDValue(sourceHitIndex);
        }
      }
    }
    iter++;
  }
}
