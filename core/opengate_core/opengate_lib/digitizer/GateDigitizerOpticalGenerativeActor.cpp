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
  CheckRequiredAttribute(fInputDigiCollection, "GlobalTime");
  // accept either crystal-local or module-local position
  if (fInputDigiCollection->IsDigiAttributeExists("PostPositionLocalModule")) {
    fPositionAttributeName = "PostPositionLocalModule";
  } else {
    CheckRequiredAttribute(fInputDigiCollection, "PostPositionLocal");
    fPositionAttributeName = "PostPositionLocal";
  }
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
  lr.fInputIter.TrackAttribute(fPositionAttributeName, &l.pos);
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

      // monotonic across the whole run (unlike iter.fIndex, which is
      // relative to the input digi collection's current buffer and gets
      // reused across events once that collection is flushed/cleared)
      const auto sourceHitIndex = static_cast<double>(l.nHitsProcessed);
      l.nHitsProcessed++;

      if (N > 0) {
        // set inputs; generator is called once per hit with batch size N
        fInputX = l.pos->x();
        fInputY = l.pos->y();
        fInputZ = l.pos->z();
        fInputTime = *l.time;
        fInputN = N;

        fGenerator(this);  // fills fOutputX/Y/DX/DY/DZ/Ekine/Time as vectors

        for (long i = 0; i < N; ++i) {
          fOutputXAttribute->FillDValue(fOutputX[i]);
          fOutputYAttribute->FillDValue(fOutputY[i]);
          fOutputDXAttribute->FillDValue(fOutputDX[i]);
          fOutputDYAttribute->FillDValue(fOutputDY[i]);
          fOutputDZAttribute->FillDValue(fOutputDZ[i]);
          fOutputEkineAttribute->FillDValue(fOutputEkine[i]);
          fOutputTimeAttribute->FillDValue(fOutputTime[i]);
          fOutputSourceHitIndexAttribute->FillDValue(sourceHitIndex);
        }
      }
    }
    iter++;
  }
}
