/* --------------------------------------------------
   Copyright (C): OpenGATE Collaboration
   This software is distributed under the terms
   of the GNU Lesser General  Public Licence (LGPL)
   See LICENSE.md for further details
   -------------------------------------------------- */

#include "GateDigitizerOpticalGenerativeActor.h"
#include "../GateHelpers.h"
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

  fOutputAttributeNames = DictGetVecStr(user_info, "_output_attribute_names");
  fOutputComponents = DictGetVecStr(user_info, "_output_components");
  fOutputColumnIndices = DictGetVecInt(user_info, "_output_column_indices");
  fOutputConstants = DictGetVecDouble(user_info, "_output_constants");
  fOutputRelativeToHit = DictGetVecInt(user_info, "_output_relative_to_hit");
  fOutputDerived = DictGetVecInt(user_info, "_output_derived");

  const auto n = fOutputAttributeNames.size();
  if (fOutputComponents.size() != n || fOutputColumnIndices.size() != n ||
      fOutputConstants.size() != n || fOutputRelativeToHit.size() != n ||
      fOutputDerived.size() != n) {
    Fatal("GateDigitizerOpticalGenerativeActor: the output schema vectors "
          "have inconsistent lengths.");
  }

  fNumberOfModelColumns = 0;
  for (const auto index : fOutputColumnIndices)
    if (index >= 0)
      fNumberOfModelColumns++;

  fNeedsVolumeID = false;
  fFromHitAttributeNames.clear();
  for (size_t i = 0; i < fOutputDerived.size(); i++) {
    if (fOutputDerived[i] == fDerivedWorldFromModule)
      fNeedsVolumeID = true;
    else if (fOutputDerived[i] == fDerivedFromHit)
      fFromHitAttributeNames.push_back(fOutputAttributeNames[i]);
  }
  // the module-local transform and a copied PreStepUniqueVolumeID read the
  // same input attribute
  for (const auto &name : fFromHitAttributeNames)
    if (name == "PreStepUniqueVolumeID")
      fNeedsVolumeID = true;
}

void GateDigitizerOpticalGenerativeActor::SetGeneratorFunction(
    GeneratorType &f) {
  fGenerator = f;
}

void GateDigitizerOpticalGenerativeActor::StartSimulationAction() {
  // Cannot use GateVDigitizerWithOutputActor::StartSimulationAction because it would
  // copy the input attributes into the output collection. For this actor another scheme
  // is needed since one input hit produces N synthetic particles with different attributes

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

  // One branch per distinct declared attribute, with the type GATE itself gives
  // it (several slots may share one attribute, as the components of a 3-vector do).
  for (const auto &name : fOutputAttributeNames) {
    if (!fOutputDigiCollection->IsDigiAttributeExists(name))
      fOutputDigiCollection->InitDigiAttributeFromName(name);
  }
  // Not part of the schema: the actor produces it itself.
  fOutputDigiCollection->InitDigiAttribute(
      new GateTDigiAttribute<double>("SourceHitIndex"));

  fOutputDigiCollection->RootInitializeTupleForMaster();
  // Since we do not call GateVDigitizerWithOutputActor::StartSimulationAction,
  // register the output tree info here
  AddOutputTreeInfo(fOutputNameRoot, fOutputDigiCollection);

  // check required attributes on the input (raw Hits) collection
  CheckRequiredAttribute(fInputDigiCollection, "TotalEnergyDeposit");
  CheckRequiredAttribute(fInputDigiCollection, "GlobalTime");
  // The exchanged reference frame is module-local (origin at the module iso-center)
  CheckRequiredAttribute(fInputDigiCollection, "PostPositionLocalModule");
  // Only needed to invert the module-local transform for a derived output; a
  // manifest that declares none works without it.
  if (fNeedsVolumeID)
    CheckRequiredAttribute(fInputDigiCollection, "PreStepUniqueVolumeID");
  // A "from_hit" output copies its value straight off the input, so fail here
  // rather than on the first hit of the run.
  for (const auto &name : fFromHitAttributeNames)
    CheckRequiredAttribute(fInputDigiCollection, name);
}

void GateDigitizerOpticalGenerativeActor::DigitInitialize(
    const std::vector<std::string> &attributes_not_in_filler) {
  // No filler is needed
  (void)attributes_not_in_filler;

  // Must call this here since we bypass GateVDigitizerWithOutputActor::DigitInitialize
  fOutputDigiCollection->RootInitializeTupleForWorker();

  // Solving the declared scheme. Slots are grouped by attribute, 
  // such that the components of a vector are together.
  // EndOfEventAction then just goes through the list and calls Fill3Value/FillDValue
  fOutputTargets.clear();
  fWorldFromModuleAttributes.clear();
  fFromHitTargets.clear();
  for (size_t i = 0; i < fOutputAttributeNames.size(); i++) {
    const auto &name = fOutputAttributeNames[i];
    auto *att = fOutputDigiCollection->GetDigiAttribute(name);

    if (fOutputDerived[i] == fDerivedWorldFromModule) {
      // computed from the module-local position, not from a model column
      fWorldFromModuleAttributes.push_back(att);
      continue;
    }
    if (fOutputDerived[i] == fDerivedFromHit) {
      // read off the input hit below, once the iterator exists
      fFromHitTargets.push_back(
          FromHitTarget{att, att->GetDigiAttributeType(), -1});
      continue;
    }

    OutputSource source;
    source.columnIndex = fOutputColumnIndices[i];
    source.constant = fOutputConstants[i];
    source.relativeToHit = fOutputRelativeToHit[i] != 0;

    OutputTarget *target = nullptr;
    for (auto &t : fOutputTargets) {
      if (t.attribute == att) {
        target = &t;
        break;
      }
    }
    if (target == nullptr) {
      fOutputTargets.emplace_back();
      target = &fOutputTargets.back();
      target->attribute = att;
      target->isVector = att->GetDigiAttributeType() == '3';
    }

    if (target->isVector) {
      const auto &component = fOutputComponents[i];
      const int c = component == "x" ? 0 : (component == "y" ? 1 : 2);
      target->components[c] = source;
    } else {
      target->scalar = source;
    }
  }

  // Look up which target holds the module-local position, compare an int instead of a name.
  fModuleLocalTargetIndex = -1;
  for (size_t i = 0; i < fOutputTargets.size(); i++) {
    if (fOutputTargets[i].attribute->GetDigiAttributeName() ==
        fModuleLocalAttributeName) {
      fModuleLocalTargetIndex = static_cast<int>(i);
      break;
    }
  }
  if (!fWorldFromModuleAttributes.empty() && fModuleLocalTargetIndex < 0) {
    std::ostringstream oss;
    oss << "GateDigitizerOpticalGenerativeActor: an output is derived from '"
        << fModuleLocalAttributeName
        << "', but that attribute is not part of the output schema.";
    Fatal(oss.str());
  }

  fOutputSourceHitIndexAttribute =
      fOutputDigiCollection->GetDigiAttribute("SourceHitIndex");

  auto &lr = fThreadLocalVDigitizerData.Get();
  lr.fInputIter = fInputDigiCollection->NewIterator();
  auto &l = fThreadLocalData.Get();
  lr.fInputIter.TrackAttribute("TotalEnergyDeposit", &l.edep);
  lr.fInputIter.TrackAttribute("PostPositionLocalModule", &l.pos);
  lr.fInputIter.TrackAttribute("GlobalTime", &l.time);
  if (fNeedsVolumeID)
    lr.fInputIter.TrackAttribute("PreStepUniqueVolumeID", &l.volumeID);

  // The iterator keeps one pointer per tracked attribute up to date, so every
  // 'I' target needs a slot of its own. Sized up front, since TrackAttribute
  // stores the address of an element.
  l.fromHitInts.assign(fFromHitTargets.size(), nullptr);
  for (size_t i = 0; i < fFromHitTargets.size(); i++) {
    auto &target = fFromHitTargets[i];
    const auto name = target.attribute->GetDigiAttributeName();
    if (target.type == 'U') {
      // PreStepUniqueVolumeID, already tracked above as the only 'U' case
    } else if (target.type == 'I') {
      target.intSlot = static_cast<int>(i);
      lr.fInputIter.TrackAttribute(name, &l.fromHitInts[i]);
    } else {
      std::ostringstream oss;
      oss << "GateDigitizerOpticalGenerativeActor: the attribute '" << name
          << "' is declared as derived from the hit, but its type '"
          << target.type << "' cannot be copied.";
      Fatal(oss.str());
    }
  }
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
      const long N = G4Poisson((*l.edep) * fScintillationYield);

      const auto sourceHitIndex = static_cast<double>(l.nHitsProcessed);
      l.nHitsProcessed++;

      if (N > 0) {
        // set inputs; generator is called once per hit with batch size N
        fInputX = l.pos->x();
        fInputY = l.pos->y();
        fInputZ = l.pos->z();
        fInputTime = *l.time;
        fInputN = N;

        fGenerator(this); // fills fOutputColumns, one vector per model column

        // The manifest does not state the number of features the model actually
        // returns. A mismatch shifts the mapping, therefore stop.
        if (static_cast<int>(fOutputColumns.size()) != fNumberOfModelColumns) {
          std::ostringstream oss;
          oss << "GateDigitizerOpticalGenerativeActor: the generator returned "
              << fOutputColumns.size() << " columns, but the declared output "
              << "schema expects " << fNumberOfModelColumns
              << ". Check the 'outputs' list of the bundle manifest against "
                 "what the model returns.";
          Fatal(oss.str());
        }

        // Column length between the features has to be the same, but can be lower than the N
        // that were requested, because of geometric efficiency etc.
        const long nRows =
            fOutputColumns.empty()
                ? 0
                : static_cast<long>(fOutputColumns.front().size());
        for (const auto &column : fOutputColumns) {
          if (static_cast<long>(column.size()) != nRows) {
            std::ostringstream oss;
            oss << "GateDigitizerOpticalGenerativeActor: the generator "
                << "returned columns of differing lengths (" << nRows << " and "
                << column.size()
                << "). Every column must have one entry per photon.";
            Fatal(oss.str());
          }
        }
        if (nRows > N) {
          std::ostringstream oss;
          oss << "GateDigitizerOpticalGenerativeActor: the generator was asked "
              << "for " << N << " photons but returned " << nRows
              << " rows. It may return fewer than requested, never more.";
          Fatal(oss.str());
        }

        // Transforms module frame back to the world, Resolved once per hit.
        const G4AffineTransform *worldFromModule = nullptr;
        if (!fWorldFromModuleAttributes.empty()) {
          const auto &volumeID = *l.volumeID;
          const auto depth = volumeID->fTouchable.GetDepth();
          const auto parentIdx = (depth >= 1) ? (G4int)depth - 1 : 0;
          worldFromModule = &volumeID->fTouchable.GetTransform(parentIdx);
        }

        const double hitTime = *l.time;
        for (long i = 0; i < nRows; ++i) {
          // keep local module local position so the world coordinates can be computed from it
          G4ThreeVector moduleLocalPosition;
          for (size_t t = 0; t < fOutputTargets.size(); t++) {
            const auto &target = fOutputTargets[t];
            if (target.isVector) {
              double v[3];
              for (int c = 0; c < 3; c++) {
                const auto &s = target.components[c];
                v[c] = s.columnIndex < 0 ? s.constant
                                         : fOutputColumns[s.columnIndex][i];
                if (s.relativeToHit)
                  v[c] += hitTime;
              }
              const G4ThreeVector value(v[0], v[1], v[2]);
              if (static_cast<int>(t) == fModuleLocalTargetIndex)
                moduleLocalPosition = value;
              target.attribute->Fill3Value(value);
            } else {
              const auto &s = target.scalar;
              double v = s.columnIndex < 0 ? s.constant
                                           : fOutputColumns[s.columnIndex][i];
              if (s.relativeToHit)
                v += hitTime;
              target.attribute->FillDValue(v);
            }
          }
          if (worldFromModule != nullptr) {
            const auto worldPosition =
                worldFromModule->InverseTransformPoint(moduleLocalPosition);
            for (auto *attribute : fWorldFromModuleAttributes)
              attribute->Fill3Value(worldPosition);
          }
          // every photon of this hit repeats the hit's own value
          for (const auto &target : fFromHitTargets) {
            if (target.type == 'U')
              target.attribute->FillUValue(*l.volumeID);
            else
              target.attribute->FillIValue(*l.fromHitInts[target.intSlot]);
          }
          fOutputSourceHitIndexAttribute->FillDValue(sourceHitIndex);
        }
      }
    }
    iter++;
  }
}
