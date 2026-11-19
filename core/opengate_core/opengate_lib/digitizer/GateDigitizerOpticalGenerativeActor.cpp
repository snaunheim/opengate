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
#include <CLHEP/Random/RandBinomial.h>
#include <G4Poisson.hh>
#include <Randomize.hh>
#include <algorithm>
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

  fOutputAttributeNames = DictGetVecStr(user_info, "_output_attribute_names");
  fOutputComponents = DictGetVecStr(user_info, "_output_components");
  fOutputColumnIndices = DictGetVecInt(user_info, "_output_column_indices");
  fOutputConstants = DictGetVecDouble(user_info, "_output_constants");
  fOutputRelativeToHit = DictGetVecInt(user_info, "_output_relative_to_hit");
  fOutputDerived = DictGetVecInt(user_info, "_output_derived");

  fTargetBatch = DictGetInt(user_info, "_target_batch");
  fPooling = fTargetBatch > 0;
  fFlushEveryEvent = DictGetBool(user_info, "flush_every_event");
  // python allows only SourceHitIndex in skip_attributes
  fWriteSourceHitIndex =
      std::find(fUserSkipDigiAttributeNames.begin(),
                fUserSkipDigiAttributeNames.end(),
                "SourceHitIndex") == fUserSkipDigiAttributeNames.end();

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

void GateDigitizerOpticalGenerativeActor::SetEfficiencyMap(
    const std::vector<double> &axisX, const std::vector<double> &axisY,
    const std::vector<double> &axisZ, const std::vector<double> &values,
    const std::vector<bool> &nearest) {
  // python checks the map already, this only guards the lookup
  fEfficiencyAxes = {axisX, axisY, axisZ};
  size_t n = 1;
  for (const auto &axis : fEfficiencyAxes) {
    if (axis.empty() || !std::is_sorted(axis.begin(), axis.end()))
      Fatal("GateDigitizerOpticalGenerativeActor: every axis of the "
            "efficiency map needs at least one value, in increasing order.");
    n *= axis.size();
  }
  if (values.size() != n || nearest.size() != 3)
    Fatal("GateDigitizerOpticalGenerativeActor: the efficiency map does not "
          "match the size of its axes.");
  fEfficiencyValues = values;
  for (size_t i = 0; i < 3; i++)
    fEfficiencyNearest[i] = nearest[i];
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
  if (fWriteSourceHitIndex)
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

  // the attributes belong to the shared collection, so the schema is resolved
  // by the first worker and only read afterwards
  std::lock_guard<std::mutex> guard(fSchemaMutex);
  if (fSchemaResolved) {
    SetUpInputIterator();
    return;
  }
  fSchemaResolved = true;

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
      fWriteSourceHitIndex
          ? fOutputDigiCollection->GetDigiAttribute("SourceHitIndex")
          : nullptr;

  // every 'I' target reads its value through a slot of its own
  for (size_t i = 0; i < fFromHitTargets.size(); i++) {
    auto &target = fFromHitTargets[i];
    if (target.type == 'I') {
      target.intSlot = static_cast<int>(i);
    } else if (target.type != 'U') {
      std::ostringstream oss;
      oss << "GateDigitizerOpticalGenerativeActor: the attribute '"
          << target.attribute->GetDigiAttributeName()
          << "' is declared as derived from the hit, but its type '"
          << target.type << "' cannot be copied.";
      Fatal(oss.str());
    }
  }

  SetUpInputIterator();
}

void GateDigitizerOpticalGenerativeActor::SetUpInputIterator() {
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
    const auto &target = fFromHitTargets[i];
    if (target.type == 'I')
      lr.fInputIter.TrackAttribute(target.attribute->GetDigiAttributeName(),
                                   &l.fromHitInts[i]);
  }
}

void GateDigitizerOpticalGenerativeActor::EndOfEventAction(
    const G4Event * /*event*/) {
  auto &lr = fThreadLocalVDigitizerData.Get();
  auto &iter = lr.fInputIter;
  auto &l = fThreadLocalData.Get();
  if (l.threadIndexOffset < 0)
    l.threadIndexOffset = fNextThreadIndex.fetch_add(1) * fSourceHitIndexStride;
  iter.GoToBegin();

  while (!iter.IsAtEnd()) {
    if (*l.edep > 0) {
      long N = SamplePhotonCount(*l.edep);
      // only the photons that reach the sensor are generated
      if (N > 0 && !fEfficiencyValues.empty())
        N = ReducePhotonCount(N, *l.pos);

      const auto sourceHitIndex =
          static_cast<double>(l.threadIndexOffset + l.nHitsProcessed);
      l.nHitsProcessed++;

      if (N > 0 && fPooling) {
        PoolHit(l, N, sourceHitIndex);
      } else if (N > 0) {
        // set inputs; generator is called once per hit with batch size N
        auto &io = GetGeneratorIO();
        io.x = l.pos->x();
        io.y = l.pos->y();
        io.z = l.pos->z();
        io.time = *l.time;
        io.n = N;

        {
          // the generator is python, pybind's std::function wrapper takes
          // the gil itself (pybind11/functional.h)
          // py::gil_scoped_acquire acquire;
          fGenerator(this); // fills io.columns, one vector per model column
        }

        const long nRows = CheckOutputColumns(N);

        PooledHit hit;
        hit.hitTime = *l.time;
        hit.sourceHitIndex = sourceHitIndex;
        if (fNeedsVolumeID)
          hit.volumeID = *l.volumeID;
        if (!fWorldFromModuleAttributes.empty())
          hit.worldFromModule = ResolveWorldFromModule(hit.volumeID);
        hit.fromHitInts.resize(fFromHitTargets.size());
        for (size_t i = 0; i < fFromHitTargets.size(); i++)
          if (fFromHitTargets[i].type == 'I')
            hit.fromHitInts[i] = *l.fromHitInts[fFromHitTargets[i].intSlot];

        FillPhotons(hit, 0, nRows);
      }
    }
    iter++;
  }

  // nothing of this event may wait in the pool for a digitizer reading it
  if (fPooling && fFlushEveryEvent)
    FlushBatch(l, l.pooledPhotons);
}

long GateDigitizerOpticalGenerativeActor::SamplePhotonCount(double edep) const {
  // same as G4Scintillation, so the count fluctuates like in the full sim
  const double mean = edep * fScintillationYield;
  if (mean > 10.) {
    const double sigma = fResolutionScale * std::sqrt(mean);
    return static_cast<long>(G4RandGauss::shoot(mean, sigma) + 0.5);
  }
  return static_cast<long>(G4Poisson(mean));
}

namespace {
// the two grid points around v on one axis and the weight of the second.
// nearest gives one point, beyond the ends the end point is used.
void AxisNeighbours(const std::vector<double> &axis, double v, bool nearest,
                    size_t &i0, size_t &i1, double &w1) {
  w1 = 0.;
  if (v <= axis.front()) {
    i0 = i1 = 0;
    return;
  }
  if (v >= axis.back()) {
    i0 = i1 = axis.size() - 1;
    return;
  }
  i1 = std::upper_bound(axis.begin(), axis.end(), v) - axis.begin();
  i0 = i1 - 1;
  const double t = (v - axis[i0]) / (axis[i1] - axis[i0]);
  if (nearest)
    i0 = i1 = (t < 0.5) ? i0 : i1;
  else
    w1 = t;
}
} // namespace

double GateDigitizerOpticalGenerativeActor::EfficiencyAt(
    const G4ThreeVector &position) const {
  size_t i[3][2];
  double w[3];
  for (int a = 0; a < 3; a++)
    AxisNeighbours(fEfficiencyAxes[a], position[a], fEfficiencyNearest[a],
                   i[a][0], i[a][1], w[a]);
  const auto ny = fEfficiencyAxes[1].size();
  const auto nz = fEfficiencyAxes[2].size();
  double efficiency = 0.;
  for (int cx = 0; cx < 2; cx++)
    for (int cy = 0; cy < 2; cy++)
      for (int cz = 0; cz < 2; cz++) {
        const double weight = (cx ? w[0] : 1. - w[0]) *
                              (cy ? w[1] : 1. - w[1]) *
                              (cz ? w[2] : 1. - w[2]);
        if (weight > 0.)
          efficiency +=
              weight * fEfficiencyValues[(i[0][cx] * ny + i[1][cy]) * nz +
                                         i[2][cz]];
      }
  return efficiency;
}

long GateDigitizerOpticalGenerativeActor::ReducePhotonCount(
    long n, const G4ThreeVector &position) const {
  // each photon arrives on its own, so the count is binomial
  const double p = EfficiencyAt(position);
  if (p <= 0.)
    return 0;
  if (p >= 1.)
    return n;
  return std::lround(
      CLHEP::RandBinomial::shoot(G4Random::getTheEngine(), n, p));
}

long GateDigitizerOpticalGenerativeActor::CheckOutputColumns(long nAsked) {
  const auto &columns = GetGeneratorIO().columns;
  // The manifest does not state the number of features the model actually
  // returns. A mismatch shifts the mapping, therefore stop.
  if (static_cast<int>(columns.size()) != fNumberOfModelColumns) {
    std::ostringstream oss;
    oss << "GateDigitizerOpticalGenerativeActor: the generator returned "
        << columns.size() << " columns, but the declared output "
        << "schema expects " << fNumberOfModelColumns
        << ". Check the 'outputs' list of the bundle manifest against "
           "what the model returns.";
    Fatal(oss.str());
  }

  // Column length between the features has to be the same, but can be lower
  // than the N that were requested, because of geometric efficiency etc.
  const long nRows =
      columns.empty() ? 0 : static_cast<long>(columns.front().size());
  for (const auto &column : columns) {
    if (static_cast<long>(column.size()) != nRows) {
      std::ostringstream oss;
      oss << "GateDigitizerOpticalGenerativeActor: the generator "
          << "returned columns of differing lengths (" << nRows << " and "
          << column.size() << "). Every column must have one entry per photon.";
      Fatal(oss.str());
    }
  }
  if (nRows > nAsked) {
    std::ostringstream oss;
    oss << "GateDigitizerOpticalGenerativeActor: the generator was asked "
        << "for " << nAsked << " photons but returned " << nRows
        << " rows. It may return fewer than requested, never more.";
    Fatal(oss.str());
  }
  return nRows;
}

const G4AffineTransform *
GateDigitizerOpticalGenerativeActor::ResolveWorldFromModule(
    const GateUniqueVolumeID::Pointer &volumeID) const {
  // the module is the parent of the crystal the hit is in
  const auto depth = volumeID->fTouchable.GetDepth();
  const auto parentIdx = (depth >= 1) ? (G4int)depth - 1 : 0;
  return &volumeID->fTouchable.GetTransform(parentIdx);
}

void GateDigitizerOpticalGenerativeActor::PoolHit(threadLocalT &l,
                                                  long nPhotons,
                                                  double sourceHitIndex) {
  PooledHit hit;
  hit.nPending = nPhotons;
  hit.position = *l.pos;
  hit.hitTime = *l.time;
  hit.sourceHitIndex = sourceHitIndex;
  if (fNeedsVolumeID)
    hit.volumeID = *l.volumeID;
  if (!fWorldFromModuleAttributes.empty())
    hit.worldFromModule = ResolveWorldFromModule(hit.volumeID);
  hit.fromHitInts.resize(fFromHitTargets.size());
  for (size_t i = 0; i < fFromHitTargets.size(); i++)
    if (fFromHitTargets[i].type == 'I')
      hit.fromHitInts[i] = *l.fromHitInts[fFromHitTargets[i].intSlot];

  l.pool.push_back(std::move(hit));
  l.pooledPhotons += nPhotons;

  // a big hit can fill more than one batch
  while (l.pooledPhotons >= fTargetBatch)
    FlushBatch(l, fTargetBatch);
}

void GateDigitizerOpticalGenerativeActor::FlushBatch(threadLocalT &l, long n) {
  if (n <= 0)
    return;

  // the generator gets the position and time of each photon's hit, the last
  // hit may only partly fit and keeps its rest for the next call
  auto &io = GetGeneratorIO();
  io.xs.clear();
  io.ys.clear();
  io.zs.clear();
  io.times.clear();
  long left = n;
  for (const auto &hit : l.pool) {
    if (left == 0)
      break;
    const long k = std::min(left, hit.nPending);
    io.xs.insert(io.xs.end(), k, hit.position.x());
    io.ys.insert(io.ys.end(), k, hit.position.y());
    io.zs.insert(io.zs.end(), k, hit.position.z());
    io.times.insert(io.times.end(), k, hit.hitTime);
    left -= k;
  }

  {
    // see EndOfEventAction, pybind takes the gil itself
    // py::gil_scoped_acquire acquire;
    fGenerator(this);  // reads the pooled input, fills the columns
  }
  const long nRows = CheckOutputColumns(n);
  if (nRows != n) {
    // a pooled call has no way to tell which hit a dropped photon came from,
    // so the model must return one row per entry it was given
    std::ostringstream oss;
    oss << "GateDigitizerOpticalGenerativeActor: the generator was given " << n
        << " pooled photons but returned " << nRows
        << " rows. In pooled mode it must return exactly one row per photon.";
    Fatal(oss.str());
  }

  long offset = 0;
  while (offset < n) {
    auto &hit = l.pool.front();
    const long k = std::min(n - offset, hit.nPending);
    FillPhotons(hit, offset, k);
    offset += k;
    hit.nPending -= k;
    if (hit.nPending == 0)
      l.pool.pop_front();
  }
  l.pooledPhotons -= n;
}

void GateDigitizerOpticalGenerativeActor::FillPhotons(const PooledHit &hit,
                                                      long offset,
                                                      long nRows) {
  const auto &columns = GetGeneratorIO().columns;
  for (long i = offset; i < offset + nRows; ++i) {
    // kept so the world coordinates can be computed from it
    G4ThreeVector moduleLocalPosition;
    for (size_t t = 0; t < fOutputTargets.size(); t++) {
      const auto &target = fOutputTargets[t];
      if (target.isVector) {
        double v[3];
        for (int c = 0; c < 3; c++) {
          const auto &s = target.components[c];
          v[c] = s.columnIndex < 0 ? s.constant
                                   : columns[s.columnIndex][i];
          if (s.relativeToHit)
            v[c] += hit.hitTime;
        }
        const G4ThreeVector value(v[0], v[1], v[2]);
        if (static_cast<int>(t) == fModuleLocalTargetIndex)
          moduleLocalPosition = value;
        target.attribute->Fill3Value(value);
      } else {
        const auto &s = target.scalar;
        double v =
            s.columnIndex < 0 ? s.constant : columns[s.columnIndex][i];
        if (s.relativeToHit)
          v += hit.hitTime;
        target.attribute->FillDValue(v);
      }
    }
    if (hit.worldFromModule != nullptr) {
      const auto worldPosition =
          hit.worldFromModule->InverseTransformPoint(moduleLocalPosition);
      for (auto *attribute : fWorldFromModuleAttributes)
        attribute->Fill3Value(worldPosition);
    }
    // every photon of this hit repeats the hit's own value
    for (size_t k = 0; k < fFromHitTargets.size(); k++) {
      const auto &target = fFromHitTargets[k];
      if (target.type == 'U')
        target.attribute->FillUValue(hit.volumeID);
      else
        target.attribute->FillIValue(hit.fromHitInts[k]);
    }
    if (fOutputSourceHitIndexAttribute != nullptr)
      fOutputSourceHitIndexAttribute->FillDValue(hit.sourceHitIndex);
  }
}

void GateDigitizerOpticalGenerativeActor::EndOfRunAction(const G4Run *run) {
  // whatever is still pooled belongs to this run
  if (fPooling) {
    auto &l = fThreadLocalData.Get();
    FlushBatch(l, l.pooledPhotons);
  }
  GateVDigitizerWithOutputActor::EndOfRunAction(run);
}
