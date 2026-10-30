#pragma once

#include "SetepDistribution.h"
#include <string>

namespace setep {
// Write PREFIX.<rank>.et Chakra execution traces (one per occupied processor)
// and the PREFIX.chakra.json manifest. Files are staged and published together;
// stale rank files of an earlier, larger export are removed.
void writeChakra(const std::string& prefix, const Graph& graph, const DistributionPlan& plan);
}
