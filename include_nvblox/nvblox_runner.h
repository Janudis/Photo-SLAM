#pragma once

namespace photoslam {

enum class NvbloxDataset {
    kTum,
    kReplica,
};

int runRgbdNvblox(NvbloxDataset dataset, int argc, char** argv);

}  // namespace photoslam
