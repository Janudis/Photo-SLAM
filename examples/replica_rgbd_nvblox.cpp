#include "include_nvblox/nvblox_runner.h"

int main(int argc, char** argv)
{
    return photoslam::runRgbdNvblox(
        photoslam::NvbloxDataset::kReplica, argc, argv);
}
