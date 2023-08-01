#include <iostream>
#include "xla/service/platform_util.h"
#include "xla/stream_executor/tpu/tpu_platform.h"

using namespace std;

int main(int argc, char *argv[]) {
  cout << "TPU: " << tensorflow::tpu::RegisterTpuPlatform() << endl;
  auto tpu = tensorflow::tpu::TpuPlatform::GetRegisteredPlatform();
  int num_tpu_devices;
  if (tpu) {
    auto version = tpu->version();
    cout << "\tVersion: " << version.version[0] << "." << version.version[1] << "." << version.version[2] << endl;
    cout << "\tName: " << tpu->Name() << endl;
    cout << "\tDevice Count: " << tpu->VisibleDeviceCount() << endl;
  } else {
    cout << "\tNo platform registered." << endl;
  }




  auto platforms_or = xla::PlatformUtil::GetSupportedPlatforms();
  if (!platforms_or.ok()) {
    cerr << "Failed: " << platforms_or.status() << endl;
    return 1;
  }
  cout << "Platforms:" << endl;
  for (auto &platform : platforms_or.value()) {
    cout << "\t" << platform->Name() << ": " << platform->VisibleDeviceCount() << " devices" << endl;
  }

}
