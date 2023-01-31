#include <string>
#include <utility>

#include "tensorflow/lite/delegates/utils/toy_mult_accelerator/sim_delegate/toy_mult_delegate.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

class ToyMultDelegateProvider : public DelegateProvider {
 public:
  ToyMultDelegateProvider() {
    default_params_.AddParam("use_toy_mult_delegate",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "ToyMultDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(ToyMultDelegateProvider);

std::vector<Flag> ToyMultDelegateProvider::CreateFlags(ToolParams* params) const {
  std::vector<Flag> flags = {CreateFlag<bool>("use_toy_mult_delegate", params,
                                              "use the toy mult delegate.")};
  return flags;
}

void ToyMultDelegateProvider::LogParams(const ToolParams& params,
                                      bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_toy_mult_delegate", "Use toy test mult delegate",
                 verbose);
}

TfLiteDelegatePtr ToyMultDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_toy_mult_delegate")) {
    auto default_options = TfLiteToyMultDelegateOptionsDefault();
    return TfLiteToyMultDelegateCreateUnique(&default_options);
  }
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

std::pair<TfLiteDelegatePtr, int>
ToyMultDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_toy_mult_delegate"));
}
}  // namespace tools
}  // namespace tflite
