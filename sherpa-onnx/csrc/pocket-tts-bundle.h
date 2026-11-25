// Bundle-specific prompt preparation shared by every Pocket TTS binding.
#ifndef SHERPA_ONNX_CSRC_POCKET_TTS_BUNDLE_H_
#define SHERPA_ONNX_CSRC_POCKET_TTS_BUNDLE_H_

#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

struct PocketTtsBundle {
  bool present = false;
  bool remove_semicolons = false;
  bool pad_short_inputs = false;
  int32_t max_tokens = 50;
  int32_t recommended_frames = 0;

  static PocketTtsBundle Parse(const std::vector<char> &bytes) {
    auto json = nlohmann::json::parse(bytes.begin(), bytes.end());
    if (!json.is_object()) throw std::invalid_argument("Invalid Pocket bundle");
    PocketTtsBundle ans;
    ans.present = true;
    ans.remove_semicolons = json.value("remove_semicolons", false);
    ans.pad_short_inputs = json.value("pad_with_spaces_for_short_inputs", false);
    auto positive_int = [&](const char *key, int32_t fallback) {
      if (!json.contains(key) || json[key].is_null()) return fallback;
      if (!json[key].is_number_integer()) {
        throw std::invalid_argument(std::string("Invalid Pocket setting: ") + key);
      }
      auto value = json[key].get<int64_t>();
      if (value < 1 || value > 10000) {
        throw std::invalid_argument(std::string("Invalid Pocket setting: ") + key);
      }
      return static_cast<int32_t>(value);
    };
    ans.max_tokens = positive_int("max_token_per_chunk", 50);
    ans.recommended_frames = positive_int("model_recommended_frames_after_eos", 0);
    return ans;
  }

  std::string Normalize(std::string text) const {
    if (!present) return text;
    std::replace(text.begin(), text.end(), '\n', ' ');
    std::replace(text.begin(), text.end(), '\r', ' ');
    if (remove_semicolons) std::replace(text.begin(), text.end(), ';', ',');
    return text;
  }

  static int32_t Words(const std::string &text) {
    std::istringstream input(text);
    std::string word;
    int32_t count = 0;
    while (input >> word) ++count;
    return count;
  }

  std::string Prepare(const std::string &text) const {
    std::string ans = PreparePocketTtsText(Normalize(text));
    if (!ans.empty() && pad_short_inputs && Words(ans) < 5) {
      ans.insert(0, 8, ' ');
    }
    return ans;
  }

  int32_t FramesAfterEos(const std::string &text) const {
    if (!present) return 3;
    if (recommended_frames) return recommended_frames;
    return Words(text) <= 4 ? 5 : 3;
  }

  // Count the final, prepared prompt, including capitalization and punctuation.
  // Prefer whole words; split an overlong word only at Unicode boundaries.
  std::vector<std::string> Split(
      const std::string &text,
      const std::function<size_t(const std::string &)> &count_tokens) const {
    std::vector<std::string> chunks;
    auto remaining = Utf8ToUtf32(Normalize(text));
    while (!remaining.empty()) {
      auto prepared = Prepare(Utf32ToUtf8(remaining));
      if (prepared.empty()) break;
      if (!present || count_tokens(prepared) <= max_tokens) {
        chunks.push_back(std::move(prepared));
        break;
      }
      size_t fits = 0;
      size_t word_end = 0;
      for (size_t i = 1; i <= remaining.size(); ++i) {
        auto prefix = Prepare(Utf32ToUtf8(remaining.substr(0, i)));
        if (prefix.empty()) continue;
        if (count_tokens(prefix) > max_tokens) break;
        fits = i;
        if (i < remaining.size() && remaining[i] == U' ') word_end = i;
      }
      size_t end = word_end ? word_end : fits;
      if (!end) {
        throw std::invalid_argument("Pocket token limit cannot fit one character");
      }
      chunks.push_back(Prepare(Utf32ToUtf8(remaining.substr(0, end))));
      remaining.erase(0, end);
    }
    return chunks;
  }
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_POCKET_TTS_BUNDLE_H_
