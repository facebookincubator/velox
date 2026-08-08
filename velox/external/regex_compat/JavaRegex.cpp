/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/external/regex_compat/JavaRegex.h"

#if VELOX_REGEX_COMPAT_HAS_JAVA

#include "velox/external/regex_compat/JvmFixture.h"

#include <mutex>
#include <stdexcept>
#include <vector>

namespace facebook::velox::regex_compat {
namespace {

// java.util.regex.Pattern flag bit constants (must match the JDK).
constexpr jint kJavaCaseInsensitive = 0x02;
constexpr jint kJavaMultiline = 0x08;
constexpr jint kJavaDotall = 0x20;
constexpr jint kJavaUnicodeCase = 0x40;

struct JavaIds {
  // Global refs to class objects so they survive across JNI local-ref frames.
  jclass patternCls = nullptr;
  jclass matcherCls = nullptr;
  jclass stringCls = nullptr;
  jclass mapCls = nullptr;
  jclass setCls = nullptr;
  jclass iteratorCls = nullptr;
  jclass entryCls = nullptr;
  jclass integerCls = nullptr;

  jmethodID compileMethod = nullptr; // static Pattern.compile(String, int)
  jmethodID matcherMethod = nullptr; // Pattern.matcher(CharSequence)
  jmethodID namedGroupsMethod = nullptr; // Pattern.namedGroups() (JDK 20+)

  jmethodID findMethod = nullptr; // Matcher.find(int)
  jmethodID findNoArgMethod = nullptr; // Matcher.find()
  jmethodID matchesMethod = nullptr; // Matcher.matches()
  jmethodID lookingAtMethod = nullptr; // Matcher.lookingAt()
  jmethodID startMethod = nullptr; // Matcher.start(int)
  jmethodID endMethod = nullptr; // Matcher.end(int)
  jmethodID groupCountMethod = nullptr; // Matcher.groupCount()
  jmethodID replaceAllMethod = nullptr; // Matcher.replaceAll(String)
  jmethodID regionMethod = nullptr; // Matcher.region(int, int)
  jmethodID useAnchoringMethod = nullptr; // Matcher.useAnchoringBounds(boolean)

  jmethodID mapEntrySetMethod = nullptr; // Map.entrySet()
  jmethodID setIteratorMethod = nullptr; // Set.iterator()
  jmethodID iteratorHasNextMethod = nullptr; // Iterator.hasNext()
  jmethodID iteratorNextMethod = nullptr; // Iterator.next()
  jmethodID entryGetKeyMethod = nullptr; // Map.Entry.getKey()
  jmethodID entryGetValueMethod = nullptr; // Map.Entry.getValue()
  jmethodID integerIntValueMethod = nullptr; // Integer.intValue()
};

std::once_flag g_idsOnce;
JavaIds g_ids;

jclass globalClassRef(JNIEnv* env, const char* name) {
  jclass local = env->FindClass(name);
  if (!local) {
    throw std::runtime_error(
        std::string("FindClass failed for ") + name);
  }
  jclass global = static_cast<jclass>(env->NewGlobalRef(local));
  env->DeleteLocalRef(local);
  return global;
}

void initIds(JNIEnv* env) {
  g_ids.patternCls = globalClassRef(env, "java/util/regex/Pattern");
  g_ids.matcherCls = globalClassRef(env, "java/util/regex/Matcher");
  g_ids.stringCls = globalClassRef(env, "java/lang/String");
  g_ids.mapCls = globalClassRef(env, "java/util/Map");
  g_ids.setCls = globalClassRef(env, "java/util/Set");
  g_ids.iteratorCls = globalClassRef(env, "java/util/Iterator");
  g_ids.entryCls = globalClassRef(env, "java/util/Map$Entry");
  g_ids.integerCls = globalClassRef(env, "java/lang/Integer");

  g_ids.compileMethod = env->GetStaticMethodID(
      g_ids.patternCls,
      "compile",
      "(Ljava/lang/String;I)Ljava/util/regex/Pattern;");
  g_ids.matcherMethod = env->GetMethodID(
      g_ids.patternCls,
      "matcher",
      "(Ljava/lang/CharSequence;)Ljava/util/regex/Matcher;");
  // Pattern.namedGroups() is JDK 20+; treat as optional.
  g_ids.namedGroupsMethod =
      env->GetMethodID(g_ids.patternCls, "namedGroups", "()Ljava/util/Map;");
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    g_ids.namedGroupsMethod = nullptr;
  }

  g_ids.findMethod = env->GetMethodID(g_ids.matcherCls, "find", "(I)Z");
  g_ids.findNoArgMethod = env->GetMethodID(g_ids.matcherCls, "find", "()Z");
  g_ids.matchesMethod = env->GetMethodID(g_ids.matcherCls, "matches", "()Z");
  g_ids.lookingAtMethod =
      env->GetMethodID(g_ids.matcherCls, "lookingAt", "()Z");
  g_ids.startMethod = env->GetMethodID(g_ids.matcherCls, "start", "(I)I");
  g_ids.endMethod = env->GetMethodID(g_ids.matcherCls, "end", "(I)I");
  g_ids.groupCountMethod =
      env->GetMethodID(g_ids.matcherCls, "groupCount", "()I");
  g_ids.replaceAllMethod = env->GetMethodID(
      g_ids.matcherCls, "replaceAll", "(Ljava/lang/String;)Ljava/lang/String;");
  g_ids.regionMethod =
      env->GetMethodID(g_ids.matcherCls, "region", "(II)Ljava/util/regex/Matcher;");
  g_ids.useAnchoringMethod = env->GetMethodID(
      g_ids.matcherCls,
      "useAnchoringBounds",
      "(Z)Ljava/util/regex/Matcher;");

  g_ids.mapEntrySetMethod =
      env->GetMethodID(g_ids.mapCls, "entrySet", "()Ljava/util/Set;");
  g_ids.setIteratorMethod =
      env->GetMethodID(g_ids.setCls, "iterator", "()Ljava/util/Iterator;");
  g_ids.iteratorHasNextMethod =
      env->GetMethodID(g_ids.iteratorCls, "hasNext", "()Z");
  g_ids.iteratorNextMethod =
      env->GetMethodID(g_ids.iteratorCls, "next", "()Ljava/lang/Object;");
  g_ids.entryGetKeyMethod =
      env->GetMethodID(g_ids.entryCls, "getKey", "()Ljava/lang/Object;");
  g_ids.entryGetValueMethod =
      env->GetMethodID(g_ids.entryCls, "getValue", "()Ljava/lang/Object;");
  g_ids.integerIntValueMethod =
      env->GetMethodID(g_ids.integerCls, "intValue", "()I");
}

jint toJavaFlags(const Options& o) {
  jint f = 0;
  if (!o.caseSensitive) {
    f |= kJavaCaseInsensitive | kJavaUnicodeCase;
  }
  if (o.dotNl) {
    f |= kJavaDotall;
  }
  if (!o.oneLine) {
    f |= kJavaMultiline;
  }
  return f;
}

// Convert a Java `String` index (a UTF-16 code-unit offset) into a byte
// offset in the given UTF-8 source.  Used to translate Matcher.start()/end()
// results — which are Java char indices — back into byte offsets in our
// std::string_view input.  Returns std::string_view::npos on bad input or
// out-of-range index.
std::size_t javaCharOffsetToByteOffset(
    std::string_view utf8,
    int javaCharOffset) {
  if (javaCharOffset < 0) {
    return std::string_view::npos;
  }
  int chars = 0;
  for (std::size_t i = 0; i < utf8.size();) {
    if (chars == javaCharOffset) {
      return i;
    }
    const unsigned char c = static_cast<unsigned char>(utf8[i]);
    if (c < 0x80) {
      i += 1;
      chars += 1;
    } else if (c < 0xC0) {
      // Stray continuation byte — advance to avoid an infinite loop.
      i += 1;
      chars += 1;
    } else if (c < 0xE0) {
      i += 2;
      chars += 1;
    } else if (c < 0xF0) {
      i += 3;
      chars += 1;
    } else {
      // 4-byte UTF-8 = U+10000..U+10FFFF, encoded as a UTF-16 surrogate
      // pair (2 code units) in Java.
      i += 4;
      chars += 2;
    }
  }
  return chars == javaCharOffset ? utf8.size() : std::string_view::npos;
}

// Inverse of the above: given a UTF-8 byte offset, return the equivalent
// Java UTF-16 char offset.  Used when we have to hand a byte offset (used
// by the caller / JavaMatcherAdapter cursor) over to Java's Matcher.region().
int byteOffsetToJavaCharOffset(
    std::string_view utf8,
    std::size_t byteOffset) {
  int chars = 0;
  std::size_t i = 0;
  while (i < utf8.size() && i < byteOffset) {
    const unsigned char c = static_cast<unsigned char>(utf8[i]);
    if (c < 0x80) {
      i += 1;
      chars += 1;
    } else if (c < 0xC0) {
      i += 1;
      chars += 1;
    } else if (c < 0xE0) {
      i += 2;
      chars += 1;
    } else if (c < 0xF0) {
      i += 3;
      chars += 1;
    } else {
      i += 4;
      chars += 2;
    }
  }
  return chars;
}

// Convert a std::string_view (UTF-8) to a JNI jstring.  Owned by caller —
// must DeleteLocalRef after use.
//
// NewStringUTF interprets its input as JNI's "modified UTF-8" — bytes >= 0x80
// are taken to be the first byte of a 2-byte sequence (essentially
// Latin-1-ish), which mangles real 3- and 4-byte UTF-8 sequences.  To
// faithfully round-trip UTF-8 we transcode to UTF-16 here and use
// NewString(jchar*, jsize) instead.
jstring toJString(JNIEnv* env, std::string_view sv) {
  std::vector<jchar> u16;
  u16.reserve(sv.size());
  for (std::size_t i = 0; i < sv.size();) {
    const unsigned char c = static_cast<unsigned char>(sv[i]);
    std::uint32_t cp = 0;
    std::size_t step = 1;
    if (c < 0x80) {
      cp = c;
      step = 1;
    } else if (c < 0xC0) {
      // Stray continuation; emit replacement to keep length sane.
      u16.push_back(0xFFFD);
      ++i;
      continue;
    } else if (c < 0xE0 && i + 1 < sv.size()) {
      cp = ((c & 0x1F) << 6) |
          (static_cast<unsigned char>(sv[i + 1]) & 0x3F);
      step = 2;
    } else if (c < 0xF0 && i + 2 < sv.size()) {
      cp = ((c & 0x0F) << 12) |
          ((static_cast<unsigned char>(sv[i + 1]) & 0x3F) << 6) |
          (static_cast<unsigned char>(sv[i + 2]) & 0x3F);
      step = 3;
    } else if (i + 3 < sv.size()) {
      cp = ((c & 0x07) << 18) |
          ((static_cast<unsigned char>(sv[i + 1]) & 0x3F) << 12) |
          ((static_cast<unsigned char>(sv[i + 2]) & 0x3F) << 6) |
          (static_cast<unsigned char>(sv[i + 3]) & 0x3F);
      step = 4;
    } else {
      u16.push_back(0xFFFD);
      ++i;
      continue;
    }
    if (cp <= 0xFFFF) {
      u16.push_back(static_cast<jchar>(cp));
    } else {
      cp -= 0x10000;
      u16.push_back(static_cast<jchar>(0xD800 | (cp >> 10)));
      u16.push_back(static_cast<jchar>(0xDC00 | (cp & 0x3FF)));
    }
    i += step;
  }
  return env->NewString(u16.data(), static_cast<jsize>(u16.size()));
}

// Read a jstring into a std::string (UTF-8).  Caller still owns the jstring.
// We use GetStringChars (UTF-16) and transcode to UTF-8 ourselves to avoid
// GetStringUTFChars's "modified UTF-8" which can't represent supplementary
// chars in their 4-byte UTF-8 form.
std::string fromJString(JNIEnv* env, jstring s) {
  if (!s) {
    return {};
  }
  const jsize len = env->GetStringLength(s);
  const jchar* u16 = env->GetStringChars(s, nullptr);
  std::string out;
  out.reserve(static_cast<std::size_t>(len));
  for (jsize i = 0; i < len; ++i) {
    std::uint32_t cp = u16[i];
    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < len) {
      const std::uint32_t lo = u16[i + 1];
      if (lo >= 0xDC00 && lo <= 0xDFFF) {
        cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
        ++i;
      }
    }
    if (cp < 0x80) {
      out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }
  env->ReleaseStringChars(s, u16);
  return out;
}

bool checkAndClearException(JNIEnv* env, std::string* outError) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  if (outError) {
    *outError = "Java exception thrown (cleared)";
  }
  env->ExceptionClear();
  return true;
}

void populateNamedFromPattern(
    JNIEnv* env,
    jobject pattern,
    std::map<std::string, int>* out) {
  if (!g_ids.namedGroupsMethod) {
    return;
  }
  jobject map = env->CallObjectMethod(pattern, g_ids.namedGroupsMethod);
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return;
  }
  if (!map) {
    return;
  }
  jobject set = env->CallObjectMethod(map, g_ids.mapEntrySetMethod);
  jobject it = env->CallObjectMethod(set, g_ids.setIteratorMethod);
  while (env->CallBooleanMethod(it, g_ids.iteratorHasNextMethod)) {
    jobject entry = env->CallObjectMethod(it, g_ids.iteratorNextMethod);
    jstring key = static_cast<jstring>(
        env->CallObjectMethod(entry, g_ids.entryGetKeyMethod));
    jobject value = env->CallObjectMethod(entry, g_ids.entryGetValueMethod);
    jint idx = env->CallIntMethod(value, g_ids.integerIntValueMethod);
    out->emplace(fromJString(env, key), static_cast<int>(idx));
    env->DeleteLocalRef(key);
    env->DeleteLocalRef(value);
    env->DeleteLocalRef(entry);
  }
  env->DeleteLocalRef(it);
  env->DeleteLocalRef(set);
  env->DeleteLocalRef(map);
}

} // namespace

JavaRegex::JavaRegex(std::string_view javaPattern, Options opt) {
  auto* env = JvmFixture::instance().env();
  std::call_once(g_idsOnce, [&]() { initIds(env); });

  jstring jPat = toJString(env, javaPattern);
  jobject pObj = env->CallStaticObjectMethod(
      g_ids.patternCls, g_ids.compileMethod, jPat, toJavaFlags(opt));
  env->DeleteLocalRef(jPat);

  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    error_ = "Java PatternSyntaxException: " + std::string(javaPattern);
    return;
  }
  pattern_ = env->NewGlobalRef(pObj);
  env->DeleteLocalRef(pObj);

  // groupCount via a throwaway empty matcher.
  jstring emptyStr = toJString(env, "");
  jobject tmpMatcher = env->CallObjectMethod(
      pattern_, g_ids.matcherMethod, emptyStr);
  env->DeleteLocalRef(emptyStr);
  captureCount_ = env->CallIntMethod(tmpMatcher, g_ids.groupCountMethod);
  env->DeleteLocalRef(tmpMatcher);

  populateNamedFromPattern(env, pattern_, &named_);
}

JavaRegex::~JavaRegex() {
  if (pattern_) {
    JvmFixture::instance().env()->DeleteGlobalRef(pattern_);
  }
}

bool JavaRegex::ok() const {
  return pattern_ != nullptr;
}
const std::string& JavaRegex::error() const {
  return error_;
}
int JavaRegex::NumberOfCapturingGroups() const {
  return captureCount_;
}
const std::map<std::string, int>& JavaRegex::NamedCapturingGroups() const {
  return named_;
}

bool JavaRegex::Match(
    std::string_view input,
    std::size_t startpos,
    std::size_t endpos,
    Anchor anchor,
    std::string_view* submatch,
    int nsubmatch) const {
  if (!pattern_) {
    return false;
  }
  auto* env = JvmFixture::instance().env();

  // Java's Matcher operates on a CharSequence we hand it; clip input to
  // [0, endpos) by materialising that prefix.  Then use region() so the
  // engine treats [startpos, endpos) as the searchable window.
  const std::string buf(input.substr(0, endpos));
  jstring jin = toJString(env, buf);
  jobject m = env->CallObjectMethod(pattern_, g_ids.matcherMethod, jin);
  env->DeleteLocalRef(jin);

  // Set region so anchors line up with [startpos, endpos).
  // Java's Matcher.region(start, end) takes UTF-16 char offsets, not bytes —
  // translate from our byte-offset parameters first.
  const jint regionStart = static_cast<jint>(
      byteOffsetToJavaCharOffset(input, startpos));
  const jint regionEnd = static_cast<jint>(
      byteOffsetToJavaCharOffset(input, endpos));
  jobject mRegion = env->CallObjectMethod(
      m, g_ids.regionMethod, regionStart, regionEnd);
  env->DeleteLocalRef(mRegion);

  jboolean matched = JNI_FALSE;
  switch (anchor) {
    case Anchor::kUnanchored:
      matched = env->CallBooleanMethod(m, g_ids.findNoArgMethod);
      break;
    case Anchor::kAnchorStart:
      matched = env->CallBooleanMethod(m, g_ids.lookingAtMethod);
      break;
    case Anchor::kAnchorBoth:
      matched = env->CallBooleanMethod(m, g_ids.matchesMethod);
      break;
  }

  if (!matched) {
    env->DeleteLocalRef(m);
    return false;
  }

  // Extract submatches: Matcher.start(i)/end(i) return UTF-16 char offsets
  // into the original CharSequence (= our `buf` = a prefix of `input`).
  // Translate each Java char offset back to a byte offset in `input` so
  // string_view substr arithmetic works for non-ASCII input.
  for (int i = 0; i < nsubmatch; ++i) {
    jint s = env->CallIntMethod(m, g_ids.startMethod, i);
    if (env->ExceptionCheck()) {
      env->ExceptionClear();
      submatch[i] = std::string_view{};
      continue;
    }
    jint e = env->CallIntMethod(m, g_ids.endMethod, i);
    if (s < 0) {
      submatch[i] = std::string_view{};
      continue;
    }
    const std::size_t sByte = javaCharOffsetToByteOffset(input, s);
    const std::size_t eByte = javaCharOffsetToByteOffset(input, e);
    if (sByte == std::string_view::npos || eByte == std::string_view::npos ||
        eByte < sByte) {
      submatch[i] = std::string_view{};
    } else {
      submatch[i] = input.substr(sByte, eByte - sByte);
    }
  }

  env->DeleteLocalRef(m);
  return true;
}

bool JavaRegex::FullMatch(std::string_view input, const JavaRegex& re) {
  std::string_view sub[1];
  return re.Match(input, 0, input.size(), Anchor::kAnchorBoth, sub, 1);
}

bool JavaRegex::PartialMatch(std::string_view input, const JavaRegex& re) {
  std::string_view sub[1];
  return re.Match(input, 0, input.size(), Anchor::kUnanchored, sub, 1);
}

int JavaRegex::GlobalReplace(
    std::string* str,
    const JavaRegex& re,
    std::string_view javaReplacement) {
  if (!re.ok() || str == nullptr) {
    return 0;
  }
  auto* env = JvmFixture::instance().env();

  // Build a Matcher on the input and call replaceAll(repl).  Matcher.replaceAll
  // is the canonical Java semantics — accepts $N / ${name} natively, returns
  // the result as a String.  We have no way to recover the *count* of
  // replacements done through the public API without manual find()-loop, so
  // we approximate: count matches first, then replaceAll.  (Tests use exact
  // count assertions, so this matters.)
  jstring jin = toJString(env, *str);
  jobject m = env->CallObjectMethod(re.pattern_, g_ids.matcherMethod, jin);

  // First: count matches by walking find().
  int count = 0;
  while (env->CallBooleanMethod(m, g_ids.findNoArgMethod)) {
    ++count;
  }

  // Second: reset matcher (recreate it — replaceAll re-walks anyway).
  env->DeleteLocalRef(m);
  m = env->CallObjectMethod(re.pattern_, g_ids.matcherMethod, jin);
  jstring jRepl = toJString(env, javaReplacement);
  jstring jOut = static_cast<jstring>(
      env->CallObjectMethod(m, g_ids.replaceAllMethod, jRepl));
  env->DeleteLocalRef(jRepl);
  env->DeleteLocalRef(m);
  env->DeleteLocalRef(jin);

  if (env->ExceptionCheck()) {
    env->ExceptionClear();
    return 0;
  }
  *str = fromJString(env, jOut);
  env->DeleteLocalRef(jOut);
  return count;
}

} // namespace facebook::velox::regex_compat

#endif // VELOX_REGEX_COMPAT_HAS_JAVA
