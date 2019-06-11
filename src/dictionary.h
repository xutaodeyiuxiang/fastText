/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;
enum class entry_type : int8_t { word = 0, label = 1 };

// words_和word2int_, entry是在words中的一个数据项
struct entry {
  std::string word;               //单词的字符串
  int64_t count;                  //该单词在训练集中出现的次数
  entry_type type;                //类型，word为0，文本分类的label为1
  std::vector<int32_t> subwords;  //该词的ngram序列对应的下标
};

class Dictionary {
 protected:
  static const int32_t MAX_VOCAB_SIZE = 30000000;
  static const int32_t MAX_LINE_SIZE = 1024;

  int32_t find(const std::string&) const;
  int32_t find(const std::string&, uint32_t h) const;
  void initTableDiscard();
  void initNgrams();
  void reset(std::istream&) const;
  void pushHash(std::vector<int32_t>&, int32_t) const;
  void addSubwords(std::vector<int32_t>&, const std::string&, int32_t) const;

  std::shared_ptr<Args> args_;
  std::vector<int32_t> word2int_;  // 单词及其对应的id
  std::vector<entry> words_;       // 单词的详细信息

  std::vector<real> pdiscard_;     // pdiscard_对应着字典中每个词word的被丢弃的概率
  int32_t size_;
  int32_t nwords_;                 // 因为词和label都存储在words_里面，为了区分word和label，所以有这个参数，前部分为word部分，有nwords_个，并且按出现次数降序排序
  int32_t nlabels_;                // 后一部分为label部分，nlables_个，也按出现次数降序排列
  int64_t ntokens_;                // ntokens_是训练预料中word和label的总数量（包含重复次数，而且清理字典时，并不改变这个值）

  int64_t pruneidx_size_;
  std::unordered_map<int32_t, int32_t> pruneidx_;
  void addWordNgrams(
      std::vector<int32_t>& line,
      const std::vector<int32_t>& hashes,
      int32_t n) const;

 public:
  static const std::string EOS;
  static const std::string BOW;
  static const std::string EOW;

  explicit Dictionary(std::shared_ptr<Args>);
  explicit Dictionary(std::shared_ptr<Args>, std::istream&);
  int32_t nwords() const;
  int32_t nlabels() const;
  int64_t ntokens() const;
  int32_t getId(const std::string&) const;
  int32_t getId(const std::string&, uint32_t h) const;
  entry_type getType(int32_t) const;
  entry_type getType(const std::string&) const;
  bool discard(int32_t, real) const;
  std::string getWord(int32_t) const;
  const std::vector<int32_t>& getSubwords(int32_t) const;
  const std::vector<int32_t> getSubwords(const std::string&) const;
  void getSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>&) const;
  void computeSubwords(
      const std::string&,
      std::vector<int32_t>&,
      std::vector<std::string>* substrings = nullptr) const;
  uint32_t hash(const std::string& str) const;
  void add(const std::string&);
  bool readWord(std::istream&, std::string&) const;
  void readFromFile(std::istream&);
  std::string getLabel(int32_t) const;
  void save(std::ostream&) const;
  void load(std::istream&);
  std::vector<int64_t> getCounts(entry_type) const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::vector<int32_t>&)
      const;
  int32_t getLine(std::istream&, std::vector<int32_t>&, std::minstd_rand&)
      const;
  void threshold(int64_t, int64_t);
  void prune(std::vector<int32_t>&);
  bool isPruned() {
    return pruneidx_size_ >= 0;
  }
  void dump(std::ostream&) const;
  void init();
};

} // namespace fasttext
