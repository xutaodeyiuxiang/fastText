/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "loss.h"
#include "utils.h"

#include <cmath>

namespace fasttext {

constexpr int64_t SIGMOID_TABLE_SIZE = 512;  // 通过查表的方法加速计算过程
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;

bool comparePairs(
    const std::pair<real, int32_t>& l,
    const std::pair<real, int32_t>& r) {
  return l.first > r.first;
}

real std_log(real x) {
  return std::log(x + 1e-5);
}

Loss::Loss(std::shared_ptr<Matrix>& wo) : wo_(wo) {
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);  // sigmoid 查找表
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
  }

  t_log_.reserve(LOG_TABLE_SIZE + 1);   // log 查找表
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log_.push_back(std::log(x));
  }
}

real Loss::log(real x) const {
  if (x > 1.0) {  // 处理不在表里面的情况
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log_[i];
}

real Loss::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i =
        int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}

void Loss::predict(
    int32_t k,
    real threshold,
    Predictions& heap,
    Model::State& state) const {
  computeOutput(state);
  findKBest(k, threshold, heap, state.output);
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Loss::findKBest(
    int32_t k,
    real threshold,
    Predictions& heap,
    const Vector& output) const {
  for (int32_t i = 0; i < output.size(); i++) {
    if (output[i] < threshold) { // 由于阈值，就跳过
      continue;
    }
    if (heap.size() == k && std_log(output[i]) < heap.front().first) {
      continue;
    }
    // heap sort
    heap.push_back(std::make_pair(std_log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

BinaryLogisticLoss::BinaryLogisticLoss(std::shared_ptr<Matrix>& wo)
    : Loss(wo) {}

real BinaryLogisticLoss::binaryLogistic(
    int32_t target,
    Model::State& state,
    bool labelIsPositive,
    real lr,
    bool backprop) const {
  real score = sigmoid(wo_->dotRow(state.hidden, target)); // sigmoid(w * hidden) 计算target score
  if (backprop) {
    real alpha = lr * (real(labelIsPositive) - score);  // 反传误差, 并存储在state中
    state.grad.addRow(*wo_, target, alpha);
    wo_->addVectorToRow(state.hidden, target, alpha);
  }
  if (labelIsPositive) {  // compute loss
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

void BinaryLogisticLoss::computeOutput(Model::State& state) const {
  // compute sigmoid(hidden * wo_)
  Vector& output = state.output;
  output.mul(*wo_, state.hidden);
  int32_t osz = output.size();
  for (int32_t i = 0; i < osz; i++) {
    output[i] = sigmoid(output[i]);
  }
}

OneVsAllLoss::OneVsAllLoss(std::shared_ptr<Matrix>& wo)
    : BinaryLogisticLoss(wo) {}

real OneVsAllLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t /* we take all targets here */,
    Model::State& state,
    real lr,
    bool backprop) {
  real loss = 0.0;
  int32_t osz = state.output.size();
  for (int32_t i = 0; i < osz; i++) {  // one-vs-all实际上是将所有二分类的loss相加，作为最后的loss
    bool isMatch = utils::contains(targets, i);
    loss += binaryLogistic(i, state, isMatch, lr, backprop);
  }

  return loss;
}

NegativeSamplingLoss::NegativeSamplingLoss(
    std::shared_ptr<Matrix>& wo,
    int neg,
    const std::vector<int64_t>& targetCounts)  // 词频
    : BinaryLogisticLoss(wo), neg_(neg), negatives_(), uniform_() {
  // 根据count计算负采样分布
  real z = 0.0;
  for (size_t i = 0; i < targetCounts.size(); i++) {
    z += pow(targetCounts[i], 0.5);
  }
  for (size_t i = 0; i < targetCounts.size(); i++) {
    real c = pow(targetCounts[i], 0.5);
    for (size_t j = 0; j < c * NegativeSamplingLoss::NEGATIVE_TABLE_SIZE / z;
         j++) {
      negatives_.push_back(i);  // 概率越大，i出现的越多，均匀采样的时候，被抽到的概率更大
    }
  }
  uniform_ = std::uniform_int_distribution<size_t>(0, negatives_.size() - 1);
}

real NegativeSamplingLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];
  // 正例的loss
  real loss = binaryLogistic(target, state, true, lr, backprop);

  // 负例的loss
  for (int32_t n = 0; n < neg_; n++) {
    auto negativeTarget = getNegative(target, state.rng);
    loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
  }
  return loss;
}

int32_t NegativeSamplingLoss::getNegative(
    int32_t target,
    std::minstd_rand& rng) {
  int32_t negative;
  // negative 采样
  do {
    negative = negatives_[uniform_(rng)];
  } while (target == negative);
  return negative;
}

HierarchicalSoftmaxLoss::HierarchicalSoftmaxLoss(
    std::shared_ptr<Matrix>& wo,
    const std::vector<int64_t>& targetCounts)
    : BinaryLogisticLoss(wo),
      paths_(),
      codes_(),
      tree_(),
      osz_(targetCounts.size()) {
  buildTree(targetCounts);
}

// 建立霍夫曼树
void HierarchicalSoftmaxLoss::buildTree(const std::vector<int64_t>& counts) {
  // counts 数组保存每个叶子节点的词频，降序排列
  tree_.resize(2 * osz_ - 1);   // 词典大小为osz_, 则树的节点个数是2 * osz_ - 1
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree_[i].parent = -1;
    tree_[i].left = -1;
    tree_[i].right = -1;
    tree_[i].count = 1e15;
    tree_[i].binary = false;
  }
  // 最前面的部分存放叶子节点，后面存放分支节点
  for (int32_t i = 0; i < osz_; i++) {
    tree_[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;   // leaf 指向当前未处理的叶子节点的最后一个，也就是权值最小的叶子节点
  int32_t node = osz_;       // node 指向当前未处理的非叶子节点的第一个，也是权值最小的非叶子节点
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) { // 逐个构造所有非叶子节点（i >= osz_, i < 2 * osz - 1)
    int32_t mini[2] = {0};  // 最小的两个节点的下标
    // 计算权值最小的两个节点，候选只可能是 leaf, leaf - 1,
    // 以及 node, node + 1  
    // 从这四个候选里找到 top-2
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree_[leaf].count < tree_[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    // 更新非叶子节点的属性
    tree_[i].left = mini[0];
    tree_[i].right = mini[1];
    tree_[i].count = tree_[mini[0]].count + tree_[mini[1]].count;
    tree_[mini[0]].parent = i;
    tree_[mini[1]].parent = i;
    tree_[mini[1]].binary = true;
  }
  // 计算霍夫曼编码, 由叶子节点向上找，一直找到根节点
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree_[j].parent != -1) {
      path.push_back(tree_[j].parent - osz_);
      code.push_back(tree_[j].binary);
      j = tree_[j].parent;
    }
    paths_.push_back(path);
    codes_.push_back(code);
  }
}

real HierarchicalSoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  real loss = 0.0;
  int32_t target = targets[targetIndex];
  const std::vector<bool>& binaryCode = codes_[target];     // 存储分支方向（相当于label是正的，还是负的）
  const std::vector<int32_t>& pathToRoot = paths_[target];  // 每个分支节点也当作一个word或label来看，也有相应的id
  // 路径上所有loss相加
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], state, binaryCode[i], lr, backprop);
  }
  return loss;
}

void HierarchicalSoftmaxLoss::predict(
    int32_t k,
    real threshold,
    Predictions& heap,
    Model::State& state) const {
  dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, state.hidden);
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}
    
// 采用dfs遍历霍夫曼树的叶子节点，实际上时间复杂度还是O(n)，但是由于用了霍夫曼树，时间复杂度会大大减少
void HierarchicalSoftmaxLoss::dfs(
    int32_t k,
    real threshold,
    int32_t node,
    real score,
    Predictions& heap,
    const Vector& hidden) const {
  if (score < std_log(threshold)) {
    return;
  }
  if (heap.size() == k && score < heap.front().first) {
    return;
  }
  
  // 只输出叶子节点的结果
  if (tree_[node].left == -1 && tree_[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }
  // 每个分支节点，都是用hidden layer作为输入向量，进行LR二分类
  real f = wo_->dotRow(hidden, node - osz_);  // 每个分支节点也有相对应的embedding，存储在wo_里面
  f = 1. / (1 + std::exp(-f));  // sigmoid

  // 向左走，负例
  dfs(k, threshold, tree_[node].left, score + std_log(1.0 - f), heap, hidden);
  dfs(k, threshold, tree_[node].right, score + std_log(f), heap, hidden);
}

SoftmaxLoss::SoftmaxLoss(std::shared_ptr<Matrix>& wo) : Loss(wo) {}

void SoftmaxLoss::computeOutput(Model::State& state) const {
  Vector& output = state.output;
  output.mul(*wo_, state.hidden);
  real max = output[0], z = 0.0;
  int32_t osz = output.size();
  for (int32_t i = 0; i < osz; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz; i++) {
    output[i] /= z;
  }
}

real SoftmaxLoss::forward(
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    Model::State& state,
    real lr,
    bool backprop) {
  // 计算输出层，归一化之后的
  computeOutput(state);

  assert(targetIndex >= 0);
  assert(targetIndex < targets.size());
  int32_t target = targets[targetIndex];

  if (backprop) {
    int32_t osz = wo_->size(0);
    for (int32_t i = 0; i < osz; i++) {
      real label = (i == target) ? 1.0 : 0.0;
      real alpha = lr * (label - state.output[i]);  // 误差
      state.grad.addRow(*wo_, i, alpha);            // 更新权重
      wo_->addVectorToRow(state.hidden, i, alpha);
    }
  }
  return -log(state.output[target]);
};

} // namespace fasttext
