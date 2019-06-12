/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "loss.h"
#include "utils.h"

#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace fasttext {

// 只是用来存储此批的中间结果，不会涉及到具体的参数，更新的参数都是在class Model里面
Model::State::State(int32_t hiddenSize, int32_t outputSize, int32_t seed)
    : lossValue_(0.0),
      nexamples_(0),
      hidden(hiddenSize),
      output(outputSize),
      grad(hiddenSize),
      rng(seed) {}

// 此批数据的平均loss
real Model::State::getLoss() const {
  return lossValue_ / nexamples_; 
}

// 每个例子增加一个loss
void Model::State::incrementNExamples(real loss) {
  lossValue_ += loss;
  nexamples_++;
}

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Loss> loss,
    bool normalizeGradient)
    : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
  Vector& hidden = state.hidden;
  hidden.zero();
  // hidden 向量保存输入词向量的均值, 也会存储到state里面
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);    // addRow 的作用是将 wi_ 矩阵的第 *it 列加到 hidden 上 
  }
  hidden.mul(1.0 / input.size()); // 求和后除以输入词个数，得到均值向量
}

void Model::predict(
    const std::vector<int32_t>& input,
    int32_t k,
    real threshold,
    Predictions& heap,
    State& state) const {
  if (k == Model::kUnlimitedPredictions) {
    k = wo_->size(0); // output size
  } else if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  heap.reserve(k + 1);
  computeHidden(input, state);

  loss_->predict(k, threshold, heap, state);
}

// 一个sample的更新过程
void Model::update(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state) {
  if (input.size() == 0) {
    return;
  }
  // 计算前向传播：输入层 -> 隐层（做完average之后的隐层）
  computeHidden(input, state);

  Vector& grad = state.grad;
  grad.zero();
  // 不仅通过前向传播算出了 loss_，还进行了反向传播，计算出了 grad_
  real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
  state.incrementNExamples(lossValue); 

  if (normalizeGradient_) {
    // 将 grad_ 除以 input_ 的大小
    grad.mul(1.0 / input.size());
  }
  // 反向传播，将 hidden_ 上的梯度传播到 wi_ 上的对应行
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addVectorToRow(grad, *it, 1.0);
  }
}

// 防止x为0的情况
real Model::std_log(real x) const {
  return std::log(x + 1e-5);
}

} // namespace fasttext
