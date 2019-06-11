/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "productquantizer.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fasttext {

// 计算欧式距离
real distL2(const real* x, const real* y, int32_t d) {
  real dist = 0;
  for (auto i = 0; i < d; i++) {
    auto tmp = x[i] - y[i];
    dist += tmp * tmp;
  }
  return dist;
}

 
ProductQuantizer::ProductQuantizer(int32_t dim, int32_t dsub)
    : dim_(dim),
      nsubq_(dim / dsub),
      dsub_(dsub),
      centroids_(dim * ksub_), // dsub_ * ksub_ * nsubq_ = dim * k_sub_
      rng(seed_) {
  lastdsub_ = dim_ % dsub;
  if (lastdsub_ == 0) {
    lastdsub_ = dsub_;
  } else {
    nsubq_++;
  }
}

// m 是划分的份数index
// i 是 第m份中的第i个聚类中心
// 函数的功能是获得指定聚类中心的向量表示
const real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) const {
  if (m == nsubq_ - 1) {
    return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];
  }
  return &centroids_[(m * ksub_ + i) * dsub_];
}

real* ProductQuantizer::get_centroids(int32_t m, uint8_t i) {
  if (m == nsubq_ - 1) {
    return &centroids_[m * ksub_ * dsub_ + i * lastdsub_];
  }
  return &centroids_[(m * ksub_ + i) * dsub_];
}

/*
 * x 是原始向量
 * c0 是聚类中心向量集合
 * code 是最近聚类中心的id
 * d 是维度
 * 返回是x到最近的聚类中心的欧式距离
 */
real ProductQuantizer::assign_centroid(
    const real* x,
    const real* c0,
    uint8_t* code,
    int32_t d) const {
  const real* c = c0;
  real dis = distL2(x, c, d);
  code[0] = 0;
  for (auto j = 1; j < ksub_; j++) {
    // 循环每个聚类中心
    c += d;  // 移到下一个聚类中心
    real disij = distL2(x, c, d);
    if (disij < dis) {
      code[0] = (uint8_t)j;
      dis = disij;
    }
  }
  return dis;
}

void ProductQuantizer::Estep(
    const real* x,
    const real* centroids,
    uint8_t* codes,
    int32_t d,
    int32_t n) const {
  for (auto i = 0; i < n; i++) {
    // n为向量总数，d为每个向量的维度，x为n个向量的起始位置，centrids为聚类中心点，codes为向量的所在簇的编号（长度为n）
    assign_centroid(x + i * d, centroids, codes + i, d); // 每个向量依次寻找最近的聚类中心
  }
}

void ProductQuantizer::MStep(
    const real* x0,
    real* centroids,
    const uint8_t* codes,
    int32_t d,
    int32_t n) {
  std::vector<int32_t> nelts(ksub_, 0);  //记录每个簇有多少个元素，初始化为0
  memset(centroids, 0, sizeof(real) * d * ksub_); // 将每个簇的中心点centroid清零
  const real* x = x0; 
  for (auto i = 0; i < n; i++) { //簇中心centroid的坐标为该簇所有点的坐标的中心，即坐标和，然后取平均
    auto k = codes[i];  //第i个节点的簇号为k
    real* c = centroids + k * d;  //第k个中心点的起始坐标
    for (auto j = 0; j < d; j++) { //d个坐标，依次相
      c[j] += x[j]; 
    }
    nelts[k]++;  //每个簇包含向量的个数增长1
    x += d;      //下一个节点
  }

  real* c = centroids;
  for (auto k = 0; k < ksub_; k++) { //对每一个中心点（总共有ksub_个，即256）
    real z = (real)nelts[k];
    if (z != 0) {
      for (auto j = 0; j < d; j++) { 
        c[j] /= z; //每个坐标值依次取平均
      }
    }
    c += d;  //下一个中心点
  }

  // 当然，如果第k个簇没有节点，则选择另外一个较大的簇m，将簇m的中心点给k并随机加减一个常数
  std::uniform_real_distribution<> runiform(0, 1);
  for (auto k = 0; k < ksub_; k++) {
    if (nelts[k] == 0) {
      int32_t m = 0;
      while (runiform(rng) * (n - ksub_) >= nelts[m] - 1) {
        m = (m + 1) % ksub_;
      }
      memcpy(centroids + k * d, centroids + m * d, sizeof(real) * d);
      for (auto j = 0; j < d; j++) {
        int32_t sign = (j % 2) * 2 - 1;
        centroids[k * d + j] += sign * eps_;
        centroids[m * d + j] -= sign * eps_;
      }
      nelts[k] = nelts[m] / 2;
      nelts[m] -= nelts[k];
    }
  }
}

void ProductQuantizer::kmeans(const real* x, real* c, int32_t n, int32_t d) {
  std::vector<int32_t> perm(n, 0);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  for (auto i = 0; i < ksub_; i++) {
    memcpy(&c[i * d], x + perm[i] * d, d * sizeof(real));
  }
  auto codes = std::vector<uint8_t>(n);
  for (auto i = 0; i < niter_; i++) {
    Estep(x, c, codes.data(), d, n);
    MStep(x, c, codes.data(), d, n);
  }
}

void ProductQuantizer::train(int32_t n, const real* x) {
  //n为原始向量数目，x指向原始向量的起始位置
  if (n < ksub_) {
    throw std::invalid_argument(
        "Matrix too small for quantization, must have at least " +
        std::to_string(ksub_) + " rows");
  }
  std::vector<int32_t> perm(n, 0);
  std::iota(perm.begin(), perm.end(), 0);
  auto d = dsub_;
  auto np = std::min(n, max_points_);  // 如果用于训练的样本个数大于预先设置的最大个数，就截断
  auto xslice = std::vector<real>(np * dsub_);
  for (auto m = 0; m < nsubq_; m++) {
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    if (np != n) {  // 如果是截断的，就shuffle数据
      std::shuffle(perm.begin(), perm.end(), rng);
    }
    for (auto j = 0; j < np; j++) {
      memcpy(
          xslice.data() + j * d,
          x + perm[j] * dim_ + m * dsub_,
          d * sizeof(real));
    }
    // 训练获取聚类中心
    kmeans(xslice.data(), get_centroids(m, 0), np, d);
  }
}

real ProductQuantizer::mulcode(
    const Vector& x,
    const uint8_t* codes,
    int32_t t,
    real alpha) const {
  real res = 0.0;
  auto d = dsub_;
  const uint8_t* code = codes + nsubq_ * t;
  for (auto m = 0; m < nsubq_; m++) {
    const real* c = get_centroids(m, code[m]);
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    for (auto n = 0; n < d; n++) {
      res += x[m * dsub_ + n] * c[n];
    }
  }
  return res * alpha;
}

void ProductQuantizer::addcode(
    Vector& x,
    const uint8_t* codes,
    int32_t t,
    real alpha) const {
  auto d = dsub_;
  const uint8_t* code = codes + nsubq_ * t;
  for (auto m = 0; m < nsubq_; m++) {
    const real* c = get_centroids(m, code[m]);
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    for (auto n = 0; n < d; n++) {
      x[m * dsub_ + n] += alpha * c[n];
    }
  }
}

// 计算编码，实际就是压缩过程
void ProductQuantizer::compute_code(const real* x, uint8_t* code) const {
  auto d = dsub_;
  for (auto m = 0; m < nsubq_; m++) {
    if (m == nsubq_ - 1) {
      d = lastdsub_;
    }
    assign_centroid(x + m * dsub_, get_centroids(m, 0), code + m, d);
  }
}

void ProductQuantizer::compute_codes(const real* x, uint8_t* codes, int32_t n)
    const {
  for (auto i = 0; i < n; i++) {
    compute_code(x + i * dim_, codes + i * nsubq_);
  }
}

void ProductQuantizer::save(std::ostream& out) const {
  out.write((char*)&dim_, sizeof(dim_));
  out.write((char*)&nsubq_, sizeof(nsubq_));
  out.write((char*)&dsub_, sizeof(dsub_));
  out.write((char*)&lastdsub_, sizeof(lastdsub_));
  out.write((char*)centroids_.data(), centroids_.size() * sizeof(real));
}

void ProductQuantizer::load(std::istream& in) {
  in.read((char*)&dim_, sizeof(dim_));
  in.read((char*)&nsubq_, sizeof(nsubq_));
  in.read((char*)&dsub_, sizeof(dsub_));
  in.read((char*)&lastdsub_, sizeof(lastdsub_));
  centroids_.resize(dim_ * ksub_);
  for (auto i = 0; i < centroids_.size(); i++) {
    in.read((char*)&centroids_[i], sizeof(real));
  }
}

} // namespace fasttext
