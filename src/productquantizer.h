/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 这个文件的目的是 压缩一个matrix，对应的是embedding矩阵。采用的算法是product quantizer, 最早用在图像检索引擎中
 假设embedding 有 512维
 （1）训练编码本：将512维度划分成16份，每份32维。对每份数据进行kmeans聚类，类别数量是256，最终我们得到 16 * 256个聚类中心
 （2）量化 ： 将512维度划分成16份，每份32维。每份找到最近的聚类中心，将聚类中心的id，存储下来，最后会有16个聚类中心id，表示原有的向量，就压缩了
 */

#pragma once

#include <cstring>
#include <istream>
#include <ostream>
#include <random>
#include <vector>

#include "real.h"
#include "vector.h"

namespace fasttext {

class ProductQuantizer {
 protected:
  const int32_t nbits_ = 8;                                    
  const int32_t ksub_ = 1 << nbits_;                            // 每份kmeans聚类中心的个数 2^nbits_ = 2^8 = 256
  const int32_t max_points_per_cluster_ = 256;
  const int32_t max_points_ = max_points_per_cluster_ * ksub_;
  const int32_t seed_ = 1234;                                   // kmenas初始化随机种子
  const int32_t niter_ = 25;                                    // kmeans迭代次数
  const real eps_ = 1e-7;                                       // kmeans训练的eps

  int32_t dim_;                                                 // 待压缩的embedding的维度
  int32_t nsubq_;                                               // 划分成多少个子空间
  int32_t dsub_;                                                // 每个空间的维度  
  int32_t lastdsub_;                                            // 最后一个子空间的维度 dsub_ * nsubq_ + lastdsub_ = dim_

  std::vector<real> centroids_;                                 // 聚类中心数据存放

  std::minstd_rand rng;

 public:
  ProductQuantizer() {}
  ProductQuantizer(int32_t, int32_t);

  real* get_centroids(int32_t, uint8_t);
  const real* get_centroids(int32_t, uint8_t) const;

  real assign_centroid(const real*, const real*, uint8_t*, int32_t) const;
  void Estep(const real*, const real*, uint8_t*, int32_t, int32_t) const;
  void MStep(const real*, real*, const uint8_t*, int32_t, int32_t);
  void kmeans(const real*, real*, int32_t, int32_t);
  void train(int, const real*);

  real mulcode(const Vector&, const uint8_t*, int32_t, real) const;
  void addcode(Vector&, const uint8_t*, int32_t, real) const;
  void compute_code(const real*, uint8_t*) const;
  void compute_codes(const real*, uint8_t*, int32_t) const;

  void save(std::ostream&) const;
  void load(std::istream&);
};

} // namespace fasttext
