/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace fasttext {

// 三种模型：cbow， skip-gram，supervise（classify）
enum class model_name : int { cbow = 1, sg, sup };
// loss 函数：Hierarchical Softmax，negative sampling，directly softmax，one-vs-all（多分类）
enum class loss_name : int { hs = 1, ns, softmax, ova };

class Args {
 protected:
  std::string lossToString(loss_name) const;
  std::string boolToString(bool) const;
  std::string modelToString(model_name) const;

 public:
  Args();
  std::string input;
  std::string output;
  double lr;            // learning rate
  int lrUpdateRate;     // learning rate updating rate
  int dim;              // dimension of word embedding
  int ws;               // the window size for skip-gram or cbow
  int epoch;    
  int minCount;         // 词典截断数（出现次数小于这个数字词，将加入都unkown中）
  int minCountLabel;    //（出现次数小于这个数的分类标签，也不会被训练）
  int neg;              // neg为negative sample时负样本的选择个数
  int wordNgrams;       // 词级别的ngram
  loss_name loss;
  model_name model;
  int bucket;           // 当ngram太多的时候，会映射到一个hash表，hash表里面的每个bucket对应一个embedding
  int minn;             // 字符级别的ngram的最小和最大长度
  int maxn;
  int thread;
  double t;             // 采用阈值,主要是随机采样的超参数，根据词频和这个参数负采样
  std::string label;
  int verbose;
  std::string pretrainedVectors;
  bool saveOutput;

  // 分类器是否需要做量化（主要用于压缩模型大小）
  bool qout;
  bool retrain;
  // 是否独立量化
  bool qnorm;
  // qout, qnorm, cutoff, dsub主要用来做模型压缩
  /*
  这两个文件分别定义了类QMatrix和类ProductQuantizer，把Matrix对象进行量化压缩。
  qmatrix.h和http://qmatrix.cc定义了类QMatrix，成员变量m_和n_依然为待压缩矩阵mat的行数和列数，压缩成QMatrix矩阵，把每个向量压缩后的编码保存在成员变量codes_中，其行数依然为m_，列数为n_除以dsub（dsub应该是dimension subspace，即子空间的维数），codesize_为QMatrix的元素数量。从构造函数可以看出，接收一个Matrix对象mat，算出codesize_大小后，适配codes_的容量，生成productquantizer指针对象，然后成员函数quantize()调用pq的方法，对mat进行压缩，压缩结果写进codes_。其他QMatrix成员函数与Matrix的函数类似。
  productquantizer.h和http://productquantizer.cc定义了乘积量化的具体操作。乘积量化简单的说，就是把较长的向量压缩成较短的向量，这样就节省了存储空间，减少了检索时间。比如我们有10million个向量，每条向量512维，那么就占有512*10m*4个字节（假设每个浮点数4个字节），大概占有20GB空间。
  如果我们把每个向量的512维，平均划分成16份，即每一份有32个维度，那么我们将10m向量的起始32维拿出来，进行k-mean聚类（依然是10m个向量，但每个向量长度只有32了），聚成256簇。每一个向量都会属于其中的某一各簇。256个簇可以用8bit的数指示了（从0000 0000到1111 1111，即用0-255来表示），即可以用一个uint8来表示。那么每一个个向量的前32个维度就可以用一个uint8来指示。然后，第二个32维也做类似处理，10m个向量，每个向量大小为32，进行k-means聚类，聚成256簇，继续用所在的簇的标号来表示。这样总共进行16次，就把一个512维的float型向量，压缩成16维的uint8型向量。这16维的每一个维度，都是所在簇的标号。这样进行压缩的话，这10m个向量空间缩小了128倍。
  如果理解了上面一段话，就可以看出来这种乘积量化属于有损压缩，但实际检索效果表现还不错。ProductQuantizer类实现了上面这个过程。该类的成员变量有：nbits_默认为8，就是用8bit来记录簇编号；ksub_是簇的大小256；niter_是k-means迭代步数；dim_是前面的n_（和前面例子的512），即原始向量的维数；dsub_就是每份的维度（就是示例中的32维，就构成了一个32维的子空间），nsubq_划分成多少份（多少个子空间）。lastdsub_记录了最后一个子空间的维度，上例中512维划分成16份，刚好可以划分成功，如果是300维划分成16份，就会有余数，这个余数就是最后一个子空间的维度；centroids_记录了所有子空间的聚类的簇的中心坐标点，大小为有dim_*ksub_大小。
  ProductQuantizer类的主要成员函数为train()，即k-means的训练函数。train中进行nsubq_步循环，每个循环对一个子空间的向量进行聚类。如果向量的总数大于65536，则进行shuffle（用变量perm打乱次序），只聚类前65536个向量。这样，保证了聚类速度，剩余没有参与聚类的向量，则可以直接选择最近的聚类中心进行划分和标识。将相应的数据用mecpy快速复制到xslice中，用了较多的指针操作。
  */
  size_t cutoff;
  // 每个子量化器的大小
  size_t dsub;

  void parseArgs(const std::vector<std::string>& args);
  void printHelp();
  void printBasicHelp();
  void printDictionaryHelp();
  void printTrainingHelp();
  void printQuantizationHelp();
  void save(std::ostream&);
  void load(std::istream&);
  void dump(std::ostream&) const;
};
} // namespace fasttext
