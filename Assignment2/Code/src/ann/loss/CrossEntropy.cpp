/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt
 * to change this license Click
 * nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this
 * template
 */

/*
 * File:   CrossEntropy.cpp
 * Author: ltsach
 *
 * Created on August 25, 2024, 2:47 PM
 */

#include "loss/CrossEntropy.h"

#include "ann/functions.h"

CrossEntropy::CrossEntropy(LossReduction reduction) : ILossLayer(reduction) {}

CrossEntropy::CrossEntropy(const CrossEntropy& orig) : ILossLayer(orig) {}

CrossEntropy::~CrossEntropy() {}

double CrossEntropy::forward(xt::xarray<double> X, xt::xarray<double> t) {
  // TODO YOUR CODE IS HERE
  this->m_aCached_Ypred = X;
  this->m_aYtarget = t;
  double loss = cross_entropy(X, t, this->m_eReduction == REDUCE_MEAN);
  return loss;
}
xt::xarray<double> CrossEntropy::backward() {
  // TODO YOUR CODE IS HERE
  xt::xarray<double> Ypred = this->m_aCached_Ypred;
  xt::xarray<double> Ytarget = this->m_aYtarget;
  xt::xarray<double> grad = -Ytarget/(Ypred + 1e-7);
  if (this->m_eReduction == REDUCE_MEAN) {
    grad /= Ypred.shape()[0];
  }
  return grad;
}