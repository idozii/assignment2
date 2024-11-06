/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt
 * to change this license Click
 * nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this
 * template
 */

/*
 * File:   ReLU.cpp
 * Author: ltsach
 *
 * Created on August 25, 2024, 2:44 PM
 */

#include "layer/ReLU.h"

#include "ann/functions.h"
#include "sformat/fmt_lib.h"

ReLU::ReLU(string name) {
  if (trim(name).size() != 0)
    m_sName = name;
  else
    m_sName = "ReLU_" + to_string(++m_unLayer_idx);
}

ReLU::ReLU(const ReLU& orig) { m_sName = "ReLU_" + to_string(++m_unLayer_idx); }

ReLU::~ReLU() {}

xt::xarray<double> ReLU::forward(xt::xarray<double> X) {
  // TODO YOUR CODE IS HERE
  m_aMask = xt::empty<bool>(X.shape());
  m_aMask = X >= 0;
  xt::xarray<double> Y = xt::zeros<double>(X.shape());
  Y = xt::where(m_aMask, X, 0);
  return Y;
}
xt::xarray<double> ReLU::backward(xt::xarray<double> DY) {
  // TODO YOUR CODE IS HERE
  xt::xarray<double> DX = xt::empty<double>(DY.shape());
  DX = xt::where(m_aMask, DY, 0);
  return DX;
}

string ReLU::get_desc() {
  string desc = fmt::format("{:<10s}, {:<15s}:", "ReLU", this->getname());
  return desc;
}