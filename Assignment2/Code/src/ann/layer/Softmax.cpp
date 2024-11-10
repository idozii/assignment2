/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt
 * to change this license Click
 * nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this
 * template
 */

/*
 * File:   Softmax.cpp
 * Author: ltsach
 *
 * Created on August 25, 2024, 2:46 PM
 */

#include "layer/Softmax.h"

#include <filesystem>  //require C++17

#include "ann/functions.h"
#include "sformat/fmt_lib.h"
namespace fs = std::filesystem;

Softmax::Softmax(int axis, string name) : m_nAxis(axis) {
  if (trim(name).size() != 0)
    m_sName = name;
  else
    m_sName = "Softmax_" + to_string(++m_unLayer_idx);
}

Softmax::Softmax(const Softmax& orig) {}

Softmax::~Softmax() {}
xt::xarray<double> Softmax::forward(xt::xarray<double> Z) {
  // TODO YOUR CODE IS HERE
  xt::xarray<double> Y = softmax(Z, this->m_nAxis);
  this->m_aCached_Y = Y;
  return Y;
}

xt::xarray<double> Softmax::backward(xt::xarray<double> DY) {
  // TODO YOUR CODE IS HERE
  xt::xarray<double> Z = xt::xarray<double>::from_shape(this->m_aCached_Y.shape());
  if (DY.dimension() < 2){
      xt::xarray<double> jacobian = xt::diag(this->m_aCached_Y) - xt::linalg::outer(this->m_aCached_Y, xt::transpose(this->m_aCached_Y));
      Z = xt::linalg::dot(jacobian, DY);
  }
  else{
    for (size_t index = 0; index < this->m_aCached_Y.shape()[0]; index++){
        xt::xarray<double> row = xt::view(this->m_aCached_Y, index, xt::all());
        xt::xarray<double> jacobian = xt::diag(row) - xt::linalg::outer(row, xt::transpose(row));
        xt::view(Z, index, xt::all()) = xt::linalg::dot(jacobian, xt::view(DY, index, xt::all()));
    }
  }
  return Z;
}

string Softmax::get_desc() {
  string desc = fmt::format("{:<10s}, {:<15s}: {:4d}", "Softmax",
                            this->getname(), m_nAxis);
  return desc;
}
