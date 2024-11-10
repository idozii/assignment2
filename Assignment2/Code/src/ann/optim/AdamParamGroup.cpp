/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/class.cc to edit this template
 */

/* 
 * File:   AdamParamGroup.cpp
 * Author: ltsach
 * 
 * Created on October 8, 2024, 1:43 PM
 */

#include "optim/AdamParamGroup.h"

AdamParamGroup::AdamParamGroup(double beta1, double beta2):
    m_beta1(beta1), m_beta2(beta2){
    //Create some maps:
    m_pParams = new xmap<string, xt::xarray<double>*>(&stringHash);
    m_pGrads = new xmap<string, xt::xarray<double>*>(&stringHash);
    m_pFirstMomment = new xmap<string, xt::xarray<double>*>(
            &stringHash,
            0.75,
            0,
            xmap<string, xt::xarray<double>*>::freeValue);
    m_pSecondMomment = new xmap<string, xt::xarray<double>*>(
            &stringHash,
            0.75,
            0,
            xmap<string, xt::xarray<double>*>::freeValue);
    //
    m_step_idx = 1;
    m_beta1_t = m_beta1;
    m_beta2_t = m_beta2;
}

AdamParamGroup::AdamParamGroup(const AdamParamGroup& orig):
    m_beta1(orig.m_beta1), m_beta2(orig.m_beta2){
    m_pParams = new xmap<string, xt::xarray<double>*>(&stringHash);
    m_pGrads = new xmap<string, xt::xarray<double>*>(&stringHash);
    m_pFirstMomment = new xmap<string, xt::xarray<double>*>(
            &stringHash,
            0.75,
            0,
            xmap<string, xt::xarray<double>*>::freeValue);
    m_pSecondMomment = new xmap<string, xt::xarray<double>*>(
            &stringHash,
            0.75,
            0,
            xmap<string, xt::xarray<double>*>::freeValue);
    //copy:
    *m_pParams = *orig.m_pParams;
    *m_pGrads = *orig.m_pGrads;
    *m_pFirstMomment = *orig.m_pFirstMomment;
    *m_pSecondMomment = *orig.m_pSecondMomment;
    //
    m_step_idx = 1;
    m_beta1_t = m_beta1;
    m_beta2_t = m_beta2;
}

AdamParamGroup::~AdamParamGroup() {
    if(m_pFirstMomment != nullptr) delete m_pFirstMomment;
    if(m_pSecondMomment != nullptr) delete m_pSecondMomment;
}

void AdamParamGroup::register_param(string param_name, 
        xt::xarray<double>* ptr_param,
        xt::xarray<double>* ptr_grad){
    //YOUR CODE IS HERE
    m_pParams->put(param_name, ptr_param);
    m_pGrads->put(param_name, ptr_grad);
    m_pFirstMomment->put(param_name, new xt::xarray<double>(ptr_param->shape(), 0.0));
    m_pSecondMomment->put(param_name, new xt::xarray<double>(ptr_param->shape(), 0.0));
}
void AdamParamGroup::register_sample_count(unsigned long long* pCounter){
    m_pCounter = pCounter;
}

void AdamParamGroup::zero_grad(){
    //YOUR CODE IS HERE
    for(auto key: m_pGrads->keys()){
        xt::xarray<double>* grad = m_pGrads->get(key);
        *grad = 0;
    }
}

void AdamParamGroup::step(double lr){
    //YOUR CODE IS HERE
    double alpha = lr * sqrt(1 - m_beta2_t) / (1 - m_beta1_t);
    for(auto key: m_pParams->keys()){
        xt::xarray<double>* param = m_pParams->get(key);
        xt::xarray<double>* grad = m_pGrads->get(key);
        xt::xarray<double>* first_moment = m_pFirstMomment->get(key);
        xt::xarray<double>* second_moment = m_pSecondMomment->get(key);
        //UPDATE first_moment:
        *first_moment = m_beta1 * (*first_moment) + (1 - m_beta1) * (*grad);
        //UPDATE second_moment:
        *second_moment = m_beta2 * (*second_moment) + (1 - m_beta2) * xt::square(*grad);
        //UPDATE param:
        *param -= alpha * (*first_moment) / (xt::sqrt(*second_moment) + 1e-7);
    }
    //UPDATE step_idx:
    m_step_idx += 1;
    m_beta1_t *= m_beta1;
    m_beta2_t *= m_beta2;
}
