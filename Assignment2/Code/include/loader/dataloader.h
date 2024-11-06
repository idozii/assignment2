/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "tensor/xtensor_lib.h"
#include "loader/dataset.h"

using namespace std;

template<typename DType, typename LType>
class DataLoader{
public:
    class Iterator; //forward declaration for class Iterator
    
private:
     Dataset<DType, LType>* ptr_dataset;
     int batch_size;
     bool shuffle;
     bool drop_last;
     int nbatch;
     ulong_tensor item_indices;
     int m_seed;
     xt::xarray<DType> new_data;
     xt::xarray<LType> new_label;
    
public:
    DataLoader(Dataset<DType, LType>* ptr_dataset, 
               int batch_size, bool shuffle=true, 
               bool drop_last=false, int seed=-1)
                    : ptr_dataset(ptr_dataset), 
                    batch_size(batch_size), 
                    shuffle(shuffle),
                    m_seed(seed){
               nbatch = ptr_dataset->len()/batch_size;
               item_indices = xt::arange(0, ptr_dataset->len());
               if(ptr_dataset->len()<batch_size){
                    if(!drop_last){
                         nbatch = 1;
                    } else {
                         nbatch = 0;
                    }
               }
               auto data = ptr_dataset->get_data();
               auto label = ptr_dataset->get_label();
               if(shuffle){
                    if(m_seed >=0){
                         xt::random::seed(m_seed);
                    }
                    xt::random::shuffle(item_indices);
               }
               new_data = xt::empty<DType>(ptr_dataset->get_data_shape());
               new_label = xt::empty<LType>(ptr_dataset->get_label_shape());
               for(int i = 0; i < ptr_dataset->len(); i++){
                    if(new_data.shape()[0]==0){
                         DataLabel<DType, LType> item = ptr_dataset->getitem(0);
                         new_data = item.getData();
                    } else {
                         DataLabel<DType, LType> item = ptr_dataset->getitem(item_indices[i]);
                         xt::view(new_data, i) = item.getData();
                    }
                    if(label.shape()[0]==0){
                         DataLabel<DType, LType> item = ptr_dataset->getitem(0);
                         new_label = item.getLabel();
                    } else {
                         DataLabel<DType, LType> item = ptr_dataset->getitem(item_indices[i]);
                         xt::view(new_label, i) = item.getLabel();
                    }
               }
    }
    virtual ~DataLoader(){}
    
    //New method: from V2: begin
    int get_batch_size(){ return batch_size; }
    int get_sample_count(){ return ptr_dataset->len(); }
    int get_total_batch(){return int(ptr_dataset->len()/batch_size); }
    
    //New method: from V2: end
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////
public:
    Iterator begin(){
        //YOUR CODE IS HERE
        return Iterator(this, 0);
    }
    Iterator end(){
        //YOUR CODE IS HERE
        return Iterator(this, nbatch);
    }
    
    //BEGIN of Iterator

    //YOUR CODE IS HERE: to define iterator
    class Iterator {
    public:
        Iterator(DataLoader* loader, int cursor) : loader(loader), cursor(cursor) {};

        Iterator& operator=(const Iterator& iterator) {
            loader = iterator.loader;
            cursor = iterator.cursor;
            return *this;
        }

        Iterator& operator++() {
            ++cursor;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++cursor;
            return tmp;
        }

        bool operator!=(const Iterator& other) const {
            return cursor != other.cursor;
        }

        Batch<DType, LType> operator*() const {
               int start = cursor * loader->batch_size;
               int end = start + loader->batch_size;
               auto data = loader->new_data;
               if(cursor == loader->nbatch-1){
                    if(loader->drop_last){
                         end = loader->ptr_dataset->len() - loader->ptr_dataset->len()%loader->batch_size;
                    } else {
                         end = loader->ptr_dataset->len();
                    }
               }
               xt::xarray<DType> batch_data = xt::view(data, xt::range(start, end), xt::all());
               xt::xarray<LType> batch_label;
               if(loader->new_label.dimension() != 0){
                    auto label = loader->new_label;
                    batch_label = xt::view(label, xt::range(start, end), xt::all());
               }
               if(data.dimension()!=0){
                    batch_data = xt::view(data, xt::range(start, end), xt::all());
               }
               return Batch<DType, LType>(batch_data, batch_label);
        }

    private:
        DataLoader* loader;
        int cursor;
    };
    //END of Iterator
    
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};


#endif /* DATALOADER_H */

