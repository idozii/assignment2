/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   Heap.h
 * Author: LTSACH
 *
 * Created on 22 August 2020, 18:18
 */

#ifndef HEAP_H
#define HEAP_H
#include <memory.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "heap/IHeap.h"
using namespace std;
/*
 * function pointer: int (*comparator)(T& lhs, T& rhs)
 *      compares objects of type T given in lhs and rhs.
 *      return: sign of (lhs - rhs)
 *              -1: lhs < rhs
 *              0 : lhs == rhs
 *              +1: ls > rhs
 *
 * function pointer: void (*deleteUserData)(Heap<T>* pHeap)
 *      remove user's data in case that T is a pointer type
 *      Users should pass &Heap<T>::free for "deleteUserData"
 *
 */
template <class T>
class Heap : public IHeap<T>
{
public:
  class Iterator; // forward declaration

protected:
  T *elements;                            // a dynamic array to contain user's data
  int capacity;                           // size of the dynamic array
  int count;                              // current count of elements stored in this heap
  int (*comparator)(T &lhs, T &rhs);      // see above
  void (*deleteUserData)(Heap<T> *pHeap); // see above

public:
  Heap(int (*comparator)(T &, T &) = 0, void (*deleteUserData)(Heap<T> *) = 0);

  Heap(const Heap<T> &heap);               // copy constructor
  Heap<T> &operator=(const Heap<T> &heap); // assignment operator

  ~Heap();

  // Inherit from IHeap: BEGIN
  void push(T item);
  T pop();
  const T peek();
  void remove(T item, void (*removeItemData)(T) = 0);
  bool contains(T item);
  int size();
  void heapify(T array[], int size);
  void clear();
  bool empty();
  string toString(string (*item2str)(T &) = 0);
  // Inherit from IHeap: END

  void println(string (*item2str)(T &) = 0)
  {
    cout << toString(item2str) << endl;
  }

  Iterator begin() { return Iterator(this, true); }
  Iterator end() { return Iterator(this, false); }

public:
  /* if T is pointer type:
   *     pass the address of method "free" to Heap<T>'s constructor:
   *     to:  remove the user's data (if needed)
   * Example:
   *  Heap<Point*> heap(&Heap<Point*>::free);
   *  => Destructor will call free via function pointer "deleteUserData"
   */
  static void free(Heap<T> *pHeap)
  {
    for (int idx = 0; idx < pHeap->count; idx++)
      delete pHeap->elements[idx];
  }

private:
  bool aLTb(T &a, T &b) { return compare(a, b) < 0; }
  int compare(T &a, T &b)
  {
    if (comparator != 0)
      return comparator(a, b);
    else
    {
      if (a < b)
        return -1;
      else if (a > b)
        return 1;
      else
        return 0;
    }
  }

  void ensureCapacity(int minCapacity);
  void swap(int a, int b);
  void reheapUp(int position);
  void reheapDown(int position);
  int getItem(T item);

  void removeInternalData();
  void copyFrom(const Heap<T> &heap);

  //////////////////////////////////////////////////////////////////////
  ////////////////////////  INNER CLASSES DEFNITION ////////////////////
  //////////////////////////////////////////////////////////////////////

public:
  // Iterator: BEGIN
  class Iterator
  {
  private:
    Heap<T> *heap;
    int cursor;

  public:
    Iterator(Heap<T> *heap = 0, bool begin = 0)
    {
      this->heap = heap;
      if (begin && (heap != 0))
        cursor = 0;
      if (!begin && (heap != 0))
        cursor = heap->size();
    }
    Iterator &operator=(const Iterator &iterator)
    {
      this->heap = iterator.heap;
      this->cursor = iterator.cursor;
      return *this;
    }

    T &operator*() { return this->heap->elements[cursor]; }
    bool operator!=(const Iterator &iterator)
    {
      return this->cursor != iterator.cursor;
    }
    // Prefix ++ overload
    Iterator &operator++()
    {
      cursor++;
      return *this;
    }
    // Postfix ++ overload
    Iterator operator++(int)
    {
      Iterator iterator = *this;
      ++*this;
      return iterator;
    }
    void remove(void (*removeItemData)(T) = 0)
    {
      this->heap->remove(this->heap->elements[cursor], removeItemData);
    }
  };
  // Iterator: END
};

//////////////////////////////////////////////////////////////////////
////////////////////////     METHOD DEFNITION      ///////////////////
//////////////////////////////////////////////////////////////////////

template <class T>
Heap<T>::Heap(int (*comparator)(T &, T &), void (*deleteUserData)(Heap<T> *))
{
  // YOUR CODE IS HERE
  this->capacity = 10;
  this->count = 0;
  this->comparator = comparator;
  this->elements = new T[capacity];
  this->deleteUserData = deleteUserData;
}
template <class T>
Heap<T>::Heap(const Heap<T> &heap)
{
  // YOUR CODE IS HERE
  this->copyFrom(heap);
  this->deleteUserData = 0;
}

template <class T>
Heap<T> &Heap<T>::operator=(const Heap<T> &heap)
{
  // YOUR CODE IS HERE
  if (this != &heap)
  {
    this->removeInternalData();
    this->copyFrom(heap);
    this->deleteUserData = 0;
  }
  return *this;
}

template <class T>
Heap<T>::~Heap()
{
  // YOUR CODE IS HERE
  this->removeInternalData();
}

template <class T>
void Heap<T>::push(T item)
{ // item  = 25
  // YOUR CODE IS HERE
  ensureCapacity(count + 1);
  elements[count] = item;
  reheapUp(count);
  count++;
}
/*
      18
     /  \
    15   13
    /
  25
 =>
      25
     /  \
    18   13
    /
  15
=> array: [25, 18, 13, 15, , ]
           0   1    2   3

 */
template <class T>
T Heap<T>::pop()
{
  // YOUR CODE IS HERE
  if (count == 0)
    throw std::underflow_error("Calling peek with an empty heap.");
  T root = elements[0];
  elements[0] = elements[count - 1];
  count--;
  reheapDown(0);
  return root;
}

/*
      15
     /  \
    18   13
 => ReheapDown
      18
     /  \
    15   13
=> Array: [18, 15, 13, , , ]
 */

template <class T>
const T Heap<T>::peek()
{
  // YOUR CODE IS HERE
  if (count == 0)
    throw std::underflow_error("Calling peek with an empty heap.");
  return elements[0];
}

template <class T>
void Heap<T>::remove(T item, void (*removeItemData)(T))
{
  // YOUR CODE IS HERE
  int idx = getItem(item);
  if (idx == -1)
    return;
  if (removeItemData != nullptr)
    removeItemData(elements[idx]);
  elements[idx] = elements[count - 1];
  count--;
  reheapDown(idx);
}

template <class T>
bool Heap<T>::contains(T item)
{
  // YOUR CODE IS HERE
  return getItem(item) != -1;
}

template <class T>
int Heap<T>::size()
{
  // YOUR CODE IS HERE
  return count;
}

template <class T>
void Heap<T>::heapify(T array[], int size)
{
  // YOUR CODE IS HERE
  for (int idx = 0; idx < size; idx++)
  {
    push(array[idx]);
  }
}

template <class T>
void Heap<T>::clear()
{
  // YOUR CODE IS HERE
  this->removeInternalData();
  this->capacity = 10;
  this->count = 0;
  this->elements = new T[capacity];
}

template <class T>
bool Heap<T>::empty()
{
  // YOUR CODE IS HERE
  return count == 0;
}

template <class T>
string Heap<T>::toString(string (*item2str)(T &))
{
  stringstream os;
  if (item2str != 0)
  {
    os << "[";
    for (int idx = 0; idx < count - 1; idx++)
      os << item2str(elements[idx]) << ",";
    if (count > 0)
      os << item2str(elements[count - 1]);
    os << "]";
  }
  else
  {
    os << "[";
    for (int idx = 0; idx < count - 1; idx++)
      os << elements[idx] << ",";
    if (count > 0)
      os << elements[count - 1];
    os << "]";
  }
  return os.str();
}

//////////////////////////////////////////////////////////////////////
//////////////////////// (private) METHOD DEFNITION //////////////////
//////////////////////////////////////////////////////////////////////

template <class T>
void Heap<T>::ensureCapacity(int minCapacity)
{
  if (minCapacity >= capacity)
  {
    // re-allocate
    int old_capacity = capacity;
    capacity = old_capacity + (old_capacity >> 2);
    try
    {
      T *new_data = new T[capacity];
      // OLD: memcpy(new_data, elements, capacity*sizeof(T));
      memcpy(new_data, elements, old_capacity * sizeof(T));
      delete[] elements;
      elements = new_data;
    }
    catch (std::bad_alloc e)
    {
      e.what();
    }
  }
}

template <class T>
void Heap<T>::swap(int a, int b)
{
  T temp = this->elements[a];
  this->elements[a] = this->elements[b];
  this->elements[b] = temp;
}

template <class T>
void Heap<T>::reheapUp(int position)
{
  // YOUR CODE IS HERE
  int parent = (position - 1) / 2;
  // position > parent
  while (position > 0 && aLTb(elements[position], elements[parent]))
  {
    swap(position, parent);
    position = parent;
    parent = (position - 1) / 2;
  }
}

template <class T>
void Heap<T>::reheapDown(int position)
{
  // YOUR CODE IS HERE
  int left = 2 * position + 1;
  int right = 2 * position + 2;
  int max = position;
  while (left < count)
  {
    if (aLTb(elements[left], elements[max]))
      max = left;
    if (right < count && aLTb(elements[right], elements[max]))
      max = right;
    if (max == position)
      break;
    swap(position, max);
    position = max;
    left = 2 * position + 1;
    right = 2 * position + 2;
  }
}

template <class T>
int Heap<T>::getItem(T item)
{
  // YOUR CODE IS HERE
  for (int idx = 0; idx < count; idx++)
  {
    if constexpr (std::is_pointer<T>::value)
    {
      if (*(elements[idx]) == *item)
        return idx;
    }
    else
    {
      if (elements[idx] == item)
        return idx;
    }
  }
  return -1;
}

template <class T>
void Heap<T>::removeInternalData()
{
  // YOUR CODE IS HERE
  if (deleteUserData != nullptr)
    deleteUserData(this);
  delete[] elements;
}

template <class T>
void Heap<T>::copyFrom(const Heap<T> &heap)
{
  capacity = heap.capacity;
  count = heap.count;
  elements = new T[capacity];
  this->comparator = heap.comparator;
  this->deleteUserData = heap.deleteUserData;

  // Copy items from heap:
  for (int idx = 0; idx < count; idx++)
  {
    this->elements[idx] = heap.elements[idx];
  }
}

#endif /* HEAP_H */
