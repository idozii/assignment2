/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/*
 * File:   xMap.h
 * Author: ltsach
 *
 * Created on October 11, 2024, 7:08 PM
 */

#ifndef XMAP_H
#define XMAP_H
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <memory.h>
using namespace std;

#include "list/DLinkedList.h"
#include "hash/IMap.h"

/*
 * xMap<K, V>:
 *  + K: key type
 *  + V: value type
 *  For example:
 *      xMap<string, int>: map from string to int
 */
template <class K, class V>
class xMap : public IMap<K, V>
{
public:
  class Entry; // forward declaration

protected:
  DLinkedList<Entry *> *table; // array of DLinkedList objects
  int capacity;                // size of table
  int count;                   // number of entries stored hash-map
  float loadFactor;            // define max number of entries can be stored (< (loadFactor * capacity))

  int (*hashCode)(K &, int);          // hasCode(K key, int tableSize): tableSize means capacity
  bool (*keyEqual)(K &, K &);         // keyEqual(K& lhs, K& rhs): test if lhs == rhs
  bool (*valueEqual)(V &, V &);       // valueEqual(V& lhs, V& rhs): test if lhs == rhs
  void (*deleteKeys)(xMap<K, V> *);   // deleteKeys(xMap<K,V>* pMap): delete all keys stored in pMap
  void (*deleteValues)(xMap<K, V> *); // deleteValues(xMap<K,V>* pMap): delete all values stored in pMap

public:
  xMap(
      int (*hashCode)(K &, int), // require
      float loadFactor = 0.75f,
      bool (*valueEqual)(V &, V &) = 0,
      void (*deleteValues)(xMap<K, V> *) = 0,
      bool (*keyEqual)(K &, K &) = 0,
      void (*deleteKeys)(xMap<K, V> *) = 0);

  xMap(const xMap<K, V> &map);                  // copy constructor
  xMap<K, V> &operator=(const xMap<K, V> &map); // assignment operator
  ~xMap();

  // Inherit from IMap:BEGIN
  V put(K key, V value);
  V &get(K key);
  V remove(K key, void (*deleteKeyInMap)(K) = 0);
  bool remove(K key, V value, void (*deleteKeyInMap)(K) = 0, void (*deleteValueInMap)(V) = 0);
  bool containsKey(K key);
  bool containsValue(V value);
  bool empty();
  int size();
  void clear();
  string toString(string (*key2str)(K &) = 0, string (*value2str)(V &) = 0);
  DLinkedList<K> keys();
  DLinkedList<V> values();
  DLinkedList<int> clashes();
  // Inherit from IMap:END

  // Show map on screen: need to convert key to string (key2str) and value2str
  void println(string (*key2str)(K &) = 0, string (*value2str)(V &) = 0)
  {
    cout << this->toString(key2str, value2str) << endl;
  }
  int getCapacity()
  {
    return capacity;
  }

  ///////////////////////////////////////////////////
  // STATIC METHODS: BEGIN
  //      * Used to create xMap objects
  ///////////////////////////////////////////////////
  /*
   * sample hash function for keys of types integer and string:
   */
  static int intKeyHash(int &key, int capacity)
  {
    return key % capacity;
  }
  static int stringKeyHash(string &key, int capacity)
  {
    long long int sum = 0;
    for (int idx = 0; idx < key.length(); idx++)
      sum += key[idx];
    return sum % capacity;
  }
  /*
   * freeKey(xMap<K,V> *pMap):
   *  Purpose: a typical function for deleting keys stored in map
   *  WHEN to use:
   *      1. K is a pointer type; AND
   *      2. Users need xMap to free keys
   */
  static void freeKey(xMap<K, V> *pMap)
  {
    for (int idx = 0; idx < pMap->capacity; idx++)
    {
      DLinkedList<Entry *> list = pMap->table[idx];
      for (auto pEntry : list)
      {
        delete pEntry->key;
      }
    }
  }
  /*
   * freeValue(xMap<K,V> *pMap):
   *  Purpose: a typical function for deleting values stored in map
   *  WHEN to use:
   *      1. V is a pointer type; AND
   *      2. Users need xMap to free values
   */
  static void freeValue(xMap<K, V> *pMap)
  {
    for (int idx = 0; idx < pMap->capacity; idx++)
    {
      DLinkedList<Entry *> list = pMap->table[idx];
      for (auto pEntry : list)
      {
        delete pEntry->value;
      }
    }
  }
  /*
   * deleteEntry(Entry* ptr): a function pointer to delete pointer to Entry
   */
  static void deleteEntry(Entry *ptr)
  {
    delete ptr;
  }
  ///////////////////////////////////////////////////
  // STATIC METHODS: END
  //      * Used to create xMap objects
  ///////////////////////////////////////////////////

protected:
  ////////////////////////////////////////////////////////
  ////////////////////////  UTILITIES ////////////////////
  ////////////////////////////////////////////////////////
  void ensureLoadFactor(int minCapacity);
  // future version:
  //   should add a method to trim table shorter when removing key (and value)
  void rehash(int newCapacity);
  void removeInternalData();
  void copyMapFrom(const xMap<K, V> &map);
  void moveEntries(
      DLinkedList<Entry *> *oldTable, int oldCapacity,
      DLinkedList<Entry *> *newTable, int newCapacity);

  /*
   * keyEQ(K& lhs, K& rhs): verify the equality of two keys
   */
  bool keyEQ(K &lhs, K &rhs)
  {
    if (keyEqual != 0)
      return keyEqual(lhs, rhs);
    else
      return lhs == rhs;
  }
  /*
   *  valueEQ(V& lhs, V& rhs): verify the equality of two values
   */
  bool valueEQ(V &lhs, V &rhs)
  {
    if (valueEqual != 0)
      return valueEqual(lhs, rhs);
    else
      return lhs == rhs;
  }

  //////////////////////////////////////////////////////////////////////
  ////////////////////////  INNER CLASSES DEFNITION ////////////////////
  //////////////////////////////////////////////////////////////////////
public:
  // Entry: BEGIN
  class Entry
  {
  private:
    K key;
    V value;
    friend class xMap<K, V>;

  public:
    Entry(K key, V value)
    {
      this->key = key;
      this->value = value;
    }
  };
  // Entry: END
};

//////////////////////////////////////////////////////////////////////
////////////////////////     METHOD DEFNITION      ///////////////////
//////////////////////////////////////////////////////////////////////

template <class K, class V>
xMap<K, V>::xMap(
    int (*hashCode)(K &, int),
    float loadFactor,
    bool (*valueEqual)(V &lhs, V &rhs),
    void (*deleteValues)(xMap<K, V> *),
    bool (*keyEqual)(K &lhs, K &rhs),
    void (*deleteKeys)(xMap<K, V> *pMap))
{
  // YOUR CODE IS HERE
  this->hashCode = hashCode;
  this->loadFactor = loadFactor;
  this->valueEqual = valueEqual;
  this->deleteValues = deleteValues;
  this->keyEqual = keyEqual;
  this->deleteKeys = deleteKeys;
  this->count = 0;
  this->capacity = 10;
  table = new DLinkedList<Entry *>[this->capacity];
}

template <class K, class V>
xMap<K, V>::xMap(const xMap<K, V> &map)
{
  // YOUR CODE IS HERE
  this->copyMapFrom(map);
}

template <class K, class V>
xMap<K, V> &xMap<K, V>::operator=(const xMap<K, V> &map)
{
  // YOUR CODE IS HERE
  if (this != &map)
  {
    this->removeInternalData();
    this->copyMapFrom(map);
    this->deleteValues = 0;
    this->deleteKeys = 0;
  }
  return *this;
}

template <class K, class V>
xMap<K, V>::~xMap()
{
  // YOUR CODE IS HERE
  this->removeInternalData();
}

//////////////////////////////////////////////////////////////////////
//////////////////////// IMPLEMENTATION of IMap    ///////////////////
//////////////////////////////////////////////////////////////////////

template <class K, class V>
V xMap<K, V>::put(K key, V value)
{
  int index = this->hashCode(key, capacity);
  V retValue = value;
  // YOUR CODE IS HERE
  DLinkedList<Entry *> &dll = this->table[index];
  bool contain = this->containsKey(key);
  if (contain)
  {
    for (auto it = dll.begin(); it != dll.end(); it++)
    {
      if (keyEQ((*it)->key, key))
      {
        retValue = (*it)->value;
        (*it)->value = value;
        break;
      }
    }
  }
  else
  {
    Entry *entry = new Entry(key, value);
    dll.add(entry);
    this->count++;
    ensureLoadFactor(this->count);
  }
  return retValue;
}

template <class K, class V>
V &xMap<K, V>::get(K key)
{
  // Calculate the index using the hash code of the key
  int index = hashCode(key, capacity);

  // Retrieve the list at the calculated index
  DLinkedList<Entry *> &dll = this->table[index];

  // Check if the list is not null and the key exists in the map
  if (!dll.empty() && this->containsKey(key)) 
  {
    // Iterate through the list to find the entry with the matching key
    for (auto it = dll.begin(); it != dll.end(); it++)
    {
      // Check if the current entry's key matches the given key
      if (keyEQ((*it)->key, key))
        return (*it)->value; // Return the value associated with the key
    }
  }

  // If the key is not found, throw a KeyNotFound exception
  stringstream os;
  os << "key (" << key << ") is not found";
  throw KeyNotFound(os.str());
}

template <class K, class V>
V xMap<K, V>::remove(K key, void (*deleteKeyInMap)(K))
{
  int index = hashCode(key, capacity);
  V retValue;

  // Retrieve the list at the calculated index
  DLinkedList<Entry *> &dll = this->table[index];

  if (this->containsKey(key))
  {
    for (auto it = dll.begin(); it != dll.end(); it++)
    {
      if (keyEQ((*it)->key, key))
      {
        // Backup the value to return
        retValue = (*it)->value;

        // Free the key if deleteKeyInMap is not NULL
        if (deleteKeyInMap)
          deleteKeyInMap(key);

        // Remove the Entry and free the memory of the Entry
        dll.removeItem(*it, this->deleteEntry);
        this->count--;
        return retValue;
      }
    }
  }

  // Key not found, throw an exception
  stringstream os;
  os << "key (" << key << ") is not found";
  throw KeyNotFound(os.str());
}

template <class K, class V>
bool xMap<K, V>::remove(K key, V value, void (*deleteKeyInMap)(K), void (*deleteValueInMap)(V))
{
  // Calculate the index using the hash code of the key
  int index = this->hashCode(key, this->capacity);

  // Retrieve the list at the calculated index
  DLinkedList<Entry *> &dll = this->table[index];

  // Check if the list is not null

  // Iterate through the list to find the entry with the matching key and value
  for (auto it = dll.begin(); it != dll.end(); it++)
  {
    // Check if the current entry's key and value match the given key and value
    if (keyEQ((*it)->key, key) && valueEQ((*it)->value, value))
    {
      // Free the key if deleteKeyInMap is not NULL
      if (deleteKeyInMap != nullptr)
        deleteKeyInMap(key);

      // Free the value if deleteValueInMap is not NULL
      if (deleteValueInMap != nullptr)
        deleteValueInMap(value);

      // Remove the Entry and free the memory of the Entry
      dll.removeItem(*it, this->deleteEntry);
      this->count--;
      // Return true indicating the key-value pair was found and removed
      return true;
    }
  }

  // Return false if the key-value pair was not found
  return false;
}

template <class K, class V>
bool xMap<K, V>::containsKey(K key)
{
  // Calculate the index using the hash code of the key
  int idx = hashCode(key, this->capacity);

  // Retrieve the list at the calculated index
  DLinkedList<Entry *> &dll = this->table[idx];

  // Check if the list is not null
  if(dll.empty()){
    return false;
  }
  // Iterate through the list to find the entry with the matching key
  for (auto it = dll.begin(); it != dll.end(); it++)
  {
    // Check if the current entry's key matches the given key
    if (keyEQ((*it)->key, key))
      return true; // Return true if the key is found
  }

  // Return false if the key is not found
  return false;
}

template <class K, class V>
bool xMap<K, V>::containsValue(V value)
{
  // Iterate through each index in the table
  for (int index = 0; index < this->capacity; index++)
  {
    // Retrieve the list at the current index
    DLinkedList<Entry *> *dll = &(this->table[index]);

    // Check if the list is not null
    if (dll != nullptr)
    {
      // Iterate through the list to find the entry with the matching value
      for (auto it = dll->begin(); it != dll->end(); it++)
      {
        // Check if the current entry's value matches the given value
        if (valueEQ((*it)->value, value))
          return true; // Return true if the value is found
      }
    }
  }

  // Return false if the value is not found
  return false;
}

template <class K, class V>
bool xMap<K, V>::empty()
{
  // YOUR CODE IS HERE
  return (this->count == 0);
}

template <class K, class V>
int xMap<K, V>::size()
{
  return this->count;
}

template <class K, class V>
void xMap<K, V>::clear()
{
  // YOUR CODE IS HERE
  this->removeInternalData();
  this->count = 0;
  this->capacity = 10;
  this->table = new DLinkedList<Entry *>[this->capacity];
}

template <class K, class V>
DLinkedList<K> xMap<K, V>::keys()
{
  // YOUR CODE IS HERE
  DLinkedList<K> list;
  if (!this->empty())
  {
    for (int index = 0; index < this->capacity; index++)
    {
      DLinkedList<Entry *> *dll = &(this->table[index]);
      if (dll != nullptr)
      {
        for (auto it = dll->begin(); it != dll->end(); it++)
          list.add((*it)->key);
      }
    }
  }
  return list;
}

template <class K, class V>
DLinkedList<V> xMap<K, V>::values()
{
  // YOUR CODE IS HERE
  DLinkedList<V> list;
  if (!this->empty())
  {
    for (int index = 0; index < this->capacity; index++)
    {
      DLinkedList<Entry *> *dll = &(this->table[index]);
      if (dll != nullptr)
      {
        for (auto it = dll->begin(); it != dll->end(); it++)
          list.add((*it)->value);
      }
    }
  }
  return list;
}

template <class K, class V>
DLinkedList<int> xMap<K, V>::clashes()
{
  // YOUR CODE IS HERE
  DLinkedList<int> list;
  if (!this->empty())
  {
    for (int index = 0; index < this->capacity; index++)
    {
      DLinkedList<Entry *> *dll = &(this->table[index]);
      if (dll != nullptr)
        list.add(dll->size());
    }
  }
  else
  {
    for (int index = 0; index < this->capacity; index++)
      list.add(0);
  }
  return list;
}

template <class K, class V>
string xMap<K, V>::toString(string (*key2str)(K &), string (*value2str)(V &))
{
  stringstream os;
  string mark(50, '=');
  os << mark << endl;
  os << setw(12) << left << "capacity: " << capacity << endl;
  os << setw(12) << left << "size: " << count << endl;
  for (int idx = 0; idx < capacity; idx++)
  {
    DLinkedList<Entry *> list = table[idx];

    os << setw(4) << left << idx << ": ";
    stringstream itemos;
    for (auto pEntry : list)
    {
      itemos << " (";

      if (key2str != 0)
        itemos << key2str(pEntry->key);
      else
        itemos << pEntry->key;
      itemos << ",";
      if (value2str != 0)
        itemos << value2str(pEntry->value);
      else
        itemos << pEntry->value;

      itemos << ");";
    }
    string valuestr = itemos.str();
    if (valuestr.length() > 0)
      valuestr = valuestr.substr(0, valuestr.length() - 1);
    os << valuestr << endl;
  }
  os << mark << endl;

  return os.str();
}

////////////////////////////////////////////////////////
//                  UTILITIES
//              Code are provided
////////////////////////////////////////////////////////

/*
 * moveEntries:
 *  Purpose: move all entries in the old hash table (oldTable) to the new table (newTable)
 */
template <class K, class V>
void xMap<K, V>::moveEntries(
    DLinkedList<Entry *> *oldTable, int oldCapacity,
    DLinkedList<Entry *> *newTable, int newCapacity)
{
  for (int old_index = 0; old_index < oldCapacity; old_index++)
  {
    DLinkedList<Entry *> &oldList = oldTable[old_index];
    for (auto oldEntry : oldList)
    {
      int new_index = this->hashCode(oldEntry->key, newCapacity);
      DLinkedList<Entry *> &newList = newTable[new_index];
      newList.add(oldEntry);
    }
  }
}

/*
 * ensureLoadFactor:
 *  Purpose: ensure the load-factor,
 *      i.e., the maximum number of entries does not exceed "loadFactor*capacity"
 */
template <class K, class V>
void xMap<K, V>::ensureLoadFactor(int current_size)
{
  int maxSize = (int)(loadFactor * capacity);

  // cout << "ensureLoadFactor: count = " << count << "; maxSize = " << maxSize << endl;
  if (current_size > maxSize)
  {
    int oldCapacity = capacity;
    // int newCapacity = oldCapacity + (oldCapacity >> 1);
    int newCapacity = 1.5 * oldCapacity;
    rehash(newCapacity);
  }
}

/*
 * rehash(int newCapacity)
 *  Purpose:
 *      1. create a new hash-table with newCapacity, and
 *      2. move all the old table to to new one
 *      3. free the old table.
 */
template <class K, class V>
void xMap<K, V>::rehash(int newCapacity)
{
  DLinkedList<Entry *> *pOldMap = this->table;
  int oldCapacity = capacity;

  // Create new table:
  this->table = new DLinkedList<Entry *>[newCapacity];
  this->capacity = newCapacity; // keep "count" not changed

  moveEntries(pOldMap, oldCapacity, this->table, newCapacity);

  // remove old data: only remove nodes in list, no entry
  for (int idx = 0; idx < oldCapacity; idx++)
  {
    DLinkedList<Entry *> &list = pOldMap[idx];
    list.clear();
  }
  // Remove oldTable
  delete[] pOldMap;
}

/*
 * removeInternalData:
 *  Purpose:
 *      1. Remove all keys and values if users require,
 *          i.e., deleteKeys and deleteValues are not nullptr
 *      2. Remove all entry
 *      3. Remove table
 */
template <class K, class V>
void xMap<K, V>::removeInternalData()
{
  // Remove user's data
  if(this->deleteKeys != nullptr)
    this->deleteKeys(this);
  if(this->deleteValues != nullptr)
    this->deleteValues(this);

  // Remove all entries in the current map
  for (int idx = 0; idx < this->capacity; idx++)
  {
    DLinkedList<Entry *> &list = this->table[idx];
    for (auto pEntry : list)
      delete pEntry;
    list.clear();
  }

  // Remove table
  delete[] table;
}

/*
 * copyMapFrom(const xMap<K,V>& map):
 *  Purpose:
 *      1. Remove all the entries of the current hash-table
 *      2. Copy (Shallow-copy only) all the entries in the input map
 *          to the current table
 */

template <class K, class V>
void xMap<K, V>::copyMapFrom(const xMap<K, V> &map)
{
  this->capacity = map.capacity;
  this->count = 0;
  this->table = new DLinkedList<Entry *>[capacity];

  this->hashCode = map.hashCode;
  this->loadFactor = map.loadFactor;

  this->valueEqual = map.valueEqual;
  this->keyEqual = map.keyEqual;
  // SHOULD NOT COPY: deleteKeys, deleteValues => delete ONLY TIME in map if needed

  // copy entries
  for (int idx = 0; idx < map.capacity; idx++)
  {
    DLinkedList<Entry *> &list = map.table[idx];
    for (auto pEntry : list)
    {
      this->put(pEntry->key, pEntry->value);
    }
  }
}
#endif /* XMAP_H */