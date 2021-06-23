/* Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tree/tree.h"

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

// logging
#include "pybind11/pybind11.h"

#ifdef LOG
#define LOG_WARNING(w) LOG(WARNING) << w;
#else
#include <iostream>
#define LOG_WARNING(w) std::cerr << w << "\n";
#endif

#ifndef DCHECK
#define DCHECK(stmt)
#endif

namespace py = pybind11;

namespace tree {
namespace {

struct DecrementsPyRefcount {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

// PyObjectPtr wraps an underlying Python object and decrements the
// reference count in the destructor.
//
// This class does not acquire the GIL in the destructor, so the GIL must be
// held when the destructor is called.
using PyObjectPtr = std::unique_ptr<PyObject, DecrementsPyRefcount>;

const int kMaxItemsInCache = 1024;

bool WarnedThatSetIsNotSequence = false;

bool IsString(PyObject* o) {
  return PyBytes_Check(o) || PyByteArray_Check(o) || PyUnicode_Check(o);
}

void AddToStringStream(std::ostringstream&) {}

template <typename T, typename... Args>
void AddToStringStream(std::ostringstream& sstream, T&& arg, Args&&... args) {
  sstream << std::forward<T>(arg);
  AddToStringStream(sstream, std::forward<Args>(args)...);
}

template <typename... Args>
std::string StrCat(Args&&... args) {
  std::ostringstream sstream;
  AddToStringStream(sstream, std::forward<Args>(args)...);
  return sstream.str();
}

template<typename T>
std::unique_ptr<T> make_unique_from_py_object_ptr(PyObject* arg)
{
    return std::unique_ptr<T>(new T(arg));
}

// Equivalent to Python's 'o.__class__.__name__'
// Note that '__class__' attribute is set only in new-style classes.
// A lot of tensorflow code uses __class__ without checks, so it seems like
// we only support new-style classes.
std::string GetClassName(PyObject* o) {
  // __class__ is equivalent to type() for new style classes.
  // type() is equivalent to PyObject_Type()
  // (https://docs.python.org/3.5/c-api/object.html#c.PyObject_Type)
  // PyObject_Type() is equivalent to o->ob_type except for Py_INCREF, which
  // we don't need here.
  PyTypeObject* type = o->ob_type;

  // __name__ is the value of `tp_name` after the last '.'
  // (https://docs.python.org/2/c-api/typeobj.html#c.PyTypeObject.tp_name)
  std::string name(type->tp_name);
  size_t pos = name.rfind('.');
  if (pos != std::string::npos) {
    name.erase(0, pos + 1);
  }
  return name;
}

std::string PyObjectToString(PyObject* o) {
  if (o == nullptr) {
    return "<null object>";
  }
  PyObject* str = PyObject_Str(o);
  if (str) {
    std::string s(PyUnicode_AsUTF8(str));
    Py_DECREF(str);
    return StrCat("type=", GetClassName(o), " str=", s);
  } else {
    return "<failed to execute str() on object>";
  }
}

class CachedTypeCheck {
 public:
  explicit CachedTypeCheck(std::function<int(PyObject*)> ternary_predicate)
      : ternary_predicate_(std::move(ternary_predicate)) {}

  ~CachedTypeCheck() {
    for (const auto& pair : type_to_sequence_map_) {
      Py_DECREF(pair.first);
    }
  }

  // Caches successful executions of the one-argument (PyObject*) callable
  // "ternary_predicate" based on the type of "o". -1 from the callable
  // indicates an unsuccessful check (not cached), 0 indicates that "o"'s type
  // does not match the predicate, and 1 indicates that it does. Used to avoid
  // calling back into Python for expensive isinstance checks.
  int CachedLookup(PyObject* o) {
    // Try not to return to Python - see if the type has already been seen
    // before.
    auto* type = Py_TYPE(o);

    {
      auto it = type_to_sequence_map_.find(type);
      if (it != type_to_sequence_map_.end()) {
        return it->second;
      }
    }

    int check_result = ternary_predicate_(o);

    if (check_result == -1) {
      return -1;  // Type check error, not cached.
    }

    // NOTE: This is never decref'd as long as the object lives, which is likely
    // forever, but we don't want the type to get deleted as long as it is in
    // the map. This should not be too much of a leak, as there should only be a
    // relatively small number of types in the map, and an even smaller number
    // that are eligible for decref. As a precaution, we limit the size of the
    // map to 1024.
    {
      if (type_to_sequence_map_.size() < kMaxItemsInCache) {
        Py_INCREF(type);
        type_to_sequence_map_.insert({type, check_result});
      }
    }

    return check_result;
  }

 private:
  std::function<int(PyObject*)> ternary_predicate_;
  std::unordered_map<PyTypeObject*, bool> type_to_sequence_map_;
};

py::object GetCollectionsSequenceType() {
  static py::object type =
      py::module::import("collections.abc").attr("Sequence");
  return type;
}

py::object GetCollectionsMappingType() {
  static py::object type =
      py::module::import("collections.abc").attr("Mapping");
  return type;
}

py::object GetCollectionsMappingViewType() {
  static py::object type =
      py::module::import("collections.abc").attr("MappingView");
  return type;
}

py::object GetWraptObjectProxyTypeUncached() {
  try {
    return py::module::import("wrapt").attr("ObjectProxy");
  } catch (const py::error_already_set& e) {
    if (e.matches(PyExc_ImportError)) return py::none();
    throw e;
  }
}

py::object GetWraptObjectProxyType() {
  static py::object type = GetWraptObjectProxyTypeUncached();
  return type;
}

// Returns 1 if `o` is considered a mapping for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsMappingHelper(PyObject* o) {
  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return PyObject_IsInstance(to_check, GetCollectionsMappingType().ptr());
  });
  if (PyDict_Check(o)) return true;
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is considered a mapping view for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsMappingViewHelper(PyObject* o) {
  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    return PyObject_IsInstance(to_check, GetCollectionsMappingViewType().ptr());
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is considered an object proxy
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsObjectProxy(PyObject* o) {
  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    auto type = GetWraptObjectProxyType();
    return !type.is_none() && PyObject_IsInstance(to_check, type.ptr()) == 1;
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is an instance of attrs-decorated class.
// Returns 0 otherwise.
int IsAttrsHelper(PyObject* o) {
  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    PyObjectPtr cls(PyObject_GetAttrString(to_check, "__class__"));
    if (cls) {
      return PyObject_HasAttrString(cls.get(), "__attrs_attrs__");
    }

    // PyObject_GetAttrString returns null on error
    PyErr_Clear();
    return 0;
  });
  return check_cache->CachedLookup(o);
}

// Returns 1 if `o` is considered a sequence for the purposes of Flatten().
// Returns 0 otherwise.
// Returns -1 if an error occurred.
int IsSequenceHelper(PyObject* o) {
  static auto* const check_cache = new CachedTypeCheck([](PyObject* to_check) {
    int is_instance =
        PyObject_IsInstance(to_check, GetCollectionsSequenceType().ptr());

    // Don't cache a failed is_instance check.
    if (is_instance == -1) return -1;

    return static_cast<int>(is_instance != 0 && !IsString(to_check));
  });  // We treat dicts and other mappings as special cases of sequences.
  if (IsMappingHelper(o)) return true;
  if (IsMappingViewHelper(o)) return true;
  if (IsAttrsHelper(o)) return true;
  if (PySet_Check(o) && !WarnedThatSetIsNotSequence) {
    LOG_WARNING(
        "Sets are not currently considered sequences, "
        "but this may change in the future, "
        "so consider avoiding using them.");
    WarnedThatSetIsNotSequence = true;
  }
  return check_cache->CachedLookup(o);
}

// ValueIterator interface
class ValueIterator {
 public:
  virtual ~ValueIterator() {}
  virtual PyObjectPtr next() = 0;

  bool valid() const { return is_valid_; }

 protected:
  void invalidate() { is_valid_ = false; }

 private:
  bool is_valid_ = true;
};

using ValueIteratorPtr = std::unique_ptr<ValueIterator>;

// Iterate through dictionaries in a deterministic order by sorting the
// keys. Notice this means that we ignore the original order of
// `OrderedDict` instances. This is intentional, to avoid potential
// bugs caused by mixing ordered and plain dicts (e.g., flattening
// a dict but using a corresponding `OrderedDict` to pack it back).
class DictValueIterator : public ValueIterator {
 public:
  explicit DictValueIterator(PyObject* dict)
      : dict_(dict), keys_(PyDict_Keys(dict)) {
    if (PyList_Sort(keys_.get()) == -1) {
      invalidate();
    } else {
      iter_.reset(PyObject_GetIter(keys_.get()));
    }
  }

  PyObjectPtr next() override {
    PyObjectPtr result;
    PyObjectPtr key(PyIter_Next(iter_.get()));
    if (key) {
      // PyDict_GetItem returns a borrowed reference.
      PyObject* elem = PyDict_GetItem(dict_, key.get());
      if (elem) {
        Py_INCREF(elem);
        result.reset(elem);
      } else {
        PyErr_SetString(PyExc_RuntimeError,
                        "Dictionary was modified during iteration over it");
      }
    }
    return result;
  }

 private:
  PyObject* dict_;
  PyObjectPtr keys_;
  PyObjectPtr iter_;
};

// Iterate over mapping objects by sorting the keys first
class MappingValueIterator : public ValueIterator {
 public:
  explicit MappingValueIterator(PyObject* mapping)
      : mapping_(mapping), keys_(PyMapping_Keys(mapping)) {
    if (!keys_ || PyList_Sort(keys_.get()) == -1) {
      invalidate();
    } else {
      iter_.reset(PyObject_GetIter(keys_.get()));
    }
  }

  PyObjectPtr next() override {
    PyObjectPtr result;
    PyObjectPtr key(PyIter_Next(iter_.get()));
    if (key) {
      // Unlike PyDict_GetItem, PyObject_GetItem returns a new reference.
      PyObject* elem = PyObject_GetItem(mapping_, key.get());
      if (elem) {
        result.reset(elem);
      } else {
        PyErr_SetString(PyExc_RuntimeError,
                        "Mapping was modified during iteration over it");
      }
    }
    return result;
  }

 private:
  PyObject* mapping_;
  PyObjectPtr keys_;
  PyObjectPtr iter_;
};

// Iterate over a sequence, by index.
class SequenceValueIterator : public ValueIterator {
 public:
  explicit SequenceValueIterator(PyObject* iterable)
      : seq_(PySequence_Fast(iterable, "")),
        size_(seq_.get() ? PySequence_Fast_GET_SIZE(seq_.get()) : 0),
        index_(0) {}

  PyObjectPtr next() override {
    PyObjectPtr result;
    if (index_ < size_) {
      // PySequence_Fast_GET_ITEM returns a borrowed reference.
      PyObject* elem = PySequence_Fast_GET_ITEM(seq_.get(), index_);
      ++index_;
      if (elem) {
        Py_INCREF(elem);
        result.reset(elem);
      }
    }

    return result;
  }

 private:
  PyObjectPtr seq_;
  const Py_ssize_t size_;
  Py_ssize_t index_;
};

class AttrsValueIterator : public ValueIterator {
 public:
  explicit AttrsValueIterator(PyObject* nested) : nested_(nested) {
    Py_INCREF(nested);
    cls_.reset(PyObject_GetAttrString(nested_.get(), "__class__"));
    if (cls_) {
      attrs_.reset(PyObject_GetAttrString(cls_.get(), "__attrs_attrs__"));
      if (attrs_) {
        iter_.reset(PyObject_GetIter(attrs_.get()));
      }
    }
    if (!iter_ || PyErr_Occurred()) invalidate();
  }

  PyObjectPtr next() override {
    PyObjectPtr result;
    PyObjectPtr item(PyIter_Next(iter_.get()));
    if (item) {
      PyObjectPtr name(PyObject_GetAttrString(item.get(), "name"));
      result.reset(PyObject_GetAttr(nested_.get(), name.get()));
    }

    return result;
  }

 private:
  PyObjectPtr nested_;
  PyObjectPtr cls_;
  PyObjectPtr attrs_;
  PyObjectPtr iter_;
};

ValueIteratorPtr GetValueIterator(PyObject* nested) {
  if (PyDict_Check(nested)) {
    return make_unique_from_py_object_ptr<DictValueIterator>(nested);
  } else if (IsMappingHelper(nested)) {
    return make_unique_from_py_object_ptr<MappingValueIterator>(nested);
  } else if (IsAttrsHelper(nested)) {
    return make_unique_from_py_object_ptr<AttrsValueIterator>(nested);
  } else {
    return make_unique_from_py_object_ptr<SequenceValueIterator>(nested);
  }
}

bool FlattenHelper(
    PyObject* nested, PyObject* list,
    const std::function<int(PyObject*)>& is_sequence_helper,
    const std::function<ValueIteratorPtr(PyObject*)>& value_iterator_getter) {
  // if nested is not a sequence, append itself and exit
  int is_seq = is_sequence_helper(nested);
  if (is_seq == -1) return false;
  if (!is_seq) {
    return PyList_Append(list, nested) != -1;
  }

  ValueIteratorPtr iter = value_iterator_getter(nested);
  if (!iter->valid()) return false;

  for (PyObjectPtr item = iter->next(); item; item = iter->next()) {
    if (Py_EnterRecursiveCall(" in flatten")) {
      return false;
    }
    const bool success = FlattenHelper(item.get(), list, is_sequence_helper,
                                       value_iterator_getter);
    Py_LeaveRecursiveCall();
    if (!success) {
      return false;
    }
  }
  return true;
}

// Sets error using keys of 'dict1' and 'dict2'.
// 'dict1' and 'dict2' are assumed to be Python dictionaries.
void SetDifferentKeysError(PyObject* dict1, PyObject* dict2,
                           std::string* error_msg, bool* is_type_error) {
  PyObjectPtr k1(PyMapping_Keys(dict1));
  if (PyErr_Occurred() || k1.get() == nullptr) {
    *error_msg =
        ("The two dictionaries don't have the same set of keys. Failed to "
         "fetch keys.");
    return;
  }
  PyObjectPtr k2(PyMapping_Keys(dict2));
  if (PyErr_Occurred() || k2.get() == nullptr) {
    *error_msg =
        ("The two dictionaries don't have the same set of keys. Failed to "
         "fetch keys.");
    return;
  }
  *is_type_error = false;
  *error_msg = StrCat(
      "The two dictionaries don't have the same set of keys. "
      "First structure has keys ",
      PyObjectToString(k1.get()), ", while second structure has keys ",
      PyObjectToString(k2.get()));
}

// Returns true iff there were no "internal" errors. In other words,
// errors that has nothing to do with structure checking.
// If an "internal" error occurred, the appropriate Python error will be
// set and the caller can propage it directly to the user.
//
// Both `error_msg` and `is_type_error` must be non-null. `error_msg` must
// be empty.
// Leaves `error_msg` empty if structures matched. Else, fills `error_msg`
// with appropriate error and sets `is_type_error` to true iff
// the error to be raised should be TypeError.
bool AssertSameStructureHelper(PyObject* o1, PyObject* o2, bool check_types,
                               std::string* error_msg, bool* is_type_error) {
  DCHECK(error_msg);
  DCHECK(is_type_error);
  const bool is_seq1 = IsSequence(o1);
  const bool is_seq2 = IsSequence(o2);
  if (PyErr_Occurred()) return false;
  if (is_seq1 != is_seq2) {
    std::string seq_str = is_seq1 ? PyObjectToString(o1) : PyObjectToString(o2);
    std::string non_seq_str =
        is_seq1 ? PyObjectToString(o2) : PyObjectToString(o1);
    *is_type_error = false;
    *error_msg = StrCat("Substructure \"", seq_str,
                        "\" is a sequence, while substructure \"", non_seq_str,
                        "\" is not");
    return true;
  }

  // Got to scalars, so finished checking. Structures are the same.
  if (!is_seq1) return true;

  if (check_types) {
    // Unwrap wrapt.ObjectProxy if needed.
    PyObjectPtr o1_wrapped;
    if (IsObjectProxy(o1)) {
      o1_wrapped.reset(PyObject_GetAttrString(o1, "__wrapped__"));
      o1 = o1_wrapped.get();
    }
    PyObjectPtr o2_wrapped;
    if (IsObjectProxy(o2)) {
      o2_wrapped.reset(PyObject_GetAttrString(o2, "__wrapped__"));
      o2 = o2_wrapped.get();
    }

    const PyTypeObject* type1 = o1->ob_type;
    const PyTypeObject* type2 = o2->ob_type;

    // We treat two different namedtuples with identical name and fields
    // as having the same type.
    const PyObject* o1_tuple = IsNamedtuple(o1, true);
    if (o1_tuple == nullptr) return false;
    const PyObject* o2_tuple = IsNamedtuple(o2, true);
    if (o2_tuple == nullptr) {
      Py_DECREF(o1_tuple);
      return false;
    }
    bool both_tuples = o1_tuple == Py_True && o2_tuple == Py_True;
    Py_DECREF(o1_tuple);
    Py_DECREF(o2_tuple);

    if (both_tuples) {
      const PyObject* same_tuples = SameNamedtuples(o1, o2);
      if (same_tuples == nullptr) return false;
      bool not_same_tuples = same_tuples != Py_True;
      Py_DECREF(same_tuples);
      if (not_same_tuples) {
        *is_type_error = true;
        *error_msg = StrCat(
            "The two namedtuples don't have the same sequence type. "
            "First structure ",
            PyObjectToString(o1), " has type ", type1->tp_name,
            ", while second structure ", PyObjectToString(o2), " has type ",
            type2->tp_name);
        return true;
      }
    } else if (type1 != type2
               /* If both sequences are list types, don't complain. This allows
                  one to be a list subclass (e.g. _ListWrapper used for
                  automatic dependency tracking.) */
               && !(PyList_Check(o1) && PyList_Check(o2))
               /* Two mapping types will also compare equal, making _DictWrapper
                  and dict compare equal. */
               && !(IsMappingHelper(o1) && IsMappingHelper(o2))) {
      *is_type_error = true;
      *error_msg = StrCat(
          "The two namedtuples don't have the same sequence type. "
          "First structure ",
          PyObjectToString(o1), " has type ", type1->tp_name,
          ", while second structure ", PyObjectToString(o2), " has type ",
          type2->tp_name);
      return true;
    }

    if (PyDict_Check(o1) && PyDict_Check(o2)) {
      if (PyDict_Size(o1) != PyDict_Size(o2)) {
        SetDifferentKeysError(o1, o2, error_msg, is_type_error);
        return true;
      }

      PyObject* key;
      Py_ssize_t pos = 0;
      while (PyDict_Next(o1, &pos, &key, nullptr)) {
        if (PyDict_GetItem(o2, key) == nullptr) {
          SetDifferentKeysError(o1, o2, error_msg, is_type_error);
          return true;
        }
      }
    } else if (IsMappingHelper(o1)) {
      // Fallback for custom mapping types. Instead of using PyDict methods
      // which stay in C, we call iter(o1).
      if (PyMapping_Size(o1) != PyMapping_Size(o2)) {
        SetDifferentKeysError(o1, o2, error_msg, is_type_error);
        return true;
      }

      PyObjectPtr iter(PyObject_GetIter(o1));
      PyObject* key;
      while ((key = PyIter_Next(iter.get())) != nullptr) {
        if (!PyMapping_HasKey(o2, key)) {
          SetDifferentKeysError(o1, o2, error_msg, is_type_error);
          Py_DECREF(key);
          return true;
        }
        Py_DECREF(key);
      }
    }
  }

  ValueIteratorPtr iter1 = GetValueIterator(o1);
  ValueIteratorPtr iter2 = GetValueIterator(o2);

  if (!iter1->valid() || !iter2->valid()) return false;

  while (true) {
    PyObjectPtr v1 = iter1->next();
    PyObjectPtr v2 = iter2->next();
    if (v1 && v2) {
      if (Py_EnterRecursiveCall(" in assert_same_structure")) {
        return false;
      }
      bool no_internal_errors = AssertSameStructureHelper(
          v1.get(), v2.get(), check_types, error_msg, is_type_error);
      Py_LeaveRecursiveCall();
      if (!no_internal_errors) return false;
      if (!error_msg->empty()) return true;
    } else if (!v1 && !v2) {
      // Done with all recursive calls. Structure matched.
      return true;
    } else {
      *is_type_error = false;
      *error_msg =
          StrCat("The two structures don't have the same number of elements. ",
                 "First structure: ", PyObjectToString(o1),
                 ". Second structure: ", PyObjectToString(o2));
      return true;
    }
  }
}

}  // namespace

bool IsSequence(PyObject* o) { return IsSequenceHelper(o) == 1; }
bool IsAttrs(PyObject* o) { return IsAttrsHelper(o) == 1; }

PyObject* Flatten(PyObject* nested) {
  PyObject* list = PyList_New(0);
  if (FlattenHelper(nested, list, IsSequenceHelper, GetValueIterator)) {
    return list;
  } else {
    Py_DECREF(list);
    return nullptr;
  }
}

PyObject* IsNamedtuple(PyObject* o, bool strict) {
  // Unwrap wrapt.ObjectProxy if needed.
  PyObjectPtr o_wrapped;
  if (IsObjectProxy(o)) {
    o_wrapped.reset(PyObject_GetAttrString(o, "__wrapped__"));
    o = o_wrapped.get();
  }

  // Must be subclass of tuple
  if (!PyTuple_Check(o)) {
    Py_RETURN_FALSE;
  }

  // If strict, o.__class__.__base__ must be tuple
  if (strict) {
    PyObject* klass = PyObject_GetAttrString(o, "__class__");
    if (klass == nullptr) return nullptr;
    PyObject* base = PyObject_GetAttrString(klass, "__base__");
    Py_DECREF(klass);
    if (base == nullptr) return nullptr;

    const PyTypeObject* base_type = reinterpret_cast<PyTypeObject*>(base);
    // built-in object types are singletons
    bool tuple_base = base_type == &PyTuple_Type;
    Py_DECREF(base);
    if (!tuple_base) {
      Py_RETURN_FALSE;
    }
  }

  // o must have attribute '_fields' and every element in
  // '_fields' must be a string.
  int has_fields = PyObject_HasAttrString(o, "_fields");
  if (!has_fields) {
    Py_RETURN_FALSE;
  }

  PyObjectPtr fields(PyObject_GetAttrString(o, "_fields"));
  int is_instance =
      PyObject_IsInstance(fields.get(), GetCollectionsSequenceType().ptr());
  if (is_instance == 0) {
    Py_RETURN_FALSE;
  } else if (is_instance == -1) {
    return nullptr;
  }

  PyObjectPtr seq(PySequence_Fast(fields.get(), ""));
  const Py_ssize_t s = PySequence_Fast_GET_SIZE(seq.get());
  for (Py_ssize_t i = 0; i < s; ++i) {
    // PySequence_Fast_GET_ITEM returns borrowed ref
    PyObject* elem = PySequence_Fast_GET_ITEM(seq.get(), i);
    if (!IsString(elem)) {
      Py_RETURN_FALSE;
    }
  }

  Py_RETURN_TRUE;
}

PyObject* SameNamedtuples(PyObject* o1, PyObject* o2) {
  PyObject* f1 = PyObject_GetAttrString(o1, "_fields");
  PyObject* f2 = PyObject_GetAttrString(o2, "_fields");
  if (f1 == nullptr || f2 == nullptr) {
    Py_XDECREF(f1);
    Py_XDECREF(f2);
    PyErr_SetString(
        PyExc_RuntimeError,
        "Expected namedtuple-like objects (that have _fields attr)");
    return nullptr;
  }

  if (PyObject_RichCompareBool(f1, f2, Py_NE)) {
    Py_RETURN_FALSE;
  }

  if (GetClassName(o1) == GetClassName(o2)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

void AssertSameStructure(PyObject* o1, PyObject* o2, bool check_types) {
  std::string error_msg;
  bool is_type_error = false;
  AssertSameStructureHelper(o1, o2, check_types, &error_msg, &is_type_error);
  if (PyErr_Occurred()) {
    // Don't hide Python exceptions while checking (e.g. errors fetching keys
    // from custom mappings).
    return;
  }
  if (!error_msg.empty()) {
    PyErr_SetString(
        is_type_error ? PyExc_TypeError : PyExc_ValueError,
        StrCat("The two structures don't have the same nested structure.\n\n",
               "First structure: ", PyObjectToString(o1),
               "\n\nSecond structure: ", PyObjectToString(o2),
               "\n\nMore specifically: ", error_msg)
            .c_str());
  }
}

namespace {

inline py::object pyo_or_throw(PyObject* ptr) {
  if (PyErr_Occurred() || ptr == nullptr) {
    throw py::error_already_set();
  }
  return py::reinterpret_steal<py::object>(ptr);
}

PYBIND11_MODULE(_tree, m) {
  // Resolve `wrapt.ObjectProxy` at import time to avoid doing
  // imports during function calls.
  tree::GetWraptObjectProxyType();

  m.def("assert_same_structure",
        [](py::handle& o1, py::handle& o2, bool check_types) {
          tree::AssertSameStructure(o1.ptr(), o2.ptr(), check_types);
          if (PyErr_Occurred()) {
            throw py::error_already_set();
          }
        });
  m.def("is_sequence", [](py::handle& o) {
    bool result = tree::IsSequence(o.ptr());
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    return result;
  });
  m.def("is_namedtuple", [](py::handle& o, bool strict) {
    return pyo_or_throw(tree::IsNamedtuple(o.ptr(), strict));
  });
  m.def("is_attrs", [](py::handle& o) {
    bool result = tree::IsAttrs(o.ptr());
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
    return result;
  });
  m.def("same_namedtuples", [](py::handle& o1, py::handle& o2) {
    return pyo_or_throw(tree::SameNamedtuples(o1.ptr(), o2.ptr()));
  });
  m.def("flatten", [](py::handle& nested) {
    return pyo_or_throw(tree::Flatten(nested.ptr()));
  });
}

}  // namespace
}  // namespace tree
