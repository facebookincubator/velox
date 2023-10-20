/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "complex.h"
#include "velox/vector/ComplexVector.h"


namespace facebook::velox::py {

using namespace velox;
namespace py = pybind11;

struct Counter {
  TypePtr type = createType(TypeKind::UNKNOWN, {});
  vector_size_t insertedElements=0; // to track the elements already in the vector
  vector_size_t totalElements=0;  
  std::vector<Counter> children;
};

template <TypeKind Kind>
void setElementInFlatVector(
    vector_size_t idx,
    const variant& v,
    VectorPtr& vector) {
  using NativeType = typename TypeTraits<Kind>::NativeType;
  auto asFlat = vector->asFlatVector<NativeType>();
  try {
      asFlat->set(idx, NativeType{v.value<NativeType>()});
  } catch (const std::exception& e) {
      throw py::type_error("size: "+std::to_string(vector->size())+", idx: "+std::to_string(idx));
  }
}

void constructType(TypePtr& type, const variant& v, Counter& counter){
    ++counter.totalElements;
    
    if (v.isNull()) {
      if (v.kind() != TypeKind::UNKNOWN && v.kind() != TypeKind::INVALID &&
          v.kind() != counter.type->kind()) {
        throw std::invalid_argument("Variant was of an unexpected kind");
      }
      return;
    } else {
      if (v.kind() == TypeKind::UNKNOWN || v.kind() == TypeKind::INVALID) {
        throw std::invalid_argument(
            "Non-null variant has unknown or invalid kind");
    } 

    switch(v.kind()){
        case TypeKind::ARRAY : {
          counter.children.resize(1);
            auto asArray = v.array();
            TypePtr children;
            for(const auto& element:asArray){
              constructType(children, element, counter.children[0]);
            }
            type = createType<TypeKind::ARRAY>({children});
            break;
        }

        default:{
            type = createScalarType(v.kind());
            break;
        }
    }

    if (counter.type->kind() == TypeKind::UNKNOWN) {
        counter.type = type;
    } else if (!(counter.type->kindEquals(type))){
     throw py::type_error("Cannot construct type tree, invalid variant for complex type"); 
    }

    }
}

static void insertVariantIntoVector(
    TypeKind& type,
    const variant& v,
    VectorPtr& vector,
    Counter& counter) {
  if (v.isNull()) {
    vector->setNull(counter.insertedElements, true);
  } else {
    switch (type) {
      case TypeKind::ARRAY: {
        auto asArray = vector->as<ArrayVector>();
        asArray->elements()->resize(counter.children[0].totalElements);
        const std::vector<variant>& elements = v.array();
        vector_size_t offset = 0;
        if (counter.insertedElements != 0) {  
          offset = asArray->offsetAt(counter.insertedElements - 1) +
              asArray->sizeAt(counter.insertedElements - 1);
        }
        asArray->setOffsetAndSize(
            counter.insertedElements, offset, elements.size());
        for (const variant& elt : elements) {
            auto elt_type = elt.kind();
          insertVariantIntoVector(
              elt_type, elt, asArray->elements(), counter.children[0]);
        }
        
        break;
      }
      default: {
        VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
            setElementInFlatVector,
            type,
            counter.insertedElements,
            v,
            vector);
        break;
      }
    }
          counter.insertedElements += 1; 
  }
}


VectorPtr variantsToVector(
    const std::vector<variant>& variants,
    velox::memory::MemoryPool* pool) {
  Counter counter;
  TypePtr type;
  for(const auto& variant: variants){
    constructType(type, variant, counter);
  }
  VectorPtr resultVector = BaseVector::create(std::move(type), variants.size(), pool);
  for (const variant& v : variants) {
    auto type = v.kind();
    insertVariantIntoVector(type, v, resultVector, counter);
  }
  return resultVector;
}

}
