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

#include <functional>

namespace facebook::velox::py {

using namespace velox;
namespace py = pybind11;

struct ElementCounter {
  vector_size_t insertedElements=0; // to track the elements already in the vector
  vector_size_t totalElements=0;  
  std::vector<ElementCounter> children;
};


void checkOrAssignType(TypePtr& type, const std::function<TypePtr()>& func) {
    if (type->kind() == TypeKind::UNKNOWN) {
        type = func();
    } else if (!(type->kindEquals(func()))) {
        throw py::type_error("Cannot construct type tree, invalid variant for complex type");
    }
}

template <TypeKind Kind>
void setElementInFlatVector(
    vector_size_t idx,
    const variant& v,
    VectorPtr& vector) {
  using NativeType = typename TypeTraits<Kind>::NativeType;
  auto asFlat = vector->asFlatVector<NativeType>();
  asFlat->set(idx, NativeType{v.value<NativeType>()});
}

void constructType(TypePtr& type, const variant& v, ElementCounter& counter){
    ++counter.totalElements;
    
    if (v.isNull()) {
      if (v.kind() != TypeKind::UNKNOWN && v.kind() != TypeKind::INVALID &&
          v.kind() != type->kind()) {
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
            TypePtr children = createType(TypeKind::UNKNOWN, {});
            for(const auto& element:asArray){
              constructType(children, element, counter.children[0]);
            }
            checkOrAssignType(type, [&children](){ return createType<TypeKind::ARRAY>({children});});
            break;
        }

        default:{
            checkOrAssignType(type, [&v]() { return createScalarType(v.kind());});
            break;
        }
    }
    }
}

static void insertVariantIntoVector(
    TypeKind& typeKind,
    const variant& v,
    VectorPtr& vector,
    ElementCounter& counter) {
  if (v.isNull()) {
    vector->setNull(counter.insertedElements, true);
  } else {
    switch (typeKind) {
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
            typeKind,
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
  ElementCounter counter;
  TypePtr type = createType(TypeKind::UNKNOWN, {});
  for(const auto& variant: variants){
    constructType(type, variant, counter);
  }
  VectorPtr resultVector = BaseVector::create(std::move(type), variants.size(), pool);
  for (const variant& v : variants) {
    auto typeKind = v.kind();
    insertVariantIntoVector(typeKind, v, resultVector, counter);
  }
  return resultVector;
}

}
