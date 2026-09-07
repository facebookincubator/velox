
Nimble Encoding Cost Model
==========================

How encoding selection estimates sizes with per-block statistics

New

Modified

Unchanged

0

Caller

EncodingFactory::encode()
^^^^^^^^^^^^^^^^^^^^^^^^^

Called by FieldWriter when flushing a stream.

.. code-block:: text

            auto statistics = Statistics<physicalType>::create(values);
            // O(1) ← data_

            auto result = selectorPolicy->select(values, statistics);

Encoding selection

EncodingSelectionPolicy::select(values, statistics)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

            hasPerBlockEncoding = any_of(encodingReadFactors_, == BlockBitPacking);

            for (const auto& [encodingType, readFactor] : encodingReadFactors_) {
              estimatedSize = EncodingSizeEstimation::estimateSize(encodingType, count, statistics, hasPerBlockEncoding);
              // dispatches by type:
              //   numeric → estimateNumericSize()
              //   bool    → estimateBoolSize()
              //   string  → estimateStringSize()
              cost = estimatedSize * readFactor;
              if (cost < minCost) {
                selectedEncoding = encodingType;
              }
            }

EncodingSizeEstimation::estimateNumericSize()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

switch (encodingType) → dispatches to each encoding's estimateSize()

UNCHANGED

no stats needed

.. code-block:: text

              return prefix + count × sizeof(T)

UNCHANGED

no stats needed

.. code-block:: text

              return prefix + sizeof(T)

UNCHANGED

.. code-block:: text

              estimateSize(rowCount, statistics, options) {
                return estimateSize(rowCount, statistics.min(), statistics.max(), options);
              }

              estimateSize(rowCount, minValue, maxValue, options) {
                bitWidth = bitsRequired(maxValue - minValue);
                if (!options.fixedBitWidthUseExactBits) bitWidth = roundUpToByte(bitWidth);
                payloadSize = nbytes(bitWidth * rowCount);
                return prefix + kPrefixSize + payloadSize;
              }

statistics.min()

← scans data\_ once

// sets both min\_ AND max\_ in one pass

// via std::minmax_element(data\_)

← cached after first call

NEW

.. code-block:: text

              estimateSize(const Statistics& statistics) {
                const auto& blocks = statistics.minMaxBlocks();

                for (const auto& b : blocks) {
                  range = b.max - b.min;
                  bw = bitsRequired(range);
                  packedSize = bufferSize(b.count, bw);
                  rawSize = b.count * sizeof(T);
                  if (packedSize < rawSize) {
                    dataSize += packedSize;
                  } else {
                    dataSize += rawSize; // skip encoding
                  }
                }

                // per-block metadata stored as nested encodings:
                //   baselines → Trivial<T>
                //   bitWidths → Trivial<uint8_t>
                //   dataOffsets → Trivial<uint32_t>
                metadataSize = kMetadataHeaderSize
                  + Trivial<T>::estimateSize(numBlocks)
                  + Trivial<u8>::estimateSize(numBlocks)
                  + Trivial<u32>::estimateSize(numBlocks);

                return prefix + metadataSize + dataSize;
              }

statistics.minMaxBlocks()

← scans data\_ once

// BlockStatsAccumulator: for each value,

// track min/max, flush every 1024 rows

// → vector < {count, min, max} >

← cached after first call
