// See https://raw.githubusercontent.com/duckdb/duckdb/master/LICENSE for licensing information

#include "duckdb.hpp"
#include "duckdb-internal.hpp"
#ifndef DUCKDB_AMALGAMATION
#error header mismatch
#endif


// LICENSE_CHANGE_BEGIN
// The following code up to LICENSE_CHANGE_END is subject to THIRD PARTY LICENSE #5
// See the end of this file for a list

// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
//
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst


// LICENSE_CHANGE_BEGIN
// The following code up to LICENSE_CHANGE_END is subject to THIRD PARTY LICENSE #5
// See the end of this file for a list

// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
// 
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files   
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,   
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is   
// furnished to do so, subject to the following conditions:
// 
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
//                 
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst 
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stddef.h>

using namespace std;

 // the official FSST API -- also usable by C mortals

/* unsigned integers */
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

inline uint64_t fsst_unaligned_load(u8 const* V) {
	uint64_t Ret;
	memcpy(&Ret, V, sizeof(uint64_t)); // compiler will generate efficient code (unaligned load, where possible)
	return Ret;
}

#define FSST_ENDIAN_MARKER ((u64) 1)
#define FSST_VERSION_20190218 20190218
#define FSST_VERSION ((u64) FSST_VERSION_20190218)

// "symbols" are character sequences (up to 8 bytes)
// A symbol is compressed into a "code" of, in principle, one byte. But, we added an exception mechanism:
// byte 255 followed by byte X represents the single-byte symbol X. Its code is 256+X.

// we represent codes in u16 (not u8). 12 bits code (of which 10 are used), 4 bits length
#define FSST_LEN_BITS       12
#define FSST_CODE_BITS      9 
#define FSST_CODE_BASE      256UL /* first 256 codes [0,255] are pseudo codes: escaped bytes */
#define FSST_CODE_MAX       (1UL<<FSST_CODE_BITS) /* all bits set: indicating a symbol that has not been assigned a code yet */
#define FSST_CODE_MASK      (FSST_CODE_MAX-1UL)   /* all bits set: indicating a symbol that has not been assigned a code yet */

struct Symbol {
   static const unsigned maxLength = 8;

   // the byte sequence that this symbol stands for
   union { char str[maxLength]; u64 num; } val; // usually we process it as a num(ber), as this is fast

   // icl = u64 ignoredBits:16,code:12,length:4,unused:32 -- but we avoid exposing this bit-field notation
   u64 icl;  // use a single u64 to be sure "code" is accessed with one load and can be compared with one comparison

   Symbol() : icl(0) { val.num = 0; }

   explicit Symbol(u8 c, u16 code) : icl((1<<28)|(code<<16)|56) { val.num = c; } // single-char symbol
   explicit Symbol(const char* begin, const char* end) : Symbol(begin, (u32) (end-begin)) {}
   explicit Symbol(u8* begin, u8* end) : Symbol((const char*)begin, (u32) (end-begin)) {}
   explicit Symbol(const char* input, u32 len) {
      val.num = 0;
      if (len>=8) {
          len = 8;
          memcpy(val.str, input, 8);
      } else {
          memcpy(val.str, input, len);
      }
      set_code_len(FSST_CODE_MAX, len);
   }
   void set_code_len(u32 code, u32 len) { icl = (len<<28)|(code<<16)|((8-len)*8); }

   u32 length() const { return (u32) (icl >> 28); }
   u16 code() const { return (icl >> 16) & FSST_CODE_MASK; }
   u32 ignoredBits() const { return (u32) icl; }

   u8 first() const { assert( length() >= 1); return 0xFF & val.num; }
   u16 first2() const { assert( length() >= 2); return 0xFFFF & val.num; }

#define FSST_HASH_LOG2SIZE 10 
#define FSST_HASH_PRIME 2971215073LL
#define FSST_SHIFT 15
#define FSST_HASH(w) (((w)*FSST_HASH_PRIME)^(((w)*FSST_HASH_PRIME)>>FSST_SHIFT))
   size_t hash() const { size_t v = 0xFFFFFF & val.num; return FSST_HASH(v); } // hash on the next 3 bytes
};

// Symbol that can be put in a queue, ordered on gain
struct QSymbol{
   Symbol symbol;
   mutable u32 gain; // mutable because gain value should be ignored in find() on unordered_set of QSymbols
   bool operator==(const QSymbol& other) const { return symbol.val.num == other.symbol.val.num && symbol.length() == other.symbol.length(); }
};

// we construct FSST symbol tables using a random sample of about 16KB (1<<14) 
#define FSST_SAMPLETARGET (1<<14)
#define FSST_SAMPLEMAXSZ ((long) 2*FSST_SAMPLETARGET)

// two phases of compression, before and after optimize():
//
// (1) to encode values we probe (and maintain) three datastructures:
// - u16 byteCodes[65536] array at the position of the next byte  (s.length==1)
// - u16 shortCodes[65536] array at the position of the next twobyte pattern (s.length==2)
// - Symbol hashtable[1024] (keyed by the next three bytes, ie for s.length>2), 
// this search will yield a u16 code, it points into Symbol symbols[]. You always find a hit, because the first 256 codes are 
// pseudo codes representing a single byte these will become escapes)
//
// (2) when we finished looking for the best symbol table we call optimize() to reshape it:
// - it renumbers the codes by length (first symbols of length 2,3,4,5,6,7,8; then 1 (starting from byteLim are symbols of length 1)
//   length 2 codes for which no longer suffix symbol exists (< suffixLim) come first among the 2-byte codes 
//   (allows shortcut during compression)
// - for each two-byte combination, in all unused slots of shortCodes[], it enters the byteCode[] of the symbol corresponding 
//   to the first byte (if such a single-byte symbol exists). This allows us to just probe the next two bytes (if there is only one
//   byte left in the string, there is still a terminator-byte added during compression) in shortCodes[]. That is, byteCodes[]
//   and its codepath is no longer required. This makes compression faster. The reason we use byteCodes[] during symbolTable construction
//   is that adding a new code/symbol is expensive (you have to touch shortCodes[] in 256 places). This optimization was
//   hence added to make symbolTable construction faster.
//
// this final layout allows for the fastest compression code, only currently present in compressBulk

// in the hash table, the icl field contains (low-to-high) ignoredBits:16,code:12,length:4
#define FSST_ICL_FREE ((15<<28)|(((u32)FSST_CODE_MASK)<<16)) // high bits of icl (len=8,code=FSST_CODE_MASK) indicates free bucket

// ignoredBits is (8-length)*8, which is the amount of high bits to zero in the input word before comparing with the hashtable key
//             ..it could of course be computed from len during lookup, but storing it precomputed in some loose bits is faster
//
// the gain field is only used in the symbol queue that sorts symbols on gain

struct SymbolTable {
   static const u32 hashTabSize = 1<<FSST_HASH_LOG2SIZE; // smallest size that incurs no precision loss

   // lookup table using the next two bytes (65536 codes), or just the next single byte
   u16 shortCodes[65536]; // contains code for 2-byte symbol, otherwise code for pseudo byte (escaped byte)

   // lookup table (only used during symbolTable construction, not during normal text compression)
   u16 byteCodes[256]; // contains code for every 1-byte symbol, otherwise code for pseudo byte (escaped byte)

   // 'symbols' is the current symbol  table symbol[code].symbol is the max 8-byte 'symbol' for single-byte 'code'
   Symbol symbols[FSST_CODE_MAX]; // x in [0,255]: pseudo symbols representing escaped byte x; x in [FSST_CODE_BASE=256,256+nSymbols]: real symbols   

   // replicate long symbols in hashTab (avoid indirection). 
   Symbol hashTab[hashTabSize]; // used for all symbols of 3 and more bytes

   u16 nSymbols;          // amount of symbols in the map (max 255)
   u16 suffixLim;         // codes higher than this do not have a longer suffix
   u16 terminator;        // code of 1-byte symbol, that can be used as a terminator during compression
   bool zeroTerminated;   // whether we are expecting zero-terminated strings (we then also produce zero-terminated compressed strings)
   u16 lenHisto[FSST_CODE_BITS]; // lenHisto[x] is the amount of symbols of byte-length (x+1) in this SymbolTable

   SymbolTable() : nSymbols(0), suffixLim(FSST_CODE_MAX), terminator(0), zeroTerminated(false) {
      // stuff done once at startup
      for (u32 i=0; i<256; i++) {
         symbols[i] = Symbol(i,i|(1<<FSST_LEN_BITS)); // pseudo symbols
      }
      Symbol unused = Symbol((u8) 0,FSST_CODE_MASK); // single-char symbol, exception code
      for (u32 i=256; i<FSST_CODE_MAX; i++) {
         symbols[i] = unused; // we start with all symbols unused
      }
      // empty hash table
      Symbol s;
      s.val.num = 0;
      s.icl = FSST_ICL_FREE; //marks empty in hashtab
      for(u32 i=0; i<hashTabSize; i++)
         hashTab[i] = s;

      // fill byteCodes[] with the pseudo code all bytes (escaped bytes)
      for(u32 i=0; i<256; i++)
         byteCodes[i] = (1<<FSST_LEN_BITS) | i;

      // fill shortCodes[] with the pseudo code for the first byte of each two-byte pattern
      for(u32 i=0; i<65536; i++)
         shortCodes[i] = (1<<FSST_LEN_BITS) | (i&255);

      memset(lenHisto, 0, sizeof(lenHisto)); // all unused
   }

   void clear() {
      // clear a symbolTable with minimal effort (only erase the used positions in it)
      memset(lenHisto, 0, sizeof(lenHisto)); // all unused
      for(u32 i=FSST_CODE_BASE; i<FSST_CODE_BASE+nSymbols; i++) {
          if (symbols[i].length() == 1) {
              u16 val = symbols[i].first();
              byteCodes[val] = (1<<FSST_LEN_BITS) | val;
          } else if (symbols[i].length() == 2) {
              u16 val = symbols[i].first2();
              shortCodes[val] = (1<<FSST_LEN_BITS) | (val&255);
          } else {
              u32 idx = symbols[i].hash() & (hashTabSize-1);
              hashTab[idx].val.num = 0;
              hashTab[idx].icl = FSST_ICL_FREE; //marks empty in hashtab
          }           
      } 
      nSymbols = 0; // no need to clean symbols[] as no symbols are used
   }
   bool hashInsert(Symbol s) {
      u32 idx = s.hash() & (hashTabSize-1);
      bool taken = (hashTab[idx].icl < FSST_ICL_FREE);
      if (taken) return false; // collision in hash table
      hashTab[idx].icl = s.icl;
      hashTab[idx].val.num = s.val.num & (0xFFFFFFFFFFFFFFFF >> (u8) s.icl);
      return true;
   }
   bool add(Symbol s) {
      assert(FSST_CODE_BASE + nSymbols < FSST_CODE_MAX);
      u32 len = s.length();
      s.set_code_len(FSST_CODE_BASE + nSymbols, len);
      if (len == 1) {
         byteCodes[s.first()] = FSST_CODE_BASE + nSymbols + (1<<FSST_LEN_BITS); // len=1 (<<FSST_LEN_BITS)
      } else if (len == 2) {
         shortCodes[s.first2()] = FSST_CODE_BASE + nSymbols + (2<<FSST_LEN_BITS); // len=2 (<<FSST_LEN_BITS)
      } else if (!hashInsert(s)) {
         return false;
      }
      symbols[FSST_CODE_BASE + nSymbols++] = s;
      lenHisto[len-1]++;
      return true;
   }
   /// Find longest expansion, return code (= position in symbol table)
   u16 findLongestSymbol(Symbol s) const {
      size_t idx = s.hash() & (hashTabSize-1);
      if (hashTab[idx].icl <= s.icl && hashTab[idx].val.num == (s.val.num & (0xFFFFFFFFFFFFFFFF >> ((u8) hashTab[idx].icl)))) {
         return (hashTab[idx].icl>>16) & FSST_CODE_MASK; // matched a long symbol 
      }
      if (s.length() >= 2) {
         u16 code =  shortCodes[s.first2()] & FSST_CODE_MASK;
         if (code >= FSST_CODE_BASE) return code; 
      }
      return byteCodes[s.first()] & FSST_CODE_MASK;
   }
   u16 findLongestSymbol(u8* cur, u8* end) const {
      return findLongestSymbol(Symbol(cur,end)); // represent the string as a temporary symbol
   }

   // rationale for finalize:
   // - during symbol table construction, we may create more than 256 codes, but bring it down to max 255 in the last makeTable()
   //   consequently we needed more than 8 bits during symbol table contruction, but can simplify the codes to single bytes in finalize()
   //   (this feature is in fact lo longer used, but could still be exploited: symbol construction creates no more than 255 symbols in each pass)
   // - we not only reduce the amount of codes to <255, but also *reorder* the symbols and renumber their codes, for higher compression perf.
   //   we renumber codes so they are grouped by length, to allow optimized scalar string compression (byteLim and suffixLim optimizations). 
   // - we make the use of byteCode[] no longer necessary by inserting single-byte codes in the free spots of shortCodes[]
   //   Using shortCodes[] only makes compression faster. When creating the symbolTable, however, using shortCodes[] for the single-byte
   //   symbols is slow, as each insert touches 256 positions in it. This optimization was added when optimizing symbolTable construction time.
   //
   // In all, we change the layout and coding, as follows..
   //
   // before finalize(): 
   // - The real symbols are symbols[256..256+nSymbols>. As we may have nSymbols > 255
   // - The first 256 codes are pseudo symbols (all escaped bytes)
   //
   // after finalize(): 
   // - table layout is symbols[0..nSymbols>, with nSymbols < 256. 
   // - Real codes are [0,nSymbols>. 8-th bit not set. 
   // - Escapes in shortCodes have the 8th bit set (value: 256+255=511). 255 because the code to be emitted is the escape byte 255
   // - symbols are grouped by length: 2,3,4,5,6,7,8, then 1 (single-byte codes last)
   // the two-byte codes are split in two sections: 
   // - first section contains codes for symbols for which there is no longer symbol (no suffix). It allows an early-out during compression
   //
   // finally, shortCodes[] is modified to also encode all single-byte symbols (hence byteCodes[] is not required on a critical path anymore).
   //
   void finalize(u8 zeroTerminated) {
       assert(nSymbols <= 255);
       u8 newCode[256], rsum[8], byteLim = nSymbols - (lenHisto[0] - zeroTerminated);

       // compute running sum of code lengths (starting offsets for each length) 
       rsum[0] = byteLim; // 1-byte codes are highest
       rsum[1] = zeroTerminated;
       for(u32 i=1; i<7; i++)
          rsum[i+1] = rsum[i] + lenHisto[i];

       // determine the new code for each symbol, ordered by length (and splitting 2byte symbols into two classes around suffixLim)
       suffixLim = rsum[1];
       symbols[newCode[0] = 0] = symbols[256]; // keep symbol 0 in place (for zeroTerminated cases only)

       for(u32 i=zeroTerminated, j=rsum[2]; i<nSymbols; i++) {  
          Symbol s1 = symbols[FSST_CODE_BASE+i];
          u32 len = s1.length(), opt = (len == 2)*nSymbols;
          if (opt) {
              u16 first2 = s1.first2();
              for(u32 k=0; k<opt; k++) {  
                 Symbol s2 = symbols[FSST_CODE_BASE+k];
                 if (k != i && s2.length() > 1 && first2 == s2.first2()) // test if symbol k is a suffix of s
                    opt = 0;
              }
              newCode[i] = opt?suffixLim++:--j; // symbols without a larger suffix have a code < suffixLim 
          } else 
              newCode[i] = rsum[len-1]++;
          s1.set_code_len(newCode[i],len);
          symbols[newCode[i]] = s1; 
       }
       // renumber the codes in byteCodes[] 
       for(u32 i=0; i<256; i++) 
          if ((byteCodes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE)
             byteCodes[i] = newCode[(u8) byteCodes[i]] + (1 << FSST_LEN_BITS);
          else 
             byteCodes[i] = 511 + (1 << FSST_LEN_BITS);
       
       // renumber the codes in shortCodes[] 
       for(u32 i=0; i<65536; i++)
          if ((shortCodes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE)
             shortCodes[i] = newCode[(u8) shortCodes[i]] + (shortCodes[i] & (15 << FSST_LEN_BITS));
          else 
             shortCodes[i] = byteCodes[i&0xFF];

       // replace the symbols in the hash table
       for(u32 i=0; i<hashTabSize; i++)
          if (hashTab[i].icl < FSST_ICL_FREE)
             hashTab[i] = symbols[newCode[(u8) hashTab[i].code()]];
   }
};

#ifdef NONOPT_FSST
struct Counters {
   u16 count1[FSST_CODE_MAX];   // array to count frequency of symbols as they occur in the sample 
   u16 count2[FSST_CODE_MAX][FSST_CODE_MAX]; // array to count subsequent combinations of two symbols in the sample 

   void count1Set(u32 pos1, u16 val) { 
      count1[pos1] = val;
   }
   void count1Inc(u32 pos1) { 
      count1[pos1]++;
   }
   void count2Inc(u32 pos1, u32 pos2) {  
      count2[pos1][pos2]++;
   }
   u32 count1GetNext(u32 &pos1) { 
      return count1[pos1];
   }
   u32 count2GetNext(u32 pos1, u32 &pos2) { 
      return count2[pos1][pos2];
   }
   void backup1(u8 *buf) {
      memcpy(buf, count1, FSST_CODE_MAX*sizeof(u16));
   }
   void restore1(u8 *buf) {
      memcpy(count1, buf, FSST_CODE_MAX*sizeof(u16));
   }
};
#else
// we keep two counters count1[pos] and count2[pos1][pos2] of resp 16 and 12-bits. Both are split into two columns for performance reasons
// first reason is to make the column we update the most during symbolTable construction (the low bits) thinner, thus reducing CPU cache pressure.
// second reason is that when scanning the array, after seeing a 64-bits 0 in the high bits column, we can quickly skip over many codes (15 or 7)
struct Counters {
   // high arrays come before low arrays, because our GetNext() methods may overrun their 64-bits reads a few bytes
   u8 count1High[FSST_CODE_MAX];   // array to count frequency of symbols as they occur in the sample (16-bits)
   u8 count1Low[FSST_CODE_MAX];    // it is split in a low and high byte: cnt = count1High*256 + count1Low
   u8 count2High[FSST_CODE_MAX][FSST_CODE_MAX/2]; // array to count subsequent combinations of two symbols in the sample (12-bits: 8-bits low, 4-bits high)
   u8 count2Low[FSST_CODE_MAX][FSST_CODE_MAX];    // its value is (count2High*256+count2Low) -- but high is 4-bits (we put two numbers in one, hence /2)
   // 385KB  -- but hot area likely just 10 + 30*4 = 130 cache lines (=8KB)
   
   void count1Set(u32 pos1, u16 val) { 
      count1Low[pos1] = val&255;
      count1High[pos1] = val>>8;
   }
   void count1Inc(u32 pos1) { 
      if (!count1Low[pos1]++) // increment high early (when low==0, not when low==255). This means (high > 0) <=> (cnt > 0)
         count1High[pos1]++; //(0,0)->(1,1)->..->(255,1)->(0,1)->(1,2)->(2,2)->(3,2)..(255,2)->(0,2)->(1,3)->(2,3)...
   }
   void count2Inc(u32 pos1, u32 pos2) {  
       if (!count2Low[pos1][pos2]++) // increment high early (when low==0, not when low==255). This means (high > 0) <=> (cnt > 0)
          // inc 4-bits high counter with 1<<0 (1) or 1<<4 (16) -- depending on whether pos2 is even or odd, repectively
          count2High[pos1][(pos2)>>1] += 1 << (((pos2)&1)<<2); // we take our chances with overflow.. (4K maxval, on a 8K sample)
   }
   u32 count1GetNext(u32 &pos1) { // note: we will advance pos1 to the next nonzero counter in register range
      // read 16-bits single symbol counter, split into two 8-bits numbers (count1Low, count1High), while skipping over zeros
	   u64 high = fsst_unaligned_load(&count1High[pos1]);

      u32 zero = high?(__builtin_ctzl(high)>>3):7UL; // number of zero bytes
      high = (high >> (zero << 3)) & 255; // advance to nonzero counter
      if (((pos1 += zero) >= FSST_CODE_MAX) || !high) // SKIP! advance pos2
         return 0; // all zero

      u32 low = count1Low[pos1];
      if (low) high--; // high is incremented early and low late, so decrement high (unless low==0)
      return (u32) ((high << 8) + low);
   }
   u32 count2GetNext(u32 pos1, u32 &pos2) { // note: we will advance pos2 to the next nonzero counter in register range
      // read 12-bits pairwise symbol counter, split into low 8-bits and high 4-bits number while skipping over zeros
	  u64 high = fsst_unaligned_load(&count2High[pos1][pos2>>1]);
      high >>= ((pos2&1) << 2); // odd pos2: ignore the lowest 4 bits & we see only 15 counters

      u32 zero = high?(__builtin_ctzl(high)>>2):(15UL-(pos2&1UL)); // number of zero 4-bits counters
      high = (high >> (zero << 2)) & 15;  // advance to nonzero counter
      if (((pos2 += zero) >= FSST_CODE_MAX) || !high) // SKIP! advance pos2
         return 0UL; // all zero

      u32 low = count2Low[pos1][pos2];
      if (low) high--; // high is incremented early and low late, so decrement high (unless low==0)
      return (u32) ((high << 8) + low);
   }
   void backup1(u8 *buf) {
      memcpy(buf, count1High, FSST_CODE_MAX);
      memcpy(buf+FSST_CODE_MAX, count1Low, FSST_CODE_MAX);
   }
   void restore1(u8 *buf) {
      memcpy(count1High, buf, FSST_CODE_MAX);
      memcpy(count1Low, buf+FSST_CODE_MAX, FSST_CODE_MAX);
   }
}; 
#endif


#define FSST_BUFSZ (3<<19) // 768KB

// an encoder is a symbolmap plus some bufferspace, needed during map construction as well as compression 
struct Encoder {
   shared_ptr<SymbolTable> symbolTable; // symbols, plus metadata and data structures for quick compression (shortCode,hashTab, etc)
   union {
      Counters counters;     // for counting symbol occurences during map construction
      u8 simdbuf[FSST_BUFSZ]; // for compression: SIMD string staging area 768KB = 256KB in + 512KB out (worst case for 256KB in) 
   };
};

// job control integer representable in one 64bits SIMD lane: cur/end=input, out=output, pos=which string (2^9=512 per call)
struct SIMDjob {
   u64 out:19,pos:9,end:18,cur:18; // cur/end is input offsets (2^18=256KB), out is output offset (2^19=512KB)  
};

extern bool 
duckdb_fsst_hasAVX512(); // runtime check for avx512 capability

extern size_t 
duckdb_fsst_compressAVX512(
   SymbolTable &symbolTable, 
   u8* codeBase,    // IN: base address for codes, i.e. compression output (points to simdbuf+256KB)
   u8* symbolBase,  // IN: base address for string bytes, i.e. compression input (points to simdbuf)
   SIMDjob* input,  // IN: input array (size n) with job information: what to encode, where to store it.
   SIMDjob* output, // OUT: output array (size n) with job information: how much got encoded, end output pointer.
   size_t n,         // IN: size of arrays input and output (should be max 512)
   size_t unroll);   // IN: degree of SIMD unrolling

// C++ fsst-compress function with some more control of how the compression happens (algorithm flavor, simd unroll degree)
size_t compressImpl(Encoder *encoder, size_t n, size_t lenIn[], u8 *strIn[], size_t size, u8 * output, size_t *lenOut, u8 *strOut[], bool noSuffixOpt, bool avoidBranch, int simd);
size_t compressAuto(Encoder *encoder, size_t n, size_t lenIn[], u8 *strIn[], size_t size, u8 * output, size_t *lenOut, u8 *strOut[], int simd);


// LICENSE_CHANGE_END


#if DUCKDB_FSST_ENABLE_INTRINSINCS && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>

#ifdef _WIN32
bool duckdb_fsst_hasAVX512() {
	int info[4];
	__cpuidex(info, 0x00000007, 0);
	return (info[1]>>16)&1;
}
#else
#include <cpuid.h>
bool duckdb_fsst_hasAVX512() {
	int info[4];
	__cpuid_count(0x00000007, 0, info[0], info[1], info[2], info[3]);
	return (info[1]>>16)&1;
}
#endif
#else
bool duckdb_fsst_hasAVX512() { return false; }
#endif

// BULK COMPRESSION OF STRINGS
//
// In one call of this function, we can compress 512 strings, each of maximum length 511 bytes.
// strings can be shorter than 511 bytes, no problem, but if they are longer we need to cut them up.
//
// In each iteration of the while loop, we find one code in each of the unroll*8 strings, i.e. (8,16,24 or 32) for resp. unroll=1,2,3,4
// unroll3 performs best on my hardware
//
// In the worst case, each final encoded string occupies 512KB bytes (512*1024; with 1024=512xexception, exception = 2 bytes).
// - hence codeBase is a buffer of 512KB (needs 19 bits jobs), symbolBase of 256KB (needs 18 bits jobs).
//
// 'jobX' controls the encoding of each string and is therefore a u64 with format [out:19][pos:9][end:18][cur:18] (low-to-high bits)
// The field 'pos' tells which string we are processing (0..511). We need this info as strings will complete compressing out-of-order.
//
// Strings will have different lengths, and when a string is finished, we reload from the buffer of 512 input strings.
// This continues until we have less than (8,16,24 or 32; depending on unroll) strings left to process.
// - so 'processed' is the amount of strings we started processing and it is between [480,512].
// Note that when we quit, there will still be some (<32) strings that we started to process but which are unfinished.
// - so 'unfinished' is that amount. These unfinished strings will be encoded further using the scalar method.
//
// Apart from the coded strings, we return in a output[] array of size 'processed' the job values of the 'finished' strings.
// In the following 'unfinished' slots (processed=finished+unfinished) we output the 'job' values of the unfinished strings.
//
// For the finished strings, we need [out:19] to see the compressed size and [pos:9] to see which string we refer to.
// For the unfinished strings, we need all fields of 'job' to continue the compression with scalar code (see SIMD code in compressBatch).
//
// THIS IS A SEPARATE CODE FILE NOT BECAUSE OF MY LOVE FOR MODULARIZED CODE BUT BECAUSE IT ALLOWS TO COMPILE IT WITH DIFFERENT FLAGS
// in particular, unrolling is crucial for gather/scatter performance, but requires registers. the #define all_* expressions however,
// will be detected to be constants by g++ -O2 and will be precomputed and placed into AVX512 registers - spoiling 9 of them.
// This reduces the effectiveness of unrolling, hence -O2 makes the loop perform worse than -O1 which skips this optimization.
// Assembly inspection confirmed that 3-way unroll with -O1 avoids needless load/stores.

size_t duckdb_fsst_compressAVX512(SymbolTable &symbolTable, u8* codeBase, u8* symbolBase, SIMDjob *input, SIMDjob *output, size_t n, size_t unroll) {
	size_t processed = 0;
	// define some constants (all_x means that all 8 lanes contain 64-bits value X)
#if defined(__AVX512F__) and DUCKDB_FSST_ENABLE_INTRINSINCS
	//__m512i all_suffixLim= _mm512_broadcastq_epi64(_mm_set1_epi64((__m64) (u64) symbolTable->suffixLim)); -- for variants b,c
	__m512i all_MASK     = _mm512_broadcastq_epi64(_mm_set1_epi64((__m64) (u64) -1));
	__m512i all_PRIME    = _mm512_broadcastq_epi64(_mm_set1_epi64((__m64) (u64) FSST_HASH_PRIME));
	__m512i all_ICL_FREE = _mm512_broadcastq_epi64(_mm_set1_epi64((__m64) (u64) FSST_ICL_FREE));
#define    all_HASH       _mm512_srli_epi64(all_MASK, 64-FSST_HASH_LOG2SIZE)
#define    all_ONE        _mm512_srli_epi64(all_MASK, 63)
#define    all_M19        _mm512_srli_epi64(all_MASK, 45)
#define    all_M18        _mm512_srli_epi64(all_MASK, 46)
#define    all_M28        _mm512_srli_epi64(all_MASK, 36)
#define    all_FFFFFF     _mm512_srli_epi64(all_MASK, 40)
#define    all_FFFF       _mm512_srli_epi64(all_MASK, 48)
#define    all_FF         _mm512_srli_epi64(all_MASK, 56)

	SIMDjob *inputEnd = input+n;
	assert(n >= unroll*8 && n <= 512); // should be close to 512
	__m512i job1, job2, job3, job4; // will contain current jobs, for each unroll 1,2,3,4
	__mmask8 loadmask1 = 255, loadmask2 = 255*(unroll>1), loadmask3 = 255*(unroll>2), loadmask4 = 255*(unroll>3); // 2b loaded new strings bitmask per unroll
	u32 delta1 = 8, delta2 = 8*(unroll>1), delta3 = 8*(unroll>2), delta4 = 8*(unroll>3); // #new loads this SIMD iteration per unroll

	if (unroll >= 4) {
		while (input+delta1+delta2+delta3+delta4 < inputEnd) {


// LICENSE_CHANGE_BEGIN
// The following code up to LICENSE_CHANGE_END is subject to THIRD PARTY LICENSE #5
// See the end of this file for a list

// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
//
//
//
//
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
//
//
//
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// furnished to do so, subject to the following conditions:
// furnished to do so, subject to the following conditions:
// furnished to do so, subject to the following conditions:
//
//
//
//
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//
//
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E1PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E2PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E3PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E4PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//
//
//
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
//
//
//
//
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask1=11111111, delta1=8).
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask2=11111111, delta2=8).
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask3=11111111, delta3=8).
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask4=11111111, delta4=8).
            job1      = _mm512_mask_expandloadu_epi64(job1, loadmask1, input); input += delta1; 
            job2      = _mm512_mask_expandloadu_epi64(job2, loadmask2, input); input += delta2; 
            job3      = _mm512_mask_expandloadu_epi64(job3, loadmask3, input); input += delta3; 
            job4      = _mm512_mask_expandloadu_epi64(job4, loadmask4, input); input += delta4; 
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
   __m512i  word1     = _mm512_i64gather_epi64(_mm512_srli_epi64(job1, 46), symbolBase, 1); 
   __m512i  word2     = _mm512_i64gather_epi64(_mm512_srli_epi64(job2, 46), symbolBase, 1); 
   __m512i  word3     = _mm512_i64gather_epi64(_mm512_srli_epi64(job3, 46), symbolBase, 1); 
   __m512i  word4     = _mm512_i64gather_epi64(_mm512_srli_epi64(job4, 46), symbolBase, 1); 
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // code1: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
                        // code2: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
                        // code3: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
                        // code4: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
   __m512i  code1     = _mm512_i64gather_epi64(_mm512_and_epi64(word1, all_FFFF), symbolTable.shortCodes, sizeof(u16));
   __m512i  code2     = _mm512_i64gather_epi64(_mm512_and_epi64(word2, all_FFFF), symbolTable.shortCodes, sizeof(u16));
   __m512i  code3     = _mm512_i64gather_epi64(_mm512_and_epi64(word3, all_FFFF), symbolTable.shortCodes, sizeof(u16));
   __m512i  code4     = _mm512_i64gather_epi64(_mm512_and_epi64(word4, all_FFFF), symbolTable.shortCodes, sizeof(u16));
                        // get the first three bytes of the string. 
                        // get the first three bytes of the string. 
                        // get the first three bytes of the string. 
                        // get the first three bytes of the string. 
   __m512i  pos1      = _mm512_mullo_epi64(_mm512_and_epi64(word1, all_FFFFFF), all_PRIME);
   __m512i  pos2      = _mm512_mullo_epi64(_mm512_and_epi64(word2, all_FFFFFF), all_PRIME);
   __m512i  pos3      = _mm512_mullo_epi64(_mm512_and_epi64(word3, all_FFFFFF), all_PRIME);
   __m512i  pos4      = _mm512_mullo_epi64(_mm512_and_epi64(word4, all_FFFFFF), all_PRIME);
                        // hash them into a random number: pos1 = pos1*PRIME; pos1 ^= pos1>>SHIFT
                        // hash them into a random number: pos2 = pos2*PRIME; pos2 ^= pos2>>SHIFT
                        // hash them into a random number: pos3 = pos3*PRIME; pos3 ^= pos3>>SHIFT
                        // hash them into a random number: pos4 = pos4*PRIME; pos4 ^= pos4>>SHIFT
            pos1      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos1,_mm512_srli_epi64(pos1,FSST_SHIFT)), all_HASH), 4);
            pos2      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos2,_mm512_srli_epi64(pos2,FSST_SHIFT)), all_HASH), 4);
            pos3      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos3,_mm512_srli_epi64(pos3,FSST_SHIFT)), all_HASH), 4);
            pos4      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos4,_mm512_srli_epi64(pos4,FSST_SHIFT)), all_HASH), 4);
                        // lookup in the 3-byte-prefix keyed hash table
                        // lookup in the 3-byte-prefix keyed hash table
                        // lookup in the 3-byte-prefix keyed hash table
                        // lookup in the 3-byte-prefix keyed hash table
   __m512i  icl1      = _mm512_i64gather_epi64(pos1, (((char*) symbolTable.hashTab) + 8), 1);
   __m512i  icl2      = _mm512_i64gather_epi64(pos2, (((char*) symbolTable.hashTab) + 8), 1);
   __m512i  icl3      = _mm512_i64gather_epi64(pos3, (((char*) symbolTable.hashTab) + 8), 1);
   __m512i  icl4      = _mm512_i64gather_epi64(pos4, (((char*) symbolTable.hashTab) + 8), 1);
                        // speculatively store the first input byte into the second position of the write1 register (in case it turns out to be an escaped byte).
                        // speculatively store the first input byte into the second position of the write2 register (in case it turns out to be an escaped byte).
                        // speculatively store the first input byte into the second position of the write3 register (in case it turns out to be an escaped byte).
                        // speculatively store the first input byte into the second position of the write4 register (in case it turns out to be an escaped byte).
   __m512i  write1    = _mm512_slli_epi64(_mm512_and_epi64(word1, all_FF), 8);
   __m512i  write2    = _mm512_slli_epi64(_mm512_and_epi64(word2, all_FF), 8);
   __m512i  write3    = _mm512_slli_epi64(_mm512_and_epi64(word3, all_FF), 8);
   __m512i  write4    = _mm512_slli_epi64(_mm512_and_epi64(word4, all_FF), 8);
                        // lookup just like the icl1 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
                        // lookup just like the icl2 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
                        // lookup just like the icl3 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
                        // lookup just like the icl4 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
   __m512i  symb1     = _mm512_i64gather_epi64(pos1, (((char*) symbolTable.hashTab) + 0), 1);
   __m512i  symb2     = _mm512_i64gather_epi64(pos2, (((char*) symbolTable.hashTab) + 0), 1);
   __m512i  symb3     = _mm512_i64gather_epi64(pos3, (((char*) symbolTable.hashTab) + 0), 1);
   __m512i  symb4     = _mm512_i64gather_epi64(pos4, (((char*) symbolTable.hashTab) + 0), 1);
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
            pos1      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl1, all_FF));
            pos2      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl2, all_FF));
            pos3      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl3, all_FF));
            pos4      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl4, all_FF));
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
   __mmask8 match1    = _mm512_cmpeq_epi64_mask(symb1, _mm512_and_epi64(word1, pos1)) & _mm512_cmplt_epi64_mask(icl1, all_ICL_FREE);
   __mmask8 match2    = _mm512_cmpeq_epi64_mask(symb2, _mm512_and_epi64(word2, pos2)) & _mm512_cmplt_epi64_mask(icl2, all_ICL_FREE);
   __mmask8 match3    = _mm512_cmpeq_epi64_mask(symb3, _mm512_and_epi64(word3, pos3)) & _mm512_cmplt_epi64_mask(icl3, all_ICL_FREE);
   __mmask8 match4    = _mm512_cmpeq_epi64_mask(symb4, _mm512_and_epi64(word4, pos4)) & _mm512_cmplt_epi64_mask(icl4, all_ICL_FREE);
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
            code1     = _mm512_mask_mov_epi64(code1, match1, _mm512_srli_epi64(icl1, 16));
            code2     = _mm512_mask_mov_epi64(code2, match2, _mm512_srli_epi64(icl2, 16));
            code3     = _mm512_mask_mov_epi64(code3, match3, _mm512_srli_epi64(icl3, 16));
            code4     = _mm512_mask_mov_epi64(code4, match4, _mm512_srli_epi64(icl4, 16));
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
            write1    = _mm512_or_epi64(write1, _mm512_and_epi64(code1, all_FF));
            write2    = _mm512_or_epi64(write2, _mm512_and_epi64(code2, all_FF));
            write3    = _mm512_or_epi64(write3, _mm512_and_epi64(code3, all_FF));
            write4    = _mm512_or_epi64(write4, _mm512_and_epi64(code4, all_FF));
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
            code1     = _mm512_and_epi64(code1, all_FFFF);
            code2     = _mm512_and_epi64(code2, all_FFFF);
            code3     = _mm512_and_epi64(code3, all_FFFF);
            code4     = _mm512_and_epi64(code4, all_FFFF);
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job1, all_M19), write1, 1);
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job2, all_M19), write2, 1);
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job3, all_M19), write3, 1);
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job4, all_M19), write4, 1);
                        // increase the job1.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
                        // increase the job2.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
                        // increase the job3.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
                        // increase the job4.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
            job1      = _mm512_add_epi64(job1, _mm512_slli_epi64(_mm512_srli_epi64(code1, FSST_LEN_BITS), 46));
            job2      = _mm512_add_epi64(job2, _mm512_slli_epi64(_mm512_srli_epi64(code2, FSST_LEN_BITS), 46));
            job3      = _mm512_add_epi64(job3, _mm512_slli_epi64(_mm512_srli_epi64(code3, FSST_LEN_BITS), 46));
            job4      = _mm512_add_epi64(job4, _mm512_slli_epi64(_mm512_srli_epi64(code4, FSST_LEN_BITS), 46));
                        // increase the job1.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
                        // increase the job2.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
                        // increase the job3.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
                        // increase the job4.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
            job1      = _mm512_add_epi64(job1, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code1, 8), all_ONE)));
            job2      = _mm512_add_epi64(job2, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code2, 8), all_ONE)));
            job3      = _mm512_add_epi64(job3, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code3, 8), all_ONE)));
            job4      = _mm512_add_epi64(job4, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code4, 8), all_ONE)));
                        // test which lanes are done now (job1.cur==job1.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job1 register)
                        // test which lanes are done now (job2.cur==job2.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job2 register)
                        // test which lanes are done now (job3.cur==job3.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job3 register)
                        // test which lanes are done now (job4.cur==job4.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job4 register)
            loadmask1 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job1, 46), _mm512_and_epi64(_mm512_srli_epi64(job1, 28), all_M18));
            loadmask2 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job2, 46), _mm512_and_epi64(_mm512_srli_epi64(job2, 28), all_M18));
            loadmask3 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job3, 46), _mm512_and_epi64(_mm512_srli_epi64(job3, 28), all_M18));
            loadmask4 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job4, 46), _mm512_and_epi64(_mm512_srli_epi64(job4, 28), all_M18));
                        // calculate the amount of lanes in job1 that are done
                        // calculate the amount of lanes in job2 that are done
                        // calculate the amount of lanes in job3 that are done
                        // calculate the amount of lanes in job4 that are done
            delta1    = _mm_popcnt_u32((int) loadmask1); 
            delta2    = _mm_popcnt_u32((int) loadmask2); 
            delta3    = _mm_popcnt_u32((int) loadmask3); 
            delta4    = _mm_popcnt_u32((int) loadmask4); 
                        // write out the job state for the lanes that are done (we need the final 'job1.out' value to compute the compressed string length)
                        // write out the job state for the lanes that are done (we need the final 'job2.out' value to compute the compressed string length)
                        // write out the job state for the lanes that are done (we need the final 'job3.out' value to compute the compressed string length)
                        // write out the job state for the lanes that are done (we need the final 'job4.out' value to compute the compressed string length)
                        _mm512_mask_compressstoreu_epi64(output, loadmask1, job1); output += delta1;
                        _mm512_mask_compressstoreu_epi64(output, loadmask2, job2); output += delta2;
                        _mm512_mask_compressstoreu_epi64(output, loadmask3, job3); output += delta3;
                        _mm512_mask_compressstoreu_epi64(output, loadmask4, job4); output += delta4;


// LICENSE_CHANGE_END

		}
	} else if (unroll == 3) {
		while (input+delta1+delta2+delta3 < inputEnd) {


// LICENSE_CHANGE_BEGIN
// The following code up to LICENSE_CHANGE_END is subject to THIRD PARTY LICENSE #5
// See the end of this file for a list

// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
//
//
//
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
//
//
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// furnished to do so, subject to the following conditions:
// furnished to do so, subject to the following conditions:
//
//
//
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E1PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E2PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E3PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//
//
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
//
//
//
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask1=11111111, delta1=8).
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask2=11111111, delta2=8).
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask3=11111111, delta3=8).
            job1      = _mm512_mask_expandloadu_epi64(job1, loadmask1, input); input += delta1; 
            job2      = _mm512_mask_expandloadu_epi64(job2, loadmask2, input); input += delta2; 
            job3      = _mm512_mask_expandloadu_epi64(job3, loadmask3, input); input += delta3; 
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
   __m512i  word1     = _mm512_i64gather_epi64(_mm512_srli_epi64(job1, 46), symbolBase, 1); 
   __m512i  word2     = _mm512_i64gather_epi64(_mm512_srli_epi64(job2, 46), symbolBase, 1); 
   __m512i  word3     = _mm512_i64gather_epi64(_mm512_srli_epi64(job3, 46), symbolBase, 1); 
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // code1: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
                        // code2: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
                        // code3: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
   __m512i  code1     = _mm512_i64gather_epi64(_mm512_and_epi64(word1, all_FFFF), symbolTable.shortCodes, sizeof(u16));
   __m512i  code2     = _mm512_i64gather_epi64(_mm512_and_epi64(word2, all_FFFF), symbolTable.shortCodes, sizeof(u16));
   __m512i  code3     = _mm512_i64gather_epi64(_mm512_and_epi64(word3, all_FFFF), symbolTable.shortCodes, sizeof(u16));
                        // get the first three bytes of the string. 
                        // get the first three bytes of the string. 
                        // get the first three bytes of the string. 
   __m512i  pos1      = _mm512_mullo_epi64(_mm512_and_epi64(word1, all_FFFFFF), all_PRIME);
   __m512i  pos2      = _mm512_mullo_epi64(_mm512_and_epi64(word2, all_FFFFFF), all_PRIME);
   __m512i  pos3      = _mm512_mullo_epi64(_mm512_and_epi64(word3, all_FFFFFF), all_PRIME);
                        // hash them into a random number: pos1 = pos1*PRIME; pos1 ^= pos1>>SHIFT
                        // hash them into a random number: pos2 = pos2*PRIME; pos2 ^= pos2>>SHIFT
                        // hash them into a random number: pos3 = pos3*PRIME; pos3 ^= pos3>>SHIFT
            pos1      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos1,_mm512_srli_epi64(pos1,FSST_SHIFT)), all_HASH), 4);
            pos2      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos2,_mm512_srli_epi64(pos2,FSST_SHIFT)), all_HASH), 4);
            pos3      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos3,_mm512_srli_epi64(pos3,FSST_SHIFT)), all_HASH), 4);
                        // lookup in the 3-byte-prefix keyed hash table
                        // lookup in the 3-byte-prefix keyed hash table
                        // lookup in the 3-byte-prefix keyed hash table
   __m512i  icl1      = _mm512_i64gather_epi64(pos1, (((char*) symbolTable.hashTab) + 8), 1);
   __m512i  icl2      = _mm512_i64gather_epi64(pos2, (((char*) symbolTable.hashTab) + 8), 1);
   __m512i  icl3      = _mm512_i64gather_epi64(pos3, (((char*) symbolTable.hashTab) + 8), 1);
                        // speculatively store the first input byte into the second position of the write1 register (in case it turns out to be an escaped byte).
                        // speculatively store the first input byte into the second position of the write2 register (in case it turns out to be an escaped byte).
                        // speculatively store the first input byte into the second position of the write3 register (in case it turns out to be an escaped byte).
   __m512i  write1    = _mm512_slli_epi64(_mm512_and_epi64(word1, all_FF), 8);
   __m512i  write2    = _mm512_slli_epi64(_mm512_and_epi64(word2, all_FF), 8);
   __m512i  write3    = _mm512_slli_epi64(_mm512_and_epi64(word3, all_FF), 8);
                        // lookup just like the icl1 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
                        // lookup just like the icl2 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
                        // lookup just like the icl3 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
   __m512i  symb1     = _mm512_i64gather_epi64(pos1, (((char*) symbolTable.hashTab) + 0), 1);
   __m512i  symb2     = _mm512_i64gather_epi64(pos2, (((char*) symbolTable.hashTab) + 0), 1);
   __m512i  symb3     = _mm512_i64gather_epi64(pos3, (((char*) symbolTable.hashTab) + 0), 1);
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
            pos1      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl1, all_FF));
            pos2      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl2, all_FF));
            pos3      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl3, all_FF));
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
   __mmask8 match1    = _mm512_cmpeq_epi64_mask(symb1, _mm512_and_epi64(word1, pos1)) & _mm512_cmplt_epi64_mask(icl1, all_ICL_FREE);
   __mmask8 match2    = _mm512_cmpeq_epi64_mask(symb2, _mm512_and_epi64(word2, pos2)) & _mm512_cmplt_epi64_mask(icl2, all_ICL_FREE);
   __mmask8 match3    = _mm512_cmpeq_epi64_mask(symb3, _mm512_and_epi64(word3, pos3)) & _mm512_cmplt_epi64_mask(icl3, all_ICL_FREE);
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
            code1     = _mm512_mask_mov_epi64(code1, match1, _mm512_srli_epi64(icl1, 16));
            code2     = _mm512_mask_mov_epi64(code2, match2, _mm512_srli_epi64(icl2, 16));
            code3     = _mm512_mask_mov_epi64(code3, match3, _mm512_srli_epi64(icl3, 16));
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
            write1    = _mm512_or_epi64(write1, _mm512_and_epi64(code1, all_FF));
            write2    = _mm512_or_epi64(write2, _mm512_and_epi64(code2, all_FF));
            write3    = _mm512_or_epi64(write3, _mm512_and_epi64(code3, all_FF));
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
            code1     = _mm512_and_epi64(code1, all_FFFF);
            code2     = _mm512_and_epi64(code2, all_FFFF);
            code3     = _mm512_and_epi64(code3, all_FFFF);
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job1, all_M19), write1, 1);
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job2, all_M19), write2, 1);
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job3, all_M19), write3, 1);
                        // increase the job1.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
                        // increase the job2.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
                        // increase the job3.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
            job1      = _mm512_add_epi64(job1, _mm512_slli_epi64(_mm512_srli_epi64(code1, FSST_LEN_BITS), 46));
            job2      = _mm512_add_epi64(job2, _mm512_slli_epi64(_mm512_srli_epi64(code2, FSST_LEN_BITS), 46));
            job3      = _mm512_add_epi64(job3, _mm512_slli_epi64(_mm512_srli_epi64(code3, FSST_LEN_BITS), 46));
                        // increase the job1.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
                        // increase the job2.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
                        // increase the job3.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
            job1      = _mm512_add_epi64(job1, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code1, 8), all_ONE)));
            job2      = _mm512_add_epi64(job2, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code2, 8), all_ONE)));
            job3      = _mm512_add_epi64(job3, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code3, 8), all_ONE)));
                        // test which lanes are done now (job1.cur==job1.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job1 register)
                        // test which lanes are done now (job2.cur==job2.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job2 register)
                        // test which lanes are done now (job3.cur==job3.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job3 register)
            loadmask1 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job1, 46), _mm512_and_epi64(_mm512_srli_epi64(job1, 28), all_M18));
            loadmask2 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job2, 46), _mm512_and_epi64(_mm512_srli_epi64(job2, 28), all_M18));
            loadmask3 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job3, 46), _mm512_and_epi64(_mm512_srli_epi64(job3, 28), all_M18));
                        // calculate the amount of lanes in job1 that are done
                        // calculate the amount of lanes in job2 that are done
                        // calculate the amount of lanes in job3 that are done
            delta1    = _mm_popcnt_u32((int) loadmask1); 
            delta2    = _mm_popcnt_u32((int) loadmask2); 
            delta3    = _mm_popcnt_u32((int) loadmask3); 
                        // write out the job state for the lanes that are done (we need the final 'job1.out' value to compute the compressed string length)
                        // write out the job state for the lanes that are done (we need the final 'job2.out' value to compute the compressed string length)
                        // write out the job state for the lanes that are done (we need the final 'job3.out' value to compute the compressed string length)
                        _mm512_mask_compressstoreu_epi64(output, loadmask1, job1); output += delta1;
                        _mm512_mask_compressstoreu_epi64(output, loadmask2, job2); output += delta2;
                        _mm512_mask_compressstoreu_epi64(output, loadmask3, job3); output += delta3;


// LICENSE_CHANGE_END

		}
	} else if (unroll == 2) {
		while (input+delta1+delta2 < inputEnd) {


// LICENSE_CHANGE_BEGIN
// The following code up to LICENSE_CHANGE_END is subject to THIRD PARTY LICENSE #5
// See the end of this file for a list

// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
//
//
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
//
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// furnished to do so, subject to the following conditions:
//
//
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E1PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E2PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
//
//
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask1=11111111, delta1=8).
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask2=11111111, delta2=8).
            job1      = _mm512_mask_expandloadu_epi64(job1, loadmask1, input); input += delta1; 
            job2      = _mm512_mask_expandloadu_epi64(job2, loadmask2, input); input += delta2; 
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
   __m512i  word1     = _mm512_i64gather_epi64(_mm512_srli_epi64(job1, 46), symbolBase, 1); 
   __m512i  word2     = _mm512_i64gather_epi64(_mm512_srli_epi64(job2, 46), symbolBase, 1); 
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // code1: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
                        // code2: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
   __m512i  code1     = _mm512_i64gather_epi64(_mm512_and_epi64(word1, all_FFFF), symbolTable.shortCodes, sizeof(u16));
   __m512i  code2     = _mm512_i64gather_epi64(_mm512_and_epi64(word2, all_FFFF), symbolTable.shortCodes, sizeof(u16));
                        // get the first three bytes of the string. 
                        // get the first three bytes of the string. 
   __m512i  pos1      = _mm512_mullo_epi64(_mm512_and_epi64(word1, all_FFFFFF), all_PRIME);
   __m512i  pos2      = _mm512_mullo_epi64(_mm512_and_epi64(word2, all_FFFFFF), all_PRIME);
                        // hash them into a random number: pos1 = pos1*PRIME; pos1 ^= pos1>>SHIFT
                        // hash them into a random number: pos2 = pos2*PRIME; pos2 ^= pos2>>SHIFT
            pos1      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos1,_mm512_srli_epi64(pos1,FSST_SHIFT)), all_HASH), 4);
            pos2      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos2,_mm512_srli_epi64(pos2,FSST_SHIFT)), all_HASH), 4);
                        // lookup in the 3-byte-prefix keyed hash table
                        // lookup in the 3-byte-prefix keyed hash table
   __m512i  icl1      = _mm512_i64gather_epi64(pos1, (((char*) symbolTable.hashTab) + 8), 1);
   __m512i  icl2      = _mm512_i64gather_epi64(pos2, (((char*) symbolTable.hashTab) + 8), 1);
                        // speculatively store the first input byte into the second position of the write1 register (in case it turns out to be an escaped byte).
                        // speculatively store the first input byte into the second position of the write2 register (in case it turns out to be an escaped byte).
   __m512i  write1    = _mm512_slli_epi64(_mm512_and_epi64(word1, all_FF), 8);
   __m512i  write2    = _mm512_slli_epi64(_mm512_and_epi64(word2, all_FF), 8);
                        // lookup just like the icl1 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
                        // lookup just like the icl2 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
   __m512i  symb1     = _mm512_i64gather_epi64(pos1, (((char*) symbolTable.hashTab) + 0), 1);
   __m512i  symb2     = _mm512_i64gather_epi64(pos2, (((char*) symbolTable.hashTab) + 0), 1);
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
            pos1      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl1, all_FF));
            pos2      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl2, all_FF));
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
   __mmask8 match1    = _mm512_cmpeq_epi64_mask(symb1, _mm512_and_epi64(word1, pos1)) & _mm512_cmplt_epi64_mask(icl1, all_ICL_FREE);
   __mmask8 match2    = _mm512_cmpeq_epi64_mask(symb2, _mm512_and_epi64(word2, pos2)) & _mm512_cmplt_epi64_mask(icl2, all_ICL_FREE);
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
            code1     = _mm512_mask_mov_epi64(code1, match1, _mm512_srli_epi64(icl1, 16));
            code2     = _mm512_mask_mov_epi64(code2, match2, _mm512_srli_epi64(icl2, 16));
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
            write1    = _mm512_or_epi64(write1, _mm512_and_epi64(code1, all_FF));
            write2    = _mm512_or_epi64(write2, _mm512_and_epi64(code2, all_FF));
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
            code1     = _mm512_and_epi64(code1, all_FFFF);
            code2     = _mm512_and_epi64(code2, all_FFFF);
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job1, all_M19), write1, 1);
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job2, all_M19), write2, 1);
                        // increase the job1.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
                        // increase the job2.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
            job1      = _mm512_add_epi64(job1, _mm512_slli_epi64(_mm512_srli_epi64(code1, FSST_LEN_BITS), 46));
            job2      = _mm512_add_epi64(job2, _mm512_slli_epi64(_mm512_srli_epi64(code2, FSST_LEN_BITS), 46));
                        // increase the job1.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
                        // increase the job2.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
            job1      = _mm512_add_epi64(job1, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code1, 8), all_ONE)));
            job2      = _mm512_add_epi64(job2, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code2, 8), all_ONE)));
                        // test which lanes are done now (job1.cur==job1.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job1 register)
                        // test which lanes are done now (job2.cur==job2.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job2 register)
            loadmask1 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job1, 46), _mm512_and_epi64(_mm512_srli_epi64(job1, 28), all_M18));
            loadmask2 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job2, 46), _mm512_and_epi64(_mm512_srli_epi64(job2, 28), all_M18));
                        // calculate the amount of lanes in job1 that are done
                        // calculate the amount of lanes in job2 that are done
            delta1    = _mm_popcnt_u32((int) loadmask1); 
            delta2    = _mm_popcnt_u32((int) loadmask2); 
                        // write out the job state for the lanes that are done (we need the final 'job1.out' value to compute the compressed string length)
                        // write out the job state for the lanes that are done (we need the final 'job2.out' value to compute the compressed string length)
                        _mm512_mask_compressstoreu_epi64(output, loadmask1, job1); output += delta1;
                        _mm512_mask_compressstoreu_epi64(output, loadmask2, job2); output += delta2;


// LICENSE_CHANGE_END

		}
	} else {
		while (input+delta1 < inputEnd) {


// LICENSE_CHANGE_BEGIN
// The following code up to LICENSE_CHANGE_END is subject to THIRD PARTY LICENSE #5
// See the end of this file for a list

// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
//
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, E1PRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst
//
                        // load new jobs in the empty lanes (initially, all lanes are empty, so loadmask1=11111111, delta1=8).
            job1      = _mm512_mask_expandloadu_epi64(job1, loadmask1, input); input += delta1; 
                        // load the next 8 input string bytes (uncompressed data, aka 'symbols').
   __m512i  word1     = _mm512_i64gather_epi64(_mm512_srli_epi64(job1, 46), symbolBase, 1); 
                        // load 16-bits codes from the 2-byte-prefix keyed lookup table. It also store 1-byte codes in all free slots.
                        // code1: Lowest 8 bits contain the code. Eleventh bit is whether it is an escaped code. Next 4 bits is length (2 or 1).
   __m512i  code1     = _mm512_i64gather_epi64(_mm512_and_epi64(word1, all_FFFF), symbolTable.shortCodes, sizeof(u16));
                        // get the first three bytes of the string. 
   __m512i  pos1      = _mm512_mullo_epi64(_mm512_and_epi64(word1, all_FFFFFF), all_PRIME);
                        // hash them into a random number: pos1 = pos1*PRIME; pos1 ^= pos1>>SHIFT
            pos1      = _mm512_slli_epi64(_mm512_and_epi64(_mm512_xor_epi64(pos1,_mm512_srli_epi64(pos1,FSST_SHIFT)), all_HASH), 4);
                        // lookup in the 3-byte-prefix keyed hash table
   __m512i  icl1      = _mm512_i64gather_epi64(pos1, (((char*) symbolTable.hashTab) + 8), 1);
                        // speculatively store the first input byte into the second position of the write1 register (in case it turns out to be an escaped byte).
   __m512i  write1    = _mm512_slli_epi64(_mm512_and_epi64(word1, all_FF), 8);
                        // lookup just like the icl1 above, but loads the next 8 bytes. This fetches the actual string bytes in the hash table.
   __m512i  symb1     = _mm512_i64gather_epi64(pos1, (((char*) symbolTable.hashTab) + 0), 1);
                        // generate the FF..FF mask with an FF for each byte of the symbol (we need to AND the input with this to correctly check equality).
            pos1      = _mm512_srlv_epi64(all_MASK, _mm512_and_epi64(icl1, all_FF));
                        // check symbol < |str| as well as whether it is an occupied slot (cmplt checks both conditions at once) and check string equality (cmpeq).
   __mmask8 match1    = _mm512_cmpeq_epi64_mask(symb1, _mm512_and_epi64(word1, pos1)) & _mm512_cmplt_epi64_mask(icl1, all_ICL_FREE);
                        // for the hits, overwrite the codes with what comes from the hash table (codes for symbols of length >=3). The rest stays with what shortCodes gave.
            code1     = _mm512_mask_mov_epi64(code1, match1, _mm512_srli_epi64(icl1, 16));
                        // write out the code byte as the first output byte. Notice that this byte may also be the escape code 255 (for escapes) coming from shortCodes.
            write1    = _mm512_or_epi64(write1, _mm512_and_epi64(code1, all_FF));
                        // zip the irrelevant 6 bytes (just stay with the 2 relevant bytes containing the 16-bits code)
            code1     = _mm512_and_epi64(code1, all_FFFF);
                        // write out the compressed data. It writes 8 bytes, but only 1 byte is relevant :-(or 2 bytes are, in case of an escape code)
                        _mm512_i64scatter_epi64(codeBase, _mm512_and_epi64(job1, all_M19), write1, 1);
                        // increase the job1.cur field in the job with the symbol length (for this, shift away 12 bits from the code) 
            job1      = _mm512_add_epi64(job1, _mm512_slli_epi64(_mm512_srli_epi64(code1, FSST_LEN_BITS), 46));
                        // increase the job1.out' field with one, or two in case of an escape code (add 1 plus the escape bit, i.e the 8th)
            job1      = _mm512_add_epi64(job1, _mm512_add_epi64(all_ONE, _mm512_and_epi64(_mm512_srli_epi64(code1, 8), all_ONE)));
                        // test which lanes are done now (job1.cur==job1.end), cur starts at bit 46, end starts at bit 28 (the highest 2x18 bits in the job1 register)
            loadmask1 = _mm512_cmpeq_epi64_mask(_mm512_srli_epi64(job1, 46), _mm512_and_epi64(_mm512_srli_epi64(job1, 28), all_M18));
                        // calculate the amount of lanes in job1 that are done
            delta1    = _mm_popcnt_u32((int) loadmask1); 
                        // write out the job state for the lanes that are done (we need the final 'job1.out' value to compute the compressed string length)
                        _mm512_mask_compressstoreu_epi64(output, loadmask1, job1); output += delta1;


// LICENSE_CHANGE_END

		}
	}

	// flush the job states of the unfinished strings at the end of output[]
	processed = n - (inputEnd - input);
	u32 unfinished = 0;
	if (unroll > 1) {
		if (unroll > 2) {
			if (unroll > 3) {
				_mm512_mask_compressstoreu_epi64(output+unfinished, loadmask4=~loadmask4, job4);
				unfinished += _mm_popcnt_u32((int) loadmask4);
			}
			_mm512_mask_compressstoreu_epi64(output+unfinished, loadmask3=~loadmask3, job3);
			unfinished += _mm_popcnt_u32((int) loadmask3);
		}
		_mm512_mask_compressstoreu_epi64(output+unfinished, loadmask2=~loadmask2, job2);
		unfinished += _mm_popcnt_u32((int) loadmask2);
	}
	_mm512_mask_compressstoreu_epi64(output+unfinished, loadmask1=~loadmask1, job1);
#else
	(void) symbolTable;
	(void) codeBase;
	(void) symbolBase;
	(void) input;
	(void) output;
	(void) n;
	(void) unroll;
#endif
	return processed;
}


// LICENSE_CHANGE_END


// LICENSE_CHANGE_BEGIN
// The following code up to LICENSE_CHANGE_END is subject to THIRD PARTY LICENSE #5
// See the end of this file for a list

// this software is distributed under the MIT License (http://www.opensource.org/licenses/MIT):
//
// Copyright 2018-2020, CWI, TU Munich, FSU Jena
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify,
// merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// - The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// You can contact the authors via the FSST source repository : https://github.com/cwida/fsst


Symbol concat(Symbol a, Symbol b) {
	Symbol s;
	u32 length = a.length()+b.length();
	if (length > Symbol::maxLength) length = Symbol::maxLength;
	s.set_code_len(FSST_CODE_MASK, length);
	s.val.num = (b.val.num << (8*a.length())) | a.val.num;
	return s;
}

namespace std {
template <>
class hash<QSymbol> {
public:
	size_t operator()(const QSymbol& q) const {
		uint64_t k = q.symbol.val.num;
		const uint64_t m = 0xc6a4a7935bd1e995;
		const int r = 47;
		uint64_t h = 0x8445d61a4e774912 ^ (8*m);
		k *= m;
		k ^= k >> r;
		k *= m;
		h ^= k;
		h *= m;
		h ^= h >> r;
		h *= m;
		h ^= h >> r;
		return h;
	}
};
}

bool isEscapeCode(u16 pos) { return pos < FSST_CODE_BASE; }

std::ostream& operator<<(std::ostream& out, const Symbol& s) {
	for (u32 i=0; i<s.length(); i++)
		out << s.val.str[i];
	return out;
}
static u64 iter = 0;

SymbolTable *buildSymbolTable(Counters& counters, vector<u8*> line, size_t len[], bool zeroTerminated=false) {
	SymbolTable *st = new SymbolTable(), *bestTable = new SymbolTable();
	int bestGain = (int) -FSST_SAMPLEMAXSZ; // worst case (everything exception)
	size_t sampleFrac = 128;

	// start by determining the terminator. We use the (lowest) most infrequent byte as terminator
	st->zeroTerminated = zeroTerminated;
	if (zeroTerminated) {
		st->terminator = 0; // except in case of zeroTerminated mode, then byte 0 is terminator regardless frequency
	} else {
		u16 byteHisto[256];
		memset(byteHisto, 0, sizeof(byteHisto));
		for(size_t i=0; i<line.size(); i++) {
			u8* cur = line[i];
			u8* end = cur + len[i];
			while(cur < end) byteHisto[*cur++]++;
		}
		u32 minSize = FSST_SAMPLEMAXSZ, i = st->terminator = 256;
		while(i-- > 0) {
			if (byteHisto[i] > minSize) continue;
			st->terminator = i;
			minSize = byteHisto[i];
		}
	}
	assert(st->terminator != 256);

	// a random number between 0 and 128
	auto rnd128 = [&](size_t i) { return 1 + (FSST_HASH((i+1UL)*sampleFrac)&127); };

	// compress sample, and compute (pair-)frequencies
	auto compressCount = [&](SymbolTable *st, Counters &counters) { // returns gain
		int gain = 0;

		for(size_t i=0; i<line.size(); i++) {
			u8* cur = line[i];
			u8* end = cur + len[i];

			if (sampleFrac < 128) {
				// in earlier rounds (sampleFrac < 128) we skip data in the sample (reduces overall work ~2x)
				if (rnd128(i) > sampleFrac) continue;
			}
			if (cur < end) {
				u8* start = cur;
				u16 code2 = 255, code1 = st->findLongestSymbol(cur, end);
				cur += st->symbols[code1].length();
				gain += (int) (st->symbols[code1].length()-(1+isEscapeCode(code1)));
				while (true) {
					// count single symbol (i.e. an option is not extending it)
					counters.count1Inc(code1);

					// as an alternative, consider just using the next byte..
					if (st->symbols[code1].length() != 1) // .. but do not count single byte symbols doubly
						counters.count1Inc(*start);

					if (cur==end) {
						break;
					}

					// now match a new symbol
					start = cur;
					if (cur<end-7) {
						u64 word = fsst_unaligned_load(cur);
						size_t code = word & 0xFFFFFF;
						size_t idx = FSST_HASH(code)&(st->hashTabSize-1);
						Symbol s = st->hashTab[idx];
						code2 = st->shortCodes[word & 0xFFFF] & FSST_CODE_MASK;
						word &= (0xFFFFFFFFFFFFFFFF >> (u8) s.icl);
						if ((s.icl < FSST_ICL_FREE) & (s.val.num == word)) {
							code2 = s.code();
							cur += s.length();
						} else if (code2 >= FSST_CODE_BASE) {
							cur += 2;
						} else {
							code2 = st->byteCodes[word & 0xFF] & FSST_CODE_MASK;
							cur += 1;
						}
					} else {
						code2 = st->findLongestSymbol(cur, end);
						cur += st->symbols[code2].length();
					}

					// compute compressed output size
					gain += ((int) (cur-start))-(1+isEscapeCode(code2));

					// now count the subsequent two symbols we encode as an extension codesibility
					if (sampleFrac < 128) { // no need to count pairs in final round
						                    // consider the symbol that is the concatenation of the two last symbols
						counters.count2Inc(code1, code2);

						// as an alternative, consider just extending with the next byte..
						if ((cur-start) > 1)  // ..but do not count single byte extensions doubly
							counters.count2Inc(code1, *start);
					}
					code1 = code2;
				}
			}
		}
		return gain;
	};

	auto makeTable = [&](SymbolTable *st, Counters &counters) {
		// hashmap of c (needed because we can generate duplicate candidates)
		unordered_set<QSymbol> cands;

		// artificially make terminater the most frequent symbol so it gets included
		u16 terminator = st->nSymbols?FSST_CODE_BASE:st->terminator;
		counters.count1Set(terminator,65535);

		auto addOrInc = [&](unordered_set<QSymbol> &cands, Symbol s, u64 count) {
			if (count < (5*sampleFrac)/128) return; // improves both compression speed (less candidates), but also quality!!
			QSymbol q;
			q.symbol = s;
			q.gain = count * s.length();
			auto it = cands.find(q);
			if (it != cands.end()) {
				q.gain += (*it).gain;
				cands.erase(*it);
			}
			cands.insert(q);
		};

		// add candidate symbols based on counted frequency
		for (u32 pos1=0; pos1<FSST_CODE_BASE+(size_t) st->nSymbols; pos1++) {
			u32 cnt1 = counters.count1GetNext(pos1); // may advance pos1!!
			if (!cnt1) continue;

			// heuristic: promoting single-byte symbols (*8) helps reduce exception rates and increases [de]compression speed
			Symbol s1 = st->symbols[pos1];
			addOrInc(cands, s1, ((s1.length()==1)?8LL:1LL)*cnt1);

			if (sampleFrac >= 128 || // last round we do not create new (combined) symbols
			    s1.length() == Symbol::maxLength || // symbol cannot be extended
			    s1.val.str[0] == st->terminator) { // multi-byte symbols cannot contain the terminator byte
				continue;
			}
			for (u32 pos2=0; pos2<FSST_CODE_BASE+(size_t)st->nSymbols; pos2++) {
				u32 cnt2 = counters.count2GetNext(pos1, pos2); // may advance pos2!!
				if (!cnt2) continue;

				// create a new symbol
				Symbol s2 = st->symbols[pos2];
				Symbol s3 = concat(s1, s2);
				if (s2.val.str[0] != st->terminator) // multi-byte symbols cannot contain the terminator byte
					addOrInc(cands, s3, cnt2);
			}
		}

		// insert candidates into priority queue (by gain)
		auto cmpGn = [](const QSymbol& q1, const QSymbol& q2) { return (q1.gain < q2.gain) || (q1.gain == q2.gain && q1.symbol.val.num > q2.symbol.val.num); };
		priority_queue<QSymbol,vector<QSymbol>,decltype(cmpGn)> pq(cmpGn);
		for (auto& q : cands)
			pq.push(q);

		// Create new symbol map using best candidates
		st->clear();
		while (st->nSymbols < 255 && !pq.empty()) {
			QSymbol q = pq.top();
			pq.pop();
			st->add(q.symbol);
		}
	};

	u8 bestCounters[512*sizeof(u16)];
#ifdef NONOPT_FSST
	for(size_t frac : {127, 127, 127, 127, 127, 127, 127, 127, 127, 128}) {
		sampleFrac = frac;
#else
	for(sampleFrac=8; true; sampleFrac += 30) {
#endif
		memset(&counters, 0, sizeof(Counters));
		long gain = compressCount(st, counters);
		if (gain >= bestGain) { // a new best solution!
			counters.backup1(bestCounters);
			*bestTable = *st; bestGain = gain;
		}
		if (sampleFrac >= 128) break; // we do 5 rounds (sampleFrac=8,38,68,98,128)
		makeTable(st, counters);
	}
	delete st;
	counters.restore1(bestCounters);
	makeTable(bestTable, counters);
	bestTable->finalize(zeroTerminated); // renumber codes for more efficient compression
	return bestTable;
}

static inline size_t compressSIMD(SymbolTable &symbolTable, u8* symbolBase, size_t nlines, size_t len[], u8* line[], size_t size, u8* dst, size_t lenOut[], u8* strOut[], int unroll) {
	size_t curLine = 0, inOff = 0, outOff = 0, batchPos = 0, empty = 0, budget = size;
	u8 *lim = dst + size, *codeBase = symbolBase + (1<<18); // 512KB temp space for compressing 512 strings
	SIMDjob input[512];  // combined offsets of input strings (cur,end), and string #id (pos) and output (dst) pointer
	SIMDjob output[512]; // output are (pos:9,dst:19) end pointers (compute compressed length from this)
	size_t jobLine[512]; // for which line in the input sequence was this job (needed because we may split a line into multiple jobs)

	while (curLine < nlines && outOff <= (1<<19)) {
		size_t prevLine = curLine, chunk, curOff = 0;

		// bail out if the output buffer cannot hold the compressed next string fully
		if (((len[curLine]-curOff)*2 + 7) > budget) break; // see below for the +7
		else budget -= (len[curLine]-curOff)*2;

		strOut[curLine] = (u8*) 0;
		lenOut[curLine] = 0;

		do {
			do {
				chunk = len[curLine] - curOff;
				if (chunk > 511) {
					chunk = 511; // large strings need to be chopped up into segments of 511 bytes
				}
				// create a job in this batch
				SIMDjob job;
				job.cur = inOff;
				job.end = job.cur + chunk;
				job.pos = batchPos;
				job.out = outOff;

				// worst case estimate for compressed size (+7 is for the scatter that writes extra 7 zeros)
				outOff += 7 + 2*(size_t)(job.end - job.cur); // note, total size needed is 512*(511*2+7) bytes.
				if (outOff > (1<<19)) break; // simdbuf may get full, stop before this chunk

				// register job in this batch
				input[batchPos] = job;
				jobLine[batchPos] = curLine;

				if (chunk == 0) {
					empty++; // detect empty chunks -- SIMD code cannot handle empty strings, so they need to be filtered out
				} else {
					// copy string chunk into temp buffer
					memcpy(symbolBase + inOff, line[curLine] + curOff, chunk);
					inOff += chunk;
					curOff += chunk;
					symbolBase[inOff++] = (u8) symbolTable.terminator; // write an extra char at the end that will not be encoded
				}
				if (++batchPos == 512) break;
			} while(curOff < len[curLine]);

			if ((batchPos == 512) || (outOff > (1<<19)) || (++curLine >= nlines)) { // cannot accumulate more?
				if (batchPos-empty >= 32) { // if we have enough work, fire off fsst_compressAVX512 (32 is due to max 4x8 unrolling)
					// radix-sort jobs on length (longest string first)
					// -- this provides best load balancing and allows to skip empty jobs at the end
					u16 sortpos[513];
					memset(sortpos, 0, sizeof(sortpos));

					// calculate length histo
					for(size_t i=0; i<batchPos; i++) {
						size_t len = input[i].end - input[i].cur;
						sortpos[512UL - len]++;
					}
					// calculate running sum
					for(size_t i=1; i<=512; i++)
						sortpos[i] += sortpos[i-1];

					// move jobs to their final destination
					SIMDjob inputOrdered[512];
					for(size_t i=0; i<batchPos; i++) {
						size_t len = input[i].end - input[i].cur;
						size_t pos = sortpos[511UL - len]++;
						inputOrdered[pos] = input[i];
					}
					// finally.. SIMD compress max 256KB of simdbuf into (max) 512KB of simdbuf (but presumably much less..)
					for(size_t done = duckdb_fsst_compressAVX512(symbolTable, codeBase, symbolBase, inputOrdered, output, batchPos-empty, unroll);
					     done < batchPos; done++) output[done] = inputOrdered[done];
				} else {
					memcpy(output, input, batchPos*sizeof(SIMDjob));
				}

				// finish encoding (unfinished strings in process, plus the few last strings not yet processed)
				for(size_t i=0; i<batchPos; i++) {
					SIMDjob job = output[i];
					if (job.cur < job.end) { // finish encoding this string with scalar code
						u8* cur = symbolBase + job.cur;
						u8* end = symbolBase + job.end;
						u8* out = codeBase + job.out;
						while (cur < end) {
							u64 word = fsst_unaligned_load(cur);
							size_t code = symbolTable.shortCodes[word & 0xFFFF];
							size_t pos = word & 0xFFFFFF;
							size_t idx = FSST_HASH(pos)&(symbolTable.hashTabSize-1);
							Symbol s = symbolTable.hashTab[idx];
							out[1] = (u8) word; // speculatively write out escaped byte
							word &= (0xFFFFFFFFFFFFFFFF >> (u8) s.icl);
							if ((s.icl < FSST_ICL_FREE) && s.val.num == word) {
								*out++ = (u8) s.code(); cur += s.length();
							} else {
								// could be a 2-byte or 1-byte code, or miss
								// handle everything with predication
								*out = (u8) code;
								out += 1+((code&FSST_CODE_BASE)>>8);
								cur += (code>>FSST_LEN_BITS);
							}
						}
						job.out = out - codeBase;
					}
					// postprocess job info
					job.cur = 0;
					job.end = job.out - input[job.pos].out; // misuse .end field as compressed size
					job.out = input[job.pos].out; // reset offset to start of encoded string
					input[job.pos] = job;
				}

				// copy out the result data
				for(size_t i=0; i<batchPos; i++) {
					size_t lineNr = jobLine[i]; // the sort must be order-preserving, as we concatenate results string in order
					size_t sz = input[i].end; // had stored compressed lengths here
					if (!strOut[lineNr]) strOut[lineNr] = dst; // first segment will be the strOut pointer
					lenOut[lineNr] += sz; // add segment (lenOut starts at 0 for this reason)
					memcpy(dst, codeBase+input[i].out, sz);
					dst += sz;
				}

				// go for the next batch of 512 chunks
				inOff = outOff = batchPos = empty = 0;
				budget = (size_t) (lim - dst);
			}
		} while (curLine == prevLine && outOff <= (1<<19));
	}
	return curLine;
}


// optimized adaptive *scalar* compression method
static inline size_t compressBulk(SymbolTable &symbolTable, size_t nlines, size_t lenIn[], u8* strIn[], size_t size, u8* out, size_t lenOut[], u8* strOut[], bool noSuffixOpt, bool avoidBranch) {
	u8 *cur = NULL, *end =  NULL, *lim = out + size;
	size_t curLine, suffixLim = symbolTable.suffixLim;
	u8 byteLim = symbolTable.nSymbols + symbolTable.zeroTerminated - symbolTable.lenHisto[0];

	u8 buf[512+7]; /* +7 sentinel is to avoid 8-byte unaligned-loads going beyond 511 out-of-bounds */
	memset(buf+511, 0, 8); /* and initialize the sentinal bytes */

	// three variants are possible. dead code falls away since the bool arguments are constants
	auto compressVariant = [&](bool noSuffixOpt, bool avoidBranch) {
		while (cur < end) {
			u64 word = fsst_unaligned_load(cur);
			size_t code = symbolTable.shortCodes[word & 0xFFFF];
			if (noSuffixOpt && ((u8) code) < suffixLim) {
				// 2 byte code without having to worry about longer matches
				*out++ = (u8) code; cur += 2;
			} else {
				size_t pos = word & 0xFFFFFF;
				size_t idx = FSST_HASH(pos)&(symbolTable.hashTabSize-1);
				Symbol s = symbolTable.hashTab[idx];
				out[1] = (u8) word; // speculatively write out escaped byte
				word &= (0xFFFFFFFFFFFFFFFF >> (u8) s.icl);
				if ((s.icl < FSST_ICL_FREE) && s.val.num == word) {
					*out++ = (u8) s.code(); cur += s.length();
				} else if (avoidBranch) {
					// could be a 2-byte or 1-byte code, or miss
					// handle everything with predication
					*out = (u8) code;
					out += 1+((code&FSST_CODE_BASE)>>8);
					cur += (code>>FSST_LEN_BITS);
				} else if ((u8) code < byteLim) {
					// 2 byte code after checking there is no longer pattern
					*out++ = (u8) code; cur += 2;
				} else {
					// 1 byte code or miss.
					*out = (u8) code;
					out += 1+((code&FSST_CODE_BASE)>>8); // predicated - tested with a branch, that was always worse
					cur++;
				}
			}
		}
	};

	for(curLine=0; curLine<nlines; curLine++) {
		size_t chunk, curOff = 0;
		strOut[curLine] = out;
		do {
			cur = strIn[curLine] + curOff;
			chunk = lenIn[curLine] - curOff;
			if (chunk > 511) {
				chunk = 511; // we need to compress in chunks of 511 in order to be byte-compatible with simd-compressed FSST
			}
			if ((2*chunk+7) > (size_t) (lim-out)) {
				return curLine; // out of memory
			}
			// copy the string to the 511-byte buffer
			memcpy(buf, cur, chunk);
			buf[chunk] = (u8) symbolTable.terminator;
			cur = buf;
			end = cur + chunk;

			// based on symboltable stats, choose a variant that is nice to the branch predictor
			if (noSuffixOpt) {
				compressVariant(true,false);
			} else if (avoidBranch) {
				compressVariant(false,true);
			} else {
				compressVariant(false, false);
			}
		} while((curOff += chunk) < lenIn[curLine]);
		lenOut[curLine] = (size_t) (out - strOut[curLine]);
	}
	return curLine;
}

#define FSST_SAMPLELINE ((size_t) 512)

// quickly select a uniformly random set of lines such that we have between [FSST_SAMPLETARGET,FSST_SAMPLEMAXSZ) string bytes
vector<u8*> makeSample(u8* sampleBuf, u8* strIn[], size_t **lenRef, size_t nlines) {
	size_t totSize = 0, *lenIn = *lenRef;
	vector<u8*> sample;

	for(size_t i=0; i<nlines; i++)
		totSize += lenIn[i];

	if (totSize < FSST_SAMPLETARGET) {
		for(size_t i=0; i<nlines; i++)
			sample.push_back(strIn[i]);
	} else {
		size_t sampleRnd = FSST_HASH(4637947);
		u8* sampleLim = sampleBuf + FSST_SAMPLETARGET;
		size_t *sampleLen = *lenRef = new size_t[nlines + FSST_SAMPLEMAXSZ/FSST_SAMPLELINE];

		while(sampleBuf < sampleLim) {
			// choose a non-empty line
			sampleRnd = FSST_HASH(sampleRnd);
			size_t linenr = sampleRnd % nlines;
			while (lenIn[linenr] == 0)
				if (++linenr == nlines) linenr = 0;

			// choose a chunk
			size_t chunks = 1 + ((lenIn[linenr]-1) / FSST_SAMPLELINE);
			sampleRnd = FSST_HASH(sampleRnd);
			size_t chunk = FSST_SAMPLELINE*(sampleRnd % chunks);

			// add the chunk to the sample
			size_t len = min(lenIn[linenr]-chunk,FSST_SAMPLELINE);
			memcpy(sampleBuf, strIn[linenr]+chunk, len);
			sample.push_back(sampleBuf);
			sampleBuf += *sampleLen++ = len;
		}
	}
	return sample;
}

extern "C" duckdb_fsst_encoder_t* duckdb_fsst_create(size_t n, size_t lenIn[], u8 *strIn[], int zeroTerminated) {
	u8* sampleBuf = new u8[FSST_SAMPLEMAXSZ];
	size_t *sampleLen = lenIn;
	vector<u8*> sample = makeSample(sampleBuf, strIn, &sampleLen, n?n:1); // careful handling of input to get a right-size and representative sample
	Encoder *encoder = new Encoder();
	encoder->symbolTable = shared_ptr<SymbolTable>(buildSymbolTable(encoder->counters, sample, sampleLen, zeroTerminated));
	if (sampleLen != lenIn) delete[] sampleLen;
	delete[] sampleBuf;
	return (duckdb_fsst_encoder_t*) encoder;
}

/* create another encoder instance, necessary to do multi-threaded encoding using the same symbol table */
extern "C" duckdb_fsst_encoder_t* duckdb_fsst_duplicate(duckdb_fsst_encoder_t *encoder) {
	Encoder *e = new Encoder();
	e->symbolTable = ((Encoder*)encoder)->symbolTable; // it is a shared_ptr
	return (duckdb_fsst_encoder_t*) e;
}

// export a symbol table in compact format.
extern "C" u32 duckdb_fsst_export(duckdb_fsst_encoder_t *encoder, u8 *buf) {
	Encoder *e = (Encoder*) encoder;
	// In ->version there is a versionnr, but we hide also suffixLim/terminator/nSymbols there.
	// This is sufficient in principle to *reconstruct* a duckdb_fsst_encoder_t from a duckdb_fsst_decoder_t
	// (such functionality could be useful to append compressed data to an existing block).
	//
	// However, the hash function in the encoder hash table is endian-sensitive, and given its
	// 'lossy perfect' hashing scheme is *unable* to contain other-endian-produced symbol tables.
	// Doing a endian-conversion during hashing will be slow and self-defeating.
	//
	// Overall, we could support reconstructing an encoder for incremental compression, but
	// should enforce equal-endianness. Bit of a bummer. Not going there now.
	//
	// The version field is now there just for future-proofness, but not used yet

	// version allows keeping track of fsst versions, track endianness, and encoder reconstruction
	u64 version = (FSST_VERSION << 32) |  // version is 24 bits, most significant byte is 0
	              (((u64) e->symbolTable->suffixLim) << 24) |
	              (((u64) e->symbolTable->terminator) << 16) |
	              (((u64) e->symbolTable->nSymbols) << 8) |
	              FSST_ENDIAN_MARKER; // least significant byte is nonzero

	/* do not assume unaligned reads here */
	memcpy(buf, &version, 8);
	buf[8] = e->symbolTable->zeroTerminated;
	for(u32 i=0; i<8; i++)
		buf[9+i] = (u8) e->symbolTable->lenHisto[i];
	u32 pos = 17;

	// emit only the used bytes of the symbols
	for(u32 i = e->symbolTable->zeroTerminated; i < e->symbolTable->nSymbols; i++)
		for(u32 j = 0; j < e->symbolTable->symbols[i].length(); j++)
			buf[pos++] = e->symbolTable->symbols[i].val.str[j]; // serialize used symbol bytes

	return pos; // length of what was serialized
}

#define FSST_CORRUPT 32774747032022883 /* 7-byte number in little endian containing "corrupt" */

extern "C" u32 duckdb_fsst_import(duckdb_fsst_decoder_t *decoder, u8 *buf) {
	u64 version = 0;
	u32 code, pos = 17;
	u8 lenHisto[8];

	// version field (first 8 bytes) is now there just for future-proofness, unused still (skipped)
	memcpy(&version, buf, 8);
	if ((version>>32) != FSST_VERSION) return 0;
	decoder->zeroTerminated = buf[8]&1;
	memcpy(lenHisto, buf+9, 8);

	// in case of zero-terminated, first symbol is "" (zero always, may be overwritten)
	decoder->len[0] = 1;
	decoder->symbol[0] = 0;

	// we use lenHisto[0] as 1-byte symbol run length (at the end)
	code = decoder->zeroTerminated;
	if (decoder->zeroTerminated) lenHisto[0]--; // if zeroTerminated, then symbol "" aka 1-byte code=0, is not stored at the end

	// now get all symbols from the buffer
	for(u32 l=1; l<=8; l++) { /* l = 1,2,3,4,5,6,7,8 */
		for(u32 i=0; i < lenHisto[(l&7) /* 1,2,3,4,5,6,7,0 */]; i++, code++)  {
			decoder->len[code] = (l&7)+1; /* len = 2,3,4,5,6,7,8,1  */
			decoder->symbol[code] = 0;
			for(u32 j=0; j<decoder->len[code]; j++)
				((u8*) &decoder->symbol[code])[j] = buf[pos++]; // note this enforces 'little endian' symbols
		}
	}
	if (decoder->zeroTerminated) lenHisto[0]++;

	// fill unused symbols with text "corrupt". Gives a chance to detect corrupted code sequences (if there are unused symbols).
	while(code<255) {
		decoder->symbol[code] = FSST_CORRUPT;
		decoder->len[code++] = 8;
	}
	return pos;
}

// runtime check for simd
inline size_t _compressImpl(Encoder *e, size_t nlines, size_t lenIn[], u8 *strIn[], size_t size, u8 *output, size_t *lenOut, u8 *strOut[], bool noSuffixOpt, bool avoidBranch, int simd) {
#ifndef NONOPT_FSST
	if (simd && duckdb_fsst_hasAVX512())
		return compressSIMD(*e->symbolTable, e->simdbuf, nlines, lenIn, strIn, size, output, lenOut, strOut, simd);
#endif
	(void) simd;
	return compressBulk(*e->symbolTable, nlines, lenIn, strIn, size, output, lenOut, strOut, noSuffixOpt, avoidBranch);
}
size_t compressImpl(Encoder *e, size_t nlines, size_t lenIn[], u8 *strIn[], size_t size, u8 *output, size_t *lenOut, u8 *strOut[], bool noSuffixOpt, bool avoidBranch, int simd) {
	return _compressImpl(e, nlines, lenIn, strIn, size, output, lenOut, strOut, noSuffixOpt, avoidBranch, simd);
}

// adaptive choosing of scalar compression method based on symbol length histogram
inline size_t _compressAuto(Encoder *e, size_t nlines, size_t lenIn[], u8 *strIn[], size_t size, u8 *output, size_t *lenOut, u8 *strOut[], int simd) {
	bool avoidBranch = false, noSuffixOpt = false;
	if (100*e->symbolTable->lenHisto[1] > 65*e->symbolTable->nSymbols && 100*e->symbolTable->suffixLim > 95*e->symbolTable->lenHisto[1]) {
		noSuffixOpt = true;
	} else if ((e->symbolTable->lenHisto[0] > 24 && e->symbolTable->lenHisto[0] < 92) &&
	           (e->symbolTable->lenHisto[0] < 43 || e->symbolTable->lenHisto[6] + e->symbolTable->lenHisto[7] < 29) &&
	           (e->symbolTable->lenHisto[0] < 72 || e->symbolTable->lenHisto[2] < 72)) {
		avoidBranch = true;
	}
	return _compressImpl(e, nlines, lenIn, strIn, size, output, lenOut, strOut, noSuffixOpt, avoidBranch, simd);
}
size_t compressAuto(Encoder *e, size_t nlines, size_t lenIn[], u8 *strIn[], size_t size, u8 *output, size_t *lenOut, u8 *strOut[], int simd) {
	return _compressAuto(e, nlines, lenIn, strIn, size, output, lenOut, strOut, simd);
}

// the main compression function (everything automatic)
extern "C" size_t duckdb_fsst_compress(duckdb_fsst_encoder_t *encoder, size_t nlines, size_t lenIn[], u8 *strIn[], size_t size, u8 *output, size_t *lenOut, u8 *strOut[]) {
	// to be faster than scalar, simd needs 64 lines or more of length >=12; or fewer lines, but big ones (totLen > 32KB)
	size_t totLen = accumulate(lenIn, lenIn+nlines, 0);
	int simd = totLen > nlines*12 && (nlines > 64 || totLen > (size_t) 1<<15);
	return _compressAuto((Encoder*) encoder, nlines, lenIn, strIn, size, output, lenOut, strOut, 3*simd);
}

/* deallocate encoder */
extern "C" void duckdb_fsst_destroy(duckdb_fsst_encoder_t* encoder) {
	Encoder *e = (Encoder*) encoder;
	delete e;
}

/* very lazy implementation relying on export and import */
extern "C" duckdb_fsst_decoder_t duckdb_fsst_decoder(duckdb_fsst_encoder_t *encoder) {
	u8 buf[sizeof(duckdb_fsst_decoder_t)];
	u32 cnt1 = duckdb_fsst_export(encoder, buf);
	duckdb_fsst_decoder_t decoder;
	u32 cnt2 = duckdb_fsst_import(&decoder, buf);
	assert(cnt1 == cnt2); (void) cnt1; (void) cnt2;
	return decoder;
}

// LICENSE_CHANGE_END
