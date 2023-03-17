#include "CompressionQpl.h"

#include <mutex>
#include <atomic>
#include <sys/time.h>

#include "velox/common/base/Exceptions.h"

#define job_size 100
#define MAX_COMPRESS_CHUNK_SIZE  262144
#define MAX_DECOMPRESS_CHUNK_SIZE  262144
#define QPL_MAX_TRANS_SIZE 2097152


static qpl_job *job_pool[job_size];
std::atomic<bool> job_status[job_size];
static bool initialized_job =false;
static int start=0;
std::mutex mtx;


bool Initjobs(qpl_path_t execution_path){

  mtx.lock();
  if(initialized_job){
    mtx.unlock();
    return true;
  }else{
    qpl_status status;
    uint32_t size;
    status = qpl_get_job_size(execution_path, &size);
    if (status != QPL_STS_OK) {
        VELOX_FAIL("QPL::An error acquired during job size getting.");
    }
    for(int i=0;i<job_size;i++){

      job_status[i]=false;
      uint8_t *job_buffer=new uint8_t[size];
      job_pool[i] = reinterpret_cast<qpl_job *>(job_buffer);
      status = qpl_init_job(execution_path, job_pool[i]);

      if (status != QPL_STS_OK) {
        if(execution_path==qpl_path_software){
          VELOX_FAIL("QPL::An error acquired during compression job initializing.");
          return false;
        }

        execution_path=qpl_path_software;
        status = qpl_get_job_size(execution_path, &size);

        if (status != QPL_STS_OK) {
          VELOX_FAIL("QPL::An error acquired during job size getting.");
        }
        i--;
        delete[] job_buffer;
      }
    }
    mtx.unlock();
    initialized_job=true;
    return true;
    
  }
}

static inline size_t Getindex()
{
    size_t tsc = 0;
    unsigned lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi) : :);
    tsc = ((((uint64_t)hi) << 32) | (uint64_t)lo);
    return ((size_t)((tsc * 44485709377909ULL) >> 4)) % job_size;
}

int Qplcodec::Getjob(){
  if(!initialized_job){
    Initjobs(execute_path_);
  }
  size_t index = Getindex();
  bool expected  = false;
  while(job_status[index].compare_exchange_strong(expected,true)==false){
    expected  = false;
    index = Getindex();
  }

  job_=job_pool[index];
  return index;
}

bool Qplcodec::Compress(int64_t input_length, const uint8_t* input,
                             int64_t output_buffer_length, uint8_t* output){
    int job_id=Getjob();
    int64_t out_size=output_buffer_length;
    job_->total_out=0;
    job_->total_in=0;

    job_->op=qpl_op_compress;
    job_->level=(qpl_compression_levels)compression_level_;
    job_->next_out_ptr=output;
    job_->next_in_ptr =const_cast<uint8_t*>(input);
    job_->flags = QPL_FLAG_FIRST | QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_OMIT_VERIFY;

    int64_t   current_chunk_size = MAX_COMPRESS_CHUNK_SIZE;
    uint32_t iteration_count    = 0;
    int64_t   source_bytes_left  = input_length;

    while (source_bytes_left > 0) {
      if (current_chunk_size >= source_bytes_left) {
          job_->flags |= QPL_FLAG_LAST;
          current_chunk_size = source_bytes_left;
      }

      job_->available_in = current_chunk_size;
      job_->available_out = QPL_MAX_TRANS_SIZE < out_size - job_->total_out ? QPL_MAX_TRANS_SIZE : out_size - job_->total_out;

      qpl_status status = qpl_execute_job(job_);

      if (status != QPL_STS_OK) {
        std::atomic_store(&job_status[job_id],false);
        VELOX_FAIL("QPL::Error while QPL compression occurred.");
        return false;
      }

      source_bytes_left=input_length-job_->total_in;
      job_->flags &= ~QPL_FLAG_FIRST;
      iteration_count++;
  }
  std::atomic_store(&job_status[job_id],false);
  return true;
}

bool Qplcodec::Decompress(int64_t input_length, const uint8_t* input,
                             int64_t output_buffer_length, uint8_t* output)  {
  if (output_buffer_length == 0) {

    return false;
  }

  int64_t out_size=output_buffer_length;
  int job_id=Getjob();
  job_->total_out=0;
  job_->total_in=0;
  job_->op=qpl_op_decompress;
  job_->next_out_ptr  = output;
  job_->next_in_ptr   = const_cast<uint8_t*>(input);
  job_->flags         = QPL_FLAG_FIRST;

  int64_t   current_chunk_size = MAX_DECOMPRESS_CHUNK_SIZE;
  uint32_t iteration_count    = 0;
  int64_t   source_bytes_left  = input_length;

  while (source_bytes_left > 0) {
    if (current_chunk_size >= source_bytes_left) {
        job_->flags |= QPL_FLAG_LAST;
        current_chunk_size = source_bytes_left;
    }
    source_bytes_left -= current_chunk_size;
    job_->available_in = current_chunk_size;
    job_->available_out = QPL_MAX_TRANS_SIZE < out_size - job_->total_out ? QPL_MAX_TRANS_SIZE : out_size - job_->total_out;

    
    qpl_status status = qpl_execute_job(job_);
    if (status != QPL_STS_OK ) 
    {   
        std::atomic_store(&job_status[job_id],false);
        VELOX_FAIL("QPL::Error while decompression occurred.");
        return false;
    }
    source_bytes_left=input_length-job_->total_in;
    job_->flags &= ~QPL_FLAG_FIRST;
    iteration_count++;
  }
  std::atomic_store(&job_status[job_id],false);
  return true;
}