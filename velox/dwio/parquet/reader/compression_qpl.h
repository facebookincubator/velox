#include <iostream>
#include <memory>

#ifdef VELOX_ENABLE_QPL  
#include <qpl/qpl.h>

namespace facebook::velox::parquet {
bool Initjobs(qpl_path_t execute_path);
class Qplcodec{
public:

  Qplcodec(qpl_path_t execute_path,qpl_compression_levels compression_level): 
    execute_path_(execute_path),compression_level_(compression_level),job_(NULL),idx_(0){};
  ~Qplcodec(){
    //Freejob();
  }
  int Getjob();
  //bool Freejob();
  bool Decompress(int64_t input_length, const uint8_t* input,
                             int64_t output_buffer_length, uint8_t* output);
  uint32_t DecompressAsync(int64_t input_length, const uint8_t* input,
                             int64_t output_buffer_length, uint8_t* output);                             
  bool Compress(int64_t input_length, const uint8_t* input,
                             int64_t output_buffer_length, uint8_t* output);                        

private:
  qpl_path_t execute_path_;
  qpl_compression_levels compression_level_;
  qpl_job *job_;
  int idx_;
  
};
}
#endif