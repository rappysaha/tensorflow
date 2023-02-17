#ifndef FC_DRIVER
#define FC_DRIVER

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <strstream>
#include <typeinfo>

#ifdef ACC_NEON
#include "arm_neon.h"
#endif

#include "acc_container.h"
#include "tensorflow/lite/delegates/utils/secda_tflite/threading_utils/utils.h"
#include "tensorflow/lite/delegates/utils/bert_sim_delegate/bert_sim_debug.h"

// FC_Driver for simulated FC-GEMM acccelerator
namespace tflite_bertsim {

/**
 * @brief Create a Weight Load object
 *
 * @param insn instruction array
 * @param idx instruction array index
 * @param wgt_start weight matrix incrementor
 * @param depth weight matrix depth after padding
 * @param m_inc
 */

void createWeightLoad(uint64_t* insn, int& idx, int wgt_start, int depth,
                      int m_inc) {
  int doffset = wgt_start * (depth / 8);
  int dstride = (depth / 8);
  int x_size = (depth / 8);
  int y_size = m_inc;

#ifdef DEBUG_FCDRIVER
  mkdir("aData/Debug", 0777);
  myfile.open("aData/Debug/DEBUG_FCDRIVER.csv",
              ios::app); 

  myfile << "createWeightLoad():" << endl;
  myfile << "doffset= " << doffset << endl;
  myfile << "dstride= " << dstride << endl;
  myfile << "x_size= " << x_size << endl;
  myfile << "y_size= " << y_size << endl;

  myfile << endl;
  myfile.close();
#endif

  uint64_t p1 = 0;
  uint64_t p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 1;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createInputLoad(uint64_t* insn, int& idx, int inp_start, int depth,
                     int n_inc) {
  int doffset = inp_start * (depth / 8);
  int dstride = (depth / 8);
  int x_size = (depth / 8);
  int y_size = n_inc;

#ifdef DEBUG_FCDRIVER
  mkdir("aData/Debug", 0777);
  myfile.open("aData/Debug/DEBUG_FCDRIVER.csv",
              ios::app); 

  myfile << "createInputLoad():" << endl;
  myfile << "doffset= " << doffset << endl;
  myfile << "dstride= " << dstride << endl;
  myfile << "x_size= " << x_size << endl;
  myfile << "y_size= " << y_size << endl;

  myfile << endl;
  myfile.close();
#endif

  uint64_t p1 = 0;
  uint64_t p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 2;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createBiasLoad(uint64_t* insn, int& idx, int bias_start, int stride,
                    int n_inc, int m_inc) {
  int doffset = bias_start / 2;
  int dstride = stride / 2;
  int x_size = n_inc / 2;
  int y_size = m_inc;

#ifdef DEBUG_FCDRIVER
  mkdir("aData/Debug", 0777);
  myfile.open("aData/Debug/DEBUG_FCDRIVER.csv",
              ios::app); 

  myfile << "createBiasLoad():" << endl;
  myfile << "doffset= " << doffset << endl;
  myfile << "dstride= " << dstride << endl;
  myfile << "x_size= " << x_size << endl;
  myfile << "y_size= " << y_size << endl;

  myfile << endl;
  myfile.close();
#endif

  uint64_t p1 = 0;
  uint64_t p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 3;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void createCompute(uint64_t* insn, int& idx, int out_start, int stride,
                   int inp_block, int wgt_block) {
  int doffset = out_start / 4;
  int dstride = stride / 4;
  int x_size = wgt_block;
  int y_size = inp_block;

#ifdef DEBUG_FCDRIVER
  mkdir("aData/Debug", 0777);
  myfile.open("aData/Debug/DEBUG_FCDRIVER.csv",
              ios::app); 

  myfile << "createCompute():" << endl;
  myfile << "doffset= " << doffset << endl;
  myfile << "dstride= " << dstride << endl;
  myfile << "x_size= " << x_size << endl;
  myfile << "y_size= " << y_size << endl;

  myfile << endl;
  myfile.close();
#endif

  uint64_t p1 = 0;
  uint64_t p2 = 0;
  p1 = dstride;
  p1 = p1 << 16;
  p1 += x_size;
  p1 = p1 << 16;
  p1 += y_size;
  p2 = doffset;
  p2 = p2 << 32;
  p2 += 0;
  insn[idx++] = p2;
  insn[idx++] = p1;
}

void BlockFC(acc_container& drv) {
  // this size concerns with max input size in bytes at any given moment
  // from a tensor to the dram memory that can be accessed by acclerator
  // inp_max= 131072, i.e. maximum input resolution can be 224*198*3 = 133056
  int inp_max = INP_SIZE;  // some size of input dram memory
  int wgt_max = WGT_SIZE;  // some size of weight memory
  int acc_max = ACC_SIZE;

#ifdef DEBUG_FCDRIVER
  mkdir("aData/Debug", 0777);
  myfile.open("aData/Debug/DEBUG_FCDRIVER.csv",
              ios::app);  // app indicates append the text at the end of file
  myfile << "Debugging BlockFC():" << endl;

  myfile << "inp_max= " << inp_max << endl;
  myfile << "wgt_max= " << wgt_max << endl;
  myfile << "acc_max= " << acc_max << endl;
#endif

  int k_inc = drv.pK;
  int m_inc = min((wgt_max), drv.pM);  // is this min() function necessary?
                                       // how it is possible to round of single
                                       // dimention can be greater than maximum no 
                                       // of element?
  int n_inc = min((inp_max), drv.pN);

#ifdef DEBUG_FCDRIVER
  myfile << "k_inc,m_inc,n_inc is making input and weight matrix dimension "
            "divisible by 16"
         << endl;
  myfile << "k_inc= " << k_inc << endl;
  myfile << "m_inc= " << m_inc << endl;
  myfile << "n_inc= " << n_inc << endl;
#endif

// inp_max= 131072, is it indicating byte value or no of elements?
// n_inc * k_inc indicates no of elements of the input matrix
// if something is bigger than (n_inc * k_inc) no of elements
// are we discarding those data by the inc of 16???

  while ((n_inc * k_inc > inp_max) && n_inc != 16) n_inc -= 16;
  while ((m_inc * k_inc > wgt_max) && m_inc != 16) m_inc -= 16;
  while ((n_inc * m_inc > acc_max) && n_inc != 16) n_inc -= 16;
  while ((n_inc * m_inc > acc_max) && m_inc != 16) m_inc -= 16;

#ifdef DEBUG_FCDRIVER
  myfile << "after while condition m_inc= " << m_inc << endl;
  myfile << "after while condition n_inc= " << n_inc << endl;
#endif

  drv.scs->sig_start_acc = ++drv.start_count;
  drv.scs->sig_crf = drv.crf;
  drv.scs->sig_crx = drv.crx;
  drv.scs->sig_ra = drv.ra;

  int32_t* wt_sum = drv.wt_sum;
  int32_t* in_sum = drv.in_sum;
  int32_t* bias_buf = new int32_t[drv.pN * drv.pM];

#ifdef DEBUG_FCDRIVER
  myfile << "!!!!!! send drv data that received from tensor to the drv scs "
            "i.e. to the accelerator !!!!!"
         << endl;
  myfile << "drv.start_count= " << (int)drv.start_count << endl;
  myfile << "Entering create_2d_biases()" << endl;
  myfile.close();
#endif

  create_2d_biases(0, drv.pN, 0, drv.pM, bias_buf, drv.bias, wt_sum, in_sum,
                   drv.rhs_offset, drv.lhs_offset, drv.K);

  unsigned int insn_count = 0;


/**
 * Counting how many instructions do we need
 *
 */
#ifdef DEBUG_FCDRIVER
  mkdir("aData/Debug", 0777);
  myfile.open("aData/Debug/DEBUG_FCDRIVER.csv", ios::app);
#endif

  for (int k = 0; k < drv.pK; k += k_inc) {  // Common Dim - inMat Col No == wgtMat Col No
    int k_b = min(k_inc, drv.pK - k); // redundant
#ifdef DEBUG_FCDRIVER
    myfile << "k_b = " << (int)k_b << endl; 
#endif
    for (int m = 0; m < drv.pM; m += m_inc) {  // Weight Dim
      int m_b = min(m_inc, drv.pM - m); // redundant
#ifdef DEBUG_FCDRIVER
      myfile << "m_b = " << (int)m_b << endl; 
#endif
      insn_count += 2;
      for (int n = 0; n < drv.pN; n += n_inc) {  // Input Dim
        int n_b = min(n_inc, drv.pN - n); // redundant
#ifdef DEBUG_FCDRIVER
        myfile << "n_b = " << (int)n_b << endl; 
#endif
        insn_count += 6;
      }
    }
  }
#ifdef DEBUG_FCDRIVER
  myfile << "No of instruction = " << (int)insn_count << endl << endl; 
  myfile.close();
#endif

  // int inDimCount =0;
  int insn_idx = 0;
  uint64_t* insn =
      static_cast<uint64_t*>(malloc(sizeof(uint64_t) * insn_count));

  for (int k = 0; k < drv.pK; k += k_inc) {  // Common Dim
    int k_b = min(k_inc, drv.pK - k);
    // cout<< "k_b= " << k_b << endl;
    for (int m = 0; m < drv.pM; m += m_inc) {  // Weight Dim
      int m_b = min(m_inc, drv.pM - m);
      // cout<< "m_b= " << m_b << endl;
      // Load Weight
      createWeightLoad(insn, insn_idx, m, drv.pK, m_b);
      for (int n = 0; n < drv.pN; n += n_inc) {  // Input Dim
        int n_b = min(n_inc, drv.pN - n);
        // cout<< "n_b= " << n_b << endl;
        // inDimCount++;
        // cout<< "inDimCount= " << inDimCount << endl;
        createInputLoad(insn, insn_idx, n, drv.pK, n_b);
        createBiasLoad(insn, insn_idx, drv.pN * m + n, drv.pN, n_b, m_b);
        createCompute(insn, insn_idx, drv.pM * n + m, drv.pM, n_b, m_b);
      }
    }
  }

  // Setting acc control signals
  drv.scs->sig_insn_count = insn_idx / 2; // rpp- dividing by 2 because we have 4-types of instruction to create
                                          // and each type create 2 member
  drv.scs->sig_insn_addr = 0; 
  drv.scs->sig_depth = drv.pK;
  drv.scs->sig_crf = drv.crf;
  drv.scs->sig_crx = drv.crx;
  drv.scs->sig_ra = drv.ra;

  // Move Input data to MMapped DMA buffer to enable accelerator access
  unsigned long long* insn_set = (unsigned long long*)insn;
  drv.scs->insn_mem.burst_write(0, insn_idx, insn_set);
  drv.scs->inp_mem.burst_write(0, drv.pN * drv.pK / 8,
                               (unsigned long long*)&drv.padded_input[0]); // rpp why 8 -- INP_ACCESS
  drv.scs->wgt_mem.burst_write(
      0, drv.pM * drv.pK / 8,
      (unsigned long long*)&drv
          .padded_weights[0]);  // rpp why 8 -- WGT_ACCESS -- due to unsigned
                                // long long. One address containg 8byte of
                                // weight data since weights are quantized by
                                // int8 -- why cannt we directly put data from
                                // drv.padded_weights[0] to global weight BRAM?????

  drv.scs->bias_mem.burst_write(0, drv.pN * drv.pM / 2,
                                (unsigned long long*)&bias_buf[0]); // rpp why 2 -- ACC_ACCESS

  // Start Accelerator Simulation
  sc_start();
  drv.profile->saveProfile(drv.acc->profiling_vars);

  // Retrive Output data from  MMapped DMA buffer
  unsigned int* out_set = (unsigned int*)drv.padded_output;
  int out_len = drv.pN * drv.pM / 4;
  drv.scs->out_mem.burst_read(0, out_len, out_set);

#ifdef DEBUG_FCDRIVER
  mkdir("aData/Debug", 0777);
  myfile.open("aData/Debug/DEBUG_FCDRIVER.csv", ios::app);
  myfile << "!!!!!! send drv data that received from tensor to the drv scs i.e. to the accelerator !!!!!" << endl;
  myfile << "!!!!!! Start Accelerator Simulation !!!!!" << endl;
  myfile << "!!!!!! drv.profile->saveProfile(drv.acc->profiling_vars); !!!!!" << endl;
  myfile << "!!!!!! Retrive Output data from  MMapped DMA buffer !!!!!" << endl;
  myfile.close();
#endif
}

void Entry(acc_container& drv) {
#ifdef DELEGATE_VERBOSE
  cout << "FC ACC - Layer: " << drv.layer << endl;
  cout << "===========================" << endl;
  cout << "Pre-ACC Info" << endl;
  cout << "padded_K: " << drv.pK << " K: " << drv.K << endl;
  cout << "padded_M: " << drv.pM << " M: " << drv.M << endl;
  cout << "padded_N: " << drv.pN << " N: " << drv.N << endl;
  cout << "===========================" << endl;
#endif
  BlockFC(drv);
}

}  // namespace tflite_bertsim
#endif  // FC_DRIVER
