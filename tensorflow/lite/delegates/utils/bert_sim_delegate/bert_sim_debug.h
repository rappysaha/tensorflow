#ifndef BERT_SIM_DEBUG_H_
#define BERT_SIM_DEBUG_H_

#include <iostream>
#include <fstream>
#include <utility>

#include <sys/stat.h>
#include <sys/types.h>

#define DEBUG_BERT_SIM

#ifdef DEBUG_BERT_SIM

#define DEBUG_BERTSIMDELEGATE
#define DEBUG_FCDRIVER
#define DEBUG_ACCCONTAINER


std::ofstream myfile;
#endif // DEBUG_BERT_SIM



#endif