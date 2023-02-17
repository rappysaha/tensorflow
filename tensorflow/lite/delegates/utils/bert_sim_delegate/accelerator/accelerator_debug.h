#ifndef ACCELERATOR_DEBUG_H_
#define ACCELERATOR_DEBUG_H_

#include <iostream>
#include <fstream>
#include <utility>

#include <sys/stat.h>
#include <sys/types.h>

#define DEBUG_ACCELERATOR

#ifdef DEBUG_ACCELERATOR

#define DEBUG_FETCH


ofstream myfileAcc;
#endif // DEBUG_ACCELERATOR



#endif