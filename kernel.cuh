#ifndef __KERNEL_H_
#define __KERNEL_H_

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cub/util_allocator.cuh"
#include <bits/stdc++.h>
#include "Graph.h"
#include "robin_map.h"
#include "IdProbPair.h"

typedef tsl::robin_map<int, double> mymapID;                        // level - node -value
typedef tsl::robin_map<int, tsl::robin_map<int, int>> mymapofmapII; // use in detect max level can be optimized
typedef tsl::robin_map<int, vector<int>> mymapOfVecI;               // store parent node
typedef tsl::robin_map<int, vector<mymapID>> mymapOfVecMapID;
typedef tsl::robin_map<int, vector<IdProbPair>> mymapOfVecIpps;

void kernel_wrapper(int source, double epsilon_h, Graph *g, double sqrtC, double sidIndegRecip, vector<mymapID> hprobsFromS, vector<mymapOfVecI> c_parentsOfNodesInGu);

#endif