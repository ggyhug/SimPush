#include "kernel.cuh"
#include <map>
#include <omp.h>
#include <cub/device/device_select.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>

static int calc_num_blocks(int total_items, int items_per_block)
{
    int num_blocks = (total_items + items_per_block - 1) / items_per_block;
    // size_t max_num_blocks = TOTAL_THREADS_GPU / items_per_block;
    // if (max_num_blocks < num_blocks) num_blocks = max_num_blocks;
    return num_blocks;
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val)
{
    unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
#define CUDA_CHECK(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

int getThreadNum()
{
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    printf("gpu num %d\n", count);
    cudaGetDeviceProperties(&prop, 0);
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

__global__ void kernel_detectmaxlevel(int *d_csr_adjs, int *d_csr_begins, int *d_indegree, int size, int vertices, int numWs,
                                      unsigned long seed, int *nodeByLevelsinThread, int source, int sIndeg, double sqrtC)
{
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadId + (blockDim.x * blockDim.y) * blockId;
    if (i < numWs)
    {
        int curV = source;
        int curIndeg = sIndeg;
        int curLev = 0;
        curandState state;
        curand_init(seed, i, 0, &state);
        while ((curand(&state) % RAND_MAX / (double)RAND_MAX) <= sqrtC)
        {
            if (curIndeg == 0)
                break;
            // curV = d_inAdjLists[curV][(curand(&state)) % curIndeg];
            curV = d_csr_adjs[d_csr_begins[curV] + (curand(&state) % curIndeg)];
            curLev++;
            atomicAdd(&nodeByLevelsinThread[curLev * vertices + curV], 1);
            curIndeg = d_indegree[curV];
            if (curLev > size)
                break;
        }
    }
}

__global__ void kernel_selectmaxlevel(int *d_result, int size, double amountPerShare, double epsilon_h)

{
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadId + (blockDim.x * blockDim.y) * blockId;
    if (i < size)
    {
        double inc = amountPerShare * (double)d_result[i] * pow(0.8, i);
        if (inc > epsilon_h / 2.0)
        {
            d_result[i] = i;
        }
        else
        {
            d_result[i] = 0;
        }
    }
}

__global__ void kernel_updatehprobsANDparents(int *d_csr_adjs, int *d_csr_begins, int *d_outdegree, int *d_begindegree,
                                              int levelOfPush, int *d_frontierQueue, double *d_frontierPushValueQueue, int *d_frontierSiz,
                                              double *d_level_hprobsFromS, int *d_level_NodesInGu, int *d_level_parentsOfNodesInGu)
{
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadId + (blockDim.x * blockDim.y) * blockId;

    int frontier_idx = blockId;

    if (frontier_idx < d_frontierSiz[0])
    {
        int front = d_frontierQueue[frontier_idx];
        double pushValue = d_frontierPushValueQueue[frontier_idx];
        int degree = d_outdegree[front];
        if (threadId < degree)
        {
            int vid = d_csr_adjs[d_csr_begins[front] + threadId];
            // atomic add
            atomicAdd(&d_level_hprobsFromS[vid], pushValue);
            // parent
            d_level_NodesInGu[d_begindegree[frontier_idx] + threadId] = vid;
            d_level_parentsOfNodesInGu[d_begindegree[frontier_idx] + threadId] = front;
        }
    }
}

__global__ void kernel_computedegree(int *d_indegree, int *d_frontierSiz, int *d_frontierQueue, int *d_degreenum)
{
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadId + (blockDim.x * blockDim.y) * blockId;
    if (i < d_frontierSiz[0])
    {
        d_degreenum[i] = d_indegree[d_frontierQueue[i]];
    }
}

__global__ void kernel_getnextfrontier(int vert, double *d_pushValue, double *d_indegRecip, double *d_level_hprobsFromS, unsigned char *d_flags, double pushValueThreshold, double sqrtC)
{
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadId + (blockDim.x * blockDim.y) * blockId;

    if (i < vert)
    {
        double prob = d_level_hprobsFromS[i];
        double pushValue = d_indegRecip[i] * prob * sqrtC;
        if (pushValue >= pushValueThreshold)
        {
            d_pushValue[i] = pushValue;
            d_flags[i] = 1;
        }
    }
}

__global__ void kernel_updatetotal(int vert, int levelOfPush, double *d_level_hprobsFromS, int *d_level_parentsOfNodesInGu, double *d_total_level_hprobsFromS, int *d_total_level_parentsOfNodesInGu)
{
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadId + (blockDim.x * blockDim.y) * blockId;
    if (i < vert)
    {
        d_total_level_hprobsFromS[levelOfPush * vert + i] = d_level_hprobsFromS[i];
        d_total_level_parentsOfNodesInGu[levelOfPush * vert + i] = d_level_parentsOfNodesInGu[i];
    }
}

void kernel_wrapper(int source, double epsilon_h, Graph *g, double sqrtC, double sidIndegRecip, vector<mymapID> hprobsFromS, vector<mymapOfVecI> c_parentsOfNodesInGu)
{
    // getThreadNum();

    // initial graph cuda
    int *d_csr_adjs = NULL;
    int *d_csr_begins = NULL;
    int *d_indegree = NULL;
    int *d_outdegree = NULL;
    double *d_indegRecip = NULL;
    CUDA_CHECK(cudaMalloc(&d_indegree, sizeof(int) * g->n));
    CUDA_CHECK(cudaMalloc(&d_outdegree, sizeof(int) * g->n));
    CUDA_CHECK(cudaMalloc(&d_indegRecip, sizeof(double) * g->n));
    CUDA_CHECK(cudaMalloc(&d_csr_adjs, sizeof(int) * g->m));
    CUDA_CHECK(cudaMalloc(&d_csr_begins, sizeof(int) * g->n));
    CUDA_CHECK(cudaMemcpy(d_indegree, g->indegree, sizeof(int) * g->n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_outdegree, g->outdegree, sizeof(int) * g->n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indegRecip, g->indegRecip, sizeof(double) * g->n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_adjs, g->csr_adjs, sizeof(int) * g->m, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_begins, g->csr_begins, sizeof(int) * g->n, cudaMemcpyHostToDevice));

    dim3 threads(32, 32);
    dim3 blocks(128, 128);

    double delta = 0.00001;
    double amountPerShare = epsilon_h / (double)log(g->n);
    int numWs = 1.0 / amountPerShare;
    int sIndeg = g->indegree[source];

    // initial detect maxlevel storage
    int maxLev = 0;
    int size = 15;
    int *max = new int[1];
    int *result = new int[size];
    int *d_max = NULL;
    int *d_result = NULL;
    int *d_level_hitnode = NULL;
    int *d_nodeByLevelsinThread = NULL;
    int *nodeByLevelsinThread = new int[size * g->n];

    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int) * size));
    CUDA_CHECK(cudaMalloc(&d_nodeByLevelsinThread, sizeof(int) * size * g->n));
    CUDA_CHECK(cudaMalloc(&d_level_hitnode, sizeof(int) * g->n));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_level_hitnode, d_max, g->n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    void *d_temp_storage1 = NULL;
    size_t temp_storage_bytes1 = 0;
    cub::DeviceReduce::Max(d_temp_storage1, temp_storage_bytes1, d_result, d_max, size);
    cudaMalloc(&d_temp_storage1, temp_storage_bytes1);

    if (sIndeg > 0)
    {
        kernel_detectmaxlevel<<<calc_num_blocks(numWs, 1024), threads>>>(d_csr_adjs, d_csr_begins, d_indegree, size, g->n, numWs,
                                                                         time(NULL), d_nodeByLevelsinThread, source, sIndeg, sqrtC);
    }
    cudaThreadSynchronize();
    for (int i = 1; i < size; i++)
    {
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, &d_nodeByLevelsinThread[i * g->n], &d_result[i], g->n);
    }
    kernel_selectmaxlevel<<<calc_num_blocks(size, 1024), threads>>>(d_result, size, amountPerShare, epsilon_h);
    cudaThreadSynchronize();
    cub::DeviceReduce::Max(d_temp_storage1, temp_storage_bytes1, d_result, d_max, size);
    CUDA_CHECK(cudaMemcpy(max, d_max, sizeof(int), cudaMemcpyDeviceToHost));
    maxLev = max[0];
    cout << "maxLev = :" << maxLev << endl;

    // parallel frontier
    int *frontierSiz = new int[1];
    int *vertices = new int[g->n];
    int levelOfPush = 0;
    double pushValueThreshold = 0.0000005;
    double spushValue = 1.0 * sqrtC * sidIndegRecip;

    int *frontierQueue = new int[g->n];
    double *frontierPushValueQueue = new double[g->n];

    double *level_hprobsFromS = new double[g->n];
    int *level_parentsOfNodesInGu = new int[g->n];
    int *degreenum = new int[g->n];

    for (int i = 0; i < g->n; i++)
    {
        vertices[i] = i;
    }

    // level_hprobsFromS[source] = 1.0;

    if (spushValue >= pushValueThreshold)
    {
        frontierSiz[0] = 0;
        frontierQueue[frontierSiz[0]] = source;
        frontierPushValueQueue[frontierSiz[0]] = spushValue;
        frontierSiz[0]++;
    }

    // CUDA initial
    int *d_frontierQueue = NULL;
    double *d_frontierPushValueQueue = NULL;
    double *d_level_hprobsFromS = NULL;
    int *d_level_NodesInGu = NULL;
    int *d_level_parentsOfNodesInGu = NULL;
    int *d_degreenum = NULL;
    int *d_begindegree = NULL;
    int *d_frontierSiz = NULL;
    int *d_vertices = NULL;
    double *d_pushValue = NULL;
    unsigned char *d_flags = NULL;
    double *d_total_level_hprobsFromS = NULL;
    int *d_total_level_parentsOfNodesInGu = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_vertices, sizeof(int) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_pushValue, sizeof(double) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_flags, sizeof(unsigned char) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_frontierSiz, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_degreenum, sizeof(int) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_begindegree, sizeof(int) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_frontierQueue, sizeof(int) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_frontierPushValueQueue, sizeof(double) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_level_hprobsFromS, sizeof(double) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_level_NodesInGu, sizeof(int) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_level_parentsOfNodesInGu, sizeof(int) * g->n));
    CUDA_CHECK(cudaMalloc((void **)&d_total_level_hprobsFromS, sizeof(double) * g->n * (maxLev + 2)));
    CUDA_CHECK(cudaMalloc((void **)&d_total_level_parentsOfNodesInGu, sizeof(int) * g->n * (maxLev + 2)));

    CUDA_CHECK(cudaMemcpy(d_vertices, vertices, sizeof(int) * g->n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontierSiz, frontierSiz, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontierQueue, frontierQueue, sizeof(int) * g->n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontierPushValueQueue, frontierPushValueQueue, sizeof(double) * g->n, cudaMemcpyHostToDevice));

    void *d_temp_storage_for_flags = NULL;
    size_t temp_storage_bytes_for_flags = 0;
    cub::DeviceSelect::Flagged(d_temp_storage_for_flags, temp_storage_bytes_for_flags, d_vertices, d_flags, d_frontierQueue, d_frontierSiz, g->n);
    CUDA_CHECK(cudaMalloc(&d_temp_storage_for_flags, temp_storage_bytes_for_flags));

    void *d_temp_storage_for_flags1 = NULL;
    size_t temp_storage_bytes_for_flags1 = 0;
    cub::DeviceSelect::Flagged(d_temp_storage_for_flags1, temp_storage_bytes_for_flags1, d_pushValue, d_flags, d_frontierPushValueQueue, d_frontierSiz, g->n);
    CUDA_CHECK(cudaMalloc(&d_temp_storage_for_flags1, temp_storage_bytes_for_flags1));

    // frontier
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (levelOfPush < maxLev)
    {
        kernel_computedegree<<<blocks, threads>>>(d_indegree, d_frontierSiz, d_frontierQueue, d_degreenum);
        void *d_temp_storage1 = NULL;
        size_t temp_storage_bytes1 = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, d_degreenum, d_begindegree, frontierSiz[0]);
        cudaMalloc(&d_temp_storage1, temp_storage_bytes1);
        cub::DeviceScan::ExclusiveSum(d_temp_storage1, temp_storage_bytes1, d_degreenum, d_begindegree, frontierSiz[0]);

        kernel_updatehprobsANDparents<<<blocks, threads>>>(d_csr_adjs, d_csr_begins, d_outdegree, d_begindegree,
                                                           levelOfPush, d_frontierQueue, d_frontierPushValueQueue, d_frontierSiz,
                                                           d_level_hprobsFromS, d_level_NodesInGu, d_level_parentsOfNodesInGu);
        kernel_getnextfrontier<<<blocks, threads>>>(g->n, d_pushValue, d_indegRecip, d_level_hprobsFromS, d_flags, pushValueThreshold, sqrtC);
        cub::DeviceSelect::Flagged(d_temp_storage_for_flags, temp_storage_bytes_for_flags, d_vertices, d_flags, d_frontierQueue, d_frontierSiz, g->n);
        cub::DeviceSelect::Flagged(d_temp_storage_for_flags1, temp_storage_bytes_for_flags1, d_pushValue, d_flags, d_frontierPushValueQueue, d_frontierSiz, g->n);
        CUDA_CHECK(cudaMemcpy(frontierSiz, d_frontierSiz, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(level_hprobsFromS, d_level_hprobsFromS, sizeof(double) * g->n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(d_flags, 0, sizeof(unsigned char) * g->n));
        CUDA_CHECK(cudaMemset(d_level_hprobsFromS, 0, sizeof(double) * g->n));
        CUDA_CHECK(cudaMemset(d_level_parentsOfNodesInGu, 0, sizeof(int) * g->n)); // check
        for (int i = 0; i < g->n; i++)
        {
            if (level_hprobsFromS[i] != 0)
            {
                count++;
                if (level_hprobsFromS[i] != hprobsFromS[levelOfPush + 1][i])
                    flag = false;
            }
        }
        if (flag)
        {
            cout << levelOfPush + 1 << "_level :yes" << endl;
        }
        else
        {
            cout << levelOfPush + 1 << "_level :no" << endl;
        }
        levelOfPush++;
    }

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "gpu frontier time:" << time << " ms" << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] frontierQueue;
    delete[] frontierPushValueQueue;
    delete[] vertices;
    delete[] frontierSiz;
    delete[] level_hprobsFromS;
    delete[] level_parentsOfNodesInGu;

    if (d_indegree)
        CUDA_CHECK(cudaFree(d_indegree));
    if (d_outdegree)
        CUDA_CHECK(cudaFree(d_outdegree));
    if (d_indegRecip)
        CUDA_CHECK(cudaFree(d_indegRecip));
    if (d_csr_adjs)
        CUDA_CHECK(cudaFree(d_csr_adjs));
    if (d_csr_begins)
        CUDA_CHECK(cudaFree(d_csr_begins));
    if (d_nodeByLevelsinThread)
        CUDA_CHECK(cudaFree(d_nodeByLevelsinThread));
    if (d_level_hitnode)
        CUDA_CHECK(cudaFree(d_level_hitnode));
    if (d_max)
        CUDA_CHECK(cudaFree(d_max));
    if (d_frontierQueue)
        CUDA_CHECK(cudaFree(d_frontierQueue));
    if (d_frontierPushValueQueue)
        CUDA_CHECK(cudaFree(d_frontierPushValueQueue));
    if (d_level_hprobsFromS)
        CUDA_CHECK(cudaFree(d_level_hprobsFromS));
    if (d_level_NodesInGu)
        CUDA_CHECK(cudaFree(d_level_NodesInGu));
    if (d_level_parentsOfNodesInGu)
        CUDA_CHECK(cudaFree(d_level_parentsOfNodesInGu));
    if (d_vertices)
        CUDA_CHECK(cudaFree(d_vertices));
    if (d_flags)
        CUDA_CHECK(cudaFree(d_flags));
    if (d_frontierSiz)
        CUDA_CHECK(cudaFree(d_frontierSiz));
    if (d_pushValue)
        CUDA_CHECK(cudaFree(d_pushValue));
    if (d_total_level_hprobsFromS)
        CUDA_CHECK(cudaFree(d_total_level_hprobsFromS));
    if (d_total_level_parentsOfNodesInGu)
        CUDA_CHECK(cudaFree(d_total_level_parentsOfNodesInGu));
}
