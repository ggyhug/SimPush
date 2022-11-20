#ifndef GRAPH_H
#define GRAPH_H

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <time.h>

using namespace std;

class Graph
{
public:
    int n;                // number of nodes
    unsigned long long m; // number of edges
    int **inAdjLists = NULL;
    int **outAdjLists = NULL;
    int *indegree = NULL;
    int *outdegree = NULL;
    double *indegRecip = NULL;
    int *csr_adjs = NULL;
    int *csr_begins = NULL;

    Graph()
    {
    }

    ~Graph()
    {
        if (indegree != NULL)
            delete[] indegree;
        if (indegRecip != NULL)
            delete[] indegRecip;
        if (outdegree != NULL)
            delete[] outdegree;
        if (inAdjLists != NULL)
        {
            for (int i = 0; i < n; ++i)
            {
                delete[] inAdjLists[i];
            }
            delete[] inAdjLists;
        }
        if (outAdjLists != NULL)
        {
            for (int i = 0; i < n; ++i)
            {
                delete[] outAdjLists[i];
            }
            delete[] outAdjLists;
        }
        if (csr_adjs != NULL)
            delete[] csr_adjs;
        if (csr_begins != NULL)
            delete[] csr_begins;
    }

    void loadGraph(string filelabel)
    {
        cout << "LOADING GRAPH\n";
        string statfilename = "graph/" + filelabel + "_stat.txt";
        cout << statfilename;
        ifstream instatfile(statfilename.c_str());
        if (!instatfile.is_open())
        {
            cerr << "file may not exist: " << statfilename << endl;
        }
        instatfile >> n >> m;
        instatfile.close();
        cout << ": n= " << n << " m= " << m << endl;

        string degfilename = "graph/" + filelabel + "_deg.txt";
        cout << degfilename;
        ifstream degfile(degfilename.c_str());
        indegree = new int[n];
        outdegree = new int[n];
        int nid = 0;
        while (degfile >> indegree[nid] >> outdegree[nid])
        {
            nid++;
        }
        degfile.close();
        cout << ": loaded degree info\n";

        inAdjLists = new int *[n];
        outAdjLists = new int *[n];
        for (int vid = 0; vid < n; ++vid)
        {
            if (indegree[vid] > 0)
                inAdjLists[vid] = new int[indegree[vid]];
            if (outdegree[vid] > 0)
                outAdjLists[vid] = new int[outdegree[vid]];
        }

        int *inAdjIdxes = new int[n];
        int *outAdjIdxes = new int[n];
        for (int i = 0; i < n; i++)
        {
            inAdjIdxes[i] = 0;
            outAdjIdxes[i] = 0;
        }
        string datafilename = "graph/" + filelabel + ".txt";
        cout << datafilename;
        ifstream datafile(datafilename.c_str());
        int from;
        int to;
        while (datafile >> from >> to)
        {
            // update from's outAdjList
            outAdjLists[from][outAdjIdxes[from]] = to;
            outAdjIdxes[from]++;
            // update to's inAdjList
            inAdjLists[to][inAdjIdxes[to]] = from;
            inAdjIdxes[to]++;
        }
        datafile.close();
        delete[] inAdjIdxes;
        delete[] outAdjIdxes;
        cout << ": loaded graph data \n";

        indegRecip = new double[n]();
        for (int i = 0; i < n; ++i)
        {
            if (indegree[i] > 0)
                indegRecip[i] = 1.0 / (double)indegree[i];
        }
        // csr
        csr_adjs = new int[m];
        csr_begins = new int[n];
        for (int vid = 0; vid < n; vid++)
        {
            int beginPos = 0;
            if (vid == 0)
            {
                beginPos = 0;
            }
            else
            {
                beginPos = csr_begins[vid - 1] + indegree[vid - 1];
            }
            csr_begins[vid] = beginPos;
        }

        for (int vid = 0; vid < n; vid++)
        {
            int beginPos = csr_begins[vid]; // inclusive
            for (int temp = 0; temp < indegree[vid]; temp++)
            {
                csr_adjs[beginPos + temp] = inAdjLists[vid][temp];
            }
        }
    }
};

#endif
