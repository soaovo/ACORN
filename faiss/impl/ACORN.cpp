/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/ACORN.h>

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>

// added
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>  
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#define _OPENMP 1
/*******************************************************
 * Added for debugging
 *******************************************************/

const int debugFlag = 0;

// const int debugSearchFlag =  std::atoi(std::getenv(debugSearchFlag));
const char* debugSearchFlagEnv = std::getenv("debugSearchFlag");
int debugSearchFlag = debugSearchFlagEnv ? std::atoi(debugSearchFlagEnv) : 0;

int getenv_int_or_default(const char* name, int default_value) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : default_value;
}

const int hybridDebugQuery = getenv_int_or_default("ACORN_DEBUG_HYBRID_QUERY", -1);
const int hybridDebugExpansionLimit =
        getenv_int_or_default("ACORN_DEBUG_HYBRID_EXPANSIONS", 4);

std::atomic<int> hybridDebugSearchCounter{0};
thread_local int hybridDebugActiveQuery = -1;
thread_local bool hybridDebugEnabled = false;
thread_local int hybridDebugExpansionCount = 0;

void hybrid_debug_log(const char* fmt, ...) {
    if (!hybridDebugEnabled) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
    fflush(stdout);
}


void debugTime() {
	if (debugFlag || debugSearchFlag) {
        struct timeval tval;
        gettimeofday(&tval, NULL);
        struct tm *tm_info = localtime(&tval.tv_sec);
        char timeBuff[25] = "";
        strftime(timeBuff, 25, "%H:%M:%S", tm_info);
        char timeBuffWithMilli[50] = "";
        sprintf(timeBuffWithMilli, "%s.%06ld ", timeBuff, tval.tv_usec);
        std::string timestamp(timeBuffWithMilli);
		std::cout << timestamp << std::flush;
    }
}

#define debug(fmt, ...) \
    do { \
        if (debugFlag == 1) { \
            fprintf(stdout, "" fmt, __VA_ARGS__);\
        } \
        if (debugFlag == 2) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)

// same as debug but for search debugging only
#define debug_search(fmt, ...) \
    do { \
        if (debugSearchFlag == 1) { \
            fprintf(stdout, "" fmt, __VA_ARGS__);\
        } \
        if (debugSearchFlag == 2) { \
            fprintf(stdout, "%d:%s(): " fmt, __LINE__, __func__, __VA_ARGS__); \
        } \
        if (debugSearchFlag == 3) { \
            debugTime(); \
            fprintf(stdout, "%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); \
        } \
    } while (0)


double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}



namespace faiss {




/**************************************************************
 * ACORN structure implementation
 **************************************************************/

int ACORN::nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no + 1] -
            cum_nneighbor_per_level[layer_no];
}

void ACORN::set_nb_neighbors(int level_no, int n) {
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int ACORN::cum_nb_neighbors(int layer_no) const {
    return cum_nneighbor_per_level[layer_no];
}

void ACORN::neighbor_range(idx_t no, int layer_no, size_t* begin, size_t* end)
        const {
    size_t o = offsets[no];
    // debug("offset o: %ld\n", o);
    *begin = o + cum_nb_neighbors(layer_no);
    // debug("begin: %ln\n", begin);
    *end = o + cum_nb_neighbors(layer_no + 1);
    // debug("end: %ln\n", end);
}



ACORN::ACORN(int M, int gamma, std::vector<int>& metadata, int M_beta) : rng(12345) {
    set_default_probas(M, 1.0 / log(M), M_beta, gamma);
    max_level = -1;
    entry_point = -1;
    efSearch = 16;
    pathwise_width = 1;
    pathwise_init_width = 1;
    pathwise_max_width = 1;
    pathwise_growth_interval = 4;
    pathwise_staged = false;
    reduced_sync = true;
    efConstruction = M * gamma; //added gamma
    upper_beam = 1;
    this->gamma = gamma;
    this->metadata = metadata.data();
    this->M = M;
    this->M_beta = M_beta;
    // gamma = gamma;
    offsets.push_back(0);
    for (int i = 0; i < assign_probas.size(); i++) nb_per_level.push_back(0);
}

int ACORN::random_level() {
    double f = rng.rand_float();
    // could be a bit faster with bissection
    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void ACORN::set_default_probas(int M, float levelMult, int M_beta, int gamma) {
    int nn = 0;
    cum_nneighbor_per_level.push_back(0);
    // printf("---set_default_probas: gamma: %d\n", this->gamma);
    // printf("---set_default_probas: gamma: %d\n", gamma);
    if (M_beta > 2*M*gamma) {
        printf("M_beta: %d, M: %d\n", M_beta, M);
        FAISS_THROW_MSG("M_beta must be less than 2*M*gamma");
    }
    for (int level = 0;; level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9)
            break;
        assign_probas.push_back(proba);
        nn += level == 0 ? (int) M_beta + 1.5*M : M * gamma;
        cum_nneighbor_per_level.push_back(nn);

    }
}

void ACORN::clear_neighbor_tables(int level) {
    for (int i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            neighbors[j] = ACORN::NeighNode(-1);
        }
    }
}

void ACORN::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}


void ACORN::print_neighbor_stats(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) // mod
                    break;
                neighset.insert(neighbors[j]); // mod
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j]; // mod
                if (i2 < 0)
                    break;
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];  //mod
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   1. nb of nodes: %zd\n", n_node);
    printf("   2. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   3. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   4. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}


// same as print_neighbor_stats with additional edge lists printed, no parallelism
void ACORN::print_edges(int level) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
    printf("\t edges lists:\n");
//#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            printf("\t\t %d: [", i);
            for (size_t j = begin; j < end; j++) {
                if (neighbors[j] < 0) // mod
                    break;
                printf("%d(%d), ", neighbors[j], metadata[neighbors[j]]); // mod
                neighset.insert(neighbors[j]); // mod
            }
            printf("]\n");
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j]; //mod
                if (i2 < 0)
                    break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2]; // mod
                    if (i3 < 0)
                        break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;

    printf("   0. level: %d\n", level);
    printf("   1. max neighbors per node: %d\n", nb_neighbors(level));
    printf("   2. nb of nodes: %zd\n", n_node);
    printf("   3. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   4. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   5. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);
}

// same as print_neighbor_stats with additional edge lists printed according to filter
void ACORN::print_edges_filtered(int level, int filter, Operation op) const {
    FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
    printf("* stats on level %d, max %d neighbors per vertex:\n",
           level,
           nb_neighbors(level));

    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
    printf("\t edges lists:\n");
//#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
  reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {   
            if (((op == EQUAL) && (metadata[i] == filter)) || (op == OR && ((metadata[i] & filter) != 0))) {
            // if (metadata[i] == filter) {
                n_node++;
                size_t begin, end;
                neighbor_range(i, level, &begin, &end);
                std::unordered_set<int> neighset;
                printf("\t\t %d (%d): [", i, metadata[i]);

                for (size_t j = begin; j < end; j++) {
                    if (neighbors[j] < 0) // mod
                        break;
                    if (((op == EQUAL) && (metadata[neighbors[j]] == filter)) || (op == OR && ((metadata[neighbors[j]] & filter) != 0))) {
                    // if (neighbors[j].second == filter) {
                        printf("%d(%d), ", neighbors[j], metadata[neighbors[j]]); // mod
                        neighset.insert(neighbors[j]); // mod
                    }
                }
                // if (metadata[i] == filter) {
                //     printf("]\n");
                // }
                printf("]\n");


                int n_neigh = neighset.size();
                int n_common = 0;
                int n_reciprocal = 0;
                for (size_t j = begin; j < end; j++) {
                    storage_idx_t i2 = neighbors[j]; //mod
                    if (i2 < 0)
                        break;
                    FAISS_ASSERT(i2 != i);
                    size_t begin2, end2;
                    neighbor_range(i2, level, &begin2, &end2);
                    for (size_t j2 = begin2; j2 < end2; j2++) {
                        storage_idx_t i3 = neighbors[j2]; // mod
                        if (i3 < 0)
                            break;
                        if (i3 == i) {
                            n_reciprocal++;
                            continue;
                        }
                        if (neighset.count(i3)) {
                            neighset.erase(i3);
                            n_common++;
                        }
                    }
                }
                tot_neigh += n_neigh;
                tot_common += n_common;
                tot_reciprocal += n_reciprocal;
            }


            
        }
    }
    float normalizer = n_node;

    printf("   0. level: %d\n", level);
    printf("   1. max neighbors per node: %d\n", nb_neighbors(level));
    printf("   2. nb of nodes: %zd\n", n_node);
    printf("   3. neighbors per node: %.2f (%zd)\n",
           tot_neigh / normalizer,
           tot_neigh);
    printf("   4. nb of reciprocal neighbors: %.2f\n",
           tot_reciprocal / normalizer);
    printf("   5. nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
           tot_common / normalizer,
           tot_common);

}




// stats for all levels 
void ACORN::print_neighbor_stats(bool edge_list, bool filtered_edge_list, int filter, Operation op) const {
    printf("========= METADATA =======\n");

    printf("\t* cumulative max num neighbors per level\n");
    for (int i = 0; i < cum_nneighbor_per_level.size() - 1; i++) {
        printf("\t\tindx %d: %d\n", i, cum_nneighbor_per_level[i]);
    }
    printf("\t* level probabilities\n");
    for (int level = 0; level < assign_probas.size(); level++) {
        printf("\t\tlevel %d: %f\n", level, assign_probas[level]);
    }
    printf("\t* efConstruction: %d\n", efConstruction);
    printf("\t* efSearch: %d\n", efSearch);
    printf("\t* max_level: %d\n", max_level);
    printf("\t* entry_point: %d\n", entry_point);
    printf("\t* gamma: %d\n", gamma);


    // per level stats
    printf("========= LEVEL STATS OF ACORN =======\n");
    if (edge_list){
        // skip bottom level since its too big
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_edges(level); // this is more detailed version with edge lists
            
        } 
    } else {
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_neighbor_stats(level);
            
        }
    }

    // optional filtered edge list print
    if (filtered_edge_list) {
        printf("========= LEVEL STATS OF SUBGRAPH WITH FILTER %d, OP %d (0 for EQUAL, 1 for OR)=======\n", filter, op);
        for (int level = 0; level <= max_level; level++) {
            printf("========= LEVEL %d =======\n", level);
            print_edges_filtered(level, filter, op); // this is more detailed version with edge lists
            
        }
    } 
    // print metadata
    // printf("========= METADATA =======\n");
    // for (int i = 0; i < metadata.size(); i++) {
    //     printf("\t%d: %d\n", i, metadata[i]);
    // }
}



void ACORN::fill_with_random_links(size_t n) {
    throw FaissException("UNIMPLEMENTED");
    // int max_level = prepare_level_tab(n);
    // RandomGenerator rng2(456);

    // for (int level = max_level - 1; level >= 0; --level) {
    //     std::vector<int> elts;
    //     for (int i = 0; i < n; i++) {
    //         if (levels[i] > level) {
    //             elts.push_back(i);
    //         }
    //     }
    //     printf("linking %zd elements in level %d\n", elts.size(), level);

    //     if (elts.size() == 1)
    //         continue;

    //     for (int ii = 0; ii < elts.size(); ii++) {
    //         int i = elts[ii];
    //         size_t begin, end;
    //         neighbor_range(i, 0, &begin, &end);
    //         for (size_t j = begin; j < end; j++) {
    //             int other = 0;
    //             do {
    //                 other = elts[rng2.rand_int(elts.size())];
    //             } while (other == i);

    //             neighbors[j] = other;
    //         }
    //     }
    // }
}

// Modified from original HNSW
// n is the number of vectors that will be added (this gets called in IndexHSNW.hnsw_add_vertices)
int ACORN::prepare_level_tab(size_t n, bool preset_levels) {
    size_t n0 = offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT(n0 + n == levels.size());
    } else {
        FAISS_ASSERT(n0 == levels.size());
        for (int i = 0; i < n; i++) {
            int pt_level = random_level();
            levels.push_back(pt_level + 1);
        }
    }

    int max_level = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = levels[i + n0] - 1;
        if (pt_level > max_level)
            max_level = pt_level;
        offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
        // neighbors.resize(offsets.back(), -1);
        neighbors.resize(offsets.back(), NeighNode(-1)); // mod
    }

    return max_level;
}

/** Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void ACORN::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size, int gamma, storage_idx_t q_id, int q_attr) {

    debug("shrink_neighbor_list: input size: %ld, max_size: %d, gamma: %d\n", input.size(), max_size, gamma);
    // new pruning method which removes neighbors are in an existing neighbors neighborhood
    std::unordered_set<storage_idx_t> neigh_of_neigh;
        // for (const NodeDistFarther& node : output) {
        //     outputSet.insert(node.id);
        // }
    int node_num = 0;    
    
    while (input.size() > 0) {
        node_num = node_num + 1;

        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        debug("shrink_neighbor_list: checking whether to keep v1: %d\n", v1.id);
        bool good = true;

        // check if current candidate is in the neighbors of output 
        if (node_num > this->M_beta  && neigh_of_neigh.count(v1.id) > 0) {
            good = false;
            debug("PRUNE v1: %d\n", v1.id);
        }
        
        
        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;
            }

            // update neigh of neigh set
            neigh_of_neigh.insert(v1.id);
            if (node_num > this->M_beta) {
                size_t begin, end;
                neighbor_range(v1.id, 0, &begin, &end);
                for (size_t j = begin; j < end; j++) {
                    if (neighbors[j] < 0) // mod
                        break;
                    neigh_of_neigh.insert(neighbors[j]); // mod
                }

            } 
            
            // break if neigh_of_neigh set is sufficiently large
            if (neigh_of_neigh.size() >= max_size) {
                break;
            }

        }
        debug("tentative edge list of size %zd: \n", output.size());
        for (NodeDistFarther v2 : output) {
            debug("\t\t node %d\n", v2.id);
        }
        debug("---- %zd: \n", output.size());
        
    }
    debug("final edge list of size %zd: \n", output.size());
        for (NodeDistFarther v2 : output) {
            debug("\t\t node %d\n", v2.id);
        }
        debug("---- %zd: \n", output.size());

}

namespace {

using storage_idx_t = ACORN::storage_idx_t;
using NodeDistCloser = ACORN::NodeDistCloser;
using NodeDistFarther = ACORN::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size, int gamma, storage_idx_t q_id, int q_attr, ACORN& hnsw) {
    debug("shrink_neighbor_list from size %ld, to max size %d\n", resultSet1.size(), max_size);
    // if (resultSet1.size() < max_size) {
    //     return;
    // }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    // ACORN::shrink_neighbor_list(qdis, resultSet, returnlist, max_size, gamma, q_id, q_attr);
    hnsw.shrink_neighbor_list(qdis, resultSet, returnlist, max_size, gamma, q_id, q_attr);


    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

// modified from normal hnsw
/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(
        ACORN& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level) {
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) { // mood
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hnsw.neighbors[i - 1] != -1) // mod
                break;
            i--;
        }
        // hnsw.neighbors[i] = dest;
        hnsw.neighbors[i] = dest; // mod
        debug("added link from %d to %d at level %d\n", src, dest, level);
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        // storage_idx_t neigh = hnsw.neighbors[i];
        // auto [neigh, metadata] = hnsw.neighbors[i]; // mod
        auto neigh = hnsw.neighbors[i];
        auto metadata = hnsw.metadata[neigh];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }


    debug("calling shrink neigbor list, src: %d, dest: %d, level: %d\n", src, dest, level);
    
    if (level == 0) {
        shrink_neighbor_list(qdis, resultSet, end - begin, hnsw.gamma, src, hnsw.metadata[src], hnsw);

    }
    
    

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        // hnsw.neighbors[i++] = resultSet.top().id;
        hnsw.neighbors[i++] = ACORN::NeighNode(resultSet.top().id); // mod
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        // hnsw.neighbors[i++] = -1;
        hnsw.neighbors[i++] = ACORN::NeighNode(-1); //mod
    }
}


// modified from normal hnsw
/// search neighbors on a single level, starting from an entry point
// this only gets called in construction
void search_neighbors_to_add(
        ACORN& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level = {}) {
    debug("search_neighbors to add, entrypoint: %d\n", entry_point);
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    // number of neighbors we want for hybrid construction
    // int M = hnsw.nb_neighbors(level);
    int M;
    if (level == 0) { // level 0 may be compressed
        M = 2* hnsw.M * hnsw.gamma;
    }
    else {
        M = hnsw.nb_neighbors(level);
    }
    debug("desired resuts size: %d, at level: %d\n", M, level);
    

    int backtrack_level = level;

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        // MOD greedy break - added break conditional on size of results
        if ((currEv.d > results.top().d && hnsw.gamma == 1) || results.size() >= M) {
            debug("greedy stop in construction, results size: %ld, desired size: %d, gamma = %d\n", results.size(), M, hnsw.gamma);
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);
        
        int numIters = 0;

        debug("checking neighbors of %d\n", currNode);
        for (size_t i = begin; i < end; i++) {
            // auto [nodeId, metadata] = hnsw.neighbors[i]; // storage_idx_t, int
            auto nodeId = hnsw.neighbors[i];
            auto metadata = hnsw.metadata[nodeId];
            // storage_idx_t nodeId = hnsw.neighbors[i];
            if (nodeId < 0)
                break;
            if (vt.get(nodeId))
                continue;
            vt.set(nodeId);

            // limit number neighbors visisted during construciton
            numIters = numIters + 1;
            if (numIters > hnsw.M) {
                break;
            }

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId);

            // debug("while checking neighbors of %d, efc: %d, results size: %ld, just visited %d\n", currNode, hnsw.efConstruction, results.size(), nodeId);
            if (results.size() < hnsw.efConstruction || results.top().d > dis) {
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }
            debug("while checking neighbors of %d, just visited %d -- efc: %d, results size: %ld, candidates size: %ld, \n", currNode, nodeId, hnsw.efConstruction, results.size(), candidates.size());

            

            // limit number neighbors visisted during construciton
            numIters = numIters + 1;
            if (numIters > hnsw.M) {
                break;
            }

        
        }
        
        debug("during BFS, gamma: %d, candidates size: %ld, results size: %ld, vt.num_visited: %d, nb on level: %d, backtrack_level: %d, level: %d\n", hnsw.gamma, candidates.size(), results.size(), vt.num_visited(), hnsw.nb_per_level[level], backtrack_level, level);
    }
    debug("search_neighbors to add finds %ld nn's\n", results.size());
    // printf("search_neighbors to add finds %ld nn's\n", results.size());
    // printf("\tdesired resuts size: %d, at level: %d\n", M, level);

    vt.advance();
}


/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
/// used for construction (other version below will be used in search)
void greedy_update_nearest(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    debug("%s\n", "reached");
    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        
        
        int numIters = 0;

        for (size_t i = begin; i < end; i++) {
            auto v = hnsw.neighbors[i];
            auto metadata = hnsw.metadata[v];
            if (v < 0) {
                break;
            }

            // limit number neighbors visisted during construciton
            numIters = numIters + 1;
            if (numIters > hnsw.M) {
                break;
            }

            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        if (nearest == prev_nearest) {
            return;
        }
    }
}


/// for hybrid search only
int hybrid_greedy_update_nearest(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    debug("%s\n", "reached"); 
    // printf("hybrid_greedy_update_nearest called with parameters: filter: %d, op: %d, regex: %s, level: %d\n", filter, op, regex.c_str(), level);
    int ndis = 0;
    for (;;) {
        int num_found = 0;
        storage_idx_t prev_nearest = nearest;
        debug_search("----hybrid_greedy_update visists current nearest: %d, d_nearest: %f\n", nearest, d_nearest);

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        debug_search("%s", "--------checking neighbors: \n");
        
        // for debugging, collect all neighbors looked at in a vector
        std::vector<std::pair<storage_idx_t, int>> neighbors_checked;
        bool keep_expanding = true;

        for (size_t i = begin; i < end; i++) {
            auto v = hnsw.neighbors[i];
            
            if (v < 0)
                break;
                
            // note that this slows down search significantly but can be useful for debugging
            // if (debugSearchFlag) {
            //     neighbors_checked.push_back(std::make_pair(v, metadata)); 
            //     debug_search("------------checking neighbor: %d, metadata: %d, metadata & filter: %d\n", v, metadata, metadata & filter);
            // }

            // filter
            // printf("---at first filter: op: %d, metadata: %s, regex: %s, check_regex result: %d\n", op, hnsw.metadata_strings[v].c_str(), regex.c_str(), CHECK_REGEX(hnsw.metadata_strings[v], regex));
            if (filter_map[v]) {
                num_found = num_found + 1;
            } else {
                // not filter & gamma > 1
                if (hnsw.gamma > 1) {
                    continue;
                }
            }
            

        
            
            // check if filter pass
            if (filter_map[v]) {
    
                float dis = qdis(v);
                ndis += 1;
                if (dis < d_nearest || !filter_map[nearest]) {
                
                    nearest = v;
                    d_nearest = dis;
                    // debug_search("----------------new nearest: %d, d_nearest: %f\n", nearest, d_nearest);
                }
                if (num_found >= hnsw.M) {
                    // debug_search("----found %d neighbors with filter %d, returning\n", num_found, filter);
                    break;
                }
            }            

          

            // expand neighbor list if gamma=1
            if (hnsw.gamma == 1) {
                size_t begin2, end2;
                hnsw.neighbor_range(v, level, &begin2, &end2);
                for (size_t j = begin2; j < end2; j++) {
                    auto v2 = hnsw.neighbors[j];
                   

                    if (v2 < 0)
                        break;


                    // check filter pass
                    if (filter_map[v2]) {
                        num_found = num_found + 1;
                        float dis2 = qdis(v2);
                        ndis += 1;
                        // debug_search("------------found: %d, metadata: %d distance to v: %f\n", v2, metadata2, dis2);
          
                        if (dis2 < d_nearest || !filter_map[nearest]) {
                            nearest = v2;
                            d_nearest = dis2;
                            // debug_search("----------------new nearest: %d, d_nearest: %f\n", nearest, d_nearest);
                        }
                        if (num_found >= hnsw.M) {
                            break;
                        }
                    } 
                   
                }
            }
        }       

        if (nearest == prev_nearest) {
            return ndis;
        }
    }
    return ndis;
}

} // namespace

// modified from normal hnsw
void ACORN::add_links_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        std::vector<storage_idx_t> ep_per_level) {
    debug("add_links_starting_from at level: %d, nearest: %d\n", level, nearest);
    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt, ep_per_level); // mod

    // added update nearest
    nearest = link_targets.top().id;

    // but we can afford only this many neighbors
    int M = nb_neighbors(level);

    debug("add_links_starting_from will shrink results list to size: %d\n", M);

    
    debug("calling shrink neigbor list, pt_id: %d, level: %d\n", pt_id, level);

    if (level == 0) {
        ::faiss::shrink_neighbor_list(ptdis, link_targets, M, gamma, pt_id, this->metadata[pt_id], *this);
        // printf("shrunk");
    }
    
    
    debug("add_links_starting_from gets edge link size: %ld\n", link_targets.size());

    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level); // mod
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level); // mod
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}



/**************************************************************
 * Building, parallel
 **************************************************************/
// mod compared to original hnsnw
void ACORN::add_with_locks(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {
    //  greedy search on upper levels
    debug("add_with_locks called for node %d, level %d\n", pt_id, pt_level);


    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
            for (int i=0; i <= max_level; i++){
                nb_per_level[i] = nb_per_level[i] + 1;
            }
            
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    // needed for backtracking in hybrid search - TODO DEPRECATED, remove backtracking things
    std::vector<storage_idx_t> ep_per_level(max_level); // idx of nearest node per level
    ep_per_level[level] = nearest;

    for (; level > pt_level; level--) {
        debug("--greedy update nearest at level: %d, ep: %d\n", level, nearest);
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
        ep_per_level[level] = nearest;
    }

    for (int i = 0; i <= max_level; i++) {
        debug("--at level: %d, ep: %d\n", i, ep_per_level[i]);
    }

    for (; level >= 0; level--) {
        debug("--add_links_starting_from at level: %d\n", level);
        add_links_starting_from(
                ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt, ep_per_level);
        nb_per_level[level] = nb_per_level[level] + 1;
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }
}

/**************************************************************
 * Searching
 **************************************************************/

namespace {

using MinimaxHeap = ACORN::MinimaxHeap;
using Node = ACORN::Node;
using NeighNode = ACORN::NeighNode;

struct PathwiseSchedule {
    int fixed_width;
    int init_width;
    int max_width;
    int growth_interval;
    bool staged;
};

struct FrontierNode {
    int id;
    float distance;
    int queue_size_after_pop;
};

PathwiseSchedule resolve_pathwise_schedule(
        const ACORN& hnsw,
        const SearchParametersACORN* params) {
    PathwiseSchedule schedule;
    schedule.fixed_width =
            params ? std::max(1, params->pathwise_width)
                   : std::max(1, hnsw.pathwise_width);
    schedule.init_width =
            params ? std::max(1, params->pathwise_init_width)
                   : std::max(1, hnsw.pathwise_init_width);
    schedule.max_width =
            params ? std::max(1, params->pathwise_max_width)
                   : std::max(1, hnsw.pathwise_max_width);
    schedule.growth_interval =
            params ? std::max(1, params->pathwise_growth_interval)
                   : std::max(1, hnsw.pathwise_growth_interval);
    schedule.staged = params ? params->pathwise_staged : hnsw.pathwise_staged;

    if (!schedule.staged) {
        schedule.init_width = schedule.fixed_width;
        schedule.max_width = schedule.fixed_width;
        return schedule;
    }

    schedule.max_width = std::max(schedule.init_width, schedule.max_width);
    return schedule;
}

int pathwise_width_for_step(const PathwiseSchedule& schedule, int step) {
    if (!schedule.staged) {
        return schedule.fixed_width;
    }

    int width = schedule.init_width;
    int phase = step / schedule.growth_interval;
    while (phase-- > 0 && width < schedule.max_width) {
        if (width > schedule.max_width / 2) {
            width = schedule.max_width;
        } else {
            width *= 2;
        }
    }
    return std::min(width, schedule.max_width);
}

template <typename ConsumeFrontier, typename OnDistanceStop>
void run_pathwise_frontier_loop(
        MinimaxHeap& candidates,
        const PathwiseSchedule& pathwise,
        bool do_dis_check,
        int efSearch,
        int& nstep,
        bool& should_stop,
        ConsumeFrontier&& consume_frontier,
        OnDistanceStop&& on_distance_stop) {
    while (candidates.size() > 0 && !should_stop) {
        int frontier_capacity = pathwise_width_for_step(pathwise, nstep);
        if (!do_dis_check) {
            frontier_capacity =
                    std::min(frontier_capacity, std::max(0, efSearch - nstep + 1));
        }
        if (frontier_capacity <= 0) {
            break;
        }

        std::vector<FrontierNode> frontier;
        frontier.reserve(frontier_capacity);

        while ((int)frontier.size() < frontier_capacity && candidates.size() > 0) {
            float d0 = 0;
            int v0 = candidates.pop_min(&d0);

            if (do_dis_check) {
                int n_dis_below = candidates.count_below(d0);
                if (n_dis_below >= efSearch) {
                    on_distance_stop(d0, n_dis_below);
                    should_stop = true;
                    break;
                }
            }

            frontier.push_back({v0, d0, candidates.size()});
        }

        if (frontier.empty()) {
            continue;
        }

        nstep += consume_frontier(frontier);
        if (!do_dis_check && nstep > efSearch) {
            should_stop = true;
        }
    }
}

/** Do a BFS on the candidates list */
// this is called in search and search_from_level_0
int search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr,
        std::vector<DistanceComputer*>* dc_pool = nullptr) {
    debug("%s\n", "reached");
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    PathwiseSchedule pathwise = resolve_pathwise_schedule(hnsw, params);
    bool reduced_sync = params ? params->reduced_sync : hnsw.reduced_sync;
    const IDSelector* sel = params ? params->sel : nullptr;

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    struct FrontierExpansion {
        int v0;
        std::vector<int> discovered;
    };

    auto direct_expand_candidate = [&](int v0) {
        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);
        size_t degree = end - begin;

        bool use_edgewise = dc_pool && dc_pool->size() > 1 &&
                degree >= dc_pool->size() * 2;

#if defined(_OPENMP)
        if (use_edgewise) {
            int worker_count = static_cast<int>(dc_pool->size());
            std::vector<std::vector<std::pair<int, float>>> local_candidates(
                    worker_count);
            std::vector<int> local_ndis(worker_count, 0);

#pragma omp parallel num_threads(worker_count)
            {
                int tid = omp_get_thread_num();
                DistanceComputer& worker = *(*dc_pool)[tid];
                auto& local = local_candidates[tid];
                int local_count = 0;

#pragma omp for schedule(static)
                for (int offset = 0; offset < (int)degree; ++offset) {
                    size_t j = begin + offset;
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0) {
                        continue;
                    }
                    bool inserted = false;
                    if (!vt.get(v1)) {
#pragma omp critical(acorn_edgewise_visit)
                        {
                            if (!vt.get(v1)) {
                                vt.set(v1);
                                inserted = true;
                            }
                        }
                    }
                    if (!inserted) {
                        continue;
                    }
                    local_count++;
                    float d = worker(v1);
                    local.emplace_back(v1, d);
                }

                local_ndis[tid] = local_count;
            }

            for (int tid = 0; tid < worker_count; ++tid) {
                ndis += local_ndis[tid];
                for (auto& entry : local_candidates[tid]) {
                    int v1 = entry.first;
                    float d = entry.second;
                    if (!sel || sel->is_member(v1)) {
                        if (nres < k) {
                            faiss::maxheap_push(++nres, D, I, d, v1);
                        } else if (d < D[0]) {
                            faiss::maxheap_replace_top(nres, D, I, d, v1);
                        }
                    }
                    candidates.push(v1, d);
                }
            }
            return;
        }
#endif

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (!sel || sel->is_member(v1)) {
                if (nres < k) {
                    faiss::maxheap_push(++nres, D, I, d, v1);
                } else if (d < D[0]) {
                    faiss::maxheap_replace_top(nres, D, I, d, v1);
                }
            }
            candidates.push(v1, d);
        }
    };

    int iqan_total_threads = reduced_sync && dc_pool
            ? std::min<int>(
                      std::max(1, pathwise.fixed_width),
                      static_cast<int>(dc_pool->size()))
            : 0;
    int iqan_workers = std::max(0, iqan_total_threads - 1);
    int iqan_local_queue_capacity =
            params ? params->iqan_local_queue_capacity : 0;
    if (iqan_local_queue_capacity <= 0) {
        iqan_local_queue_capacity = efSearch;
    }
    int iqan_seq_iterations = params ? params->iqan_seq_iterations : 1;
    iqan_seq_iterations = std::max(1, iqan_seq_iterations);

    if (level == 0 && iqan_total_threads >= 2) {
        struct IQANCandidate {
            int id;
            float distance;
            bool checked;

            bool operator<(const IQANCandidate& other) const {
                if (distance != other.distance) {
                    return distance < other.distance;
                }
                return id < other.id;
            }
        };

        struct QueueInsertResult {
            int index;
            bool inserted;
        };

        const int master_queue_capacity = std::max(k, efSearch);
        std::vector<IQANCandidate> master_queue;
        master_queue.reserve(master_queue_capacity);
        for (int i = 0; i < candidates.size(); ++i) {
            idx_t id = candidates.ids[i];
            if (id < 0) {
                continue;
            }
            float distance = candidates.dis[i];
            master_queue.push_back({static_cast<int>(id), distance, false});
        }
        std::sort(master_queue.begin(), master_queue.end());
        if ((int)master_queue.size() > master_queue_capacity) {
            master_queue.resize(master_queue_capacity);
        }

        std::vector<std::vector<IQANCandidate>> local_queues(iqan_workers);
        for (std::vector<IQANCandidate>& local_queue : local_queues) {
            local_queue.reserve(iqan_local_queue_capacity);
        }

        auto sort_and_dedup_queue = [&](std::vector<IQANCandidate>& queue) {
            if (queue.empty()) {
                return;
            }
            std::sort(queue.begin(), queue.end());
            size_t out = 0;
            for (size_t idx = 0; idx < queue.size(); ++idx) {
                if (out == 0 || queue[idx].id != queue[out - 1].id) {
                    queue[out++] = queue[idx];
                } else if (!queue[idx].checked && queue[out - 1].checked) {
                    queue[out - 1].checked = false;
                }
            }
            queue.resize(out);
        };

        auto merge_sorted_into_queue =
                [&](std::vector<IQANCandidate>& queue,
                    int capacity,
                    std::vector<IQANCandidate>& incoming) {
                    if (incoming.empty()) {
                        return capacity;
                    }
                    sort_and_dedup_queue(incoming);
                    std::vector<IQANCandidate> merged;
                    merged.reserve(std::min<int>(
                            capacity,
                            static_cast<int>(queue.size() + incoming.size())));
                    size_t q_idx = 0;
                    size_t in_idx = 0;
                    int best_insert = capacity;
                    while ((q_idx < queue.size() || in_idx < incoming.size()) &&
                           (int)merged.size() < capacity) {
                        bool take_incoming = false;
                        bool duplicate = false;
                        if (q_idx >= queue.size()) {
                            take_incoming = true;
                        } else if (in_idx >= incoming.size()) {
                            take_incoming = false;
                        } else if (incoming[in_idx] < queue[q_idx]) {
                            take_incoming = true;
                        } else if (queue[q_idx] < incoming[in_idx]) {
                            take_incoming = false;
                        } else {
                            duplicate = true;
                        }

                        if (duplicate) {
                            IQANCandidate merged_candidate = queue[q_idx];
                            if (!incoming[in_idx].checked &&
                                merged_candidate.checked) {
                                merged_candidate.checked = false;
                            }
                            merged.push_back(merged_candidate);
                            ++q_idx;
                            ++in_idx;
                            continue;
                        }

                        if (take_incoming) {
                            best_insert = std::min<int>(
                                    best_insert,
                                    static_cast<int>(merged.size()));
                            merged.push_back(incoming[in_idx++]);
                        } else {
                            merged.push_back(queue[q_idx++]);
                        }
                    }
                    queue.swap(merged);
                    incoming.clear();
                    return best_insert;
                };

        auto merge_local_queue_into_master =
                [&](std::vector<IQANCandidate>& local_queue) {
                    return merge_sorted_into_queue(
                            master_queue,
                            master_queue_capacity,
                            local_queue);
                };

        auto expand_candidate_into_queue =
                [&](DistanceComputer& worker,
                    int v0,
                    std::vector<IQANCandidate>& queue,
                    int queue_capacity,
                    int& local_ndis) {
                    std::vector<IQANCandidate> discovered;
                    size_t begin, end;
                    hnsw.neighbor_range(v0, level, &begin, &end);
                    for (size_t j = begin; j < end; ++j) {
                        int v1 = hnsw.neighbors[j];
                        if (v1 < 0) {
                            break;
                        }
                        if (!vt.try_set(v1)) {
                            continue;
                        }
                        ++local_ndis;
                        float distance = worker(v1);
                        discovered.push_back({v1, distance, false});
                    }
                    return merge_sorted_into_queue(queue, queue_capacity, discovered);
                };

        size_t k_master = 0;
        auto advance_master_cursor = [&]() {
            while (k_master < master_queue.size() && master_queue[k_master].checked) {
                ++k_master;
            }
        };

        auto should_stop_iqan = [&]() {
            advance_master_cursor();
            return k_master >= master_queue.size();
        };

        DistanceComputer* master_worker = &qdis;

        auto pick_top_master_to_workers = [&]() {
            bool assigned = false;
            int dest_queue = 0;
            for (size_t idx = k_master; idx < master_queue.size(); ++idx) {
                IQANCandidate& candidate = master_queue[idx];
                if (candidate.checked) {
                    continue;
                }
                local_queues[dest_queue].push_back(candidate);
                candidate.checked = true;
                assigned = true;
                if ((int)local_queues[dest_queue].size() ==
                    iqan_local_queue_capacity) {
                    break;
                }
                ++dest_queue;
                if (dest_queue == iqan_workers) {
                    dest_queue = 0;
                }
            }
            return assigned;
        };

        const int seq_iter_bound = 1;
        int seq_iter = 0;
        while (seq_iter < seq_iter_bound && !should_stop_iqan()) {
            IQANCandidate& candidate = master_queue[k_master];
            candidate.checked = true;
            int local_ndis = 0;
            int insert_index = expand_candidate_into_queue(
                    *master_worker,
                    candidate.id,
                    master_queue,
                    master_queue_capacity,
                    local_ndis);
            ndis += local_ndis;
            if (insert_index <= (int)k_master) {
                k_master = insert_index;
            } else {
                ++k_master;
            }
            ++seq_iter;
        }

        while (!should_stop_iqan()) {
            if (!pick_top_master_to_workers()) {
                break;
            }

            std::vector<int> local_ndis(iqan_total_threads, 0);
            size_t next_master = k_master;
            const int subsearch_iterations = iqan_seq_iterations;

#if defined(_OPENMP)
#pragma omp parallel num_threads(iqan_total_threads) shared(next_master, local_ndis)
            {
                int tid = omp_get_thread_num();
                if (tid == iqan_workers) {
                    size_t k_local = k_master;
                    int worker_iter = 0;
                    while (worker_iter < subsearch_iterations &&
                           k_local < master_queue.size()) {
                        IQANCandidate& candidate = master_queue[k_local];
                        if (candidate.checked) {
                            ++k_local;
                            continue;
                        }
                        candidate.checked = true;
                        ++worker_iter;
                        int insert_index = expand_candidate_into_queue(
                                *master_worker,
                                candidate.id,
                                master_queue,
                                master_queue_capacity,
                                local_ndis[tid]);
                        if (insert_index <= (int)k_local) {
                            k_local = static_cast<size_t>(std::max(0, insert_index));
                        } else {
                            ++k_local;
                        }
                    }
                    next_master = k_local;
                } else {
                    DistanceComputer& worker = *(*dc_pool)[tid + 1];
                    std::vector<IQANCandidate>& local_queue = local_queues[tid];
                    size_t k_local = 0;
                    int worker_iter = 0;
                    while (worker_iter < subsearch_iterations &&
                           k_local < local_queue.size()) {
                        IQANCandidate& candidate = local_queue[k_local];
                        if (candidate.checked) {
                            ++k_local;
                            continue;
                        }
                        candidate.checked = true;
                        ++worker_iter;
                        int insert_index = expand_candidate_into_queue(
                                worker,
                                candidate.id,
                                local_queue,
                                iqan_local_queue_capacity,
                                local_ndis[tid]);
                        if (insert_index <= (int)k_local) {
                            k_local = static_cast<size_t>(std::max(0, insert_index));
                        } else {
                            ++k_local;
                        }
                    }
                }
            }
#else
            for (int worker_id = 0; worker_id < iqan_workers; ++worker_id) {
                DistanceComputer& worker = *(*dc_pool)[worker_id + 1];
                std::vector<IQANCandidate>& local_queue = local_queues[worker_id];
                size_t k_local = 0;
                int worker_iter = 0;
                while (worker_iter < subsearch_iterations &&
                       k_local < local_queue.size()) {
                    IQANCandidate& candidate = local_queue[k_local];
                    if (candidate.checked) {
                        ++k_local;
                        continue;
                    }
                    candidate.checked = true;
                    ++worker_iter;
                    int insert_index = expand_candidate_into_queue(
                            worker,
                            candidate.id,
                            local_queue,
                            iqan_local_queue_capacity,
                            local_ndis[worker_id]);
                    if (insert_index <= (int)k_local) {
                        k_local = static_cast<size_t>(std::max(0, insert_index));
                    } else {
                        ++k_local;
                    }
                }
            }
            {
                size_t k_local = k_master;
                int worker_iter = 0;
                while (worker_iter < subsearch_iterations &&
                       k_local < master_queue.size()) {
                    IQANCandidate& candidate = master_queue[k_local];
                    if (candidate.checked) {
                        ++k_local;
                        continue;
                    }
                    candidate.checked = true;
                    ++worker_iter;
                    int insert_index = expand_candidate_into_queue(
                            *master_worker,
                            candidate.id,
                            master_queue,
                            master_queue_capacity,
                            local_ndis[iqan_workers]);
                    if (insert_index <= (int)k_local) {
                        k_local = static_cast<size_t>(std::max(0, insert_index));
                    } else {
                        ++k_local;
                    }
                }
                next_master = k_local;
            }
#endif

            for (int value : local_ndis) {
                ndis += value;
            }

            int best_insert = master_queue_capacity;
            for (std::vector<IQANCandidate>& local_queue : local_queues) {
                if (local_queue.empty()) {
                    continue;
                }
                best_insert =
                        std::min(best_insert, merge_local_queue_into_master(local_queue));
            }

            k_master = next_master;
            if (best_insert <= (int)k_master) {
                k_master = static_cast<size_t>(std::max(0, best_insert));
            } else {
                advance_master_cursor();
            }
        }

        if (level == 0) {
            stats.n1++;
            if (should_stop_iqan()) {
                stats.n2++;
            }
            stats.n3 += ndis;
        }

        nres = 0;
        for (const IQANCandidate& candidate : master_queue) {
            if (!sel || sel->is_member(candidate.id)) {
                if (nres < k) {
                    faiss::maxheap_push(++nres, D, I, candidate.distance, candidate.id);
                } else if (candidate.distance < D[0]) {
                    faiss::maxheap_replace_top(nres, D, I, candidate.distance, candidate.id);
                }
                if (nres >= k) {
                    break;
                }
            }
        }

        return nres;
    }

    auto collect_expansion = [&](const FrontierNode& node) {
        int v0 = node.id;
        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);
        FrontierExpansion expansion{v0, {}};
        expansion.discovered.reserve(end - begin);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            if (!vt.get(v1)) {
                expansion.discovered.push_back(v1);
            }
        }

        return expansion;
    };

    auto merge_frontier_round = [&](const std::vector<FrontierExpansion>& frontier) {
        size_t reserve = 0;
        for (const FrontierExpansion& expansion : frontier) {
            reserve += expansion.discovered.size();
        }

        std::unordered_set<int> round_seen;
        round_seen.reserve(std::max<size_t>(8, reserve));

        std::vector<int> merged_ids;
        merged_ids.reserve(reserve);
        for (const FrontierExpansion& expansion : frontier) {
            for (int id : expansion.discovered) {
                if (vt.get(id)) {
                    continue;
                }
                if (!round_seen.insert(id).second) {
                    continue;
                }
                vt.set(id);
                merged_ids.push_back(id);
            }
        }
        if (merged_ids.empty()) {
            return;
        }

#if defined(_OPENMP)
        int worker_count = dc_pool
                ? std::min<int>(
                          static_cast<int>(dc_pool->size()),
                          static_cast<int>(merged_ids.size()))
                : 0;
        if (worker_count >= 2) {
            std::vector<float> distances(merged_ids.size());

#pragma omp parallel num_threads(worker_count)
            {
                int tid = omp_get_thread_num();
                DistanceComputer& worker = *(*dc_pool)[tid];

#pragma omp for schedule(static)
                for (int idx = 0; idx < (int)merged_ids.size(); ++idx) {
                    distances[idx] = worker(merged_ids[idx]);
                }
            }

            ndis += merged_ids.size();
            for (size_t idx = 0; idx < merged_ids.size(); ++idx) {
                int id = merged_ids[idx];
                float d = distances[idx];
                if (!sel || sel->is_member(id)) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, id);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, id);
                    }
                }
                candidates.push(id, d);
            }
            return;
        }
#endif

        for (int id : merged_ids) {
            ndis++;
            float d = qdis(id);
            if (!sel || sel->is_member(id)) {
                if (nres < k) {
                    faiss::maxheap_push(++nres, D, I, d, id);
                } else if (d < D[0]) {
                    faiss::maxheap_replace_top(nres, D, I, d, id);
                }
            }
            candidates.push(id, d);
        }
    };

    int nstep = 0;
    bool should_stop = false;
    run_pathwise_frontier_loop(
            candidates,
            pathwise,
            do_dis_check,
            efSearch,
            nstep,
            should_stop,
            [&](const std::vector<FrontierNode>& frontier) {
                if (!reduced_sync) {
                    for (const FrontierNode& node : frontier) {
                        direct_expand_candidate(node.id);
                    }
                    return static_cast<int>(frontier.size());
                }
                std::vector<FrontierExpansion> expansions(frontier.size());

#if defined(_OPENMP)
                int worker_count = dc_pool
                        ? std::min<int>(
                                  static_cast<int>(dc_pool->size()),
                                  static_cast<int>(frontier.size()))
                        : 0;
                if (worker_count >= 2) {
#pragma omp parallel for schedule(static) num_threads(worker_count)
                    for (int idx = 0; idx < (int)frontier.size(); ++idx) {
                        expansions[idx] = collect_expansion(frontier[idx]);
                    }
                } else
#endif
                {
                    for (size_t idx = 0; idx < frontier.size(); ++idx) {
                        expansions[idx] = collect_expansion(frontier[idx]);
                    }
                }

                merge_frontier_round(expansions);
                return static_cast<int>(frontier.size());
            },
            [&](float, int) {});

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }

    return nres;
}

// has a filter arg for hybrid search, this only gets called on level 0
int hybrid_search_from_candidates(
        const ACORN& hnsw,
        DistanceComputer& qdis,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        ACORNStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersACORN* params = nullptr,
        std::vector<DistanceComputer*>* dc_pool = nullptr) {
    // debug("%s\n", "reached");
    // printf("----hybrid_search_from_candidates called with filter: %d, k: %d, op: %d, regex: %s\n", filter, k, op, regex.c_str());
    // debug_search("----hybrid_search_from_candidates called with filter: %d, k: %d\n", filter, k);
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    PathwiseSchedule pathwise = resolve_pathwise_schedule(hnsw, params);
    bool reduced_sync = params ? params->reduced_sync : hnsw.reduced_sync;
    const IDSelector* sel = params ? params->sel : nullptr;

    struct DiscoveredNode {
        int id;
        const char* hop;
    };

    struct FrontierExpansion {
        int v0;
        bool trace_expansion;
        std::vector<DiscoveredNode> discovered;
    };
    std::vector<FrontierExpansion> frontier_single(1);

    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, d, v1);
            }
        }
        vt.set(v1);
    }

    int iqan_total_threads = reduced_sync && dc_pool
            ? std::min<int>(
                      std::max(1, pathwise.fixed_width),
                      static_cast<int>(dc_pool->size()))
            : 0;
    int iqan_workers = std::max(0, iqan_total_threads - 1);
    int iqan_local_queue_capacity =
            params ? params->iqan_local_queue_capacity : 0;
    if (iqan_local_queue_capacity <= 0) {
        iqan_local_queue_capacity = efSearch;
    }
    int iqan_seq_iterations = params ? params->iqan_seq_iterations : 1;
    iqan_seq_iterations = std::max(1, iqan_seq_iterations);

    if (level == 0 && iqan_total_threads >= 2) {
        static std::atomic<int> iqan_entry_count{0};
        int entry_id = iqan_entry_count.fetch_add(1);
        if (entry_id < 3) {
            printf("[PATHWISE] hybrid_search_from_candidates: ENTERED iQAN path "
                   "(threads=%d, workers=%d, efSearch=%d, seq_iters=%d)\n",
                   iqan_total_threads, iqan_workers, efSearch, iqan_seq_iterations);
            fflush(stdout);
        }

        // ====== iQAN-style in-place sorted queue implementation ======
        struct Candidate {
            int id;
            float distance;
            bool checked;
            bool operator<(const Candidate& o) const {
                return distance < o.distance ||
                       (distance == o.distance && id < o.id);
            }
        };

        const int L = std::max(k, efSearch);
        const int local_cap = iqan_local_queue_capacity;
        const int total_slots = iqan_workers * local_cap + L;
        std::vector<Candidate> set_L(total_slots);
        std::vector<int> queue_starts(iqan_total_threads);
        std::vector<int> queue_sizes(iqan_total_threads, 0);
        for (int i = 0; i < iqan_workers; ++i) {
            queue_starts[i] = i * local_cap;
        }
        const int master_start = iqan_workers * local_cap;
        queue_starts[iqan_workers] = master_start;

        // --- add_into_queue: binary search + memmove, returns insert pos ---
        auto add_into_queue = [](std::vector<Candidate>& q, int q_start,
                                 int& q_size, int q_cap,
                                 const Candidate& cand) -> int {
            if (q_size == 0) {
                q[q_start] = cand;
                ++q_size;
                return 0;
            }
            auto it = std::lower_bound(
                    q.begin() + q_start,
                    q.begin() + q_start + q_size, cand);
            int insert_loc = static_cast<int>(it - (q.begin() + q_start));
            if (insert_loc < q_size) {
                if (it->id == cand.id) return q_cap; // duplicate
                if (q_size >= q_cap) --q_size; // full: drop last
            } else {
                if (q_size < q_cap) {
                    q[q_start + q_size] = cand;
                    ++q_size;
                    return q_size - 1;
                }
                return q_cap; // worse than all, queue full
            }
            int abs_loc = q_start + insert_loc;
            std::memmove(&q[abs_loc + 1], &q[abs_loc],
                         (q_size - insert_loc) * sizeof(Candidate));
            q[abs_loc] = cand;
            ++q_size;
            return insert_loc;
        };

        // --- merge_into_master: merge queue2 into fixed-size master (size=L) ---
        // Follows iQAN merge_two_queues_into_1st_queue_seq_fixed exactly.
        auto merge_into_master = [&](int q2_start, int& q2_size) -> int {
            if (q2_size == 0) return L;
            const int q1_start = master_start;
            const int q1_size = L; // master is always full (padded with FLT_MAX)
            auto it = std::lower_bound(
                    set_L.begin() + q1_start,
                    set_L.begin() + q1_start + q1_size,
                    set_L[q2_start]);
            int insert_idx = static_cast<int>(it - (set_L.begin() + q1_start));
            if (insert_idx >= q1_size) { q2_size = 0; return q1_size; }
            // Insert first element
            if (set_L[q2_start].id != it->id) {
                int a = q1_start + insert_idx;
                std::memmove(&set_L[a + 1], &set_L[a],
                             (q1_size - insert_idx - 1) * sizeof(Candidate));
                set_L[a] = set_L[q2_start];
            } else if (!set_L[q2_start].checked && it->checked) {
                it->checked = false;
            }
            if (q2_size == 1) { q2_size = 0; return insert_idx; }
            // Merge remaining elements
            int qi1 = q1_start + insert_idx + 1;
            int qi2 = q2_start + 1;
            int qi1_end = q1_start + q1_size;
            int qi2_end = q2_start + q2_size;
            for (int ins_i = insert_idx + 1; ins_i < q1_size; ++ins_i) {
                if (qi1 >= qi1_end || qi2 >= qi2_end) break;
                if (set_L[qi1] < set_L[qi2]) {
                    ++qi1;
                } else if (set_L[qi2] < set_L[qi1]) {
                    int a = q1_start + ins_i;
                    std::memmove(&set_L[a + 1], &set_L[a],
                                 (q1_size - ins_i - 1) * sizeof(Candidate));
                    set_L[a] = set_L[qi2++];
                    ++qi1;
                } else {
                    if (!set_L[qi2].checked && set_L[qi1].checked)
                        set_L[qi1].checked = false;
                    ++qi2; ++qi1;
                }
            }
            q2_size = 0;
            return insert_idx;
        };

        // --- Initialize master queue: fill L slots, pad with sentinel ---
        int& master_size = queue_sizes[iqan_workers];
        for (int i = 0; i < candidates.size(); ++i) {
            idx_t id = candidates.ids[i];
            if (id < 0) continue;
            Candidate c{static_cast<int>(id), candidates.dis[i], false};
            add_into_queue(set_L, master_start, master_size, L, c);
        }
        // Pad remaining slots with sentinel so master is always size L
        for (int i = master_size; i < L; ++i) {
            set_L[master_start + i] = {-1, std::numeric_limits<float>::max(), false};
        }
        master_size = L;

        // --- expand_one: expand neighbors in-place, no dist_bound when queue not "real-full" ---
        auto expand_one = [&](DistanceComputer& worker, int v0,
                              int q_start, int& q_size, int q_cap,
                              float dist_bound,
                              int& local_ndis) -> int {
            int nk = q_cap;
            size_t begin, end;
            hnsw.neighbor_range(v0, level, &begin, &end);
            int num_found = 0;
            bool keep_expanding = true;
            for (size_t j = begin; j < end; ++j) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0) break;
                if (filter_map[v1]) {
                    ++num_found;
                    if (vt.try_set(v1)) {
                        ++local_ndis;
                        float d = worker(v1);
                        if (d <= dist_bound) {
                            Candidate c{v1, d, false};
                            int r = add_into_queue(set_L, q_start, q_size, q_cap, c);
                            if (r < nk) nk = r;
                        }
                    }
                    if (num_found >= hnsw.M * 2) { keep_expanding = false; break; }
                }
                if ((((int)(j - begin) >= hnsw.M_beta) && keep_expanding) ||
                    hnsw.gamma == 1) {
                    size_t b2, e2;
                    hnsw.neighbor_range(v1, level, &b2, &e2);
                    for (size_t j2 = b2; j2 < e2; ++j2) {
                        int v2 = hnsw.neighbors[j2];
                        if (v2 < 0) break;
                        if (!filter_map[v2]) continue;
                        ++num_found;
                        if (vt.try_set(v2)) {
                            ++local_ndis;
                            float d = worker(v2);
                            if (d <= dist_bound) {
                                Candidate c{v2, d, false};
                                int r = add_into_queue(set_L, q_start, q_size, q_cap, c);
                                if (r < nk) nk = r;
                            }
                        }
                        if (num_found >= hnsw.M * 2) { keep_expanding = false; break; }
                    }
                }
            }
            return nk;
        };

        // dist_bound: last REAL element of master queue (skip sentinels)
        auto get_dist_bound = [&]() -> float {
            return set_L[master_start + L - 1].distance;
        };

        int k_master = 0;
        auto advance_cursor = [&]() {
            while (k_master < L && set_L[master_start + k_master].checked)
                ++k_master;
        };
        auto stopped = [&]() {
            advance_cursor();
            return k_master >= L;
        };

        auto pick_to_workers = [&]() -> bool {
            bool assigned = false;
            int dest = 0;
            for (int idx = k_master; idx < L; ++idx) {
                Candidate& c = set_L[master_start + idx];
                if (c.checked || c.id < 0) continue;
                int& ws = queue_sizes[dest];
                if (ws < local_cap) {
                    set_L[queue_starts[dest] + ws] = c;
                    ++ws;
                }
                c.checked = true;
                assigned = true;
                if (ws >= local_cap) break;
                ++dest;
                if (dest >= iqan_workers) dest = 0;
            }
            return assigned;
        };

        // Sequential start (1 iteration)
        {
            int local_ndis = 0;
            if (!stopped()) {
                Candidate& c = set_L[master_start + k_master];
                c.checked = true;
                int r = expand_one(qdis, c.id, master_start, master_size,
                                   L, get_dist_bound(), local_ndis);
                ndis += local_ndis;
                if (r <= k_master) k_master = r; else ++k_master;
            }
        }

        // Parallel phase
        while (!stopped()) {
            if (!pick_to_workers()) break;

            std::vector<int> local_ndis(iqan_total_threads, 0);
            int next_master = k_master;
            const int X = iqan_seq_iterations;
            float db = get_dist_bound();

#if defined(_OPENMP)
#pragma omp parallel num_threads(iqan_total_threads) shared(next_master, local_ndis, db)
            {
                int tid = omp_get_thread_num();
                int qs = queue_starts[tid];
                int& qsz = queue_sizes[tid];
                int qcap = (tid == iqan_workers) ? L : local_cap;
                int k_uc = (tid == iqan_workers) ? k_master : 0;
                DistanceComputer& w = (tid == iqan_workers)
                        ? qdis : *(*dc_pool)[tid + 1];
                int iters = 0;
                while (iters < X && k_uc < qsz) {
                    Candidate& c = set_L[qs + k_uc];
                    if (c.checked || c.id < 0) { ++k_uc; continue; }
                    c.checked = true;
                    ++iters;
                    int r = expand_one(w, c.id, qs, qsz, qcap, db,
                                       local_ndis[tid]);
                    if (r <= k_uc) k_uc = r; else ++k_uc;
                }
                if (tid == iqan_workers) next_master = k_uc;
            }
#else
            for (int wid = 0; wid < iqan_workers; ++wid) {
                int qs = queue_starts[wid];
                int& qsz = queue_sizes[wid];
                DistanceComputer& w = *(*dc_pool)[wid + 1];
                int k_uc = 0, iters = 0;
                while (iters < X && k_uc < qsz) {
                    Candidate& c = set_L[qs + k_uc];
                    if (c.checked) { ++k_uc; continue; }
                    c.checked = true; ++iters;
                    int r = expand_one(w, c.id, qs, qsz, local_cap, db,
                                       local_ndis[wid]);
                    if (r <= k_uc) k_uc = r; else ++k_uc;
                }
            }
            {
                int k_uc = k_master, iters = 0;
                while (iters < X && k_uc < master_size) {
                    Candidate& c = set_L[master_start + k_uc];
                    if (c.checked || c.id < 0) { ++k_uc; continue; }
                    c.checked = true; ++iters;
                    int r = expand_one(qdis, c.id, master_start, master_size,
                                       L, db, local_ndis[iqan_workers]);
                    if (r <= k_uc) k_uc = r; else ++k_uc;
                }
                next_master = k_uc;
            }
#endif
            for (int v : local_ndis) ndis += v;

            // Merge all worker queues into master (master is always size L)
            int best_insert = L;
            for (int wid = 0; wid < iqan_workers; ++wid) {
                int& ws = queue_sizes[wid];
                if (ws == 0) continue;
                int r = merge_into_master(queue_starts[wid], ws);
                if (r < best_insert) best_insert = r;
            }

            k_master = next_master;
            if (best_insert <= k_master) {
                k_master = best_insert;
            } else {
                advance_cursor();
            }
        }

        if (level == 0) {
            stats.n1++;
            if (stopped()) stats.n2++;
            stats.n3 += ndis;
        }

        nres = 0;
        for (int i = 0; i < L; ++i) {
            const Candidate& c = set_L[master_start + i];
            if (c.id < 0) continue; // skip sentinel
            if (!sel || sel->is_member(c.id)) {
                if (nres < k) {
                    faiss::maxheap_push(++nres, D, I, c.distance, c.id);
                } else if (c.distance < D[0]) {
                    faiss::maxheap_replace_top(nres, D, I, c.distance, c.id);
                }
                if (nres >= k) break;
            }
        }

        return nres;
    }

    {
        static std::atomic<int> fallback_count{0};
        int fb_id = fallback_count.fetch_add(1);
        if (fb_id < 3) {
            printf("[PATHWISE] hybrid_search_from_candidates: FALLBACK to frontier loop "
                   "(level=%d, iqan_threads=%d, reduced_sync=%d, dc_pool=%s, pathwise_width=%d)\n",
                   level, iqan_total_threads, reduced_sync ? 1 : 0,
                   dc_pool ? "non-null" : "NULL",
                   pathwise.fixed_width);
            fflush(stdout);
        }
    }

    auto merge_frontier_round = [&](const std::vector<FrontierExpansion>& frontier) {
        std::vector<DiscoveredNode> round_nodes;
        for (const FrontierExpansion& expansion : frontier) {
            round_nodes.insert(
                    round_nodes.end(),
                    expansion.discovered.begin(),
                    expansion.discovered.end());
        }
        if (round_nodes.empty()) {
            return;
        }

        std::vector<DiscoveredNode> merged_nodes;
        merged_nodes.reserve(round_nodes.size());
        std::unordered_set<int> round_seen;
        round_seen.reserve(std::max<size_t>(8, round_nodes.size()));
        for (const DiscoveredNode& node : round_nodes) {
            int id = node.id;
            if (vt.get(id)) {
                continue;
            }
            if (!round_seen.insert(id).second) {
                continue;
            }
            vt.set(id);
            merged_nodes.push_back(node);
        }
        if (merged_nodes.empty()) {
            return;
        }

        std::vector<int> ids;
        ids.reserve(merged_nodes.size());
        for (const DiscoveredNode& node : merged_nodes) {
            ids.push_back(node.id);
        }

#if defined(_OPENMP)
        int worker_count = dc_pool
                ? std::min<int>(
                          static_cast<int>(dc_pool->size()),
                          static_cast<int>(ids.size()))
                : 0;
        bool use_edgewise = worker_count >= 2;
        if (use_edgewise) {
            std::vector<float> distances(ids.size());

            if (hybridDebugEnabled) {
                hybrid_debug_log(
                        "  [hybrid-debug q=%d] round batch=%zu edgewise=1 workers=%d\n",
                        hybridDebugActiveQuery,
                        ids.size(),
                        worker_count);
            }

#pragma omp parallel num_threads(worker_count)
            {
                int tid = omp_get_thread_num();
                DistanceComputer& worker = *(*dc_pool)[tid];

#pragma omp for schedule(static)
                for (int idx = 0; idx < (int)ids.size(); ++idx) {
                    distances[idx] = worker(ids[idx]);
                }
            }

            ndis += ids.size();
            for (size_t idx = 0; idx < ids.size(); ++idx) {
                int id = ids[idx];
                float distance = distances[idx];
                const char* hop = merged_nodes[idx].hop;
                if (!sel || sel->is_member(id)) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, distance, id);
                    } else if (distance < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, distance, id);
                    }
                }
                candidates.push(id, distance);
                if (hybridDebugEnabled) {
                    hybrid_debug_log(
                            "    [hybrid-debug q=%d] %s push id=%d dis=%g nres=%d cand=%d\n",
                            hybridDebugActiveQuery,
                            hop,
                            id,
                            distance,
                            nres,
                            candidates.size());
                }
            }
            return;
        }
#endif

        if (hybridDebugEnabled) {
            hybrid_debug_log(
                    "  [hybrid-debug q=%d] round batch=%zu edgewise=0\n",
                    hybridDebugActiveQuery,
                    ids.size());
        }

        ndis += ids.size();
        for (size_t idx = 0; idx < ids.size(); ++idx) {
            int id = ids[idx];
            float distance = qdis(id);
            const char* hop = merged_nodes[idx].hop;
            if (!sel || sel->is_member(id)) {
                if (nres < k) {
                    faiss::maxheap_push(++nres, D, I, distance, id);
                } else if (distance < D[0]) {
                    faiss::maxheap_replace_top(nres, D, I, distance, id);
                }
            }
            candidates.push(id, distance);
            if (hybridDebugEnabled) {
                hybrid_debug_log(
                        "    [hybrid-debug q=%d] %s push id=%d dis=%g nres=%d cand=%d\n",
                        hybridDebugActiveQuery,
                        hop,
                        id,
                        distance,
                        nres,
                        candidates.size());
            }
        }
    };

    auto collect_expansion = [&](const FrontierNode& node) {
        int v0 = node.id;
        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        int num_found = 0;
        bool keep_expanding = true;
        bool trace_expansion = hybridDebugEnabled &&
                hybridDebugExpansionCount < hybridDebugExpansionLimit;
        FrontierExpansion expansion{v0, trace_expansion, {}};

        if (trace_expansion) {
            hybrid_debug_log(
                    "[hybrid-debug q=%d] expand #%d v0=%d queue_before=%d nres=%d range=%zu\n",
                    hybridDebugActiveQuery,
                    hybridDebugExpansionCount,
                    v0,
                    node.queue_size_after_pop,
                    nres,
                    end - begin);
            hybridDebugExpansionCount++;
        }

        for (size_t j = begin; j < end; j++) {
            auto v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }

            if (filter_map[v1]) {
                num_found++;
            }

            if (vt.get(v1)) {
                if (trace_expansion) {
                    hybrid_debug_log(
                            "  [hybrid-debug q=%d] L1 skip id=%d filter=%d visited=1 num_found=%d\n",
                            hybridDebugActiveQuery,
                            v1,
                            filter_map[v1],
                            num_found);
                }
                continue;
            }

            if (filter_map[v1]) {
                expansion.discovered.push_back({v1, "L1"});
                if (trace_expansion) {
                    hybrid_debug_log(
                            "  [hybrid-debug q=%d] L1 collect id=%d num_found=%d\n",
                            hybridDebugActiveQuery,
                            v1,
                            num_found);
                }

                if (num_found >= hnsw.M * 2) {
                    keep_expanding = false;
                    if (trace_expansion) {
                        hybrid_debug_log(
                                "  [hybrid-debug q=%d] L1 stop num_found=%d threshold=%d\n",
                                hybridDebugActiveQuery,
                                num_found,
                                hnsw.M * 2);
                    }
                    break;
                }
            }

            if ((((int)(j - begin) >= hnsw.M_beta) && keep_expanding) ||
                hnsw.gamma == 1) {
                size_t begin2, end2;
                std::vector<int> second_hop_hits;
                hnsw.neighbor_range(v1, level, &begin2, &end2);
                if (trace_expansion) {
                    hybrid_debug_log(
                            "  [hybrid-debug q=%d] expand L2 from id=%d offset=%zu keep=%d\n",
                            hybridDebugActiveQuery,
                            v1,
                            j - begin,
                            keep_expanding ? 1 : 0);
                }
                for (size_t j2 = begin2; j2 < end2; j2 += 1) {
                    auto v2 = hnsw.neighbors[j2];

                    if (v2 < 0) {
                        break;
                    }

                    if (filter_map[v2]) {
                        num_found++;
                    } else {
                        continue;
                    }

                    if (vt.get(v2)) {
                        if (trace_expansion) {
                            hybrid_debug_log(
                                    "    [hybrid-debug q=%d] L2 skip id=%d visited=1 num_found=%d\n",
                                    hybridDebugActiveQuery,
                                    v2,
                                    num_found);
                        }
                        continue;
                    }

                    second_hop_hits.push_back(v2);
                    if (num_found >= hnsw.M * 2) {
                        keep_expanding = false;
                        if (trace_expansion) {
                            hybrid_debug_log(
                                    "    [hybrid-debug q=%d] L2 stop num_found=%d threshold=%d\n",
                                    hybridDebugActiveQuery,
                                    num_found,
                                    hnsw.M * 2);
                        }
                        break;
                    }
                }

                if (!second_hop_hits.empty()) {
                    for (int v2 : second_hop_hits) {
                        expansion.discovered.push_back({v2, "L2"});
                    }
                }
            }
        }

        return expansion;
    };

    int nstep = 0;
    bool should_stop = false;
    run_pathwise_frontier_loop(
            candidates,
            pathwise,
            do_dis_check,
            efSearch,
            nstep,
            should_stop,
            [&](const std::vector<FrontierNode>& frontier_nodes) {
                if (!reduced_sync) {
                    for (const FrontierNode& node : frontier_nodes) {
                        frontier_single[0] = collect_expansion(node);
                        merge_frontier_round(frontier_single);

                        if (hybridDebugEnabled) {
                            hybrid_debug_log(
                                    "[hybrid-debug q=%d] post-expand v0=%d queue=%d nres=%d ndis=%d\n",
                                    hybridDebugActiveQuery,
                                    frontier_single[0].v0,
                                    candidates.size(),
                                    nres,
                                    ndis);
                        }
                    }
                    return static_cast<int>(frontier_nodes.size());
                }
                std::vector<FrontierExpansion> frontier(frontier_nodes.size());

#if defined(_OPENMP)
                int worker_count = dc_pool
                        ? std::min<int>(
                                  static_cast<int>(dc_pool->size()),
                                  static_cast<int>(frontier_nodes.size()))
                        : 0;
                if (hybridDebugEnabled) {
                    worker_count = 0;
                }
                if (worker_count >= 2) {
#pragma omp parallel for schedule(static) num_threads(worker_count)
                    for (int idx = 0; idx < (int)frontier_nodes.size(); ++idx) {
                        frontier[idx] = collect_expansion(frontier_nodes[idx]);
                    }
                } else
#endif
                {
                    for (size_t idx = 0; idx < frontier_nodes.size(); ++idx) {
                        frontier[idx] = collect_expansion(frontier_nodes[idx]);
                    }
                }

                merge_frontier_round(frontier);

                for (const FrontierExpansion& expansion : frontier) {
                    if (hybridDebugEnabled) {
                        hybrid_debug_log(
                                "[hybrid-debug q=%d] post-expand v0=%d queue=%d nres=%d ndis=%d\n",
                                hybridDebugActiveQuery,
                                expansion.v0,
                                candidates.size(),
                                nres,
                                ndis);
                    }
                }

                return static_cast<int>(frontier.size());
            },
            [&](float d0, int n_dis_below) {
                if (hybridDebugEnabled) {
                    hybrid_debug_log(
                            "[hybrid-debug q=%d] stop by dis_check d0=%g below=%d ef=%d queue=%d nres=%d\n",
                            hybridDebugActiveQuery,
                            d0,
                            n_dis_below,
                            efSearch,
                            candidates.size(),
                            nres);
                }
            });

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.n3 += ndis;
    }


    return nres;
}








} // anonymous namespace

ACORNStats ACORN::search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        const SearchParametersACORN* params,
        std::vector<DistanceComputer*>* dc_pool) const {
    debug("%s\n", "reached");
    ACORNStats stats;
    if (entry_point == -1) {
        return stats;
    }
    if (upper_beam == 1) { // common branch
        debug("%s\n", "reached upper beam == 1");

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        for (int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }

        
        int ef = std::max(efSearch, k);
        if (search_bounded_queue) { // this is the most common branch
            debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);

            search_from_candidates(
                    *this,
                    qdis,
                    k,
                    I,
                    D,
                    candidates,
                    vt,
                    stats,
                    0,
                    0,
                    params,
                    dc_pool);
        } else {
            debug("%s\n", "reached search_bounded_queue == False");
            throw FaissException("UNIMPLEMENTED search unbounded queue");
            
        }

        vt.advance();

    } else {
        debug("%s\n", "reached upper beam != 1");

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = search_from_candidates(
                        *this,
                        qdis,
                        k,
                        I,
                        D,
                        candidates,
                        vt,
                        stats,
                        0,
                        0,
                        params,
                        dc_pool);
            } else {
                nres = search_from_candidates(
                        *this,
                        qdis,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level,
                        0,
                        params,
                        dc_pool);
            }
            vt.advance();
        }
    }

    return stats;
}


// hybrid search TODO
ACORNStats ACORN::hybrid_search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt,
        char* filter_map,
        // int filter,
        // Operation op,
        // std::string regex,
        const SearchParametersACORN* params,
        std::vector<DistanceComputer*>* dc_pool) const {
    debug("%s\n", "reached");
    // debug_search("Hybrid Search, params -- k: %d, filter: %d\n", k, filter);
    ACORNStats stats;
    PathwiseSchedule pathwise = resolve_pathwise_schedule(*this, params);
    if (entry_point == -1) {
        return stats;
    }

    static std::atomic<int> hs_dbg_count{0};
    int hs_dbg = hs_dbg_count.fetch_add(1);

    if (upper_beam == 1) { // common branch
        debug("%s\n", "reached upper beam == 1");

        hybridDebugActiveQuery = hybridDebugSearchCounter.fetch_add(1);
        hybridDebugEnabled = hybridDebugActiveQuery == hybridDebugQuery;
        hybridDebugExpansionCount = 0;
        if (hybridDebugEnabled) {
            hybrid_debug_log(
                    "[hybrid-debug q=%d] start k=%d efSearch=%d pathwise_width=%d staged=%d init=%d max=%d growth=%d gamma=%d M=%d M_beta=%d\n",
                    hybridDebugActiveQuery,
                    k,
                    efSearch,
                    pathwise.fixed_width,
                    pathwise.staged ? 1 : 0,
                    pathwise.init_width,
                    pathwise.max_width,
                    pathwise.growth_interval,
                    gamma,
                    M,
                    M_beta);
        }

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);
        if (hs_dbg < 1) { printf("[PATHWISE-DBG] hybrid_search: entry_point=%d d_nearest=%g max_level=%d dc_pool=%s\n", nearest, d_nearest, max_level, dc_pool ? "non-null" : "NULL"); fflush(stdout); }

        debug_search("-starting at ep: %d, d: %f, metadata: %d\n", nearest, d_nearest, metadata[nearest]);

        int ndis_upper = 0;
        for (int level = max_level; level >= 1; level--) {
            debug_search("-at level %d, searching for greedy nearest from current nearest: %d, dist: %f, metadata: %d\n", level, nearest, d_nearest, metadata[nearest]);
            ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter_map, level, nearest, d_nearest);
            // ndis_upper += hybrid_greedy_update_nearest(*this, qdis, filter, op, regex, level, nearest, d_nearest);
            debug_search("-at level %d, new nearest: %d, d: %f, metadata: %d\n", level, nearest, d_nearest, metadata[nearest]);


        }
        stats.n3 += ndis_upper;
        if (hs_dbg < 1) { printf("[PATHWISE-DBG] hybrid_search: upper levels done, nearest=%d, calling search_from_candidates\n", nearest); fflush(stdout); }

        int ef = std::max(efSearch, k);
        if (search_bounded_queue) { // this is the most common branch
            debug("%s\n", "reached search bounded queue");

            MinimaxHeap candidates(ef);

            candidates.push(nearest, d_nearest);
            debug_search("-starting BFS at level 0 with ef: %d, nearest: %d, d: %f, metadata: %d\n", ef, nearest, d_nearest, metadata[nearest]);
            hybrid_search_from_candidates(
                    *this,
                    qdis,
                    filter_map,
                    k,
                    I,
                    D,
                    candidates,
                    vt,
                    stats,
                    0,
                    0,
                    params,
                    dc_pool);

            if (hybridDebugEnabled) {
                hybrid_debug_log(
                        "[hybrid-debug q=%d] finished n3=%zu first_label=%lld first_dis=%g\n",
                        hybridDebugActiveQuery,
                        stats.n3,
                        (long long)I[0],
                        D[0]);
            }
            

        } else {
            // TODO
            printf("UNIMPLEMENTED BRANCH for hybid search\n");
            debug("%s\n", "reached search_bounded_queue == False");
            throw FaissException("UNIMPLEMENTED search unbounded queue");

            
        }

        vt.advance();
        hybridDebugEnabled = false;
        hybridDebugActiveQuery = -1;

    } else {
        debug("%s\n", "reached upper beam != 1");

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for (int level = max_level; level >= 0; level--) {
            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = hybrid_search_from_candidates(
                        *this,
                        qdis,
                        filter_map,
                        k,
                        I,
                        D,
                        candidates,
                        vt,
                        stats,
                        0,
                        0,
                        params,
                        dc_pool);
            
                
            } else {
                nres = hybrid_search_from_candidates(
                        *this,
                        qdis,
                        filter_map,
                        // filter,
                        // op,
                        // regex,
                        candidates_size,
                        I_to_next.data(),
                        D_to_next.data(),
                        candidates,
                        vt,
                        stats,
                        level,
                        0,
                        params,
                        dc_pool);
            }
            vt.advance();
        }
    }

    return stats;
}






/**************************************************************
 * MinimaxHeap
 **************************************************************/

void ACORN::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0])
            return;
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        --nvalid;
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float ACORN::MinimaxHeap::max() const {
    return dis[0];
}

int ACORN::MinimaxHeap::size() const {
    return nvalid;
}

void ACORN::MinimaxHeap::clear() {
    nvalid = k = 0;
}

int ACORN::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    // returns min. This is an O(n) operation
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1)
            break;
        i--;
    }
    if (i == -1)
        return -1;
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out)
        *vmin_out = vmin;
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;

    return ret;
}

int ACORN::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}

} // namespace faiss
//test
