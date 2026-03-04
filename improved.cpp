#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;

#define DEPOT 0

typedef unsigned node_t;
typedef double weight_t;
typedef float demand_t;

struct Node {
  node_t id;
  weight_t x, y;
  demand_t demand;
};

struct Edge {
  node_t to;
  weight_t length;
  Edge(node_t t, weight_t l) : to(t), length(l) {}
};

struct Params {
  short nThreads = 20;
  bool toRound = true;
};

class VRP {
  size_t size;
  size_t type;
  demand_t capacity;
  std::vector<weight_t> distanceMatrix;

public:
  std::vector<Node> node;
  Params params;

  unsigned read(const string &filename);
  void precompute_distances();
  weight_t get_dist(node_t n1, node_t n2) const;
  std::vector<std::vector<Edge>> cal_graph_dist() const;
  size_t getSize() const { return size; }
  demand_t getCapacity() const { return capacity; }
};

unsigned VRP::read(const string &filename) {
  ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "Could not open the file \"" << filename << "\"" << std::endl;
    exit(1);
  }
  
  string line;
  for (int i = 0; i < 3; ++i) getline(in, line);

  getline(in, line);
  size = stof(line.substr(line.find(":") + 2));
  getline(in, line);
  type = line.find(":");
  getline(in, line);
  capacity = stof(line.substr(line.find(":") + 2));
  getline(in, line); 

  node.resize(size);
  stringstream iss; 
  for (size_t i = 0; i < size; ++i) {
    getline(in, line);
    iss.clear(); iss.str(line);
    size_t id;
    iss >> id >> node[i].x >> node[i].y;
  }
  getline(in, line); 
  for (size_t i = 0; i < size; ++i) {
    getline(in, line);
    iss.clear(); iss.str(line);
    size_t id;
    iss >> id >> node[i].demand;
  }
  in.close();
  precompute_distances();
  return capacity;
}

void VRP::precompute_distances() {
  distanceMatrix.resize(size * size);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      if (i == j) distanceMatrix[i * size + j] = 0.0;
      else {
        weight_t dx = node[i].x - node[j].x;
        weight_t dy = node[i].y - node[j].y;
        weight_t dist = std::sqrt(dx * dx + dy * dy);
        distanceMatrix[i * size + j] = params.toRound ? std::round(dist) : dist;
      }
    }
  }
}

weight_t VRP::get_dist(node_t n1, node_t n2) const {
  return distanceMatrix[n1 * size + n2];
}

// FIX 1 — cal_graph_dist now stores RAW (unrounded) edge weights for Prim's,
// exactly matching the original. get_dist() is NOT called here because it
// returns rounded values, which would produce a different MST on tie-breaking edges.
std::vector<std::vector<Edge>> VRP::cal_graph_dist() const {
  std::vector<std::vector<Edge>> cG(size);
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      if (i != j) {
        weight_t dx = node[i].x - node[j].x;
        weight_t dy = node[i].y - node[j].y;
        weight_t w = std::sqrt(dx * dx + dy * dy);  // unrounded, matches original
        cG[i].push_back(Edge(j, w));
      }
    }
  }
  return cG;
}

weight_t calRouteValue(const VRP &vrp, const std::vector<node_t> &aRoute) {
  if (aRoute.empty()) return 0.0;
  weight_t routeVal = vrp.get_dist(DEPOT, aRoute.front());
  for (size_t i = 1; i < aRoute.size(); ++i) routeVal += vrp.get_dist(aRoute[i - 1], aRoute[i]);
  return routeVal + vrp.get_dist(aRoute.back(), DEPOT);
}

weight_t calCost(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes) {
  weight_t total_cost = 0.0;
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    total_cost += calRouteValue(vrp, final_routes[ii]);
  }
  return total_cost;
}

// Reverted to original O(V^2) array search for exact tie-breaking match
std::vector<std::vector<Edge>> PrimsAlgo(const VRP &vrp, const std::vector<std::vector<Edge>> &graph) {
  auto N = graph.size();
  const node_t INIT = -1;
  std::vector<weight_t> key(N, DBL_MAX);
  std::vector<weight_t> toEdges(N, INIT);
  std::vector<bool> visited(N, false);

  key[DEPOT] = 0.0;

  for (size_t count = 0; count < N; ++count) {
    weight_t min = DBL_MAX;
    node_t u = INIT;

    for (size_t v = 0; v < N; ++v) {
      if (!visited[v] && key[v] < min) {
        min = key[v];
        u = v;
      }
    }

    if (u == INIT) break;
    visited[u] = true;

    for (const Edge &E : graph[u]) {
      if (!visited[E.to] && E.length < key[E.to]) {
        key[E.to] = E.length;
        toEdges[E.to] = u;
      }
    }
  }

  std::vector<std::vector<Edge>> nG(N);
  for (node_t v = 0; v < N; ++v) {
    if (toEdges[v] != INIT) {
      weight_t w = vrp.get_dist(v, toEdges[v]);
      nG[toEdges[v]].push_back(Edge(v, w));
      nG[v].push_back(Edge(toEdges[v], w));
    }
  }
  return nG;
}

void ShortCircutTour(const std::vector<std::vector<Edge>> &g, std::vector<bool> &visited, node_t start, std::vector<node_t> &out) {
  std::stack<node_t> s;
  s.push(start);
  while (!s.empty()) {
    node_t u = s.top();
    s.pop();
    if (!visited[u]) {
      visited[u] = true;
      out.push_back(u);
      for (auto it = g[u].rbegin(); it != g[u].rend(); ++it) {
        if (!visited[it->to]) s.push(it->to);
      }
    }
  }
}

std::vector<std::vector<node_t>> convertToVrpRoutes(const VRP &vrp, const std::vector<node_t> &singleRoute) {
  std::vector<std::vector<node_t>> routes;
  demand_t residueCap = vrp.getCapacity();
  std::vector<node_t> aRoute;

  for (auto v : singleRoute) {
    if (v == DEPOT) continue;
    if (residueCap - vrp.node[v].demand >= 0) {
      aRoute.push_back(v);
      residueCap -= vrp.node[v].demand;
    } else {
      routes.push_back(aRoute);
      aRoute.clear();
      aRoute.push_back(v);
      residueCap = vrp.getCapacity() - vrp.node[v].demand;
    }
  }
  if (!aRoute.empty()) routes.push_back(aRoute);
  return routes;
}

// FIX 2 — tsp_approx now faithfully replicates the original algorithm:
//   1. Rotates the input so the depot anchors position 0.
//   2. Uses a backward-scanning selection-sort from position 1 onwards.
//   3. Compares raw SQUARED Euclidean distance (no sqrt, no rounding) to
//      preserve the original tie-breaking behaviour exactly.
// The new signature (route in, optimized out) keeps the cleaner improved.cpp
// calling convention while matching the original's mathematical output.
void tsp_approx(const VRP &vrp, const std::vector<node_t> &route, std::vector<node_t> &optimized) {
  unsigned sz = route.size();
  if (sz == 0) { optimized.clear(); return; }

  // Build working array: [DEPOT, route[0], ..., route[sz-1]]
  // This mirrors the original rotation: tour[0]=cities[ncities-1]=DEPOT,
  // tour[1..sz]=cities[0..sz-1]=route[0..sz-1]
  std::vector<node_t> tour(sz + 1);
  tour[0] = DEPOT;
  for (unsigned i = 0; i < sz; ++i) tour[i + 1] = route[i];

  unsigned ncities = sz + 1;

  for (unsigned i = 1; i < ncities; ++i) {
    weight_t ThisX = vrp.node[tour[i - 1]].x;
    weight_t ThisY = vrp.node[tour[i - 1]].y;
    weight_t CloseDist = DBL_MAX;
    unsigned ClosePt = i;

    // Backward scan: identical to the original j-decrement loop
    for (int j = (int)ncities - 1; ; --j) {
      weight_t dx = vrp.node[tour[j]].x - ThisX;
      // Early-exit on x alone, exactly like the original two-stage check
      weight_t ThisDist = dx * dx;
      if (ThisDist <= CloseDist) {
        weight_t dy = vrp.node[tour[j]].y - ThisY;
        ThisDist += dy * dy;
        if (ThisDist <= CloseDist) {
          if (j < (int)i) break;  // stop once we've passed the unsorted region
          CloseDist = ThisDist;
          ClosePt = (unsigned)j;
        }
      }
    }
    std::swap(tour[i], tour[ClosePt]);
  }

  // Return nodes only (skip depot at index 0)
  optimized.assign(tour.begin() + 1, tour.end());
}

// FIX 3 — tsp_2opt termination condition restored to `while (improve < 2)`.
// The original performed TWO consecutive full passes with no improvement before
// stopping. The previous improved.cpp stopped after ONE such pass (`while
// (improvement)`), potentially exiting at a suboptimal point and changing cost.
void tsp_2opt(const VRP &vrp, std::vector<node_t> &tour) {
  unsigned ncities = tour.size();
  if (ncities < 2) return;
  unsigned improve = 0;

  while (improve < 2) {
    weight_t best_distance = calRouteValue(vrp, tour);
    for (unsigned i = 0; i < ncities - 1; ++i) {
      for (unsigned k = i + 1; k < ncities; ++k) {
        std::reverse(tour.begin() + i, tour.begin() + k + 1);
        weight_t new_distance = calRouteValue(vrp, tour);
        if (new_distance < best_distance) {
          improve = 0;  // reset counter on any improvement, exactly as original
          best_distance = new_distance;
        } else {
          std::reverse(tour.begin() + i, tour.begin() + k + 1);
        }
      }
    }
    improve++;  // one full scan done; if no improvement found, this accumulates to 2
  }
}

std::vector<std::vector<node_t>> postProcessIt(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes, weight_t &minCost) {
  unsigned nroutes = final_routes.size();
  std::vector<std::vector<node_t>> postprocessed_final_routes(nroutes);

  #pragma omp parallel for schedule(static)
  for (unsigned zzz = 0; zzz < nroutes; ++zzz) {
    std::vector<node_t> route1, route2 = final_routes[zzz], route3 = final_routes[zzz];
    tsp_approx(vrp, final_routes[zzz], route1);
    route2 = route1;
    tsp_2opt(vrp, route2);
    tsp_2opt(vrp, route3);

    weight_t cost2 = calRouteValue(vrp, route2);
    weight_t cost3 = calRouteValue(vrp, route3);
    postprocessed_final_routes[zzz] = (cost2 < cost3) ? route2 : route3;
  }

  minCost = calCost(vrp, postprocessed_final_routes);
  return postprocessed_final_routes;
}

bool verify_sol(const VRP &vrp, const vector<vector<node_t>> &final_routes, unsigned capacity) {
  std::vector<unsigned> hist(vrp.getSize(), 0);
  for (unsigned i = 0; i < final_routes.size(); ++i) {
    unsigned route_sum_of_demands = 0;
    for (unsigned j = 0; j < final_routes[i].size(); ++j) {
      route_sum_of_demands += vrp.node[final_routes[i][j]].demand;
      hist[final_routes[i][j]] += 1;
    }
    if (route_sum_of_demands > capacity) return false;
  }
  for (unsigned i = 1; i < vrp.getSize(); ++i) {
    if (hist[i] != 1) return false;
  }
  return true;
}

void printOutput(const VRP &vrp, const std::vector<std::vector<node_t>> &final_routes, weight_t total_cost) {
  for (unsigned ii = 0; ii < final_routes.size(); ++ii) {
    std::cout << "Route #" << ii + 1 << ":";
    for (unsigned jj = 0; jj < final_routes[ii].size(); ++jj) std::cout << " " << final_routes[ii][jj];
    std::cout << '\n';
  }
  std::cout << "Cost " << total_cost << std::endl;
}

int main(int argc, char *argv[]) {
  VRP vrp;
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " toy.vrp [-nthreads <n>]" << '\n';
    exit(1);
  }

  for (int ii = 2; ii < argc; ii += 2) {
    if (std::string(argv[ii]) == "-round") vrp.params.toRound = atoi(argv[ii + 1]);
    else if (std::string(argv[ii]) == "-nthreads") vrp.params.nThreads = atoi(argv[ii + 1]);
  }

  vrp.read(argv[1]);
  auto start = std::chrono::high_resolution_clock::now();

  auto cG = vrp.cal_graph_dist();
  auto mstCopy = PrimsAlgo(vrp, cG);

  weight_t global_minCost = DBL_MAX;
  int global_best_thread = INT_MAX;
  std::vector<std::vector<node_t>> global_minRoute;
  short PARLIMIT = vrp.params.nThreads;

  #pragma omp parallel num_threads(PARLIMIT)
  {
    int thread_id = omp_get_thread_num();
    std::default_random_engine local_rng(std::chrono::high_resolution_clock::now().time_since_epoch().count() + thread_id);

    std::vector<node_t> local_singleRoute;
    std::vector<bool> local_visited(mstCopy.size(), false);

    weight_t thread_minCost = DBL_MAX;
    std::vector<std::vector<node_t>> thread_minRoute;

    #pragma omp for schedule(static)
    for (int i = 0; i < 100000; i++) {
      auto local_mst = mstCopy;
      for (auto &list : local_mst) std::shuffle(list.begin(), list.end(), local_rng);

      local_singleRoute.clear();
      std::fill(local_visited.begin(), local_visited.end(), false);

      ShortCircutTour(local_mst, local_visited, DEPOT, local_singleRoute);
      auto aRoutes = convertToVrpRoutes(vrp, local_singleRoute);
      weight_t aCostRoute = calCost(vrp, aRoutes);

      if (aCostRoute < thread_minCost) {
        thread_minCost = aCostRoute;
        thread_minRoute = aRoutes;
      }
    }

    #pragma omp critical
    {
      if (thread_minCost < global_minCost || (thread_minCost == global_minCost && thread_id < global_best_thread)) {
        global_minCost = thread_minCost;
        global_minRoute = thread_minRoute;
        global_best_thread = thread_id;
      }
    }
  }

  auto postRoutes = postProcessIt(vrp, global_minRoute, global_minCost);

  auto end = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1.E-9;
  bool verified = verify_sol(vrp, postRoutes, vrp.getCapacity());

  std::cerr << argv[1] << " Cost " << global_minCost << " Time(s) " << total_time
            << " parLimit " << PARLIMIT << (verified ? " VALID" : " INVALID") << std::endl;

  printOutput(vrp, postRoutes, global_minCost);
  return 0;
}