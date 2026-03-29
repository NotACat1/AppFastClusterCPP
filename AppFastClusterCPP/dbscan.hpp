#pragma once

#include <vector>
#include <cstdint>

namespace fc::algorithms {

    /**
     * @class DBSCAN
     * @brief High-performance implementation of the Density-Based Spatial Clustering of Applications with Noise.
     * * This implementation decouples the clustering logic from the spatial indexing mechanism.
     * By utilizing a functional template for region queries, it supports various backends
     * including Brute-force SIMD, KD-Trees, or R-Trees without inheritance overhead.
     */
    class DBSCAN {
    public:
        // Point state constants
        static constexpr int32_t UNCLASSIFIED = -2; ///< Initial state for all points
        static constexpr int32_t NOISE = -1;        ///< Points that do not satisfy density requirements

        /**
         * @struct Result
         * @brief Container for clustering output.
         */
        struct Result {
            std::vector<int32_t> labels; ///< Cluster IDs for each point (matching input indices)
            int32_t num_clusters;        ///< Total count of discovered clusters
        };

        /**
         * @brief Executes the DBSCAN algorithm.
         * * @tparam RegionQueryFunc A functor or lambda with signature: std::vector<std::size_t>(std::size_t index, float radius).
         * @param num_points Total number of points in the dataset.
         * @param eps The distance threshold (epsilon) for neighborhood searches.
         * @param min_pts Minimum number of points required to form a dense region (core point).
         * @param region_query A user-provided search function that returns indices of neighbors.
         * @return Result object containing labels and cluster statistics.
         */
        template <typename RegionQueryFunc>
        Result run(std::size_t num_points, float eps, std::size_t min_pts, RegionQueryFunc&& region_query) {
            Result result;
            result.labels.assign(num_points, UNCLASSIFIED);
            result.num_clusters = 0;

            // Cluster expansion frontier buffer. 
            // Pre-allocated once to minimize heap fragmentation and reallocations during runtime.
            std::vector<std::size_t> seed_set;
            seed_set.reserve(num_points / 10); // Heuristic initial capacity based on typical cluster density

            for (std::size_t i = 0; i < num_points; ++i) {
                if (result.labels[i] != UNCLASSIFIED) continue;

                // Invoke spatial query. Expected to leverage NRVO (Named Return Value Optimization).
                auto neighbors = region_query(i, eps);

                if (neighbors.size() < min_pts) {
                    result.labels[i] = NOISE;
                    continue;
                }

                // Core point detected: initialize a new cluster expansion
                int32_t cluster_id = result.num_clusters++;
                result.labels[i] = cluster_id;

                // Prepare the seed set for recursive-like expansion
                seed_set.clear();
                for (auto n_idx : neighbors) {
                    if (n_idx != i) seed_set.push_back(n_idx);
                }

                expand_cluster(seed_set, cluster_id, eps, min_pts, result.labels, region_query);
            }

            return result;
        }

    private:
        /**
         * @brief Spreads the cluster ID to all reachable points in the density-connected component.
         * * Uses a LIFO (Last-In-First-Out) traversal which is generally more cache-friendly
         * than FIFO (queue-based) approaches for spatial datasets.
         */
        template <typename RegionQueryFunc>
        void expand_cluster(
            std::vector<std::size_t>& seed_set,
            int32_t cluster_id,
            float eps,
            std::size_t min_pts,
            std::vector<int32_t>& labels,
            RegionQueryFunc& region_query
        ) {
            while (!seed_set.empty()) {
                std::size_t current_idx = seed_set.back();
                seed_set.pop_back();

                // Boundary point check: points previously marked as NOISE 
                // are transitioned to the current cluster but do not trigger further expansion.
                if (labels[current_idx] == NOISE) {
                    labels[current_idx] = cluster_id;
                    continue;
                }

                // Skip points already processed or assigned to other clusters
                if (labels[current_idx] != UNCLASSIFIED) {
                    continue;
                }

                // Assign point to the current cluster
                labels[current_idx] = cluster_id;

                // Perform density check for the current neighbor
                auto neighbors = region_query(current_idx, eps);

                // If the neighbor is also a core point, add its neighbors to the frontier
                if (neighbors.size() >= min_pts) {
                    for (auto n_idx : neighbors) {
                        if (labels[n_idx] == UNCLASSIFIED || labels[n_idx] == NOISE) {
                            seed_set.push_back(n_idx);
                        }
                    }
                }
            }
        }
    };
} // namespace fc::algorithms