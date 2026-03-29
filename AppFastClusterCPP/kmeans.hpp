#pragma once

#include <vector>
#include <cstdint>
#include <limits>
#include <random>
#include <algorithm>

namespace fc::algorithms {

    /**
     * @class KMeans
     * @brief A high-performance K-Means clustering implementation.
     * * Designed with a Data-Oriented approach using flat arrays to maximize spatial locality
     * and minimize cache misses. The distance calculation is decoupled via a template
     * functor, allowing for seamless integration of SIMD-accelerated (AVX2/AVX-512)
     * distance metrics without modifying the core logic.
     */
    class KMeans {
    public:
        /**
         * @struct Result
         * @brief Encapsulates clustering output and execution metadata.
         */
        struct Result {
            std::vector<int32_t> labels;    ///< Cluster assignment for each input point
            std::vector<float> centroids;   ///< Flattened centroid coordinates [k * dim]
            std::size_t num_iterations;     ///< Actual iterations performed until convergence
        };

        /**
         * @brief Executes the K-Means algorithm.
         * * @tparam DistanceFunc Functor type. Signature: float(const float* p1, const float* p2, size_t dim)
         * @param data Pointer to the input feature matrix [num_points * dim] in row-major order.
         * @param num_points Total number of observations.
         * @param dim Number of features (dimensions) per point.
         * @param k Target number of clusters.
         * @param max_iters Maximum allowed iterations to prevent infinite loops in non-converging scenarios.
         * @param distance_func Metric used for cluster assignment (typically L2 squared).
         * @param seed PRNG seed for reproducible centroid initialization.
         * @return Result structure containing finalized labels and centroids.
         */
        template <typename DistanceFunc>
        Result run(const float* data, std::size_t num_points, std::size_t dim,
            std::size_t k, std::size_t max_iters,
            DistanceFunc&& distance_func, uint32_t seed = 42) {

            Result result;
            if (num_points == 0 || k == 0 || dim == 0) return result;

            // Clamping k to num_points to prevent over-segmentation in small datasets
            k = std::min(k, num_points);

            result.labels.assign(num_points, -1);
            result.centroids.resize(k * dim);
            result.num_iterations = 0;

            std::mt19937 gen(seed);

            // 1. Initialization (Forgy Method)
            // Centroids are initialized by randomly sampling points from the dataset.
            // This method is computationally inexpensive and effective for most distributions.
            std::vector<std::size_t> indices(num_points);
            for (std::size_t i = 0; i < num_points; ++i) indices[i] = i;

            // Partial Fisher-Yates shuffle to pick k unique starting points
            for (std::size_t i = 0; i < k; ++i) {
                std::uniform_int_distribution<std::size_t> dist(i, num_points - 1);
                std::swap(indices[i], indices[dist(gen)]);

                // Deep copy coordinates to the centroid buffer
                for (std::size_t d = 0; d < dim; ++d) {
                    result.centroids[i * dim + d] = data[indices[i] * dim + d];
                }
            }

            // Workspace buffers to avoid frequent heap allocations during the iterative process
            std::vector<std::size_t> cluster_counts(k, 0);
            std::vector<float> new_centroids(k * dim, 0.0f);

            for (std::size_t iter = 0; iter < max_iters; ++iter) {
                bool changed = false;

                // 2. Assignment Step
                // Complexity: O(N * K * D). This is the hot path of the algorithm.
                for (std::size_t i = 0; i < num_points; ++i) {
                    const float* point = data + i * dim;
                    float min_dist = std::numeric_limits<float>::max();
                    int32_t best_cluster = 0;

                    for (std::size_t c = 0; c < k; ++c) {
                        const float* centroid = result.centroids.data() + c * dim;
                        float dist = distance_func(point, centroid, dim);

                        if (dist < min_dist) {
                            min_dist = dist;
                            best_cluster = static_cast<int32_t>(c);
                        }
                    }

                    // Check for centroid migration to determine convergence
                    if (result.labels[i] != best_cluster) {
                        result.labels[i] = best_cluster;
                        changed = true;
                    }
                }

                // Early exit if the system has reached a stable configuration
                if (!changed) {
                    result.num_iterations = iter + 1;
                    break;
                }

                // 3. Update Step
                // Recompute centroids by averaging the coordinates of assigned points.
                std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
                std::fill(cluster_counts.begin(), cluster_counts.end(), 0);

                // Accumulate coordinates per cluster
                for (std::size_t i = 0; i < num_points; ++i) {
                    int32_t cluster_id = result.labels[i];
                    cluster_counts[cluster_id]++;

                    const float* point = data + i * dim;
                    float* centroid_accum = new_centroids.data() + cluster_id * dim;

                    for (std::size_t d = 0; d < dim; ++d) {
                        centroid_accum[d] += point[d];
                    }
                }

                // Normalize sums to compute new means
                for (std::size_t c = 0; c < k; ++c) {
                    float* centroid = result.centroids.data() + c * dim;

                    if (cluster_counts[c] > 0) {
                        const float* centroid_accum = new_centroids.data() + c * dim;
                        for (std::size_t d = 0; d < dim; ++d) {
                            centroid[d] = centroid_accum[d] / cluster_counts[c];
                        }
                    }
                    else {
                        // Handling the "Empty Cluster Problem":
                        // If a cluster loses all its points, reassign its centroid to 
                        // a random point in the dataset to keep k constant and resume exploration.
                        std::uniform_int_distribution<std::size_t> dist(0, num_points - 1);
                        std::size_t rand_idx = dist(gen);
                        const float* rand_point = data + rand_idx * dim;

                        for (std::size_t d = 0; d < dim; ++d) {
                            centroid[d] = rand_point[d];
                        }
                    }
                }

                result.num_iterations = iter + 1;
            }

            return result;
        }
    };

    /**
     * @struct L2SquaredDistance
     * @brief Squared Euclidean Distance metric.
     * * Optimizes K-Means by avoiding the costly square root operation (sqrt),
     * which is monotonic and does not change the result of centroid assignment comparisons.
     */
    struct L2SquaredDistance {
        inline float operator()(const float* p1, const float* p2, std::size_t dim) const {
            float dist = 0.0f;
            // Hot loop: Candidate for auto-vectorization by the compiler
            for (std::size_t i = 0; i < dim; ++i) {
                float diff = p1[i] - p2[i];
                dist += diff * diff;
            }
            return dist;
        }
    };
} // namespace fc::algorithms