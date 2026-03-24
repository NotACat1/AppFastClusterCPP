#pragma once

#include <concepts>

namespace fc {

    /**
     * @brief Concept defining the requirements for machine learning coordinates.
     * * Restricts coordinate types to standard floating-point representations (float, double, etc.)
     * to ensure numerical stability and compatibility with hardware-accelerated SIMD instructions.
     */
    template <typename T>
    concept MLCoordinate = std::floating_point<T>;

}