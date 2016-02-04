#pragma once

#ifndef _VC_COMMON_H_
#define _VC_COMMON_H_

#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

#define BGR_RED cv::Scalar(0, 0, 0xFF)
#define BGR_GREEN cv::Scalar(0, 0xFF, 0)
#define BGR_BLUE cv::Scalar(0xFF, 0, 0)
#define BGR_YELLOW cv::Scalar(0, 0xFF, 0xFF)
#define BGR_MAGENTA cv::Scalar(0xFF, 0, 0xFF)

using IndexIntensityPair = std::pair<int32_t, double>;
using IndexIntensityPairVec = typename std::vector<IndexIntensityPair>;
template <typename T>
using vec = std::vector<T>;
using Voxel = cv::Vec3d;
using Pixel = cv::Vec2d;

// Helpful for printing out vector. Only for debug.
template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& s, std::pair<T1, T2> p)
{
    return s << "(" << std::get<0>(p) << ", " << std::get<1>(p) << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& s, std::vector<T> v)
{
    s << "[";
    if (v.size() == 0) {
        return s << "]";
    } else if (v.size() == 1) {
        return s << v[0] << "]";
    }
    // Need - 2 because v.end() points to one past the end of v.
    std::for_each(v.begin(), v.end() - 2, [&s](const T& t) { s << t << ", "; });
    return s << *(v.end() - 1) << "]";
}

#endif  // VC_COMMON_H
