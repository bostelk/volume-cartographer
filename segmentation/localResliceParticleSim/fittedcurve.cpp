#include <algorithm>
#include <cmath>
#include <iostream>
#include "fittedcurve.h"

using namespace volcart::segmentation;

double calcArcLength(const std::vector<Voxel>& vs);

FittedCurve::FittedCurve(const std::vector<Voxel>& vs, const int32_t zIndex)
    : npoints_(vs.size()), zIndex_(zIndex), seedPoints_(vs)
{
    std::vector<double> xs, ys;
    xs.reserve(vs.size());
    ys.reserve(vs.size());
    xs.push_back(vs.front()(0));
    ys.push_back(vs.front()(1));

    double arcLength = calcArcLength(vs);

    // Calculate new ts_
    // Initial start t = 0
    double accumulatedLength = 0;
    ts_.reserve(vs.size());
    ts_.push_back(0);
    for (size_t i = 1; i < vs.size(); ++i) {
        xs.push_back(vs[i](0));
        ys.push_back(vs[i](1));
        accumulatedLength += std::sqrt(std::pow(vs[i](0) - vs[i - 1](0), 2) +
                                       std::pow(vs[i](1) - vs[i - 1](1), 2));
        ts_.push_back(accumulatedLength / arcLength);
    }

    spline_ = CubicSpline<double>(xs, ys);

    // Calculate new voxel positions from the spline
    points_.reserve(vs.size());
    for (const auto t : ts_) {
        auto p = spline_.eval(t);
        points_.emplace_back(p(0), p(1));
        xs_.push_back(p(0));
        ys_.push_back(p(1));
    }
}

std::vector<Voxel> FittedCurve::resample(const double resamplePerc)
{
    ts_.clear();
    xs_.clear();
    ys_.clear();
    npoints_ = std::round(resamplePerc * npoints_);
    ts_.resize(npoints_, 0.0);

    // Calculate new knot positions in t-space
    double sum = 0;
    for (int32_t i = 0; i < npoints_ && sum <= 1;
         ++i, sum += 1.0 / (npoints_ - 1)) {
        ts_[i] = sum;
    }

    // Get new positions
    std::vector<Voxel> rs;
    rs.reserve(npoints_);
    points_.clear();
    for (const auto t : ts_) {
        auto p = spline_.eval(t);
        points_.emplace_back(p(0), p(1));
        xs_.push_back(p(0));
        ys_.push_back(p(1));
        rs.emplace_back(p(0), p(1), zIndex_);
    }
    return rs;
}

Voxel FittedCurve::operator()(const int32_t index) const
{
    Pixel p = spline_.eval(ts_[index]);
    return {p(0), p(1), double(zIndex_)};
}

double calcArcLength(const std::vector<Voxel>& vs)
{
    double length = 0;
    for (size_t i = 1; i < vs.size(); ++i) {
        length += std::sqrt(std::pow(vs[i](0) - vs[i - 1](0), 2) +
                            std::pow(vs[i](1) - vs[i - 1](1), 2));
    }
    return length;
}

std::vector<double> FittedCurve::curvature(const int32_t hstep,
                                           const double scaleFactor) const
{
    const auto dx1 = d1(xs_, hstep);
    const auto dy1 = d1(ys_, hstep);
    const auto dx2 = d2(xs_, hstep);
    const auto dy2 = d2(ys_, hstep);

    // Calculate curvature
    // according to: http://mathworld.wolfram.com/Curvature.html
    std::vector<double> k;
    k.reserve(points_.size());
    for (size_t i = 0; i < points_.size(); ++i) {
        k.push_back((dx1[i] * dy2[i] - dy1[i] * dx2[i]) /
                    std::pow(dx1[i] * dx1[i] + dy1[i] * dy1[i], 3.0 / 2.0) *
                    scaleFactor);
    }

    return k;
}
