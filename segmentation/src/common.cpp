#include "segmentation/lrps/common.h"

std::vector<double> volcart::segmentation::squareDiff(
    const std::vector<Voxel>& v1, const std::vector<Voxel>& v2)
{
    assert(v1.size() == v2.size() && "src and target must be the same size");
    std::vector<double> res(v1.size());
    const auto zipped = zip(v1, v2);
    std::transform(
        std::begin(zipped), std::end(zipped), std::begin(res),
        [](auto p) { return cv::norm(p.first, p.second); });
    return res;
}

double volcart::segmentation::sumSquareDiff(
    const std::vector<double>& v1, const std::vector<double>& v2)
{
    assert(v1.size() == v2.size() && "v1 and v2 must be the same size");
    double res = 0;
    for (uint32_t i = 0; i < v1.size(); ++i) {
        res += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return std::sqrt(res);
}
