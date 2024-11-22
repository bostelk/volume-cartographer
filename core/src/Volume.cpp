#include "vc/core/types/Volume.hpp"

#include <iomanip>
#include <sstream>

#include <opencv2/imgcodecs.hpp>

#include "vc/core/io/TIFFIO.hpp"

#define VESUVIUS_IMPL
#include "vc/core/io/vesuvius-c.h"

namespace fs = volcart::filesystem;
namespace tio = volcart::tiffio;

using namespace volcart;


volume* vol = nullptr;

// Load a Volume from disk
Volume::Volume(fs::path path) : DiskBasedObjectBaseClass(std::move(path))
{
    if (metadata_.get<std::string>("type") != "vol") {
        throw std::runtime_error("File not of type: vol");
    }

    width_ = metadata_.get<int>("width");
    height_ = metadata_.get<int>("height");
    slices_ = metadata_.get<int>("slices");
    numSliceCharacters_ = std::to_string(slices_).size();

    std::vector<std::mutex> init_mutexes(slices_);

    slice_mutexes_.swap(init_mutexes);

    std::string ZarrCache = "./54keV_7.91um_Scroll1A.zarr/0/";
    std::string ZarrUrl = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/";

    ZarrCache = "./s1-surface-regular.zarr/";
    ZarrUrl = "https://dl.ash2txt.org/community-uploads/bruniss/Fiber-and-Surface-Models/Predictions/s1/full-scroll-preds/s1-surface-regular.zarr/";

    vol = vs_vol_new(ZarrCache.c_str(), ZarrUrl.c_str());
}

// Setup a Volume from a folder of slices
Volume::Volume(fs::path path, std::string uuid, std::string name)
    : DiskBasedObjectBaseClass(
          std::move(path), std::move(uuid), std::move(name)),
          slice_mutexes_(slices_)
{
    metadata_.set("type", "vol");
    metadata_.set("width", width_);
    metadata_.set("height", height_);
    metadata_.set("slices", slices_);
    metadata_.set("voxelsize", double{});
    metadata_.set("min", double{});
    metadata_.set("max", double{});
}

// Load a Volume from disk, return a pointer
auto Volume::New(fs::path path) -> Volume::Pointer
{
    return std::make_shared<Volume>(path);
}

// Set a Volume from a folder of slices, return a pointer
auto Volume::New(fs::path path, std::string uuid, std::string name)
    -> Volume::Pointer
{
    return std::make_shared<Volume>(path, uuid, name);
}

auto Volume::sliceWidth() const -> int { return width_; }
auto Volume::sliceHeight() const -> int { return height_; }
auto Volume::numSlices() const -> int { return slices_; }
auto Volume::voxelSize() const -> double
{
    return metadata_.get<double>("voxelsize");
}
auto Volume::min() const -> double { return metadata_.get<double>("min"); }
auto Volume::max() const -> double { return metadata_.get<double>("max"); }

void Volume::setSliceWidth(int w)
{
    width_ = w;
    metadata_.set("width", w);
}

void Volume::setSliceHeight(int h)
{
    height_ = h;
    metadata_.set("height", h);
}

void Volume::setNumberOfSlices(std::size_t numSlices)
{
    slices_ = numSlices;
    numSliceCharacters_ = std::to_string(numSlices).size();
    metadata_.set("slices", numSlices);
}

void Volume::setVoxelSize(double s) { metadata_.set("voxelsize", s); }
void Volume::setMin(double m) { metadata_.set("min", m); }
void Volume::setMax(double m) { metadata_.set("max", m); }

auto Volume::bounds() const -> Volume::Bounds
{
    return {
        {0, 0, 0},
        {static_cast<double>(width_), static_cast<double>(height_),
         static_cast<double>(slices_)}};
}

auto Volume::isInBounds(double x, double y, double z) const -> bool
{
    return x >= 0 && x < width_ && y >= 0 && y < height_ && z >= 0 &&
           z < slices_;
}

auto Volume::isInBounds(const cv::Vec3d& v) const -> bool
{
    return isInBounds(v(0), v(1), v(2));
}

auto Volume::getSlicePath(int index) const -> fs::path
{
    std::stringstream ss;
    ss << std::setw(numSliceCharacters_) << std::setfill('0') << index
       << ".tif";
    return path_ / ss.str();
}

auto Volume::getSliceData(int index) const -> cv::Mat
{
    if (cacheSlices_) {
        return cache_slice_(index);
    }
    return load_slice_(index);
}

auto Volume::getSliceDataCopy(int index) const -> cv::Mat
{
    return getSliceData(index).clone();
}

auto Volume::getSliceDataRect(int index, cv::Rect rect) const -> cv::Mat
{
    auto whole_img = getSliceData(index);
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return whole_img(rect);
}

auto Volume::getSliceDataRectCopy(int index, cv::Rect rect) const -> cv::Mat
{
    auto whole_img = getSliceData(index);
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return whole_img(rect).clone();
}

void Volume::setSliceData(int index, const cv::Mat& slice, bool compress)
{
    auto slicePath = getSlicePath(index);
    tio::WriteTIFF(
        slicePath.string(), slice,
        (compress) ? tiffio::Compression::LZW : tiffio::Compression::NONE);
}

auto Volume::intensityAt(int x, int y, int z) const -> std::uint16_t
{
    // clang-format off
    if (x < 0 || x >= sliceWidth() ||
        y < 0 || y >= sliceHeight() ||
        z < 0 || z >= numSlices()) {
        return 0;
    }
    // clang-format on
    return getSliceData(z).at<std::uint16_t>(y, x);
}

// Trilinear Interpolation
// From: https://en.wikipedia.org/wiki/Trilinear_interpolation
auto Volume::interpolateAt(double x, double y, double z) const -> std::uint16_t
{
    // insert safety net
    if (!isInBounds(x, y, z)) {
        return 0;
    }

    double intPart;
    double dx = std::modf(x, &intPart);
    auto x0 = static_cast<int>(intPart);
    int x1 = x0 + 1;
    double dy = std::modf(y, &intPart);
    auto y0 = static_cast<int>(intPart);
    int y1 = y0 + 1;
    double dz = std::modf(z, &intPart);
    auto z0 = static_cast<int>(intPart);
    int z1 = z0 + 1;

    auto c00 =
        intensityAt(x0, y0, z0) * (1 - dx) + intensityAt(x1, y0, z0) * dx;
    auto c10 =
        intensityAt(x0, y1, z0) * (1 - dx) + intensityAt(x1, y0, z0) * dx;
    auto c01 =
        intensityAt(x0, y0, z1) * (1 - dx) + intensityAt(x1, y0, z1) * dx;
    auto c11 =
        intensityAt(x0, y1, z1) * (1 - dx) + intensityAt(x1, y1, z1) * dx;

    auto c0 = c00 * (1 - dy) + c10 * dy;
    auto c1 = c01 * (1 - dy) + c11 * dy;

    auto c = c0 * (1 - dz) + c1 * dz;
    return static_cast<std::uint16_t>(cvRound(c));
}

auto Volume::reslice(
    const cv::Vec3d& center,
    const cv::Vec3d& xvec,
    const cv::Vec3d& yvec,
    int width,
    int height) const -> Reslice
{
    auto xnorm = cv::normalize(xvec);
    auto ynorm = cv::normalize(yvec);
    auto origin = center - ((width / 2) * xnorm + (height / 2) * ynorm);

    cv::Mat m(height, width, CV_16UC1);
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            m.at<std::uint16_t>(h, w) =
                interpolateAt(origin + (h * ynorm) + (w * xnorm));
        }
    }

    return Reslice(m, origin, xnorm, ynorm);
}

auto Volume::load_slice_(int index) const -> cv::Mat
{
    {
        std::unique_lock<std::shared_mutex> lock(print_mutex_);
        std::cout << "Requested to load slice " << index << std::endl;
    }
    auto slicePath = getSlicePath(index);
    cv::Mat mat;
    try {
        // mat = tio::ReadTIFF(slicePath.string());
        
        std::vector<std::tuple<int, int, int>> umbilicus_scroll1a_zyx = {
            {0, 2503, 4082},     {629, 2420, 4039},   {1256, 2338, 4031},
            {1407, 2337, 4032},  {1517, 2312, 4031},  {1581, 2308, 3990},
            {1820, 2307, 4004},  {2008, 2279, 3959},  {2245, 2247, 3908},
            {2422, 2227, 3939},  {2446, 2216, 3942},  {2469, 2205, 3971},
            {2481, 2201, 3996},  {2499, 2194, 4000},  {2536, 2190, 3989},
            {2565, 2178, 3987},  {2600, 2171, 3989},  {2625, 2173, 3975},
            {2682, 2172, 3934},  {2784, 2139, 3931},  {2811, 2131, 3933},
            {2826, 2127, 3934},  {2844, 2126, 3936},  {2855, 2125, 3934},
            {2867, 2126, 3934},  {2877, 2121, 3932},  {2897, 2118, 3930},
            {2909, 2116, 3932},  {2929, 2111, 3936},  {2942, 2105, 3936},
            {2958, 2099, 3936},  {2967, 2096, 3936},  {2979, 2095, 3935},
            {2993, 2093, 3935},  {3009, 2092, 3935},  {3028, 2099, 3926},
            {3045, 2100, 3928},  {3063, 2102, 3929},  {3079, 2100, 3934},
            {3094, 2101, 3935},  {3106, 2098, 3936},  {3119, 2094, 3936},
            {3130, 2097, 3934},  {3141, 2098, 3931},  {3153, 2100, 3930},
            {3168, 2102, 3926},  {3182, 2103, 3923},  {3191, 2114, 3923},
            {3201, 2118, 3922},  {3216, 2118, 3922},  {3229, 2115, 3923},
            {3243, 2107, 3923},  {3259, 2110, 3921},  {3269, 2111, 3918},
            {3284, 2119, 3917},  {3299, 2121, 3916},  {3314, 2123, 3915},
            {3323, 2123, 3916},  {3335, 2123, 3916},  {3346, 2122, 3919},
            {3357, 2121, 3919},  {3372, 2133, 3916},  {3382, 2136, 3919},
            {3394, 2139, 3925},  {3406, 2143, 3927},  {3422, 2150, 3928},
            {3438, 2152, 3929},  {3453, 2156, 3933},  {3466, 2160, 3933},
            {3479, 2165, 3934},  {3505, 2165, 3931},  {3526, 2167, 3931},
            {3544, 2168, 3932},  {3565, 2172, 3930},  {3582, 2170, 3929},
            {3596, 2177, 3926},  {3613, 2184, 3923},  {3639, 2183, 3928},
            {3652, 2185, 3926},  {3664, 2183, 3928},  {3683, 2184, 3927},
            {3702, 2187, 3925},  {3727, 2190, 3923},  {3747, 2192, 3922},
            {3761, 2196, 3917},  {3778, 2200, 3914},  {3790, 2202, 3912},
            {3797, 2205, 3912},  {3807, 2211, 3912},  {3818, 2216, 3911},
            {3831, 2218, 3911},  {3842, 2218, 3911},  {3855, 2216, 3908},
            {3881, 2216, 3906},  {3907, 2214, 3905},  {3929, 2218, 3905},
            {3943, 2220, 3899},  {3972, 2220, 3897},  {3988, 2220, 3901},
            {4014, 2218, 3901},  {4033, 2211, 3901},  {4053, 2211, 3901},
            {4068, 2211, 3900},  {4083, 2214, 3899},  {5186, 2494, 3839},
            {5206, 2504, 3844},  {5223, 2513, 3841},  {5245, 2525, 3830},
            {5268, 2546, 3820},  {5294, 2552, 3810},  {6022, 3194, 3907},
            {5997, 3177, 3905},  {5973, 3135, 3919},  {5940, 3111, 3951},
            {5893, 3070, 3952},  {5844, 3037, 3953},  {5742, 2929, 3867},
            {5678, 2865, 3878},  {5618, 2828, 3850},  {5578, 2789, 3838},
            {5472, 2655, 3850},  {5411, 2615, 3835},  {5379, 2592, 3822},
            {5334, 2567, 3813},  {5525, 2730, 3842},  {5800, 2958, 3914},
            {4549, 2293, 3839},  {4395, 2256, 3839},  {4353, 2247, 3842},
            {4313, 2240, 3842},  {4291, 2225, 3847},  {4233, 2227, 3842},
            {4209, 2228, 3842},  {4187, 2222, 3845},  {6092, 3240, 3944},
            {6137, 3256, 3960},  {6203, 3250, 3957},  {7285, 3605, 3743},
            {7366, 3576, 3717},  {7484, 3599, 3731},  {7569, 3577, 3723},
            {7616, 3582, 3715},  {7716, 3568, 3710},  {7778, 3562, 3688},
            {7856, 3552, 3657},  {8009, 3548, 3615},  {8119, 3526, 3583},
            {8345, 3505, 3524},  {8493, 3475, 3467},  {8585, 3479, 3435},
            {8599, 3475, 3436},  {8618, 3472, 3431},  {8632, 3471, 3426},
            {8644, 3469, 3416},  {8654, 3466, 3412},  {8668, 3466, 3420},
            {8681, 3465, 3420},  {8690, 3454, 3420},  {8700, 3451, 3430},
            {8723, 3431, 3444},  {8739, 3360, 3451},  {8771, 3311, 3451},
            {8829, 3312, 3451},  {8892, 3360, 3451},  {8938, 3409, 3451},
            {9020, 3499, 3390},  {9105, 3502, 3385},  {9147, 3490, 3364},
            {9242, 3525, 3370},  {9354, 3561, 3365},  {9442, 3577, 3364},
            {9603, 3572, 3342},  {9944, 3670, 3332},  {10022, 3722, 3316},
            {10075, 3745, 3292}, {10149, 3792, 3292}, {10224, 3820, 3292},
            {10315, 3803, 3283}, {10424, 3801, 3268}, {10577, 3818, 3239},
            {10873, 3852, 3170}, {10954, 3868, 3168}, {11010, 3878, 3199},
            {11096, 3879, 3218}, {11718, 3986, 3002}, {11624, 3930, 3028},
            {11527, 3918, 2992}, {11445, 3951, 2983}, {11354, 3950, 3064},
            {11311, 3988, 3063}, {11145, 3948, 3163}, {11796, 4033, 3011},
            {11833, 4029, 3018}, {11860, 4018, 3015}, {11867, 4024, 2978},
            {11877, 4030, 2957}, {11894, 4048, 2955}, {11932, 4058, 2945},
            {11950, 4069, 2941}, {11988, 4074, 2938}, {12008, 4079, 2936},
            {12030, 4081, 2933}, {12094, 4099, 2941}, {12183, 4139, 2951},
            {12309, 4281, 2959}, {12391, 4348, 2941}, {12764, 4445, 2947},
            {13102, 4539, 2933}, {13196, 4595, 2906}, {13382, 4694, 2889},
            {13427, 4722, 2877}, {13495, 4738, 2854}, {13557, 4772, 2822},
            {13603, 4800, 2857}, {13641, 4825, 2839}, {13675, 4822, 2839},
            {11199, 3968, 3134}, {11251, 3975, 3100}, {7216, 3589, 3702},
            {7106, 3611, 3703},  {7045, 3583, 3713},  {6969, 3549, 3726},
            {6883, 3529, 3745},  {6785, 3551, 3740},  {6664, 3538, 3756},
            {6588, 3534, 3781},  {6246, 3385, 3902},  {6319, 3460, 3883},
            {6523, 3480, 3865},  {6433, 3507, 3876},  {4121, 2230, 3866},
            {4100, 2224, 3889},  {4156, 2225, 3856},  {5127, 2499, 3840},
            {4958, 2444, 3840},  {4830, 2389, 3840},  {4705, 2323, 3840},
            {4594, 2293, 3840},  {4505, 2273, 3840},  {4456, 2268, 3840},
            {4392, 2257, 3840}};

        std::tuple<int, int, int> umbilicus = {0, 2503, 4082};
        int min_dist = 9999999999;
        for (const auto& point : umbilicus_scroll1a_zyx)
        { 
            int dist = abs(std::get<0>(point) - index);
            if (dist < min_dist)
            {
                umbilicus = point;
                min_dist = dist;
            }
        }

        printf("Found umbilicus %i %i %i (z y x)\n", std::get<0>(umbilicus),std::get<1>(umbilicus),std::get<2>(umbilicus));

        int size = 1024;

        int nearest_x = (std::get<2>(umbilicus) / size) * size;
        int nearest_y = (std::get<1>(umbilicus) / size) * size;
        //int nearest_x = ((std::get<2>(umbilicus) - (size / 2)) / size) * size;
        //int nearest_y = ((std::get<1>(umbilicus) - (size / 2)) / size) * size;

        int nearest_z = (index / size) * size;
        int nearest_z_remainder = index % size;

        printf("Nearest %i %i %i (z y x)\n", nearest_z, nearest_y, nearest_x);

        s32 vol_start[3] = {nearest_z, nearest_y, nearest_x};  // z y x
        s32 chunk_dims[3] = {size, size, size};                // z y x

        ChunkLoadState* state = vs_vol_get_chunk_start(vol, vol_start, chunk_dims);
        chunk* mychunk = NULL;
        while (!vs_vol_get_chunk_poll(state, &mychunk)) {
            // Do other work or sleep briefly
            usleep(10000);
        }

        slice* myslice = vs_slice_extract(mychunk, nearest_z_remainder);

        {
            // Convert slice data type: f32 to char.
            unsigned char* data = (unsigned char*)malloc(myslice->dims[0] * myslice->dims[1]);
            for (int y = 0; y < myslice->dims[1]; y++)
            {
                for (int x = 0; x < myslice->dims[0];x++)
                {
                    data[y * myslice->dims[1] + x] = (char)(myslice->data[y * myslice->dims[1] + x]);
                }
            }

            cv::Mat matSlice(myslice->dims[0], myslice->dims[1], CV_8UC1);
            std::memcpy(matSlice.data, data, myslice->dims[0] * myslice->dims[1]);
            
            mat = cv::Mat::zeros(sliceHeight(), sliceWidth(), CV_8UC1);
            cv::Rect dstRect(cv::Point(vol_start[2], vol_start[1]), cv::Size(chunk_dims[2], chunk_dims[1]));

            matSlice.copyTo(mat(dstRect));

            free(data);
        }

        vs_slice_free(myslice);
        vs_chunk_free(mychunk);
    } catch (std::runtime_error) {
    }
    return mat;
}

auto Volume::cache_slice_(int index) const -> cv::Mat
{
    // Check if the slice is in the cache.
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (cache_->contains(index)) {
            return cache_->get(index);
        }
    }

    {
        // Get the lock for this slice.
        auto& mutex = slice_mutexes_[index];

        // If the slice is not in the cache, get exclusive access to this slice's mutex.
        std::unique_lock<std::mutex> lock(mutex);
        // Check again to ensure the slice has not been added to the cache while waiting for the lock.
        {
            std::shared_lock<std::shared_mutex> lock(cache_mutex_);
            if (cache_->contains(index)) {
                return cache_->get(index);
            }
        }
        // Load the slice and add it to the cache.
        {
            auto slice = load_slice_(index);
            std::unique_lock<std::shared_mutex> lock(cache_mutex_);
            cache_->put(index, slice);
            return slice;
        }
    }

}


void Volume::cachePurge() const 
{
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    cache_->purge();
}

