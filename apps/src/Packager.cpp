//
// Created by Seth Parker on 7/30/15.
//

#include <iostream>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include "apps/SliceImage.hpp"
#include "vc/core/types/VolumePkg.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace vc = volcart;

enum class Flip { None, Horizontal, Vertical, Both };

struct VolumeInfo {
    fs::path path;
    std::string name;
    double voxelsize;
    Flip flipOption{Flip::None};
};

VolumeInfo GetVolumeInfo(fs::path slicesPath);
void AddVolume(vc::VolumePkg& volpkg, VolumeInfo info);

int main(int argc, char* argv[])
{
    ///// Parse the command line options /////
    // All command line options
    // clang-format off
    po::options_description options("Options");
    options.add_options()
        ("help,h", "Show this message")
        ("volpkg,v", po::value<std::string>()->required(),
           "Path for the output volume package")
        ("material-thickness,m", po::value<double>()->required(),
           "Estimated thickness of a material layer (in microns)")
        ("slices,s", po::value<std::vector<std::string>>(),
           "Directory of input slice data. Can be specified multiple times to "
           "add multiple volumes");

    // Useful transforms for origin adjustment
    po::options_description extras("Metadata");
    extras.add_options()
        ("name", po::value<std::string>(),
           "Set a descriptive name for the VolumePkg. "
           "Default: Filename specified by --volpkg");
    // clang-format on
    po::options_description all("Usage");
    all.add(options).add(extras);

    // parsed will hold the values of all parsed options as a Map
    po::variables_map parsed;
    po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);

    // Show the help message
    if (parsed.count("help") || argc < 2) {
        std::cout << all << std::endl;
        return EXIT_SUCCESS;
    }

    // Warn of missing options
    try {
        po::notify(parsed);
    } catch (po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    ///// New VolumePkg /////
    // Get the output volpkg path
    fs::path volpkgPath = parsed["volpkg"].as<std::string>();

    // Check the extension and make sure the pkg doesn't already exist
    if (volpkgPath.extension().string() != ".volpkg")
        volpkgPath.replace_extension(".volpkg");
    if (fs::exists(volpkgPath)) {
        std::cerr << "ERROR: Volume package already exists at path specified."
                  << std::endl;
        std::cerr << "This program does not currently allow for modification "
                     "of existing volume packages."
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Get volpkg name
    std::string vpkgName = volpkgPath.stem().string();
    if (parsed.count("name")) {
        vpkgName = parsed["name"].as<std::string>();
    }

    // Get material thickness
    auto thickness = parsed["material-thickness"].as<double>();

    // Generate an empty volpkg and save it to disk
    vc::VolumePkg volpkg(volpkgPath, vc::VOLPKG_VERSION_LATEST);
    volpkg.setMetadata("name", vpkgName);
    volpkg.setMetadata("materialthickness", thickness);
    volpkg.saveMetadata();

    ///// Add Volumes /////
    // Get the input slices dir
    std::vector<std::string> volumesPaths;
    if (parsed.count("slices")) {
        volumesPaths = parsed["slices"].as<std::vector<std::string>>();
    }

    // Get info
    std::vector<VolumeInfo> volumesList;
    for (auto& v : volumesPaths) {
        volumesList.emplace_back(GetVolumeInfo(v));
    }

    // Add them in sequence
    for (auto& v : volumesList) {
        AddVolume(volpkg, v);
    }
}

VolumeInfo GetVolumeInfo(fs::path slicesPath)
{
    VolumeInfo info;
    info.path = slicesPath;

    std::cout << "Describing Volume: " << slicesPath << std::endl;

    // Volume Name
    std::cout << "Enter a descriptive name for the volume: ";
    std::getline(std::cin, info.name);

    // get voxel size
    std::string input;
    do {
        std::cout << "Enter the voxel size of the volume in microns "
                     "(e.g. 13.546): ";
        std::getline(std::cin, input);
    } while (!boost::conversion::try_lexical_convert(input, info.voxelsize));

    // Flip options
    std::cout << "Flip options: Vertical flip (vf), horizontal flip (hf), "
                 "both, [none] : ";
    std::getline(std::cin, input);

    if (input == "vf") {
        info.flipOption = Flip::Vertical;
    } else if (input == "hf") {
        info.flipOption = Flip::Horizontal;
    } else if (input == "both") {
        info.flipOption = Flip::Both;
    }

    return info;
}

void AddVolume(vc::VolumePkg& volpkg, VolumeInfo info)
{
    std::cout << "Adding Volume: " << info.path << std::endl;

    // Filter the slice path directory by extension and sort the vector of files
    std::cout << "Reading the slice directory..." << std::endl;
    std::vector<volcart::SliceImage> slices;
    if (!fs::exists(info.path) || !fs::is_directory(info.path)) {
        std::cerr
            << "ERROR: Slices directory does not exist/is not a directory."
            << std::endl;
        std::cerr << "Please provide a directory of slice images." << std::endl;
        return;
    }

    // Filter out subfiles that aren't TIFs
    // To-Do: #177
    fs::directory_iterator subfile(info.path);
    fs::directory_iterator dirEnd;
    while (subfile != dirEnd) {
        auto ext(subfile->path().extension().string());
        ext = boost::to_upper_copy<std::string>(ext);
        if (fs::is_regular_file(subfile->path()) &&
            (ext == ".TIF" || ext == ".TIFF")) {
            volcart::SliceImage temp;
            temp.path = *subfile;
            slices.push_back(temp);
        }
        ++subfile;
    }

    if (slices.empty()) {
        std::cerr << "ERROR: No supported image files found in provided slices "
                     "directory."
                  << std::endl;
        return;
    }

    // Sort the Slices by their filenames
    std::sort(slices.begin(), slices.end(), SlicePathLessThan);
    std::cout << "Slice images found: " << slices.size() << std::endl;

    ///// Analyze the slices /////
    bool vol_consistent = true;
    double vol_min{}, vol_max{};
    uint64_t counter = 1;
    for (auto slice = slices.begin(); slice != slices.end(); ++slice) {
        std::cout << "Analyzing slice: " << counter << "/" << slices.size()
                  << "\r" << std::flush;
        if (!slice->analyze())
            continue;  // skip if we can't analyze

        // Compare all slices to the properties of the first slice
        if (slice == slices.begin()) {
            vol_min = slice->min();
            vol_max = slice->max();
        } else {
            // Check for consistency of slices
            if (*slice != *slices.begin()) {
                vol_consistent = false;
                std::cerr << std::endl
                          << slice->path.filename()
                          << " does not match the initial slice of the volume."
                          << std::endl;
                continue;
            }

            // Update the volume's min and max
            if (slice->min() < vol_min)
                vol_min = slice->min();
            if (slice->max() > vol_max)
                vol_max = slice->max();
        }

        ++counter;
    }
    std::cout << std::endl;
    if (!vol_consistent) {
        std::cerr << "ERROR: Slices in slice directory do not have matching "
                     "properties (width/height/depth)."
                  << std::endl;
        return;
    }

    ///// Add data to the volume /////
    // Metadata
    auto volume = volpkg.newVolume(info.name);
    volume->setNumberOfSlices(slices.size());
    volume->setSliceWidth(slices.front().width());
    volume->setSliceHeight(slices.front().height());
    volume->setVoxelSize(info.voxelsize);

    // Scale 8-bit min/max values
    // To-Do: Handle other bit depths
    if (slices.begin()->depth() == 0) {
        vol_min = vol_min * 65535.00 / 255.00;
        vol_max = vol_max * 65535.00 / 255.00;
    }
    volume->setMin(vol_min);
    volume->setMax(vol_max);
    volume->saveMetadata();  // Save final metadata changes to disk

    // Do we need to flip?
    auto needsFlip = info.flipOption == Flip::Horizontal ||
                     info.flipOption == Flip::Vertical ||
                     info.flipOption == Flip::Both;

    counter = 0;
    for (auto& slice : slices) {
        std::cout << "Saving slice image to volume package: " << counter + 1
                  << "/" << slices.size() << "\r" << std::flush;
        if (slice.needsConvert() || needsFlip) {
            // Get slice
            auto tmp = slice.conformedImage();

            // Apply flips
            switch (info.flipOption) {
                case Flip::Both:
                    cv::flip(tmp, tmp, -1);
                    break;
                case Flip::Vertical:
                    cv::flip(tmp, tmp, 0);
                    break;
                case Flip::Horizontal:
                    cv::flip(tmp, tmp, 1);
                    break;
                case Flip::None:
                    // Do nothing
                    break;
            }

            // Add to volume
            volume->setSliceData(counter, tmp);
        } else {
            fs::copy(slice.path, volume->getSlicePath(counter));
        }

        ++counter;
    }
    std::cout << std::endl;

    return;
}