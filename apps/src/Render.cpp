// render.cpp
// Abigail Coleman Feb. 2015

#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vtkCleanPolyData.h>

#include "core/io/OBJWriter.hpp"
#include "core/io/PLYReader.hpp"
#include "core/io/PLYWriter.hpp"
#include "core/types/VolumePkg.hpp"
#include "core/util/MeshMath.hpp"
#include "core/vc_defines.hpp"
#include "external/GetMemorySize.hpp"
#include "meshing/ACVD.hpp"
#include "meshing/ITK2VTK.hpp"
#include "meshing/SmoothNormals.hpp"
#include "texturing/AngleBasedFlattening.hpp"
#include "texturing/CompositeTextureV2.hpp"

using namespace volcart;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

// Volpkg version required by this app
static constexpr int VOLPKG_SUPPORTED_VERSION = 3;

// Min. number of points required to do flattening
static constexpr uint16_t CLEANER_MIN_REQ_POINTS = 100;

int main(int argc, char* argv[])
{
    std::cout << "vc_render" << std::endl;
    ///// Parse the command line options /////
    fs::path volpkgPath, outputPath;
    std::string segID;
    double radius;
    CompositeOption aFilterOption;
    DirectionOption aDirectionOption;

    try {
        // All command line options
        // clang-format off
        po::options_description options("Options");
        options.add_options()
            ("help,h", "Show this message")
            ("volpkg,v", po::value<std::string>()->required(),
                "Path to the volume package")
            ("seg,s", po::value<std::string>()->required(),
                "Segmenation ID number")
            ("radius,r", po::value<int>()->required(), "Texture search radius")
            ("method,m", po::value<int>()->default_value(1),
                "Texture method:\n"
                "  0 = Intersection\n"
                "  1 = Non-Maximum Suppression\n"
                "  2 = Maximum\n"
                "  3 = Minimum\n"
                "  4 = Median w/ Averaging\n"
                "  5 = Median\n"
                "  6 = Mean\n")
            ("direction,d", po::value<int>()->default_value(0),
                "Sample Direction:\n"
                "  0 = Omni\n"
                "  1 = Positive\n"
                "  2 = Negative\n")
            ("output-file,o", po::value<std::string>(),
                "Output file path. If not specified, file will be saved to"
                " volume package.");
        // clang-format on

        // parsedOptions will hold the values of all parsed options as a Map
        po::variables_map parsedOptions;
        po::store(
            po::command_line_parser(argc, argv).options(options).run(),
            parsedOptions);

        // Show the help message
        if (parsedOptions.count("help") || argc < 2) {
            std::cout << options << std::endl;
            return EXIT_SUCCESS;
        }

        // Warn of missing options
        try {
            po::notify(parsedOptions);
        } catch (po::error& e) {
            std::cerr << "ERROR: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }

        // Get the parsed options
        volpkgPath = parsedOptions["volpkg"].as<std::string>();
        segID = parsedOptions["seg"].as<std::string>();
        radius = parsedOptions["radius"].as<int>();
        aFilterOption =
            static_cast<CompositeOption>(parsedOptions["method"].as<int>());
        aDirectionOption =
            static_cast<DirectionOption>(parsedOptions["direction"].as<int>());

        // Check for output file
        if (parsedOptions.count("output-file")) {
            outputPath = parsedOptions["output-file"].as<std::string>();
            if (fs::exists(fs::canonical(outputPath.parent_path())))
                outputPath = fs::canonical(outputPath.parent_path()).string() +
                             "/" + outputPath.filename().string();
            else
                std::cerr << "ERROR: Cannot write to provided output file. "
                             "Output directory does not exist."
                          << std::endl;
        }

    } catch (std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    ///// Load the volume package /////
    if (fs::exists(volpkgPath) ||
        (fs::canonical(volpkgPath).extension() != ".volpkg")) {
        volpkgPath = fs::canonical(volpkgPath);
    } else {
        std::cerr << "ERROR: Volume package does not exist/not recognized at "
                     "provided path: "
                  << volpkgPath << std::endl;
        return EXIT_FAILURE;
    }

    VolumePkg vpkg(volpkgPath);
    if (vpkg.getVersion() != VOLPKG_SUPPORTED_VERSION) {
        std::cerr << "ERROR: Volume package is version " << vpkg.getVersion()
                  << " but this program requires a version "
                  << std::to_string(VOLPKG_SUPPORTED_VERSION) << "."
                  << std::endl;
        return EXIT_FAILURE;
    }
    double cacheBytes = 0.75 * SystemMemorySize();
    vpkg.volume().setCacheMemoryInBytes(static_cast<size_t>(cacheBytes));

    ///// Set the segmentation ID /////
    vpkg.setActiveSegmentation(segID);
    fs::path meshName = vpkg.getMeshPath();

    // try to convert the ply to an ITK mesh
    volcart::io::PLYReader reader(meshName);
    try {
        reader.read();
    } catch (volcart::IOException e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    auto input = reader.getMesh();

    // Calculate sampling density
    double voxelsize = vpkg.getVoxelSize();
    double sa = volcart::meshmath::SurfaceArea(input) *
                (voxelsize * voxelsize) *
                (0.001 * 0.001);  // convert vx^2 -> mm^2;
    double densityFactor = 50;
    auto numberOfVertices =
        static_cast<uint16_t>(std::round(densityFactor * sa));
    numberOfVertices = (numberOfVertices < CLEANER_MIN_REQ_POINTS)
                           ? CLEANER_MIN_REQ_POINTS
                           : numberOfVertices;

    // Convert to polydata
    auto vtkMesh = vtkSmartPointer<vtkPolyData>::New();
    volcart::meshing::ITK2VTK(input, vtkMesh);

    // Decimate using ACVD
    std::cout << "Resampling mesh..." << std::endl;
    auto acvdMesh = vtkSmartPointer<vtkPolyData>::New();
    volcart::meshing::ACVD(vtkMesh, acvdMesh, numberOfVertices);

    // Merge Duplicates
    // Note: This merging has to be the last in the process chain for some
    // really weird reason. - SP
    auto Cleaner = vtkSmartPointer<vtkCleanPolyData>::New();
    Cleaner->SetInputData(acvdMesh);
    Cleaner->Update();

    auto itkACVD = volcart::ITKMesh::New();
    volcart::meshing::VTK2ITK(Cleaner->GetOutput(), itkACVD);

    // ABF flattening
    std::cout << "Computing parameterization..." << std::endl;
    volcart::texturing::AngleBasedFlattening abf(itkACVD);
    // abf.setABFMaxIterations(5);
    abf.compute();

    // Get uv map
    volcart::UVMap uvMap = abf.getUVMap();
    auto width = static_cast<int>(std::ceil(uvMap.ratio().width));
    auto height = static_cast<int>(
        std::ceil(static_cast<double>(width) / uvMap.ratio().aspect));

    volcart::texturing::CompositeTextureV2 result(
        itkACVD, vpkg, uvMap, radius, width, height, aFilterOption,
        aDirectionOption);

    // Setup rendering
    volcart::Rendering rendering;
    rendering.setTexture(result.texture());
    rendering.setMesh(itkACVD);

    if (outputPath.extension() == ".PLY" || outputPath.extension() == ".ply") {
        std::cout << "Writing to PLY..." << std::endl;
        volcart::io::PLYWriter writer(
            outputPath.string(), itkACVD, result.texture());
        writer.write();
    } else if (
        outputPath.extension() == ".OBJ" || outputPath.extension() == ".obj") {
        std::cout << "Writing to OBJ..." << std::endl;
        volcart::io::OBJWriter writer;
        writer.setMesh(itkACVD);
        writer.setRendering(rendering);
        writer.setPath(outputPath.string());
        writer.write();
    } else if (
        outputPath.extension() == ".PNG" || outputPath.extension() == ".png") {
        std::cout << "Writing to PNG..." << std::endl;
        cv::imwrite(outputPath.string(), rendering.getTexture().image(0));
    } else {
        std::cout << "Writing to Volume Package..." << std::endl;
        vpkg.saveMesh(itkACVD, result.texture());
    }

    return EXIT_SUCCESS;
}  // end main