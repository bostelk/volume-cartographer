#ifndef TENSORPARTICLE_H
#define TENSORPARTICLE_H

#include <iostream>

#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>

#include "volumepkg.h"

#include "field.h"
#include "particle.h"
#include "chain.h"
#include "slice.h"

namespace volcart {
    namespace segmentation {
        pcl::PointCloud<pcl::PointXYZRGB> localResliceParticleSim(pcl::PointCloud<pcl::PointXYZRGB>::Ptr segPath, VolumePkg volpkg, double gravity_scale = 0.5, int threshold = 1, int endOffset = -1);
    }
}

#endif