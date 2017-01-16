#pragma once

#ifdef VC_USE_BULLET

#include <BulletSoftBody/btSoftBody.h>
#include <BulletSoftBody/btSoftRigidDynamicsWorld.h>

#include "core/vc_defines.h"

namespace volcart
{
namespace meshing
{
/**
 * @class ITK2Bullet
 * @author Abigail Coleman
 * @date 10/21/15
 *
 * @brief Convert from an ITKMesh to a btSoftBody.
 *
 * Copy vertex and face information from a btSoftBody to an ITKMesh.
 *
 * @ingroup Meshing
 */
class ITK2Bullet
{
public:
    /**
    * @param worldInfo Soft Body world configuration
    */
    ITK2Bullet(
        ITKMesh::Pointer input,
        btSoftBodyWorldInfo& worldInfo,
        btSoftBody** output);
};

/**
 * @class Bullet2ITK
 * @author Abigail Coleman
 * @date 10/21/15
 *
 * @brief Convert from a btSoftBody to an ITKMesh.
 *
 * Copy vertex and face information from a btSoftBody to an ITKMesh.
 *
 * @ingroup Meshing
 */
class Bullet2ITK
{
public:
    Bullet2ITK(btSoftBody* input, ITKMesh::Pointer output);
};
}
}

#endif  // VC_USE_BULLET
