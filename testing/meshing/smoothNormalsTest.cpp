//
// Created by Ryan Taber on 1/29/16.
//

#ifndef VC_PREBUILT_LIBS
#define BOOST_TEST_DYN_LINK
#endif
#define BOOST_TEST_MODULE smoothNormals

#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_log.hpp>
#include "vc_defines.h"
#include "shapes.h"
#include "parsingHelpers.h"
#include "smoothNormals.h"


/************************************************************************************
 *                                                                                  *
 *  smoothNormalsTest.cpp - tests the functionality of meshing/smoothNormals.cpp    *
 *  The ultimate goal of this file is the following:                                *
 *                                                                                  *
 *        1. confirm volcart::meshing::smoothNormals() works as expected            *
 *                                                                                  *
 *  This file is broken up into a test fixture (SmoothNormalsFixture) which         *
 *  intitializes the objects used for the test case.                                *
 *                                                                                  *
 *  1. CompareSmoothedPlane (fixture test case)                                     *
 *  2. CompareSmoothedCube (fixture test case)                                      *
 *  3. CompareSmoothedSphere (fixture test case)                                    *
 *  4. CompareSmoothedArch (fixture test case)                                      *
 *  5. SmoothWithZeroRadiusTest (fixture test case)                                 *
 *                                                                                  *
 * Input:                                                                           *
 *     No required inputs for this sample test. All test objects are created        *
 *     internally.                                                                  *
 *                                                                                  *
 * Test-Specific Output:                                                            *
 *     Specific test output only given on failure of any tests. Otherwise, general  *
 *     number of testing errors is output.                                          *
 *                                                                                  *
 * Miscellaneous:                                                                   *
 *     See the /testing/meshing wiki for more information on this test              *
 * **********************************************************************************/


/*
 * This builds objects for the case below that reference
 * the fixture as their second argument
 *
 */

struct SmoothNormalsFixture {

    SmoothNormalsFixture() {

        //smoothing radius and _Tolerance value for later comparisons
        _SmoothingFactor = 2;
        _Tolerance = 0.00001;

        //assign input meshes that will be smoothed
        _in_PlaneMesh = _Plane.itkMesh();
        _in_CubeMesh = _Cube.itkMesh();
        _in_ArchMesh = _Arch.itkMesh();
        _in_SphereMesh = _Sphere.itkMesh();
        _in_ConeMesh = _Cone.itkMesh();

        //call smoothNormals and assign results to output meshes
        _out_SmoothedPlaneMesh = volcart::meshing::smoothNormals(_in_PlaneMesh, _SmoothingFactor);
        _out_SmoothedCubeMesh = volcart::meshing::smoothNormals(_in_CubeMesh, _SmoothingFactor);
        _out_SmoothedArchMesh = volcart::meshing::smoothNormals(_in_ArchMesh, _SmoothingFactor);
        _out_SmoothedSphereMesh = volcart::meshing::smoothNormals(_in_SphereMesh, _SmoothingFactor);
        _out_SmoothedConeMesh = volcart::meshing::smoothNormals(_in_ConeMesh, _SmoothingFactor);

        //read in saved obj files created by SmoothNormalsExample.cpp
        volcart::testing::ParsingHelpers::parseObjFile("PlaneWithSmoothedNormals.obj", _SavedPlanePoints, _SavedPlaneCells);
        volcart::testing::ParsingHelpers::parseObjFile("CubeWithSmoothedNormals.obj", _SavedCubePoints, _SavedCubeCells);
        volcart::testing::ParsingHelpers::parseObjFile("ArchWithSmoothedNormals.obj", _SavedArchPoints, _SavedArchCells);
        volcart::testing::ParsingHelpers::parseObjFile("SphereWithSmoothedNormals.obj", _SavedSpherePoints, _SavedSphereCells);
        volcart::testing::ParsingHelpers::parseObjFile("ConeWithSmoothedNormals.obj", _SavedConePoints, _SavedConeCells);

        std::cerr << "setting up smoothNormals objects" << std::endl;
    }

    ~SmoothNormalsFixture(){ std::cerr << "cleaning up smoothNormals objects" << std::endl; }

    //init input and output mesh ptrs
    VC_MeshType::Pointer _in_PlaneMesh, _in_CubeMesh, _in_SphereMesh, _in_ArchMesh, _in_ConeMesh;
    VC_MeshType::Pointer _out_SmoothedPlaneMesh, _out_SmoothedCubeMesh, _out_SmoothedSphereMesh,
                         _out_SmoothedArchMesh, _out_SmoothedConeMesh;
    //init shapes
    volcart::shapes::Plane _Plane;
    volcart::shapes::Cube _Cube;
    volcart::shapes::Sphere _Sphere;
    volcart::shapes::Arch _Arch;
    volcart::shapes::Cone _Cone;

    double _Tolerance;
    double _SmoothingFactor;

    //init vectors to hold points and cells from savedITK data files
    std::vector<VC_Vertex> _SavedPlanePoints, _SavedCubePoints, _SavedArchPoints, _SavedSpherePoints, _SavedConePoints;
    std::vector<VC_Cell> _SavedPlaneCells, _SavedCubeCells, _SavedArchCells, _SavedSphereCells, _SavedConeCells;
};


/*
 * The next four tests use the obj files representing the smoothed shapes created by smoothingExample.cpp
 * and compares these files to test-specific calls smoothNormals() using the same shape objects.
 *
 * Smoothing factor should be 2 for each test case
 *
 * Split the tests into four cases for log purposes and pinpointing errors faster if there should be
 * an issue in the future.
 *
 */

BOOST_FIXTURE_TEST_CASE(CompareFixtureSmoothedPlaneWithSavedPlaneTest, SmoothNormalsFixture){

    //Check number of points in each mesh
    BOOST_CHECK_EQUAL( _out_SmoothedPlaneMesh->GetNumberOfPoints(), _SavedPlanePoints.size() );
    BOOST_CHECK_EQUAL(_out_SmoothedPlaneMesh->GetNumberOfCells(), _SavedPlaneCells.size());

    //points
    for ( size_t p_id = 0; p_id < _out_SmoothedPlaneMesh ->GetNumberOfPoints(); ++p_id) {


        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedPlaneMesh->GetPoint(p_id)[0], _SavedPlanePoints[p_id].x, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedPlaneMesh->GetPoint(p_id)[1], _SavedPlanePoints[p_id].y, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedPlaneMesh->GetPoint(p_id)[2], _SavedPlanePoints[p_id].z, _Tolerance);
    }

    //normals
    int p = 0;
    VC_PointsInMeshIterator point = _out_SmoothedPlaneMesh->GetPoints()->Begin();
    for ( ; point != _out_SmoothedPlaneMesh->GetPoints()->End(); ++point ) {

        VC_PixelType out_PlaneNormal;
        _out_SmoothedPlaneMesh->GetPointData(point.Index(), &out_PlaneNormal);

        //Now compare the normals for the two meshes
        BOOST_CHECK_CLOSE_FRACTION(out_PlaneNormal[0], _SavedPlanePoints[p].nx, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_PlaneNormal[1], _SavedPlanePoints[p].ny, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_PlaneNormal[2], _SavedPlanePoints[p].nz, _Tolerance);

        p++;

    }

    //cells
    VC_CellIterator out_PlaneCell = _out_SmoothedPlaneMesh->GetCells()->Begin();

    int c = 0;

    while (out_PlaneCell != _out_SmoothedPlaneMesh->GetCells()->End()) {

        //Initialize Iterators for Points in a Cell
        VC_PointsInCellIterator out_PlaneMeshPointId = out_PlaneCell.Value()->PointIdsBegin();

        int counter = 0;
        //while we have points in the cell
        while ( out_PlaneMeshPointId != out_PlaneCell.Value()->PointIdsEnd() ) {

            //Now to check the points within the cells
            if (counter == 0)
                BOOST_CHECK_EQUAL(*out_PlaneMeshPointId, _SavedPlaneCells[c].v1);
            else if(counter == 1)
                BOOST_CHECK_EQUAL(*out_PlaneMeshPointId, _SavedPlaneCells[c].v2);
            else if (counter == 2)
                BOOST_CHECK_EQUAL(*out_PlaneMeshPointId, _SavedPlaneCells[c].v3);

            //increment points
            out_PlaneMeshPointId++;
            counter++;
        }

        //increment cells
        ++out_PlaneCell;
        ++c;
    }
}


//           //
//           //
//   CUBE    //
//           //
//           //

BOOST_FIXTURE_TEST_CASE(CompareFixtureSmoothedCubeWithSavedCubeTest, SmoothNormalsFixture){

    //Check number of points and cells in each mesh
    BOOST_CHECK_EQUAL( _out_SmoothedCubeMesh->GetNumberOfPoints(), _SavedCubePoints.size() );
    BOOST_CHECK_EQUAL(_out_SmoothedCubeMesh->GetNumberOfCells(), _SavedCubeCells.size());
    
    //points
    for ( size_t p_id = 0; p_id < _out_SmoothedCubeMesh ->GetNumberOfPoints(); ++p_id) {


        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedCubeMesh->GetPoint(p_id)[0], _SavedCubePoints[p_id].x, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedCubeMesh->GetPoint(p_id)[1], _SavedCubePoints[p_id].y, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedCubeMesh->GetPoint(p_id)[2], _SavedCubePoints[p_id].z, _Tolerance);
    }

    //normals
    int p = 0;
    VC_PointsInMeshIterator point = _out_SmoothedCubeMesh->GetPoints()->Begin();
    for ( ; point != _out_SmoothedCubeMesh->GetPoints()->End(); ++point ) {

        VC_PixelType out_CubeNormal;
        _out_SmoothedCubeMesh->GetPointData(point.Index(), &out_CubeNormal);

        //Now compare the normals for the two meshes
        BOOST_CHECK_CLOSE_FRACTION(out_CubeNormal[0], _SavedCubePoints[p].nx, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_CubeNormal[1], _SavedCubePoints[p].ny, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_CubeNormal[2], _SavedCubePoints[p].nz, _Tolerance);

        p++;

    }

    //cells
    VC_CellIterator out_CubeCell = _out_SmoothedCubeMesh->GetCells()->Begin();

    int c = 0;

    while (out_CubeCell != _out_SmoothedCubeMesh->GetCells()->End()) {

        //Initialize Iterators for Points in a Cell
        VC_PointsInCellIterator out_CubeMeshPointId = out_CubeCell.Value()->PointIdsBegin();

        int counter = 0;
        //while we have points in the cell
        while (out_CubeMeshPointId != out_CubeCell.Value()->PointIdsEnd() ) {

            //Now to check the points within the cells
            if (counter == 0)
                BOOST_CHECK_EQUAL(*out_CubeMeshPointId, _SavedCubeCells[c].v1);
            else if(counter == 1)
                BOOST_CHECK_EQUAL(*out_CubeMeshPointId, _SavedCubeCells[c].v2);
            else if (counter == 2)
                BOOST_CHECK_EQUAL(*out_CubeMeshPointId, _SavedCubeCells[c].v3);

            //increment points
            out_CubeMeshPointId++;
            counter++;

        }

        //increment cells
        ++out_CubeCell;
        ++c;
    }
}

//             //
//             //
//   SPHERE    //
//             //
//             //

BOOST_FIXTURE_TEST_CASE(CompareFixtureSmoothedSphereWithSavedSphereTest, SmoothNormalsFixture){

    //Check number of points in each mesh
    BOOST_CHECK_EQUAL( _out_SmoothedSphereMesh->GetNumberOfPoints(), _SavedSpherePoints.size() );
    BOOST_CHECK_EQUAL(_out_SmoothedSphereMesh->GetNumberOfCells(), _SavedSphereCells.size());
    
    //points
    for ( size_t p_id = 0; p_id < _out_SmoothedSphereMesh ->GetNumberOfPoints(); ++p_id) {


        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedSphereMesh->GetPoint(p_id)[0], _SavedSpherePoints[p_id].x, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedSphereMesh->GetPoint(p_id)[1], _SavedSpherePoints[p_id].y, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedSphereMesh->GetPoint(p_id)[2], _SavedSpherePoints[p_id].z, _Tolerance);
    }

    //normals
    int p = 0;
    VC_PointsInMeshIterator point = _out_SmoothedSphereMesh->GetPoints()->Begin();
    for ( ; point != _out_SmoothedSphereMesh->GetPoints()->End(); ++point ) {

        VC_PixelType out_SphereNormal;
        _out_SmoothedSphereMesh->GetPointData(point.Index(), &out_SphereNormal);

        //Now compare the normals for the two meshes
        BOOST_CHECK_CLOSE_FRACTION(out_SphereNormal[0], _SavedSpherePoints[p].nx, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_SphereNormal[1], _SavedSpherePoints[p].ny, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_SphereNormal[2], _SavedSpherePoints[p].nz, _Tolerance);

        p++;

    }

    //cells
    VC_CellIterator out_SphereCell = _out_SmoothedSphereMesh->GetCells()->Begin();

    int c = 0;

    while (out_SphereCell != _out_SmoothedSphereMesh->GetCells()->End()) {

        //Initialize Iterators for Points in a Cell
        VC_PointsInCellIterator out_SphereMeshPointId = out_SphereCell.Value()->PointIdsBegin();

        int counter = 0;
        //while we have points in the cell
        while (out_SphereMeshPointId != out_SphereCell.Value()->PointIdsEnd() ) {

            //Now to check the points within the cells
            if (counter == 0)
                BOOST_CHECK_EQUAL(*out_SphereMeshPointId, _SavedSphereCells[c].v1);
            else if(counter == 1)
                BOOST_CHECK_EQUAL(*out_SphereMeshPointId, _SavedSphereCells[c].v2);
            else if (counter == 2)
                BOOST_CHECK_EQUAL(*out_SphereMeshPointId, _SavedSphereCells[c].v3);

            //increment points
            out_SphereMeshPointId++;
            counter++;

        }

        //increment cells
        ++out_SphereCell;
        ++c;
    }
}


//           //
//   ARCH    //
//           //


BOOST_FIXTURE_TEST_CASE(CompareFixtureSmoothedArchWithSavedArchTest, SmoothNormalsFixture){

    //Check number of points in each mesh
    BOOST_CHECK_EQUAL( _out_SmoothedArchMesh->GetNumberOfPoints(), _SavedArchPoints.size() );
    BOOST_CHECK_EQUAL(_out_SmoothedArchMesh->GetNumberOfCells(), _SavedArchCells.size());

    //points
    for ( size_t p_id = 0; p_id < _out_SmoothedArchMesh ->GetNumberOfPoints(); ++p_id) {


        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedArchMesh->GetPoint(p_id)[0], _SavedArchPoints[p_id].x, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedArchMesh->GetPoint(p_id)[1], _SavedArchPoints[p_id].y, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedArchMesh->GetPoint(p_id)[2], _SavedArchPoints[p_id].z, _Tolerance);
    }

    //normals
    int p = 0;
    VC_PointsInMeshIterator point = _out_SmoothedArchMesh->GetPoints()->Begin();
    for ( ; point != _out_SmoothedArchMesh->GetPoints()->End(); ++point ) {

        VC_PixelType out_ArchNormal;
        _out_SmoothedArchMesh->GetPointData(point.Index(), &out_ArchNormal);

        //Now compare the normals for the two meshes
        BOOST_CHECK_CLOSE_FRACTION(out_ArchNormal[0], _SavedArchPoints[p].nx, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_ArchNormal[1], _SavedArchPoints[p].ny, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_ArchNormal[2], _SavedArchPoints[p].nz, _Tolerance);

        p++;

    }

    // Initialize Cell Iterators
    VC_CellIterator out_ArchCell = _out_SmoothedArchMesh->GetCells()->Begin();

    int c = 0;

    while (out_ArchCell != _out_SmoothedArchMesh->GetCells()->End()) {

        //Initialize Iterators for Points in a Cell
        VC_PointsInCellIterator out_ArchMeshPointId = out_ArchCell.Value()->PointIdsBegin();

        int counter = 0;
        //while we have points in the cell
        while ( out_ArchMeshPointId != out_ArchCell.Value()->PointIdsEnd() ) {

            //Now to check the points within the cells
            if (counter == 0)
                BOOST_CHECK_EQUAL(*out_ArchMeshPointId, _SavedArchCells[c].v1);
            else if(counter == 1)
                BOOST_CHECK_EQUAL(*out_ArchMeshPointId, _SavedArchCells[c].v2);
            else if (counter == 2)
                BOOST_CHECK_EQUAL(*out_ArchMeshPointId, _SavedArchCells[c].v3);

            //increment points
            out_ArchMeshPointId++;
            counter++;

        }

        //increment cells
        ++out_ArchCell;
        ++c;
    }
}


//           //
//   CONE    //
//           //

BOOST_FIXTURE_TEST_CASE(CompareFixtureSmoothedConeWithSavedConeTest, SmoothNormalsFixture){

    //Check number of points in each mesh
    BOOST_CHECK_EQUAL( _out_SmoothedConeMesh->GetNumberOfPoints(), _SavedConePoints.size() );

    //compare point values
    for ( size_t p_id = 0; p_id < _out_SmoothedConeMesh ->GetNumberOfPoints(); ++p_id) {

        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedConeMesh->GetPoint(p_id)[0], _SavedConePoints[p_id].x, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedConeMesh->GetPoint(p_id)[1], _SavedConePoints[p_id].y, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(_out_SmoothedConeMesh->GetPoint(p_id)[2], _SavedConePoints[p_id].z, _Tolerance);
    }

    //normals
    int p = 0;
    VC_PointsInMeshIterator point = _out_SmoothedConeMesh->GetPoints()->Begin();
    for ( ; point != _out_SmoothedConeMesh->GetPoints()->End(); ++point ) {

        VC_PixelType out_ConeNormal;
        _out_SmoothedConeMesh->GetPointData(point.Index(), &out_ConeNormal);

        //Now compare the normals for the two meshes
        BOOST_CHECK_CLOSE_FRACTION(out_ConeNormal[0], _SavedConePoints[p].nx, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_ConeNormal[1], _SavedConePoints[p].ny, _Tolerance);
        BOOST_CHECK_CLOSE_FRACTION(out_ConeNormal[2], _SavedConePoints[p].nz, _Tolerance);

        p++;

    }

    //               //
    // compare cells //
    //               //

    BOOST_CHECK_EQUAL(_out_SmoothedConeMesh->GetNumberOfCells(), _SavedConeCells.size());

    // Initialize Cell Iterators
    VC_CellIterator out_ConeCell = _out_SmoothedConeMesh->GetCells()->Begin();

    int c_id = 0;

    while ( out_ConeCell != _out_SmoothedConeMesh->GetCells()->End()) {

        //Initialize Iterators for Points in a Cell
        VC_PointsInCellIterator out_ConeMeshPointId = out_ConeCell.Value()->PointIdsBegin();

        int counter = 0;
        //while we have points in the cell
        while ( out_ConeMeshPointId != out_ConeCell.Value()->PointIdsEnd() ) {

            //Now to check the points within the cells
            if (counter == 0)
                BOOST_CHECK_EQUAL(*out_ConeMeshPointId, _SavedConeCells[c_id].v1);
            else if(counter == 1)
                BOOST_CHECK_EQUAL(*out_ConeMeshPointId, _SavedConeCells[c_id].v2);
            else if (counter == 2)
                BOOST_CHECK_EQUAL(*out_ConeMeshPointId, _SavedConeCells[c_id].v3);

            //increment points
            out_ConeMeshPointId++;
            counter++;

        }

        //increment cells
        ++out_ConeCell;
        ++c_id;
    }
}

/*
 * Testing a zero radius smoothing factor
 *
 * Expected output would be an unchanged mesh
 *
 */

BOOST_FIXTURE_TEST_CASE(SmoothWithZeroRadiusTest, SmoothNormalsFixture){

    //call smoothNormals() and assign results
    VC_MeshType::Pointer ZeroRadiusSmoothedMesh = volcart::meshing::smoothNormals(_in_ArchMesh, 0);

    //check number of points and cells are equivalent between the two meshes
    BOOST_CHECK_EQUAL(_in_ArchMesh->GetNumberOfPoints(), ZeroRadiusSmoothedMesh->GetNumberOfPoints());
    BOOST_CHECK_EQUAL(_in_ArchMesh->GetNumberOfCells(), ZeroRadiusSmoothedMesh->GetNumberOfCells());


    //compare point values
    for ( size_t p_id = 0; p_id < _in_ArchMesh ->GetNumberOfPoints(); ++p_id) {

        BOOST_CHECK_EQUAL(_in_ArchMesh->GetPoint(p_id)[0], ZeroRadiusSmoothedMesh->GetPoint(p_id)[0]);
        BOOST_CHECK_EQUAL(_in_ArchMesh->GetPoint(p_id)[1], ZeroRadiusSmoothedMesh->GetPoint(p_id)[1]);
        BOOST_CHECK_EQUAL(_in_ArchMesh->GetPoint(p_id)[2], ZeroRadiusSmoothedMesh->GetPoint(p_id)[2]);
    }

    //compare normals
    VC_PointsInMeshIterator point = _in_ArchMesh->GetPoints()->Begin();

    for ( ; point != _in_ArchMesh->GetPoints()->End(); ++point ) {

        VC_PixelType in_ArchNormal, ZeroRadiusNormal;
        _in_ArchMesh->GetPointData(point.Index(), &in_ArchNormal);
        ZeroRadiusSmoothedMesh->GetPointData(point.Index(), &ZeroRadiusNormal);

        //Now compare the normals for the two meshes
        BOOST_CHECK_EQUAL(in_ArchNormal[0], ZeroRadiusNormal[0]);
        BOOST_CHECK_EQUAL(in_ArchNormal[1], ZeroRadiusNormal[1]);
        BOOST_CHECK_EQUAL(in_ArchNormal[2], ZeroRadiusNormal[2]);

    }

    //               //
    // compare cells //
    //               //

    // Initialize Cell Iterators
    VC_CellIterator in_ArchCell = _in_ArchMesh->GetCells()->Begin();
    VC_CellIterator ZeroRadiusSmoothedCell = ZeroRadiusSmoothedMesh->GetCells()->Begin();

    int c = 0;

    while ( in_ArchCell != _in_ArchMesh->GetCells()->End()) {

        //Initialize Iterators for Points in a Cell
        VC_PointsInCellIterator in_ArchMeshPointId = in_ArchCell.Value()->PointIdsBegin();
        VC_PointsInCellIterator ZeroRadiusMeshPointId = ZeroRadiusSmoothedCell.Value()->PointIdsBegin();

        int counter = 0;
        //while we have points in the cell
        while ( in_ArchMeshPointId != in_ArchCell.Value()->PointIdsEnd() ) {

            //Now to check the points within the cells
            if (counter == 0)
                BOOST_CHECK_EQUAL(*in_ArchMeshPointId, *ZeroRadiusMeshPointId);
            else if(counter == 1)
                BOOST_CHECK_EQUAL(*in_ArchMeshPointId, *ZeroRadiusMeshPointId);
            else if (counter == 2)
                BOOST_CHECK_EQUAL(*in_ArchMeshPointId, *ZeroRadiusMeshPointId);

            //increment points
            ++in_ArchMeshPointId; ++ZeroRadiusMeshPointId; ++counter;

        }

        //increment cells
        ++in_ArchCell; ++ZeroRadiusSmoothedCell;
    }

}