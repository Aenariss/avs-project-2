/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Vojtech Fiala <xfiala61@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    9.12.2022
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

#define edgeSizeCutoff 1

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

bool TreeMeshBuilder::condition(const double edgeSize, const ParametricScalarField &field, const Vec3_t<float> pos, double half) {
    Vec3_t<float> center = getCenter(pos, half);
    auto value = evaluateFieldAt(center, field);
    // if its less or equal, its not empty
    return (value <= (mIsoLevel + ((sqrt(3) / 2) * edgeSize)));
}

Vec3_t<float> TreeMeshBuilder::getCenter(Vec3_t<float> pos, double half) {
    // center of the edge is in the half, aka start + half, but has to be in original continuous space 
    // i can get that by multiplying the position by the resolution
    Vec3_t<float> center( (pos.x + half) * mGridResolution,
                          (pos.y + half) * mGridResolution,
                          (pos.z + half) * mGridResolution );
    return center;
}

Vec3_t<float> TreeMeshBuilder::posInSpace(Vec3_t<float> pos, double half, size_t it) {
    // I need to calculate a new position from which I continue
    // because I divide the block into 8 parts, I can use the sc_vertexNormPos which 
    // has defined 8 vertexes for 8 corners of the block
    Vec3_t<float> newpos( pos.x + half * sc_vertexNormPos[it].x,
                          pos.y + half * sc_vertexNormPos[it].y,
                          pos.z + half * sc_vertexNormPos[it].z );
    return newpos;
}

unsigned int TreeMeshBuilder::tree(const Vec3_t<float> pos, double edgeSize, const ParametricScalarField &field) {

    double half = edgeSize / 2;
    unsigned totalTriangles = 0;

    // if condition is satisfied, the block is not empty
    if (!(condition(edgeSize * mGridResolution, field, pos, half)))
        return 0;
    
    // if im already at the bottom, its time to buildCube()
    // 2 is too much, 1-1.5 seems ok, but 1 is fancier
    if (edgeSize <= edgeSizeCutoff)
        totalTriangles += buildCube(pos, field);
    // else I go through 8 parts
    else {
        for (int i = 0; i < 8; i++) {
                // get new position from which it recursively launches again
                Vec3_t<float> newpos = posInSpace(pos, half, i);
                // add task to queue
                #pragma omp task default(none) shared(field, totalTriangles, half) firstprivate(newpos)
                {
                    // atomic operation
                    #pragma omp atomic
                    // recursively count again, using new position and half the size of the edge (8 parts, each has half the size of original)
                    totalTriangles += tree(newpos, half, field);
                }
        }
    }
    // parent waits for child tasks to finish
    #pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    
    double origEdge = (double) mGridSize;  // length of the grid side
    unsigned triangles;
    #pragma omp parallel default(none) shared(triangles, origEdge, field)    // initialize parallel threads
    {
        #pragma omp single   // only do this once
        {
            triangles = tree(sc_vertexNormPos[0], origEdge, field);
        }
    }
    return triangles;

}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    // Critical, because i need the threads to push one by one, only 1 at the time
    // I dont need to care about the order, cuz it can be written however
    // (only the coords need to be correct)
    #pragma omp critical
    mTriangles.push_back(triangle);
}
