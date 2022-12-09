/**
 * @file    loop_mesh_builder.h
 *
 * @author  Vojtech Fiala <xfiala61@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    9.12.2022
 **/

#ifndef LOOP_MESH_BUILDER_H
#define LOOP_MESH_BUILDER_H

#include <vector>
#include "base_mesh_builder.h"

class LoopMeshBuilder : public BaseMeshBuilder
{
public:
    LoopMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);

    std::vector<Triangle_t> mTriangles; ///< Temporary array of triangles

    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }
};

#endif // LOOP_MESH_BUILDER_H
