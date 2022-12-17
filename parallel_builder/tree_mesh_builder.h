/**
 * @file    tree_mesh_builder.h
 *
 * @author  Vojtech Fiala <xfiala61@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    17.12.2022
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    unsigned int tree(const Vec3_t<float> &pos, const double edgeSize, const ParametricScalarField &field);

    
    bool condition(const double edgeSize, const ParametricScalarField &field, const Vec3_t<float> &pos, const double half);
    Vec3_t<float> getCenter(const Vec3_t<float> &pos, const double half);
    Vec3_t<float> posInSpace(const Vec3_t<float> &pos, const double half, const size_t it);
    std::vector<Triangle_t> mTriangles; ///< Temporary array of triangles

    const Triangle_t *getTrianglesArray() const { return mTriangles.data(); }

};

#endif // TREE_MESH_BUILDER_H
