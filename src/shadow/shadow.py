#!/usr/bin/env python3

import numpy as np
import vtk


class Shadow:

  def __init__(self):
    print("Shadow class initialized")

  class BVHNode:
    def __init__(self, bounding_box, left=None, right=None, triangles=None, triangle_indices=None):
      self.bounding_box = bounding_box
      self.left = left
      self.right = right
      self.triangles = triangles
      self.triangle_indices = triangle_indices


  def compute_bounding_box(self, triangles):
    min_corner = np.min(triangles.reshape(-1, 3), axis=0)
    max_corner = np.max(triangles.reshape(-1, 3), axis=0)
    return min_corner, max_corner


  def build_bvh(self, triangles, triangle_indices, depth=0, max_depth=20):
    if len(triangles) == 0:
      return None

    bounding_box = self.compute_bounding_box(triangles)
    
    if len(triangles) <= 2 or depth >= max_depth:
      return self.BVHNode(bounding_box, triangles=triangles, triangle_indices=triangle_indices)
    
    min_corner, max_corner = bounding_box
    extents = max_corner - min_corner
    split_axis = np.argmax(extents)
    #split_axis = depth % 3
    sorted_indices = np.argsort([np.mean(tri[:, split_axis]) for tri in triangles])
    sorted_triangles = triangles[sorted_indices]
    sorted_triangle_indices = triangle_indices[sorted_indices]
    mid = len(sorted_triangles) // 2
    
    left_triangles = sorted_triangles[:mid]
    left_indices = sorted_triangle_indices[:mid]
    right_triangles = sorted_triangles[mid:]
    right_indices = sorted_triangle_indices[mid:]
    
    left_node = self.build_bvh(left_triangles, left_indices, depth + 1, max_depth)
    right_node = self.build_bvh(right_triangles, right_indices, depth + 1, max_depth)
    
    return self.BVHNode(bounding_box, left=left_node, right=right_node)


  def intersect_ray_triangle(self, ray_origin, ray_direction, triangle):
    epsilon = 1e-8

    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)
    if -epsilon < a < epsilon:
        return False, None # This ray is parallel to this triangle.

    f = 1.0 / a
    s = ray_origin - triangle[0]
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)
    if v < 0.0 or u + v > 1.0:
        return False, None
    
    # At this stage we can compute t to find out where the intersection point is on the line.
    t = f * np.dot(edge2, q)

    if t > epsilon: # ray intersection
        intersection_point = ray_origin + ray_direction * t
        return True, intersection_point
    else: # This means that there is a line intersection but not a ray intersection.
        return False, None


  def traverse_bvh(self, node, ray_origin, ray_direction):

    if node is None:
      return False
    min_corner, max_corner = node.bounding_box
    if not self.ray_intersects_aabb(ray_origin, ray_direction, min_corner, max_corner):
      return False
    if node.triangles is not None:
      for triangle in node.triangles:
        hit, _ = self.intersect_ray_triangle(ray_origin, ray_direction, triangle)
        if hit:
          return True
      return False
    return self.traverse_bvh(node.left, ray_origin, ray_direction) or self.traverse_bvh(node.right, ray_origin, ray_direction)


  def ray_intersects_aabb(self, ray_origin, ray_direction, aabb_min, aabb_max):
    eps_tmp = np.array([1.e-30,1.e-30,1.e-30])
    tmin = (aabb_min - ray_origin) / (ray_direction+eps_tmp)
    tmax = (aabb_max - ray_origin) / (ray_direction+eps_tmp)
    #tmin = (aabb_min - ray_origin) / (ray_direction)
    #tmax = (aabb_max - ray_origin) / (ray_direction)
    tmin_final = np.max(np.minimum(tmin, tmax))
    tmax_final = np.min(np.maximum(tmin, tmax))
    return tmax_final >= max(tmin_final, 0.0)


  def output_bvh_structure(self):

    def add_bvh_nodes_to_vtk(node, vtk_points, vtk_cells):
    # Function to add BVH nodes to a VTK polydata
      if node is not None:
        min_corner, max_corner = node.bounding_box
        
        # Add the bounding box vertices to vtkPoints
        bbox_points = [
            [min_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]]
        ]
        bbox_point_ids = []
        for point in bbox_points:
            pid = vtk_points.InsertNextPoint(point)
            bbox_point_ids.append(pid)
        
        # Add the bounding box edges to vtkCells
        bbox_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], # top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # vertical edges
        ]
        for edge in bbox_edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, bbox_point_ids[edge[0]])
            line.GetPointIds().SetId(1, bbox_point_ids[edge[1]])
            vtk_cells.InsertNextCell(line)
        
        # Recurse for child nodes
        add_bvh_nodes_to_vtk(node.left, vtk_points, vtk_cells)
        add_bvh_nodes_to_vtk(node.right, vtk_points, vtk_cells)

    # Create VTK structures for storing points and cells
    vtk_points = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()

    # Add BVH nodes to the VTK structures
    add_bvh_nodes_to_vtk(bvh_root, vtk_points, vtk_cells)

    # Create a polydata object
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(vtk_cells)

    # Write the polydata to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename_output )
    writer.SetInputData(polydata)
    writer.Write()

    print("BVH structure written to",filename_output)

    # Function to print the BVH structure for debugging
    #def print_bvh(node, depth=0):
    #    if node is not None:
    #        print("  " * depth + f"Node depth {depth}: Bounding box {node.bounding_box}, Triangles: {len(node.triangles) if node.triangles is not None else 'N/A'}, Indices: {node.triangle_indices if node.triangle_indices is not None else 'N/A'}")
    #        print_bvh(node.left, depth + 1)
    #        print_bvh(node.right, depth + 1)
    #
    # Print the BVH
    #print_bvh(bvh_root)