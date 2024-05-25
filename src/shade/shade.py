#!/usr/bin/env python3

import numpy as np


class Shade():

  def __init__(self):
    print("Shade class initialized")

  def calculate_incidence_angle_iter(self, stl_mesh_normal, light_direction):

    def normalize(v):
      norm = np.linalg.norm(v)
      if norm == 0: 
        return v
      return v / norm

    # Normalize the vectors
    mesh_normal = normalize(stl_mesh_normal)
    light_normal = normalize(light_direction)
    
    # Calculate the dot product
    dot_product = np.dot(mesh_normal, light_normal)
    
    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


  def calculate_incidence_angles(self, normals, light_direction):
    # Normalize the normals and light direction
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    light_direction = light_direction / np.linalg.norm(light_direction)

    # Calculate the dot product
    dot_products = np.dot(normals, light_direction)

    # Clip the dot products to avoid numerical errors
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angles_rad = np.arccos(dot_products)
    angles_deg = np.degrees(angles_rad)

    return angles_deg