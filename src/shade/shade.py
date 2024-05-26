#!/usr/bin/env python3

import numpy as np


class Shade():

  def __init__(self):
    print("Shade class initialized")


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

    # Incidence angle
    incidence_angle = np.pi - angles_rad

    #angles_deg = np.degrees(angles_rad)

    return incidence_angle


  def calculate_brightness(self, incidence_angle):
    brightness = np.cos(incidence_angle)
    brightness[brightness < 0.0] = 0.0
    return brightness