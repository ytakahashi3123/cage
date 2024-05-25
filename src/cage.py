#!/usr/bin/env python3

# Cages: Script to calculate brightness on STL surface using ray tracing method
# Version 1.0.0
# Date: 2024/05/31

# Author: Yusuke Takahashi, Hokkaido University
# Contact: ytakahashi@eng.hokudai.ac.jp

code_name = "Cages"
version = "1.0.0"

import numpy as np
from orbital.orbital import orbital
from handler_mesh.handler_mesh import handler_mesh
from shade.shade import Shade
from shadow.shadow import Shadow


def find_probe_index(probe,stl_data):
  # Find index for probe
  probe_index_list = []
  for n in range(0,len(probe)):
    probe_index = handler_mesh.get_index_stldata(stl_data, probe[n]['coordinate'])
    probe_index_list.append(probe_index)
  return probe_index_list

def compute_brightness(config,stl_data):
    
  # Find index for probe
  probe = config['probe']
  probe_index_list = find_probe_index(probe,stl_data)
  print(probe_index_list)

  # Define light direction
  light_direction = np.array( config['light_direction'] )

  # Define rotation angular (e.g., [1, 1, 1] degrees for x, y, z axes)
  angular_velocity = np.array( config['angular_velocity'] )
  if config['angular_velocity_unit'] == 'degree':
    convert_unit = np.pi/180
  else:
    convert_unit = 1.0

  # Rotation period
  rotation_period = handler_mesh.get_rotation_period(angular_velocity*convert_unit)

  # Time step [s]
  time_step = config['time_step']

  # Number of steps
  num_steps = (rotation_period/time_step).astype(int)

  # Calculate the rotation per step
  rotation_per_step = angular_velocity*time_step

  # Create a copy of the mesh to rotate
  rotated_mesh = handler_mesh.copy_mesh(stl_data)

  # Set rotation matrix
  rotation_matrix = handler_mesh.get_rotation_matrix(rotation_per_step, order='ZYX', bydegrees=True)

  # Light direction normalized (emitted from vertex) for shadow calculation
  shadow_ray = -light_direction / np.linalg.norm(light_direction)

  num_normals = len(stl_data.normals)
  brightness = np.ones( num_normals )
  brightness_ave = np.zeros( num_normals )

  for n in range(0,num_steps):

    # Apply the rotation to each vertex in the mesh
    rotated_mesh.vectors = np.dot(rotated_mesh.vectors, rotation_matrix.T)

    # Apply the rotation to each normal vectors in the mesh
    rotated_mesh.normals = np.dot(rotated_mesh.normals, rotation_matrix.T)

    if config['flag_shade']:
      # Ray tracing to evaluate incidence angle
      incidence_angle = shade.calculate_incidence_angles(rotated_mesh.normals, light_direction)
      brightness = incidence_angle/180.0

    if config['flag_shadow']:
      # BVH building
      bvh_root = shadow.build_bvh(rotated_mesh.vectors, np.arange(len(rotated_mesh.vectors)))

      # 重複を排除した頂点を取得
      unique_vertices, inverse_indices = np.unique(rotated_mesh.vectors.reshape(-1, 3), axis=0, return_inverse=True)
      dimness = np.zeros(len(unique_vertices))
      # 重複のない頂点に対して影計算を行う
      for m, vertex in enumerate(unique_vertices):
        ray_origin = vertex + 1e-3 * shadow_ray
        dimness[m] = shadow.traverse_bvh(bvh_root, ray_origin, shadow_ray)
      triangle_dimness = dimness[inverse_indices].reshape(-1, 3).max(axis=1)

      brightness = brightness * (1.0-triangle_dimness)

    # Average
    brightness_ave += (brightness - brightness_ave) / (n + 1)

    print('Step',n)
    for m in range(0,len(probe)):
      triangle_index = probe_index_list[m][0]
      print('--Probe',m, probe[m]['name'],'Brightness',brightness[triangle_index])

    # Output series data
    if config['flag_filename_vtk_series'] :
      filename_output = config['directory_output'] + '/' + orbit.insert_suffix(config['filename_vtk_series'],'_'+str(n).zfill(config['step_digit']),'.')
      print('--Output',n,filename_output)
      variable_name = config['variable_name_series']
      variable_data = brightness * config['variable_scale_series']
      handler_mesh.output_vtk(rotated_mesh, filename_output, variable_name, variable_data)

  # Output average data
  if config['flag_filename_vtk_ave'] :
    filename_output = config['directory_output'] + '/' + config['filename_vtk_ave']
    print('Output',filename_output)
    variable_name = config['variable_name_ave']
    variable_data = brightness_ave * config['variable_scale_series']
    handler_mesh.output_vtk(stl_data, filename_output, variable_name, variable_data)

  return 


def main():

  # Read control file
  file_control = orbit.file_control_default
  config       = orbit.read_config_yaml(file_control)

  # Make result directory
  orbit.make_directory_rm(config['directory_output'])

  # Load the STL file
  stl_data = handler_mesh.load_stl(config['filename_input_stl'])

  # Ray tracing for rotated STL
  compute_brightness(config, stl_data)

  return


if __name__ == '__main__':

  print('Program name:',code_name, 'version:', version)
  print('Initializing computation process')

  # Call Classes
  orbit = orbital()
  handler_mesh = handler_mesh()
  shade = Shade()
  shadow = Shadow()

  main()

  print('Finalizing computation process')
  exit()