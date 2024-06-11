#!/usr/bin/env python3

import numpy as np
import time
import vtk
from vtk.util import numpy_support


def run_raytracing(config,stl_data,orbital,mesh_stl,shade,shadow):
    
  print('Ray tracing routine started')
  
  # Find index for probe
  probe = config['probe']
  num_probes = len(probe)
  if probe is not None:
    probe_index_list = find_probe_index(probe,stl_data,mesh_stl)

  # Define rotation angular (e.g., [1, 1, 1] degrees for x, y, z axes)
  angular_velocity = np.array( config['angular_velocity'] ).astype(float)

  # Define rotation center
  rotation_center = np.array(config['rotation_center']).astype(float)

  # Define body axis
  euler_angle_bodyaxis = np.array( config['euler_angle_bodyaxis'] ).astype(float)
  rotation_object = mesh_stl.get_rotation_object_from_euler_angle(euler_angle_bodyaxis)
  body_axis = mesh_stl.get_facing_axis_after_rotation(rotation_object)
  print('Body axis:')
  print('--x:', body_axis[0])
  print('--y:', body_axis[1])
  print('--z:', body_axis[2])

  # Initial rotation of STL data
  #quaternion = mesh_stl.get_quaternion_combined(body_axis, euler_angle_bodyaxis)
  #stl_data = mesh_stl.rotate_stl_quaternion(stl_data, rotation_center, quaternion)
  #if config['flag_filename_initial_stl']:
  #  filename_output = config['directory_output'] + '/' + config['filename_initial_stl']
  #  print('--Output STL data rotated by initial Euler angle',filename_output)
  #  mesh_stl.save_stl(stl_data, filename_output)

  # Define rotation axis
  polar_angle = np.array( config['polar_angle_rotaionaxis'] ).astype(float)
  rotation_axis = mesh_stl.get_unitvector_from_polar_angle(polar_angle)
  print('Rotation axis:', rotation_axis)

  # Number of steps
  num_step = config['number_step']
  if num_step <= 0:
    print('Error, number of steps is less than zero.', num_step)
    exit()

  # Time step
  if np.all(angular_velocity == 0): 
    # Non-Rotation case
    num_step = 1
    time_step = 1.0 
  else: 
    # Rotation case
    rotation_period = mesh_stl.get_rotation_period( angular_velocity )
    print('Rotation period, s:', rotation_period )
    time_step = rotation_period/float(num_step)

    quaternion = mesh_stl.get_quaternion(rotation_axis, angular_velocity*time_step)
    rotation_object = mesh_stl.get_rotation_object_from_quaternion(quaternion)
    angular_velocity_tmp, axis_tmp, angle_tmp = mesh_stl.get_euler_and_axis_from_rotaion_object(rotation_object)
    print('Euler angle per second, degree/s:', angular_velocity_tmp)
    print('Angle, degree:', angle_tmp)
    print('Rotation axis:', axis_tmp)

  print('Time step, s:', time_step)

  # Calculate the rotation per step
  rotation_per_step = angular_velocity*time_step
  print('Rotation per step:', rotation_per_step)

  # Create a copy of the mesh to rotate
  rotated_mesh = mesh_stl.copy_mesh(stl_data)

  # Define light direction
  light_direction = np.array( config['light_direction'] )

  # Light direction normalized (emitted from vertex) for shadow calculation
  shadow_ray = -light_direction / np.linalg.norm(light_direction)

  # Initial settings
  num_normals = len(stl_data.normals)
  brightness = np.ones( num_normals )
  brightness_ave = np.zeros( num_normals )
  num_probes = num_probes + 1
  probe_data = np.zeros(num_step*num_probes).reshape(num_step,num_probes)

  for n in range(0,num_step):
    start_time = time.time()

    # Rotation
    quaternion = mesh_stl.get_quaternion_combined(rotation_axis, rotation_per_step)
    rotation_object = mesh_stl.get_rotation_object_from_quaternion(quaternion)
    rotated_mesh = mesh_stl.rotate_stl(rotated_mesh, rotation_center, rotation_object)

    if config['flag_shade']:
      # Ray tracing to evaluate incidence angle (rad)
      incidence_angle = shade.calculate_incidence_angles(rotated_mesh.normals, light_direction)
      # Brightness
      brightness = shade.calculate_brightness(incidence_angle)

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
    # Output series data
    if config['flag_filename_vtk_series'] :
      filename_output = config['directory_output'] + '/' + orbital.insert_suffix(config['filename_vtk_series'],'_'+str(n).zfill(config['step_digit']),'.')
      print('--Output',n,filename_output)
      if config['variable_mesh_rotate_series']:
        variable_mesh = rotated_mesh
      else:
        variable_mesh = stl_data
      variable_name = config['variable_name_series']
      variable_data = brightness * config['variable_scale_series']
      output_vtk(filename_output, variable_mesh, variable_name, variable_data)
    # Probe data
    if probe is not None:
      for m in range(0,len(probe)):
        triangle_index = probe_index_list[m][0]
        probe_data[n,m] = brightness[triangle_index]
        print('--Probe',m, probe[m]['name'],'Brightness',probe_data[n,m])

    # Elapsed time
    elapsed_time = time.time()-start_time
    if config['display_verbose']:
      print('--Elapsed time [s]',elapsed_time)

  # Output average data
  if config['flag_filename_vtk_ave'] :
    filename_output = config['directory_output'] + '/' + config['filename_vtk_ave']
    print('Output',filename_output)
    variable_mesh = stl_data
    variable_name = config['variable_name_ave']
    variable_data = brightness_ave * config['variable_scale_series']
    output_vtk(filename_output, variable_mesh, variable_name, variable_data)
  if probe is not None:
    for m in range(0,len(probe)):
      triangle_index = probe_index_list[m][0]
      probe_data[-1,m] = brightness_ave[triangle_index]
      print('Probe',m, probe[m]['name'],'Brightness',probe_data[-1,m])

    # Output probe data
    write_probe_data(config, num_step, time_step, probe, probe_data)

  print('Ray tracing routine finished')

  return 


def output_vtk(filename_output, stl_data, variable_name, variable_data):

#  print('Writing brightness on STL to:',filename_output)

  # 三角形要素の数を取得
  num_triangles = len(stl_data.vectors)

  # Input parameter
  x = variable_data

  # VTKのポリゴンデータを作成
  points = vtk.vtkPoints()
  cells = vtk.vtkCellArray()
  unstructured_grid = vtk.vtkUnstructuredGrid()

  # 頂点座標を追加
  # STLメッシュから頂点座標を抽出し、ポイントに追加
  for vector in stl_data.vectors:
    for point in vector:
      points.InsertNextPoint(point)

  # 三角形セルを追加
  for i in range(num_triangles):
    triangle = vtk.vtkTriangle()
    for j in range(3):
      triangle.GetPointIds().SetId(j, 3 * i + j)
    cells.InsertNextCell(triangle)

  # ポリゴンデータにポイントとセルをセット
  unstructured_grid.SetPoints(points)
  unstructured_grid.SetCells(vtk.VTK_TRIANGLE, cells)

  # NumPy配列をVTK配列に変換
  x_vtk_array = numpy_support.numpy_to_vtk(x, deep=True)
  x_vtk_array.SetName(variable_name)

  # ポリゴンデータに要素データを追加
  unstructured_grid.GetCellData().AddArray(x_vtk_array)
  #unstructured_grid.GetPointData().AddArray(x_vtk_array)

  # VTK/VTUファイルとして保存
  writer = vtk.vtkXMLUnstructuredGridWriter()
  writer.SetFileName(filename_output)
  writer.SetInputData(unstructured_grid)
  writer.Write()

  return


def find_probe_index(probe,stl_data,mesh_stl):
  # Find index for probe
  probe_index_list = []
  for n in range(0,len(probe)):
    probe_index = mesh_stl.get_index_stldata(stl_data, probe[n]['coordinate'])
    probe_index_list.append(probe_index)
  return probe_index_list

def write_probe_data(config, num_step, time_step, probe, probe_data):

  num_probes = len(probe)

  # Write series data
  filename_tmp = config['directory_output'] + '/' + config['filename_probe']
  file_output = open( filename_tmp , 'w')
  header_tmp = "Variables = Time, "
  for n in range(0, num_probes):
    header_tmp = header_tmp + probe[n]['name'] + ', '
  header_tmp = header_tmp.rstrip(',') + '\n'
  file_output.write( header_tmp )
  text_tmp = ''
  for n in range(0, num_step):
    text_tmp = text_tmp + str( float(n)*time_step ) + ', '
    for m in range(0, num_probes):
      text_tmp = text_tmp  + str( probe_data[n,m] ) + ', '
    text_tmp = text_tmp.rstrip(', ')  + '\n'
  file_output.write( text_tmp )
  file_output.close()
  
  # Write average data
  filename_tmp = config['directory_output'] + '/' + config['filename_probe_ave']
  file_output = open( filename_tmp , 'w')
  header_tmp = "Variables = Time, "
  for n in range(0, num_probes):
    header_tmp = header_tmp + probe[n]['name'] + ', '
  header_tmp = header_tmp.rstrip(',') + '\n'
  file_output.write( header_tmp )
  text_tmp = ''
  text_tmp = text_tmp + str( float(0) ) + ', '
  for m in range(0, num_probes):
    text_tmp = text_tmp + str( probe_data[-1,m] ) + ', '
  text_tmp = text_tmp.rstrip(', ')  + '\n'
  file_output.write( text_tmp )
  file_output.close()

  return