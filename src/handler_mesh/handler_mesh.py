#!/usr/bin/env python3

import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation as R
import vtk
from vtk.util import numpy_support
from orbital.orbital import orbital


class handler_mesh(orbital):

  def __init__(self):
    print("Constructing class: Mesh")
    return


  def load_stl(self,filename):
    return mesh.Mesh.from_file(filename)


  def copy_mesh(self,stl_data):
    return mesh.Mesh(np.copy(stl_data.data))


  def get_parameter_stl(self,stl_data):
    # Coordinate
    points = stl_data.points.reshape([-1, 3])
    # Volume, Center of gravity, and Intertia tensor
    volume, cog, inertia = stl_data.get_mass_properties()
    #print(volume, cog, inertia)
    return  points, volume, cog, inerti


  def get_rotation_matrix(self, angle, order='ZYX', bydegrees=True):
    r = R.from_euler(order, angle, degrees=bydegrees)
    rotation_matrix = r.as_matrix()
    return rotation_matrix 


  def get_rotation_period(self, angular_velocity):
    # 角速度ベクトルの大きさを計算
    magnitude = np.linalg.norm(angular_velocity)
    # 1周分の時間を計算
    rotation_period = 2 * np.pi / magnitude
    print('Rotation period [s]:', rotation_period )    
    return rotation_period

  def get_index_stldata(self, stl_data, target_points):
    # 頂点座標をフラットなリストに変換
    points = stl_data.vectors.reshape(-1, 3)  # 形状を (N*3, 3) に変換
    # 各頂点と任意の座標の間の距離を計算
    distances = np.linalg.norm(points - target_points, axis=1)
    # 最小距離のインデックスを取得
    closest_index = np.argmin(distances)
    # Convert
    closest_triangle_index = closest_index // 3  # 三角形のインデックス
    closest_vertex_index = closest_index % 3     # 三角形内の頂点のインデックス
    return [closest_triangle_index, closest_vertex_index]

  def output_vtk(self, stl_mesh, filename_output, variable_name, variable_data):

    print('Writing brightness on STL to:',filename_output)

    # 三角形要素の数を取得
    num_triangles = len(stl_mesh.vectors)

    # Input parameter
    x = variable_data

    # VTKのポリゴンデータを作成
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    unstructured_grid = vtk.vtkUnstructuredGrid()

    # 頂点座標を追加
    # STLメッシュから頂点座標を抽出し、ポイントに追加
    for vector in stl_mesh.vectors:
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
    #poly_data.GetCellData().AddArray(x_vtk_array)
    unstructured_grid.GetCellData().AddArray(x_vtk_array)

    # VTK/VTUファイルとして保存
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename_output)
    writer.SetInputData(unstructured_grid)
    writer.Write()

    return

