#!/usr/bin/env python3

import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation as R


class Handler_mesh():

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


  def rotate_stl(self, rotation_data, rotation_center, rotation_angle):
    rotation_matrix = self.get_rotation_matrix(rotation_angle, order='XYZ', bydegrees=True)
    # Translate to rotation center
    rotation_data.translate(rotation_center)
    # Rotation
    # --Apply the rotation to each vertex in the mesh
    rotation_data.vectors = np.dot(rotation_data.vectors, rotation_matrix.T)
    # --Apply the rotation to each normal vectors in the mesh
    rotation_data.normals = np.dot(rotation_data.normals, rotation_matrix.T)
    # Translate from rotation center
    rotation_data.translate(-rotation_center)
    return rotation_data


  def get_rotation_matrix(self, angle, order='XYZ', bydegrees=True):
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

  