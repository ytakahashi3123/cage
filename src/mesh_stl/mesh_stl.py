#!/usr/bin/env python3

import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation as R


class Mesh_stl():

  def __init__(self):
    print("Mesh_stl class initialized")
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


  def get_rotation_matrix(self, angle, order='xyz', bydegrees=True):
    r = R.from_euler(order, angle, degrees=bydegrees)
    rotation_matrix = r.as_matrix()
    return rotation_matrix 


  def rotate_stl(self, rotation_data, rotation_center, rotation_angle):
    # Translate to rotation center
    rotation_data.translate(-rotation_center)
    # Rotation
    rotation_matrix = self.get_rotation_matrix(rotation_angle, order='xyz', bydegrees=True)
    # --Apply the rotation to each vertex in the mesh
    rotation_data.vectors = np.dot(rotation_data.vectors, rotation_matrix.T)
    # --Apply the rotation to each normal vectors in the mesh
    rotation_data.normals = np.dot(rotation_data.normals, rotation_matrix.T)
    # Translate from rotation center
    rotation_data.translate(rotation_center)
    return rotation_data


  def get_quaternion(self, rotation_axis, rotation_angle):
    # Quaternion
    unit_vector_x = rotation_axis
    unit_vector_y = np.array( [0.0,1.0,0.0] )
    unit_vector_z = np.array( [0.0,0.0,1.0] )
    theta = (rotation_angle * np.pi/180.0) /2.0
    quaternion_x = np.append( unit_vector_x*np.sin(theta[0]), np.cos(theta[0]) )
    quaternion_y = np.append( unit_vector_y*np.sin(theta[1]), np.cos(theta[1]) )
    quaternion_z = np.append( unit_vector_z*np.sin(theta[2]), np.cos(theta[2]) )
    quaternion = [quaternion_x, quaternion_y, quaternion_z]
    #quaternion = np.append( rotation_axis*np.sin(theta), np.cos(theta) )
    return quaternion


  def get_rotation_quaternion(self, quaternion):
    # Rotation module based on Quaternion
    # (座標系の回転だとrot_z * rot_y * rot_xになるはず?オブジェクトの回転はそのままrot_x * rot_y * rot_z)
    rot_x = R.from_quat(quaternion[0])
    rot_y = R.from_quat(quaternion[1])
    rot_z = R.from_quat(quaternion[2])
    rotation = rot_x * rot_y * rot_z
    # 結果のクォータニオンを取得するならば
    #quaternion_combined = r.as_quat()
    #print(quaternion_combined)
    #euler_angles, axis, angle = slt.quaternion_to_euler_and_axis(quaternion_combined)
    #print(f"オイラー角 (度): {np.degrees(euler_angles)}")
    #print(f"回転軸: {axis}")
    #print(f"回転角度 (度): {np.degrees(angle)}")
    return rotation


  def rotate_stl_quaternion(self, rotation_data, rotation_center, rotation_axis, rotation_angle):
    # Define quaternions and rotation modules
    # -- rot_x回転ののち、回転後の座標系（ローカル座標系:y）でrot_y回転、回転後の座標系（ローカル座標系:z）でrot_z回転,
    quaternion = self.get_quaternion(rotation_axis, rotation_angle)
    r = self.get_rotation_quaternion(quaternion)
    # Translate to rotation center
    rotation_data.translate(-rotation_center)
    # Rotation
    rotation_data.vectors = r.apply( rotation_data.vectors.reshape(-1, 3) ).reshape(-1, 3, 3)
    rotation_data.normals = r.apply( rotation_data.normals )
    # Translate from rotation center
    rotation_data.translate(rotation_center)
    return rotation_data


  def get_rotation_period(self, angular_velocity):
    # "angular_velocity": degree/s
    # 角速度ベクトルの大きさを計算
    magnitude = np.linalg.norm(angular_velocity)
    # 1周分の時間を計算
    rotation_period = 360.0 / magnitude
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

  
  def quaternion_to_euler_and_axis(self, quaternion, order='xyz'):
    """
    クォータニオンからオイラー角と回転軸に変換する。

    Parameters:
    - quaternion: リストまたは配列 [x, y, z, w]
    - order: オイラー角の順序（デフォルトは 'xyz'）

    Returns:
    - euler_angle: オイラー角のリスト [roll, pitch, yaw] (単位はdeg)
    - axis: 回転軸のベクトル [x, y, z]
    - angle: 回転角度 (単位はdeg)
    """
    r = R.from_quat(quaternion)

    # オイラー角を計算
    euler_angle = r.as_euler(order, degrees=True)

    # 回転軸と回転角度を計算
    angle = r.magnitude()
    axis = r.as_rotvec() / angle if angle != 0 else np.array([0, 0, 1])

    return euler_angle, axis, np.degrees(angle)