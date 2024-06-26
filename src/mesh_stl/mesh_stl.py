#!/usr/bin/env python3

import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation as R


class Mesh_stl():

  def __init__(self):
    print("Mesh_stl class initialized")
    return

  def load_stl(self,filename):
    print('Loading STL file:', filename)
    return mesh.Mesh.from_file(filename)

  def save_stl(self, stl_data, filename='initial_data.stl'):
    print('Saving STL file:', filename)
    stl_data.save(filename)
    return 

  def copy_mesh(self,stl_data):
    return mesh.Mesh(np.copy(stl_data.data))

  def get_parameter_stl(self,stl_data):
    # Coordinate
    points = stl_data.points.reshape([-1, 3])
    # Volume, Center of gravity, and Intertia tensor
    volume, cog, inertia = stl_data.get_mass_properties()
    #print(volume, cog, inertia)
    return  points, volume, cog, inerti

  def get_rotation_period(self, angular_velocity):
    # "angular_velocity": degree/s
    # 角速度ベクトルの大きさを計算
    magnitude = np.linalg.norm(angular_velocity)
    # 1周分の時間を計算1
    rotation_period = 360.0 /magnitude
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

  def get_rotation_matrix(self, angle, order='xyz', bydegrees=True):
    r = R.from_euler(order, angle, degrees=bydegrees)
    rotation_matrix = r.as_matrix()
    return rotation_matrix 

  def get_quaternion(self, rotation_axis, rotation_angle):
    # Quaternion
    unit_vector = rotation_axis
    theta = (rotation_angle * np.pi/180.0) /2.0
    sin_theta = np.sin(theta).reshape(-1, 1)
    cos_theta = np.cos(theta).reshape(-1, 1)
    sin_components = unit_vector * sin_theta
    quaternion = np.hstack((sin_components, cos_theta))
    return quaternion

  def get_quaternion_combined(self, rotation_axis, rotation_angle):
    # Quaternion
    unit_vector = rotation_axis
    theta = (rotation_angle * np.pi/180.0) /2.0
    sin_theta = np.sin(theta).reshape(-1, 1)
    cos_theta = np.cos(theta).reshape(-1, 1)
    sin_components = unit_vector * sin_theta
    quaternion = np.hstack((sin_components, cos_theta))
    return quaternion

  def get_rotation_object_from_quaternion(self, quaternion):
    return R.from_quat(quaternion)

#  def get_rotation_object_from_quaternion_combined(self, quaternion):
#    # Rotation module based on Quaternion, Order in intrinsic: "xyz"
#    # (座標系の回転だとrot_z * rot_y * rot_xになるはず?オブジェクトの回転はそのままrot_x * rot_y * rot_z)
#    rot_x = R.from_quat(quaternion[0])
#    rot_y = R.from_quat(quaternion[1])
#    rot_z = R.from_quat(quaternion[2])
#    rotation = rot_x * rot_y * rot_z
#    return rotation

  def get_rotation_object_from_euler_angle(self ,euler_angle, order='XYZ', degrees=True):
    # オイラー角から回転オブジェクトを計算
    # 'XYZ': Captal letter: intrinsic
    rotation = R.from_euler(order, euler_angle, degrees=degrees)
    return rotation

  def rotate_stl(self, rotated_stl_data, rotation_center, rotation_object):
    # Translate to rotation center
    rotated_stl_data.translate(-rotation_center)
    # Rotation
    rotated_stl_data.vectors = rotation_object.apply( rotated_stl_data.vectors.reshape(-1, 3) ).reshape(-1, 3, 3)
    rotated_stl_data.normals = rotation_object.apply( rotated_stl_data.normals )
    # Translate from rotation center
    rotated_stl_data.translate(rotation_center)
    return rotated_stl_data

#  def rotate_stl_quaternion(self, rotation_data, rotation_center, quaternion):
#    # Rotation object based on quaternion
#    rotation_object = self.get_rotation_object_from_quaternion(quaternion)
#    # Translate to rotation center
#    rotation_data.translate(-rotation_center)
#    # Rotation
#    rotation_data.vectors = rotation_object.apply( rotation_data.vectors.reshape(-1, 3) ).reshape(-1, 3, 3)
#    rotation_data.normals = rotation_object.apply( rotation_data.normals )
#    # Translate from rotation center
#    rotation_data.translate(rotation_center)
#    return rotation_data

#  def rotate_stl_euler_angle(self, rotation_data, rotation_center, euler_angle, order='xyz', degrees=True):
#    # オイラー角から回転オブジェクトを計算
#    rotation_object = self.get_rotation_object_from_euler_angle(euler_angle)
#    # Translate to rotation center
#    rotation_data.translate(-rotation_center)
#    # 回転を適用
#    rotation_data.vectors = rotation_object.apply( rotation_data.vectors.reshape(-1, 3) ).reshape(-1, 3, 3)
#    rotation_data.normals = rotation_object.apply( rotation_data.normals )
#    # Translate from rotation center
#    rotation_data.translate(rotation_center)
#    return rotation_data#

#  def rotate_stl_rotation_matrix(self, rotation_data, rotation_center, rotation_angle):
#    # Translate to rotation center
#    rotation_data.translate(-rotation_center)
#    # Rotation
#    rotation_matrix = self.get_rotation_matrix(rotation_angle, order='xyz', bydegrees=True)
#    # --Apply the rotation to each vertex in the mesh
#    rotation_data.vectors = np.dot(rotation_data.vectors, rotation_matrix.T)
#    # --Apply the rotation to each normal vectors in the mesh
#    rotation_data.normals = np.dot(rotation_data.normals, rotation_matrix.T)
#    # Translate from rotation center
#    rotation_data.translate(rotation_center)
#    return rotation_data

  def extract_rotation_axis(self, rotation):
    # 回転クォータニオンを抽出
    quat = rotation.as_quat()
    # 回転軸を計算 (q = [x, y, z, w] の形式)
    axis = quat[:3]
    # 回転軸ベクトルを正規化
    normalized_axis = axis #normalize(axis)    
    return normalized_axis

  def get_facing_axis_after_rotation(self,rotation):
    # 標準座標系の軸（単位ベクトル）
    initial_axes = np.eye(3)
    # 回転を適用して、回転後の軸を取得
    rotated_axes = rotation.apply(initial_axes)
    return rotated_axes

  def get_euler_and_axis_from_rotaion_object(self, rotation, order='XYZ'):
    # 回転オブジェクトからオイラー角と回転軸を求める
    #Input parameters:
    #- rotation
    #- order: オイラー角の順序（デフォルトは 'xyz'）
    #Returns:
    #- euler_angle: オイラー角のリスト [roll, pitch, yaw] (単位はdeg)
    #- axis: 回転軸のベクトル [x, y, z]
    #- angle: 回転角度 (単位はdeg)

    # オイラー角を計算
    euler_angle = rotation.as_euler(order, degrees=True)

    #quaternion = rotation.as_quat()
    #x, y, z, w = quaternion
    ## 回転角度を計算
    ## 2 * arccos(w) により回転角度を得る（ラジアン）
    #angle_rad = 2 * np.arccos(w)
    ## 角度を度に変換
    #angle_deg_q = np.degrees(angle_rad)

    # 回転軸と回転角度を計算
    angle = rotation.magnitude()
    angle_deg = np.degrees( angle )
    if angle != 0 :
      axis = rotation.as_rotvec() / angle  
    else : 
      axis = np.array([1, 0, 0])

    #print(angle_deg,angle_deg_q)

    return euler_angle, axis, angle_deg


  def get_unitvector_from_polar_angle(self, polar_angle):
    theta = np.deg2rad( polar_angle[0] )
    phi   = np.deg2rad( polar_angle[1] )
    unit_x = np.sin( theta ) * np.cos( phi )
    unit_y = np.sin( theta ) * np.sin( phi )
    unit_z = np.cos( theta )
    return np.array( [unit_x, unit_y, unit_z] )