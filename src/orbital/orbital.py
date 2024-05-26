#!/usr/bin/env python3

import numpy as np
import os as os
import shutil as shutil
import yaml as yaml
import sys as sys
import time as time
from functools import wraps


class Orbital():

  file_control_default = "cage.yml"

  def __init__(self):
    print("Calling class: orbital")

  def argument(self, filename_default):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', action='store', type=str, default=filename_default)
    args = parser.parse_args()
    return args

  def read_config_yaml(self, file_control):
    #import pprint as pprint
    print("Reading control file...:", file_control)
    try:
      with open(file_control) as file:
        config = yaml.safe_load(file)
#        pprint.pprint(config)
    except Exception as e:
      print('Exception occurred while loading YAML...', file=sys.stderr)
      print(e, file=sys.stderr)
      sys.exit(1)
    return config

  def make_directory(self, dir_path):
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)
    return

  def make_directory_rm(self, dir_path):  
    if not os.path.exists(dir_path):
      os.mkdir(dir_path)
    else:
      shutil.rmtree(dir_path)
      os.mkdir(dir_path)
    return

  def split_file(self, filename, addfile, splitchar):
    splitchar_tmp   = splitchar
    filename_split  = filename.rsplit(splitchar_tmp, 1)
    filename_result = filename_split[0]+addfile+splitchar+filename_split[1]
    return filename_result

  def insert_suffix(self, filename, suffix, splitchar):
    parts = filename.split(splitchar)
    if len(parts) == 2:
      new_filename = f"{parts[0]}{suffix}.{parts[1]}"
      return new_filename
    else:
      # ファイル名が拡張子を含まない場合の処理
      return filename + suffix

  def get_file_extension(self, filename):
    # ドットで分割し、最後の要素が拡張子となる
    parts = filename.split(".")
    if len(parts) > 1:
      return parts[-1].lower()
    else:
      # ドットが含まれていない場合は拡張子が存在しない
      return None

  def closest_value_index(self, numbers, value):
    closest_index = min(range(len(numbers)), key=lambda i: abs(numbers[i] - value))
    closest_value = numbers[closest_index]
    return closest_value, closest_index

  def get_directory_path(self, path_specify, default_path, manual_path):
    script_directory = os.path.dirname(os.path.realpath(__file__))
    if path_specify == 'auto' or path_specify == 'default':
      directory_path = script_directory + default_path
    elif path_specify == 'manual':
      directory_path = manual_path
    else :
      directory_path = script_directory + default_path
    return directory_path

  def read_header_tecplot(self, filename, headerline, headername, var_list):
    # Set header
    with open(filename) as f:
      lines = f.readlines()
    # リストとして取得
    lines_strip = [line.strip() for line in lines]
    # ”Variables ="を削除した上で、カンマとスペースを削除
    variables_line = lines_strip[headerline].replace(headername, '')
    variables_line = variables_line.replace(',', ' ').replace('"', ' ')
    # 空白文字で分割して単語のリストを取得
    words = variables_line.split()

    # set variables
    result_var   = var_list
    result_index = []
    for i in range( 0,len(result_var) ):
      for n in range( 0,len(words) ):
        if result_var[i] == words[n] :
          result_index.append(n)
          break

    return result_index


  # Decorator for time measurement
  def time_measurement_decorated(func):
    @wraps(func)
    def wrapper(*args, **kargs) :
      #text_blue = '\033[94m'
      #text_green = '\033[92m'
      text_yellow = '\033[93m'
      text_end = '\033[0m'
      flag_time_measurement = False
      if flag_time_measurement :
        start_time = time.time()
        result = func(*args,**kargs)
        elapsed_time = time.time() - start_time
        print('Elapsed time of '+str(func.__name__)+str(':'),text_yellow + str(elapsed_time) + text_end,'s')
      else :
        result = func(*args,**kargs)
      return result 
    return wrapper
