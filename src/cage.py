#!/usr/bin/env python3

# Cage: Script to calculate brightness on STL surface using ray tracing method
# Version 1.1.0
# Date: 2024/06/30

# Author: Yusuke Takahashi, Hokkaido University
# Contact: ytakahashi@eng.hokudai.ac.jp

code_name = "Cage"
version = "1.1.0"

import numpy as np
from orbital.orbital import Orbital
from mesh_stl.mesh_stl import Mesh_stl
from shade.shade import Shade
from shadow.shadow import Shadow
from raytracing import raytracing


def main():

  # Call Classes
  orbital = Orbital()
  mesh_stl = Mesh_stl()
  shade = Shade()
  shadow = Shadow()

  # Read control file
  file_control = orbital.file_control_default
  config       = orbital.read_config_yaml(file_control)

  # Make result directory
  orbital.make_directory_rm(config['directory_output'])

  # Load the STL file
  stl_data = mesh_stl.load_stl(config['filename_input_stl'])

  # Ray tracing for rotated STL to calculate brightness on STL surface
  raytracing.run_raytracing(config,stl_data,orbital,mesh_stl,shade,shadow)

  return


if __name__ == '__main__':

  print('Program name:',code_name, 'version:', version)
  print('Initializing computation process')

  main()

  print('Finalizing computation process')
  exit()