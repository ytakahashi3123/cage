#!/usr/bin/env python3

# Cage: Script to calculate brightness on STL surface using ray tracing method
# Version 1.0.0
# Date: 2024/05/31

# Author: Yusuke Takahashi, Hokkaido University
# Contact: ytakahashi@eng.hokudai.ac.jp

code_name = "Cage"
version = "1.0.0"

import numpy as np
from orbital.orbital import Orbital
from handler_mesh.handler_mesh import Handler_mesh
from shade.shade import Shade
from shadow.shadow import Shadow
from raytracing import raytracing


def main():

  # Call Classes
  orbital = Orbital()
  handler_mesh = Handler_mesh()
  shade = Shade()
  shadow = Shadow()

  # Read control file
  file_control = orbital.file_control_default
  config       = orbital.read_config_yaml(file_control)

  # Make result directory
  orbital.make_directory_rm(config['directory_output'])

  # Load the STL file
  stl_data = handler_mesh.load_stl(config['filename_input_stl'])

  # Ray tracing for rotated STL to calculate brightness on STL surface
  raytracing.run_raytracing(config,stl_data,orbital,handler_mesh,shade,shadow)

  return


if __name__ == '__main__':

  print('Program name:',code_name, 'version:', version)
  print('Initializing computation process')

  main()

  print('Finalizing computation process')
  exit()