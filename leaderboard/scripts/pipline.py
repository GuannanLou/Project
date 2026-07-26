import subprocess
import time
import csv
import datetime
import numpy as np
import os

from ..leaderboard.SBT.GA_search import search_based_testing

# current_datetime = datetime.datetime.now()
# formatted_datetime = current_datetime.strftime("%Y-%m-%d|%H:%M:%S")
# vector_path = '{}.npz'.format('vectorPath|'+formatted_datetime)
# with open('{}.csv'.format('processLog|'+formatted_datetime), mode='w', newline='') as log_file:
#     writer = csv.writer(log_file)
#     writer.writerow(['time', 'ADS', 'section', 'level'])
#     # for ADS in ['TCP', 'InterFuser']:
#     # for ADS in ['TCP']:
#     for ADS in ['InterFuser']:
#         # for section in ['Curve', 'Straight']:
#         for section in ['Curve']:
#             # for level in [0,1,2,3,4]:
#             for level in [4]:

#                 current_time = datetime.datetime.now()
#                 writer.writerow([current_time, ADS, section, level])
#                 # command = 'cd ~/Projects/carla;./CarlaUE4.sh --world-port=2000'
#                 # command = 'cd ~/Projects/carla;./CarlaUE4.sh --world-port=2000 -quality-level=Low -opengl'
#                 # command = 'cd ~/Projects/CARLA_0.9.15;./CarlaUE4.sh --world-port=2000'
#                 command = 'cd ~/Projects/CARLA_0.9.15;./CarlaUE4.sh --world-port=2000 -RenderOffScreen'
#                 try:
#                     subprocess.Popen(command, shell=True)
#                     # subprocess.run(command, shell=True, check=True)
#                     print("Run carla")
#                 except subprocess.CalledProcessError as e:
#                     print(f"Run carla fail: {e.returncode}")

#                 time.sleep(10)

#                 print('Prepare Scenario Vector')
#                 vector = np.random.random((1,15,71))
#                 np.savez(vector_path, np.concatenate((np.load(vector_path),vector)))
#                 print('Ready to Test')

#                 command = 'cd ~/Projects/TCP-Interfuser;sh leaderboard/scripts/test_basement.sh {} {} {} {}'.format(ADS, section, level, vector_path)
#                 try:
#                     subprocess.run(command, shell=True)
#                     print("Run ADS")
#                 except subprocess.CalledProcessError as e:
#                     print(f"Run ADS fail: {e.returncode}")



# with open('{}.csv'.format('processLog|'+formatted_datetime), mode='w', newline='') as log_file:
#     writer = csv.writer(log_file)
#     writer.writerow(['time', 'ADS', 'section', 'level'])

#     for i in range(5):
#         for ADS in ['InterFuser']:
#             for section in ['Curve']:
#                 for level in [0]:

#                     current_time = datetime.datetime.now()
#                     writer.writerow([current_time, ADS, section, level])
#                     command = 'cd ~/Projects/CARLA_0.9.15;./CarlaUE4.sh --world-port=2000 -RenderOffScreen'
#                     print()
#                     print(command)
#                     try:
#                         subprocess.Popen(command, shell=True)
#                         print("Run carla")
#                     except subprocess.CalledProcessError as e:
#                         print(f"Run carla fail: {e.returncode}")

#                     time.sleep(10)

#                     print()
#                     print('Prepare Scenario Vector')
#                     vector = np.random.random((1,15,71))
#                     print(vector_path)
#                     if os.path.exists(vector_path):
#                         np.savez(vector_path, np.concatenate((np.load(vector_path)['arr_0'],vector)))
#                     else:
#                         np.savez(vector_path, vector)

#                     print('Ready to Test')

#                     command = 'cd ~/Projects/TCP-Interfuser;sh leaderboard/scripts/test_basement.sh {} {} {} "{}"'.format(ADS, section, level, vector_path)
#                     print()
#                     print(command)
#                     try:
#                         subprocess.run(command, shell=True)
#                         print("Run ADS")
#                     except subprocess.CalledProcessError as e:
#                         print(f"Run ADS fail: {e.returncode}")


# ADS = 'InterFuser'
# section = 'Curve'
# level = 0

# current_time = datetime.datetime.now()
# command = 'cd ~/Projects/CARLA_0.9.15;./CarlaUE4.sh --world-port=2000 -RenderOffScreen'
# print()
# print(command)
# try:
#     subprocess.Popen(command, shell=True)
#     print("Run carla")
# except subprocess.CalledProcessError as e:
#     print(f"Run carla fail: {e.returncode}")

# time.sleep(10)

# print()
# print('Prepare Scenario Vector')
# vector = np.random.random((1,15,71))
# print(vector_path)
# if os.path.exists(vector_path):
#     np.savez(vector_path, np.concatenate((np.load(vector_path)['arr_0'],vector)))
# else:
#     np.savez(vector_path, vector)

# print('Ready to Test')

# command = 'cd ~/Projects/TCP-Interfuser;sh leaderboard/scripts/test_basement.sh {} {} {} "{}"'.format(ADS, section, level, vector_path)
# print()
# print(command)
# try:
#     subprocess.run(command, shell=True)
#     print("Run ADS")
# except subprocess.CalledProcessError as e:
#     print(f"Run ADS fail: {e.returncode}")


# print('Ready to Test')
# ADS = 'InterFuser'
# section = 'Curve'
# level = 0


# command = 'cd ~/Projects/TCP-Interfuser;sh leaderboard/scripts/test_basement.sh {} {} {} "{}"'.format(ADS, section, level, vector_path)
# print()
# print(command)
# try:
#     subprocess.run(command, shell=True)
#     print("Run ADS")
# except subprocess.CalledProcessError as e:
#     print(f"Run ADS fail: {e.returncode}")


# current_datetime = datetime.datetime.now()
# formatted_datetime = current_datetime.strftime("%Y-%m-%d|%H:%M:%S")
# vector_path = '{}.npz'.format('vectorPath|'+formatted_datetime)

# data_root = '/home/guannan/Projects/SBT-data/InterFuser'
# data_root = ../SBT-data/${MODEL}/${current_time}--${SAVE_IMG_TEXT}
search_based_testing()