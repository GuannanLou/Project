import subprocess
import time
import carla
import gc 
import psutil
from pathlib import Path

def is_port_free(port):
    connections = psutil.net_connections()
    for conn in connections:
        if conn.laddr and conn.laddr.port == port:
            return False
    return True

def get_free_port(last_port, start, end):
    next_port = last_port
    while not is_port_free(next_port):
        next_port += 1
        if next_port == end:
            next_port = start
    return next_port

def run_carla(carla_path = '../CARLA_0.9.15/CarlaUE4.sh', rander=True, port=2000):
    rander_setting = '' if rander else ' -RenderOffScreen -opengl -nosound'
    command =  carla_path + rander_setting + ' --world-port={}'.format(port)
    print()
    print(command)
    try:
        print("Run carla")
        subprocess.Popen(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Run carla fail: {e.returncode}")

    time.sleep(10)


def run_agent(ADS, section, vector_path, current_time, port):
    level = 0
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    RUN_CASES_SCRIPT = PROJECT_ROOT / "leaderboard" / "scripts" / "run_cases.sh"

    # command = 'cd ~/Projects/TCP-Interfuser;sh leaderboard/scripts/run_cases.sh {} {} {} "{}" "{}" {}'.format(ADS, section, level, vector_path, current_time, port)
    command = f'cd {PROJECT_ROOT};sh {RUN_CASES_SCRIPT} {ADS} {section} {level} "{vector_path}" "{current_time}" {port}'
    
    print()
    print(command)
    try:
        print("Run ADS")
        subprocess.run(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Run ADS fail: {e.returncode}")
    
    time.sleep(10)

def clean_carla(port):
    print('Cleaner try to connect Carla')
    client = carla.Client('localhost', port)
    client.set_timeout(10.0)
    world = client.get_world()

    print('Clean all actors')
    actors = world.get_actors()
    for actor in actors:
        actor.destroy()

    gc.collect()
    print('Finished')

def kill_carla():
    command = 'pkill -9 -f "CarlaUE4-Linux-Shipping|CarlaUE4.sh"'
    print()
    print(command)
    try:
        print("Stop carla")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fail to stop carla: {e.returncode}")
    time.sleep(10)

def kill_agent():
    command = 'pkill -9 -f "python3"'
    print()
    print(command)
    try:
        print("Stop agent")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fail to stop agent: {e.returncode}")
    time.sleep(10)

def kill_by_port(port):
    # command = 'kill -9 $(lsof -t -i:{})'.format(port)
    command = 'fuser -k {}/tcp'.format(port)
    print()
    print(command)
    try:
        print("Stop by port")
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fail to stop by port: {e.returncode}")
    time.sleep(10)