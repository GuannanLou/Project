import argparse
from argparse import RawTextHelpFormatter
import os
import pathlib
import random
# random.seed(3.1415926)

from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer

# from SBT.GA_search import search_based_testing
from execute_cases import TestCase

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def log_experiment_configs(json_path):
    import json

    ## Experiment Controls
    AGENT_MODE = int(os.environ.get('AGENT_MODE', float('inf')))
    ADS_MODEL = os.environ['ADS_MODEL']==True
    GA = os.environ['GA']==True
    SURROGATE = os.environ['SURROGATE']==True
    SURROGATE_MODEL = os.environ.get('SURROGATE_MODEL', '')
    TIMEOUT = int(os.environ.get('TIMEOUT', float('inf')))
    REGION = int(os.environ.get('REGION', 7))
              
    ## Information Collection
    SAVE_IMG = os.environ['SAVE_IMG']==True
    LOG = os.environ['LOG']==True
    SAVE_PATH = os.environ.get('SAVE_PATH', '')

    ## Route File
    ROUTE_FILE = os.environ.get('ROUTE_FILE', '')

    data = {
        'AGENT_MODE' : AGENT_MODE,
        'ADS_MODEL' : ADS_MODEL,
        'GA' : GA,
        'SURROGATE' : SURROGATE,
        'SURROGATE_MODEL' : SURROGATE_MODEL,
        'TIMEOUT' : TIMEOUT,
        'REGION' : REGION,
        'SAVE_IMG' : SAVE_IMG,
        'LOG' : LOG,
        'SAVE_PATH' : SAVE_PATH,
        'ROUTE_FILE' : ROUTE_FILE
    }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)

    print("Log experiment configs")

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default=60.0, type=int,
                        help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--log', default="1",
                        help='Whether print log to console')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--weather',
                        type=str, default='none',
                        help='Name of the weahter to be executed',
                        )
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--fitness_path", type=str,
                        default='./fitness.csv',
                        help="Path for fitness.csv")
    parser.add_argument('--agent_mode', type=int, help='Run with debug output', default=1)

    parser.add_argument('--vector_path', type=str,
                        default='./vector.csv',
                        help="Path for vector.csv")

    arguments = parser.parse_args()
    print("init statistics_manager")
    
    # surrogate = os.environ['SURROGATE']==True
    arguments.log = os.environ['LOG']==True
    if not os.path.exists(os.environ['SAVE_PATH']):
        pathlib.Path(os.environ['SAVE_PATH']).mkdir()

    print('PORT:', arguments.port)

    log_experiment_configs(os.environ['SAVE_PATH']+'experiment_config.json')

    statistics_manager = StatisticsManager()
    route_indexer = RouteIndexer(arguments.routes, arguments.scenarios, arguments.repetitions)
    leaderboard_evaluator = TestCase(arguments, statistics_manager)


    config = None
    while route_indexer.peek():
        config = route_indexer.next()
    config.original_trajectory = [config.trajectory[0], config.trajectory[1]]

    leaderboard_evaluator.run_cases(config)
    # print('Config in Problem')
    # print(config.__dict__)

    # search_based_testing(arguments, leaderboard_evaluator, route_indexer)

if __name__ == '__main__':
    main()