
import os
import sys
import traceback
import numpy as np
import subprocess

from SBT.problem import CustomizedProblem, SurrogateProblem, CollisionProblem
from utils.utils import mkdir, savepath_parser

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize



# GA = os.environ['GA']==True
LOG = os.environ['LOG']==True
REGION = int(os.environ.get('REGION', 7))
SURROGATE = os.environ['SURROGATE']==True
save_surrogate_log = True


def random_search(arguments, leaderboard_evaluator, route_indexer, case_number=3000, scenario_vecs=None):
    print("begin random_search")
    # print('LOG:', arguments.log)
    arguments.log=LOG
    # print('LOG:', arguments.log)
    config = None
    while route_indexer.peek():
        config = route_indexer.next()
    config.original_trajectory = [config.trajectory[0], config.trajectory[1]]

    # if scenario_vecs==[]:
    #     scenario_vecs = np.random.rand(case_number, 9+3+2)

    # leaderboard_evaluator.run_cases(scenario_vecs, config)

    for scenario_vec in scenario_vecs:
        leaderboard_evaluator.run_one_case(scenario_vec, config)



def GA_search(arguments, leaderboard_evaluator, route_indexer, pop_size = 50, n_offsprings = 10, generations = 76):
    print("begin")
    arguments.log=LOG
    config = None
    while route_indexer.peek():
        config = route_indexer.next()
    config.original_trajectory = [config.trajectory[0], config.trajectory[1]]

    savepath = savepath_parser(arguments.fitness_path)
    print(savepath)

    mkdir(savepath)
    if save_surrogate_log:
        output_file = savepath+'/console.log'
        sys.stdout = open(output_file, 'w')

    problem = None
    if SURROGATE:
        print('surrogate')
        problem = SurrogateProblem(config, surrogate_path='./data/'+arguments.fitness_path.split('/')[1]+'/')
    else:
        problem = CustomizedProblem(arguments.fitness_path,
                                    arguments.fitness_path.replace('fitness.csv','criterion.csv'),
                                    leaderboard_evaluator.run_one_case,
                                    config)
    # algorithm = NSGA2(
    #     pop_size=pop_size,
    #     n_offsprings=n_offsprings,
    #     sampling=FloatRandomSampling(),
    #     crossover=SBX(prob=0.9, eta=15),
    #     mutation=PM(eta=20),
    #     eliminate_duplicates=True
    # )

    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=(np.arange(pop_size)*(1/pop_size)).reshape((pop_size,1))*np.ones((1,71)),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", generations)

    res = minimize(problem,
        algorithm,
        termination,
        seed=1,
        save_history=False,
        verbose=True)

    X = res.X
    F = res.F
    
    print(X)
    print(F)

    np.savez('./data/'+arguments.fitness_path.split('/')[1]+'/output.npz', X, F)


    if save_surrogate_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__


def GA_search_collision(arguments, leaderboard_evaluator, route_indexer, pop_size = 20, n_offsprings = 1, generations = 50):
    print("begin GA_search_collision")
    arguments.log=LOG
    config = None
    while route_indexer.peek():
        config = route_indexer.next()
    config.original_trajectory = [config.trajectory[0], config.trajectory[1]]

    # savepath = savepath_parser(arguments.fitness_path)
    # print('savepath:',savepath)

    # mkdir(savepath)
    # if save_surrogate_log:
    #     output_file = savepath+'/console.log'
    #     sys.stdout = open(output_file, 'w')

    print()
    print('CHECK n_var')
    print()
    print()
    print('####Problem')
    problem = CollisionProblem(arguments.fitness_path,
                               arguments.fitness_path.replace('fitness.csv','criterion.csv'),
                               leaderboard_evaluator.run_one_case,
                               config,
                               n_var=11+15*4)

    print('####GA')
    algorithm = GA(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", generations)

    res = minimize(problem,
        algorithm,
        termination,
        seed=1,
        save_history=False,
        verbose=False)

    X = res.X
    F = res.F
    
    print('X:',X)
    print('F:',F)

    np.savez('./data/'+arguments.fitness_path.split('/')[1]+'/output.npz', X, F)


    if save_surrogate_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__


def search_based_testing(arguments, leaderboard_evaluator, route_indexer):
    arguments.region = REGION
    try:
        GA_search_collision(
            arguments, 
            leaderboard_evaluator, 
            route_indexer, 
            pop_size     = 2, 
            n_offsprings = 0, 
            generations  = 3
        )
        # if GA: 
        #     GA_search(arguments, 
        #             leaderboard_evaluator, 
        #             route_indexer, 
        #             pop_size     = 50, 
        #             n_offsprings = 10, 
        #             generations  = 76)
        # else:
            
        #     print('CHECK###:', arguments.vector_path)
        #     scenario_vecs = np.load(arguments.vector_path)['arr_0'][-1]
        #     # scenario_vecs = np.random.rand(300,71)
        #     # scenario_vecs = np.random.rand(300,14)
        #     print(scenario_vecs)
        #     random_search(arguments, 
        #                 leaderboard_evaluator, 
        #                 route_indexer, 
        #                 case_number=3000, 
        #                 scenario_vecs=scenario_vecs)
    except Exception as e:
        traceback.print_exc()
    finally:
        
        if not SURROGATE:
            del leaderboard_evaluator
    
    command = 'pkill -9 -f "CarlaUE4-Linux-Shipping|CarlaUE4.sh"'
    try:
        subprocess.run(command, shell=True, check=True)
        print("Stop carla")
    except subprocess.CalledProcessError as e:
        print(f"Fail to stop carla: {e.returncode}")

    command = 'pkill -9 -f "python3"'
    try:
        subprocess.run(command, shell=True, check=True)
        print("Stop carla")
    except subprocess.CalledProcessError as e:
        print(f"Fail to stop carla: {e.returncode}")