import os
import sys
import time
import torch
import random
import datetime
import traceback
import subprocess
import numpy as np
import pandas as pd

from .surrogate_train import train
from .simulator_utils import run_carla, kill_carla, run_agent, kill_agent, kill_by_port, get_free_port
from .AutoFuzz_utils import parse_fuzzing_arguments, is_distinct_vectorized
from .pgd_attack import pgd_attack, train_net

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

save_surrogate_log = True

fuzzing_arguments = parse_fuzzing_arguments()

class MyProblem(Problem):
    def __init__(self, fuzzing_arguments, 
                #  sim_specific_arguments, 
                #  fuzzing_content, 
                #  run_simulation, 
                #  dt_arguments
                 ):    
        self.fuzzing_arguments = fuzzing_arguments

        self.fuzzing_arguments = fuzzing_arguments
        # self.sim_specific_arguments = sim_specific_arguments
        # self.fuzzing_content = fuzzing_content
        # self.run_simulation = run_simulation
        # self.dt_arguments = dt_arguments


        self.ego_car_model = fuzzing_arguments.ego_car_model
        self.scheduler_port = fuzzing_arguments.scheduler_port
        self.dashboard_address = fuzzing_arguments.dashboard_address
        self.ports = fuzzing_arguments.ports
        self.episode_max_time = fuzzing_arguments.episode_max_time
        self.objective_weights = fuzzing_arguments.objective_weights
        self.check_unique_coeff = fuzzing_arguments.check_unique_coeff
        self.consider_interested_bugs = fuzzing_arguments.consider_interested_bugs
        self.record_every_n_step = fuzzing_arguments.record_every_n_step
        self.use_single_objective = fuzzing_arguments.use_single_objective
        self.simulator = fuzzing_arguments.simulator


        if self.fuzzing_arguments.sample_avoid_ego_position and hasattr(self.sim_specific_arguments, 'ego_start_position'):
            self.ego_start_position = self.sim_specific_arguments.ego_start_position
        else:
            self.ego_start_position = None


        # self.call_from_dt = dt_arguments.call_from_dt
        # self.dt = dt_arguments.dt
        # self.estimator = dt_arguments.estimator
        # self.critical_unique_leaves = dt_arguments.critical_unique_leaves
        # self.cumulative_info = dt_arguments.cumulative_info
        # cumulative_info = dt_arguments.cumulative_info

        # if cumulative_info:
        #     self.counter = cumulative_info['counter']
        #     self.has_run = cumulative_info['has_run']
        #     self.start_time = cumulative_info['start_time']
        #     self.time_list = cumulative_info['time_list']
        #     self.bugs = cumulative_info['bugs']
        #     self.unique_bugs = cumulative_info['unique_bugs']
        #     self.interested_unique_bugs = cumulative_info['interested_unique_bugs']
        #     self.bugs_type_list = cumulative_info['bugs_type_list']
        #     self.bugs_inds_list = cumulative_info['bugs_inds_list']
        #     self.bugs_num_list = cumulative_info['bugs_num_list']
        #     self.unique_bugs_num_list = cumulative_info['unique_bugs_num_list']
        #     self.has_run_list = cumulative_info['has_run_list']
        # else:
        #     self.counter = 0
        #     self.has_run = 0
        #     self.start_time = time.time()
        #     self.time_list = []
        #     self.bugs = []
        #     self.unique_bugs = []
        #     self.interested_unique_bugs = []
        #     self.bugs_type_list = []
        #     self.bugs_inds_list = []
        #     self.bugs_num_list = []
        #     self.unique_bugs_num_list = []
        #     self.has_run_list = []

        self.counter = 0
        self.has_run = 0
        self.start_time = time.time()
        self.time_list = []
        self.bugs = []
        self.unique_bugs = []
        self.interested_unique_bugs = []
        self.bugs_type_list = []
        self.bugs_inds_list = []
        self.bugs_num_list = []
        self.unique_bugs_num_list = []
        self.has_run_list = []


        # self.labels = fuzzing_content.labels
        # self.mask = fuzzing_content.mask
        # self.parameters_min_bounds = fuzzing_content.parameters_min_bounds
        # self.parameters_max_bounds = fuzzing_content.parameters_max_bounds
        # self.parameters_distributions = fuzzing_content.parameters_distributions
        # self.customized_constraints = fuzzing_content.customized_constraints
        # self.customized_center_transforms = fuzzing_content.customized_center_transforms
        # xl = [pair[1] for pair in self.parameters_min_bounds.items()]
        # xu = [pair[1] for pair in self.parameters_max_bounds.items()]
        # n_var = fuzzing_content.n_var

        n_var = 71
        xl = np.zeros(n_var)
        xu = np.ones(n_var)


        self.p, self.c, self.th = self.check_unique_coeff
        self.launch_server = True
        self.objectives_list = []
        self.trajectory_vector_list = []
        self.x_list = []
        self.y_list = []
        self.F_list = []

        super().__init__(n_var=n_var, n_obj=4, n_constr=0, xl=xl, xu=xu)

        self.current_port = 2000
        self.generations = 0
        self.simulations = 0
        self.surrogate_model = None

    def _evaluate(self, X, out, *args, **kwargs):
        
        self.generations += 1
        self.current_port = get_free_port(self.current_port, 2000,3000)

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d|%H:%M:%S")

        print("Current generation:", self.generations)

        fitness = []
        while len(fitness) != len(X):
            fitness = execute(X[len(fitness):], self.current_port, formatted_datetime)

        self.simulations += len(X)
        out['F'] = fitness
        self.F_list.append(fitness)
        """TODO:Handle Error"""
        
        time_elapsed = time.time() - self.start_time
        self.time_list.append(time_elapsed)

        """TODO:Get unique bug
        get_unique_bugs()"""

        # objective_weights = self.objective_weights
        # customized_center_transforms = self.customized_center_transforms

        # episode_max_time = self.episode_max_time

        # default_objectives = self.fuzzing_arguments.default_objectives
        # standardize_objective = self.fuzzing_arguments.standardize_objective
        # normalize_objective = self.fuzzing_arguments.normalize_objective
        # traj_dist_metric = self.fuzzing_arguments.traj_dist_metric


        # all_final_generated_transforms_list = []

        # tmp_run_info_list = []
        # x_sublist = []
        # objectives_sublist_non_traj = []
        # trajectory_vector_sublist = []

        # for i in range(X.shape[0]):
        #     if self.counter == 0:
        #         launch_server = True
        #     else:
        #         launch_server = False

        #     cur_i = i
        #     total_i = self.counter

        #     port = self.ports[0]
        #     x = X[cur_i]

        #     manager = Manager()
        #     return_dict = manager.dict()
        #     try:
        #         p = Process(target=fun, args=(self, x, launch_server, self.counter, port, return_dict))
        #         p.start()
        #         p.join(240)
        #         if p.is_alive():
        #             print("Function is hanging!")
        #             p.terminate()
        #             print("Kidding, just terminated!")
        #         if 'returned_data' in return_dict:
        #             objectives, run_info, has_run = return_dict['returned_data']
        #         else:
        #             raise
        #     except:
        #         traceback.print_exc()
        #         objectives, run_info, has_run = default_objectives, None, 0

        #     print('get job result for', total_i)
        #     if run_info and 'all_final_generated_transforms' in run_info:
        #         all_final_generated_transforms_list.append(run_info['all_final_generated_transforms'])

        #     self.has_run_list.append(has_run)
        #     self.has_run += has_run

        #     # record bug
        #     if run_info and run_info['is_bug']:
        #         self.bugs.append(X[cur_i].astype(float))
        #         self.bugs_inds_list.append(total_i)
        #         self.bugs_type_list.append(run_info['bug_type'])

        #         self.y_list.append(run_info['bug_type'])
        #     else:
        #         self.y_list.append(0)



        #     self.counter += 1
        #     tmp_run_info_list.append(run_info)
        #     x_sublist.append(x)
        #     objectives_sublist_non_traj.append(objectives)
        #     if run_info and 'trajectory_vector' in run_info:
        #         trajectory_vector_sublist.append(run_info['trajectory_vector'])
        #     else:
        #         trajectory_vector_sublist.append(None)


        # job_results, self.x_list, self.objectives_list, self.trajectory_vector_list = get_job_results(tmp_run_info_list, x_sublist, objectives_sublist_non_traj, trajectory_vector_sublist, self.x_list, self.objectives_list, self.trajectory_vector_list, traj_dist_metric)
        # print('self.objectives_list', self.objectives_list)


        # # hack:
        # if run_info and 'all_final_generated_transforms' in run_info:
        #     with open('carla_lbc/tmp_folder/total.pickle', 'wb') as f_out:
        #         pickle.dump(all_final_generated_transforms_list, f_out)

        # record time elapsed and bug numbers
        # time_elapsed = time.time() - self.start_time
        # self.time_list.append(time_elapsed)


        # current_F = get_F(job_results, self.objectives_list, objective_weights, self.use_single_objective, standardize=standardize_objective, normalize=normalize_objective)

        # out["F"] = current_F
        # self.F_list.append(fitness)
        # print('\n'*3, 'self.F_list', len(self.F_list), self.F_list, '\n'*3)

        # print('\n'*10, '+'*100)



        # bugs_type_list_tmp = self.bugs_type_list
        # bugs_tmp = self.bugs
        # bugs_inds_list_tmp = self.bugs_inds_list

        # self.unique_bugs, unique_bugs_inds_list, self.interested_unique_bugs, bugcounts = get_unique_bugs(self.x_list, self.objectives_list, self.mask, self.xl, self.xu, self.check_unique_coeff, objective_weights, return_mode='unique_inds_and_interested_and_bugcounts', consider_interested_bugs=1, bugs_type_list=bugs_type_list_tmp, bugs=bugs_tmp, bugs_inds_list=bugs_inds_list_tmp, trajectory_vector_list=self.trajectory_vector_list)


        # time_elapsed = time.time() - self.start_time
        # num_of_bugs = len(self.bugs)
        # num_of_unique_bugs = len(self.unique_bugs)
        # num_of_interested_unique_bugs = len(self.interested_unique_bugs)

        # self.bugs_num_list.append(num_of_bugs)
        # self.unique_bugs_num_list.append(num_of_unique_bugs)
        # mean_objectives_this_generation = np.mean(np.array(self.objectives_list[-X.shape[0]:]), axis=0)

        # with open(self.fuzzing_arguments.mean_objectives_across_generations_path, 'a') as f_out:

        #     info_dict = {
        #         'counter': self.counter,
        #         'has_run': self.has_run,
        #         'time_elapsed': time_elapsed,
        #         'num_of_bugs': num_of_bugs,
        #         'num_of_unique_bugs': num_of_unique_bugs,
        #         'num_of_interested_unique_bugs': num_of_interested_unique_bugs,
        #         'bugcounts and unique bug counts': bugcounts, 'mean_objectives_this_generation': mean_objectives_this_generation.tolist(),
        #         'current_F': current_F
        #     }

        #     f_out.write(str(info_dict))
        #     f_out.write(';'.join([str(ind) for ind in unique_bugs_inds_list])+' objective_weights : '+str(self.objective_weights)+'\n')
        # print(info_dict)
        # print('+'*100, '\n'*10)


class CollisionProblem(Problem):
    def __init__(self, n_var):

        super().__init__(n_var=n_var, 
                         n_obj=4, 
                         n_constr=0, 
                         xl=np.zeros(n_var),
                         xu=np.ones(n_var))
        # super().__init__(n_var=n_var,
        #                  n_obj=1,
        #                  xl=np.zeros(n_var),
        #                  xu=np.ones(n_var))
        self.current_port = 2000
        self.generations = 0
        self.surrogate_model = None

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        self.generations += 1
        self.current_port = get_free_port(self.current_port, 2000,3000)

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d|%H:%M:%S")

        print("Current generation:", self.generations)
        # if self.generations >= 7 and self.generations % 4 == 1:
        if self.generations % 4 == 1 or self.generations <= 7:
        # if False:
            
            fitness = []
            while len(fitness) != len(x):
                fitness = execute(x[len(fitness):], self.current_port, formatted_datetime)
                # print(len(fitness), len(x))
            self.surrogate_model = train(root  = '/home/guannan/Projects/SBT-data/InterFuser', 
                                         start = '2025-03-18|21:00:00', 
                                         end   = '2025-03-21|09:30:00')
        else:
            self.surrogate_model.eval()
            with torch.no_grad():
                fitness = np.array(
                    self.surrogate_model(
                        torch.tensor(np.array(x).astype(np.float32))
                    )
                ).tolist()

        out['F'] = fitness

def execute(scenario_vector, current_port, formatted_datetime):
    print()
    print('Save Scenario Vector')
    vector_path = '/home/guannan/Projects/TCP-Interfuser/temp.npz'
    print(vector_path)
    np.savez(vector_path, scenario_vector)

    run_carla(rander=False, port=current_port)
    run_agent('InterFuser', 'Curve', vector_path, formatted_datetime, current_port)

    kill_by_port(current_port)

    data_root = '/home/guannan/Projects/SBT-data/InterFuser/{}/'.format(formatted_datetime)
    fitness_file = data_root+'fitness.csv'
    cirtion_file = data_root+'criterion.csv'
    
    fitness = get_fitness(cirtion_file, fitness_file, col='CollisionTest')

    print()
    print()

    if fitness is None:
        return np.array([])
    else:
        return fitness

def get_folder(folder, indexs):
    result = []
    dirs = os.listdir(folder)
    dirs.sort()
    dirs = dirs[3:-1]
    for i in indexs:
        result.append(dirs[i])
    return result

def get_max_gear(case):
    return pd.read_csv(case+'control.csv')['gear'].max()

def get_fitness(criterion_dir, fitness_dir, col, length=False, direction=True):
    folder = criterion_dir.replace('criterion.csv','')

    with open(criterion_dir, 'r') as f:
        total_lines = sum(1 for _ in f)

    criterion_header = ["RouteCompletionTest",   
                "RouteCompletionTest_figure",
                "OutsideRouteLanesTest", 
                "OutsideRouteLanesTest_figure",
                "CollisionTest",         
                "CollisionTest_figure",
                "RunningRedLightTest",   
                "RunningRedLightTest_figure",
                "RunningStopTest",       
                "RunningStopTest_figure",
                "InRouteTest", 
                "InRouteTest_figure",          
                "AgentBlockedTest",
                "AgentBlockedTest_figure",      
                "Timeout"]
    
    fitness_header = ["DOL","DVE","DPD","DSM","DFD"]
    if direction:
        fitness_header = ["DOL","DVE",'DVE-d',"DPD","DSM","DFD"]

    if length:
        criterion = pd.read_csv(criterion_dir,names=criterion_header, skiprows=range(total_lines-length))
        fitness = pd.read_csv(fitness_dir,names=fitness_header, skiprows=range(total_lines-length))
    else:
        criterion = pd.read_csv(criterion_dir,names=criterion_header)
        fitness = pd.read_csv(fitness_dir,names=fitness_header)

    if direction:
        fitness['DVE'] = fitness['DVE'].apply(lambda data: float(data[1:]))
        fitness['DVE-d'] = fitness['DVE-d'].apply(lambda data: float(data[:-1]))
    else:
        fitness['DVE-d'] = np.zeros_like(fitness['DVE'])
    result = pd.DataFrame()

    result['RouteCompletionTest']   =   criterion["RouteCompletionTest_figure"]/100
    result['OutsideRouteLanesTest'] = 1-criterion["OutsideRouteLanesTest_figure"]/100
    result['CollisionTest']         =   criterion["CollisionTest"]/2*2
    result['RunningRedLightTest']   = 1-criterion["RunningRedLightTest"]
    result['RunningStopTest']       = 1-criterion["RunningStopTest"]
    result['InRouteTest']           = 1-criterion["InRouteTest"]
    result['AgentBlockedTest']      = 1-criterion["AgentBlockedTest"]
    result['Timeout']               = 1-criterion["Timeout"]

    DVE = fitness['DVE'].copy()/2
    DVE[fitness['DVE'] >= 2] = 1
    collisionTest = result['CollisionTest'].copy()
    collisionTest[result['CollisionTest']==0] = DVE[result['CollisionTest']==0]
    collisionTest[result['CollisionTest']==1] = 0
    result.loc[:,'CollisionTest'] = collisionTest
    result.loc[:,'CollisionDirection'] = fitness['DVE-d']

    DOL = fitness['DOL'].copy() - 0.5
    DOL[fitness['DOL'] >= 1.5] = 1
    DOL[fitness['DOL'] <= 0.5] = 0
    result.loc[:,'OutsideRouteLanesTest'] = 1-DOL

    DPD = fitness['DPD'].copy()/10
    DPD[fitness['DPD'] >= 10] = 1
    result['PedestrianTest'] = DPD

    change = []
    for i, case in enumerate(get_folder(folder, result[result['CollisionTest'] == 0].index.to_numpy())):
        if get_max_gear(folder+case+'/') == 0:
            index = result[result['CollisionTest'] == 0].index.to_numpy()[i]
            change.append(index)
            print('ERROR: Wrong Collision -', folder+case)

    for index in change:
        result.loc[index,'CollisionTest'] = 1
        result.loc[index,'RouteCompletionTest'] = 1

    return result[col].to_numpy()

def GA_search_collision(pop_size = 20, n_offsprings = 1, generations = 50):
    print("begin GA_search_collision")

    problem = CollisionProblem(n_var=11+15*4)

    algorithm = GA(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=(np.arange(1,pop_size+1)*(1/pop_size)).reshape((pop_size,1))*np.ones((1,71)),
        crossover=SBX(prob=0.8, eta=5),
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
    
    print('X:', X.shape, X)
    print('F:', F.shape, F)
    
    np.savez('./data/output.npz', X, F)

    if save_surrogate_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

def search_based_testing():
    print('GA_search')
    try:
        GA_search_collision(
            pop_size     = 20, 
            n_offsprings = 10, 
            generations  = 30
        )

    except Exception as e:
        traceback.print_exc()

import copy
from pymoo.core.initialization import Initialization
from pymoo.core.individual import Individual
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.population import Population

"""TODO: import"""
# from customized_utils import rand_real,  make_hierarchical_dir, exit_handler, is_critical_region, if_violate_constraints, filter_critical_regions, encode_fields, remove_fields_not_changing, get_labels_to_encode, customized_fit, customized_standardize, customized_inverse_standardize, decode_fields, encode_bounds, recover_fields_not_changing, process_X, inverse_process_X, calculate_rep_d, select_batch_max_d_greedy, if_violate_constraints_vectorized, is_distinct_vectorized, eliminate_repetitive_vectorized, get_sorted_subfolders, load_data, get_F, set_general_seed, emptyobject, get_job_results, choose_farthest_offs
# from customized_utils import pretrain_regression_nets
"""END TODO: import"""



class NSGA2_CUSTOMIZED(NSGA2):
    def __init__(self, dt=False, X=None, F=None, fuzzing_arguments=None, plain_sampling=None, local_mating=None, **kwargs):
        self.dt = dt
        self.X = X
        self.F = F
        self.plain_sampling = plain_sampling

        """TODO:add"""
        self.sampling = kwargs['sampling']
        self.pop_size = fuzzing_arguments.pop_size
        self.n_offsprings = fuzzing_arguments.n_offsprings

        self.survival_multiplier = fuzzing_arguments.survival_multiplier
        self.algorithm_name = fuzzing_arguments.algorithm_name
        self.emcmc = fuzzing_arguments.emcmc
        self.initial_fit_th = fuzzing_arguments.initial_fit_th
        self.rank_mode = fuzzing_arguments.rank_mode
        self.min_bug_num_to_fit_dnn = fuzzing_arguments.min_bug_num_to_fit_dnn
        self.ranking_model = fuzzing_arguments.ranking_model
        self.use_unique_bugs = fuzzing_arguments.use_unique_bugs
        self.pgd_eps = fuzzing_arguments.pgd_eps
        self.adv_conf_th = fuzzing_arguments.adv_conf_th
        self.attack_stop_conf = fuzzing_arguments.attack_stop_conf
        self.uncertainty = fuzzing_arguments.uncertainty
        self.warm_up_path = fuzzing_arguments.warm_up_path
        self.warm_up_len = fuzzing_arguments.warm_up_len
        self.regression_nn_use_running_data = fuzzing_arguments.regression_nn_use_running_data
        self.only_run_unique_cases = fuzzing_arguments.only_run_unique_cases
        """END TODO:add"""

        super().__init__(pop_size=self.pop_size, n_offsprings=self.n_offsprings, **kwargs)

        self.plain_initialization = Initialization(self.plain_sampling, 
                                                   individual=Individual(), 
                                                   repair=self.repair, 
                                                   eliminate_duplicates= NoDuplicateElimination())


        # heuristic: we keep up about 1 times of each generation's population
        self.survival_size = self.pop_size * self.survival_multiplier

        self.all_pop_run_X = []

        # hack: defined separately w.r.t. MyMating
        self.mating_max_iterations = 1

        self.tmp_off = []
        self.tmp_off_type_1_len = 0
        # self.tmp_off_type_1and2_len = 0

        self.high_conf_configs_stack = []
        self.high_conf_configs_ori_stack = []

        self.device_name = 'cuda'


        # avfuzzer variables
        self.best_y_gen = []
        self.global_best_y = [None, 10000]
        self.restart_best_y = [None, 10000]
        self.local_best_y = [None, 10000]

        self.pop_before_local = None

        self.local_gen = -1
        self.restart_gen = 0
        self.cur_gen = -1

        self.local_mating = local_mating
        self.mutation = kwargs['mutation']

        self.minLisGen = 2


    def set_off(self):
        self.tmp_off = []

        print('len(self.pop)', len(self.pop))
        # do the mating using the current population
        if len(self.pop) > 0:
            self.tmp_off, parents = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        print('\n'*3, 'after mating len 0', len(self.tmp_off), 'self.n_offsprings', self.n_offsprings, '\n'*3)

        if len(self.tmp_off) < self.n_offsprings:
            remaining_num = self.n_offsprings - len(self.tmp_off)
            remaining_off = self.initialization.do(self.problem, remaining_num, algorithm=self)
            remaining_parrents = remaining_off
            if len(self.tmp_off) == 0:
                self.tmp_off = remaining_off
                parents = remaining_parrents
            else:
                self.tmp_off = Population.merge(self.tmp_off, remaining_off)
                parents = Population.merge(parents, remaining_parrents)

            print('\n'*3, 'unique after random generation len 1', len(self.tmp_off), '\n'*3)

        self.tmp_off_type_1_len = len(self.tmp_off)

        if len(self.tmp_off) < self.n_offsprings:
            remaining_num = self.n_offsprings - len(self.tmp_off)
            remaining_off = self.plain_initialization.do(self.problem, remaining_num, algorithm=self)
            remaining_parrents = remaining_off

            self.tmp_off = Population.merge(self.tmp_off, remaining_off)
            parents = Population.merge(parents, remaining_parrents)

            print('\n'*3, 'random generation len 2', len(self.tmp_off), '\n'*3)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        # if len(self.tmp_off) == 0 or (not self.problem.call_from_dt and 
        #                               self.problem.fuzzing_arguments.finish_after_has_run and 
        #                               self.problem.has_run >= self.problem.fuzzing_arguments.has_run_num):
        if len(self.tmp_off) == 0 or (self.problem.has_run >= self.problem.fuzzing_arguments.has_run_num):
            self.termination.force_termination = True
            print("Mating cannot generate new springs, terminate earlier.")
            print('self.tmp_off', len(self.tmp_off), self.tmp_off)
            return
        # if not the desired number of offspring could be created
        elif len(self.tmp_off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")


        # additional step to rank and select self.off after gathering initial population
        # if ((self.rank_mode == 'none') or 
        #     (self.rank_mode in ['nn', 'adv_nn'] and 
        #         (len(self.problem.objectives_list) < self.initial_fit_th or 
        #         np.sum(determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)) < self.min_bug_num_to_fit_dnn)) or 
        #     (self.rank_mode in ['regression_nn'] and 
        #         len(self.problem.objectives_list) < self.pop_size)
        #     ):

        """TODO: problem.objectives_list, problem.objective_weights"""
        if (self.rank_mode in ['nn', 'adv_nn'] and 
                (len(self.problem.objectives_list) < self.initial_fit_th or 
                np.sum(determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)) < self.min_bug_num_to_fit_dnn)):
            self.off = self.tmp_off[:self.pop_size]
        else:
           
            standardize_prev = None

            X_train_ori = self.all_pop_run_X
            X_test_ori = self.tmp_off.get("X")

            initial_X = np.concatenate([X_train_ori, X_test_ori])
            cutoff = X_train_ori.shape[0]
            cutoff_end = initial_X.shape[0]
            partial = True

            X_train, X_test, xl, xu, labels_used, standardize, one_hot_fields_len, param_for_recover_and_decode = process_X(initial_X, 
                                                                                                                            self.problem.labels, 
                                                                                                                            self.problem.xl, 
                                                                                                                            self.problem.xu, 
                                                                                                                            cutoff, 
                                                                                                                            cutoff_end, 
                                                                                                                            partial, 
                                                                                                                            len(self.problem.interested_unique_bugs), 
                                                                                                                            self.problem.fuzzing_content.keywords_dict, 
                                                                                                                            standardize_prev=standardize_prev)

            (X_removed, kept_fields, removed_fields, enc, inds_to_encode, inds_non_encode, encoded_fields, _, _, unique_bugs_len) = param_for_recover_and_decode

            print('process_X finished')

            adv_conf_th = self.adv_conf_th
            attack_stop_conf = self.attack_stop_conf

            y_train = determine_y_upon_weights(self.problem.objectives_list, self.problem.objective_weights)

            if self.ranking_model == 'nn_pytorch':
                print(X_train.shape, y_train.shape)
                clf = train_net(X_train, y_train, [], [], batch_train=200, device_name=self.device_name)
            else:
                raise ValueError('invalid ranking model', self.ranking_model)
            print('X_train', X_train.shape)
            print('clf.predict_proba(X_train)', clf.predict_proba(X_train).shape)

            if self.ranking_model != 'adaboost':
                prob_train = clf.predict_proba(X_train)[:, 1].squeeze()
            cur_y = y_train

            if self.adv_conf_th < 0 and self.rank_mode in ['adv_nn']:
                adv_conf_th = sorted(prob_train, reverse=True)[int(np.sum(cur_y)//np.abs(self.adv_conf_th))]
                attack_stop_conf = np.max([self.attack_stop_conf, adv_conf_th])
            if self.adv_conf_th > attack_stop_conf:
                self.adv_conf_th = attack_stop_conf


            pred = clf.predict_proba(X_test)
            if len(pred.shape) == 1:
                pred = np.expand_dims(pred, axis=0)
            scores = pred[:, 1]

            print('initial scores', scores)

            if self.rank_mode == 'adv_nn':
                X_test_pgd_ori = None
                X_test_pgd = None


            if self.use_unique_bugs:
                print('self.tmp_off_type_1_len', self.tmp_off_type_1_len)
                scores[:self.tmp_off_type_1_len] += np.max(scores)
            scores *= -1

            inds = np.argsort(scores)[:self.pop_size]
            print('scores', scores)
            print('sorted(scores)', sorted(scores))
            print('chosen indices', inds)

            X_test_pgd_ori = X_test_ori[inds]
            X_test_pgd = X_test[inds]
            associated_clf_id = []

            # conduct pgd with constraints differently for different types of inputs
            # self.use_unique_bugs == 1
            unique_coeff = (self.problem.p, self.problem.c, self.problem.th)
            mask = self.problem.mask

            y_zeros = np.zeros(X_test_pgd.shape[0])
            X_test_adv, new_bug_pred_prob_list, initial_bug_pred_prob_list = pgd_attack(clf, 
                                                                                        X_test_pgd, 
                                                                                        y_zeros, 
                                                                                        xl, xu, 
                                                                                        encoded_fields, 
                                                                                        labels_used, 
                                                                                        self.problem.customized_constraints, 
                                                                                        standardize, 
                                                                                        prev_X=self.problem.interested_unique_bugs, 
                                                                                        base_ind=0, 
                                                                                        unique_coeff=unique_coeff, 
                                                                                        mask=mask, 
                                                                                        param_for_recover_and_decode=param_for_recover_and_decode, 
                                                                                        eps=self.pgd_eps, 
                                                                                        adv_conf_th=adv_conf_th, 
                                                                                        attack_stop_conf=attack_stop_conf, 
                                                                                        associated_clf_id=associated_clf_id, 
                                                                                        X_test_pgd_ori=X_test_pgd_ori, 
                                                                                        consider_uniqueness=True, 
                                                                                        device_name=self.device_name)

            X_test_adv_processed = inverse_process_X(X_test_adv, 
                                                     standardize, 
                                                     one_hot_fields_len, 
                                                     partial, 
                                                     X_removed, 
                                                     kept_fields, 
                                                     removed_fields, 
                                                     enc, 
                                                     inds_to_encode, 
                                                     inds_non_encode, 
                                                     encoded_fields)
            X_off = X_test_adv_processed

            pop = Population(X_off.shape[0], individual=Individual())
            pop.set("X", X_off)
            pop.set("F", [None for _ in range(X_off.shape[0])])
            self.off = pop


        if self.only_run_unique_cases:
            X_off = [off_i.X for off_i in self.off]
            remaining_inds = is_distinct_vectorized(X_off, 
                                                    self.problem.interested_unique_bugs, 
                                                    self.problem.mask, 
                                                    self.problem.xl, 
                                                    self.problem.xu, 
                                                    self.problem.p, 
                                                    self.problem.c, 
                                                    self.problem.th, 
                                                    verbose=False)
            self.off = self.off[remaining_inds]

        self.off.set("n_gen", self.n_gen)

        print('\n'*2, 'self.n_gen', self.n_gen, '\n'*2)

        if len(self.all_pop_run_X) == 0:
            self.all_pop_run_X = self.off.get("X")
        else:
            if len(self.off.get("X")) > 0:
                self.all_pop_run_X = np.concatenate([self.all_pop_run_X, self.off.get("X")])

    # mainly used to modify survival
    def _next(self):

        # set self.off
        self.set_off()
        # evaluate the offspring
        if len(self.off) > 0:
            self.evaluator.eval(self.problem, self.off, algorithm=self)
        
        # merge the offsprings with the current population
        self.pop = Population.merge(self.pop, self.off)

        # the do survival selection
        if self.survival:
            print('\n'*3)
            print('len(self.pop) before', len(self.pop))
            print('survival')
            self.pop = self.survival.do(self.problem, self.pop, self.survival_size, algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)
            print('len(self.pop) after', len(self.pop))
            print(self.pop_size, self.survival_size)
            print('\n'*3)


    def _initialize(self):
        if (self.warm_up_path and 
            ((self.dt and not self.problem.cumulative_info) or 
             (not self.dt))):
            subfolders = get_sorted_subfolders(self.warm_up_path)
            X, _, objectives_list, mask, _, _ = load_data(subfolders)

            if self.warm_up_len > 0:
                X = X[:self.warm_up_len]
                objectives_list = objectives_list[:self.warm_up_len]
            else:
                self.warm_up_len = len(X)

            xl = self.problem.xl
            xu = self.problem.xu
            p, c, th = self.problem.p, self.problem.c, self.problem.th
            unique_coeff = (p, c, th)


            self.problem.unique_bugs, (self.problem.bugs, self.problem.bugs_type_list, self.problem.bugs_inds_list, self.problem.interested_unique_bugs) = get_unique_bugs(
                X, objectives_list, mask, xl, xu, unique_coeff, self.problem.objective_weights, return_mode='return_bug_info', consider_interested_bugs=self.problem.consider_interested_bugs
            )

            print('\n'*10)
            print('self.problem.bugs', len(self.problem.bugs))
            print('self.problem.unique_bugs', len(self.problem.unique_bugs))
            print('\n'*10)

            self.all_pop_run_X = np.array(X)
            self.problem.objectives_list = objectives_list.tolist()

        if self.dt:
            X_list = list(self.X)
            F_list = list(self.F)
            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])
            self.pop = pop
            self.set_off()
            pop = self.off

        elif self.warm_up_path:
            X_list = X[-self.pop_size:]
            current_objectives = objectives_list[-self.pop_size:]


            F_list = get_F(current_objectives, objectives_list, self.problem.objective_weights, self.problem.use_single_objective)


            pop = Population(len(X_list), individual=Individual())
            pop.set("X", X_list, "F", F_list, "n_gen", self.n_gen, "CV", [0 for _ in range(len(X_list))], "feasible", [[True] for _ in range(len(X_list))])

            self.pop = pop
            self.set_off()
            pop = self.off

        else:
            # create the initial population
            if self.use_unique_bugs:
                pop = self.initialization.do(self.problem, self.problem.fuzzing_arguments.pop_size, algorithm=self)
            else:
                pop = self.plain_initialization.do(self.problem, self.pop_size, algorithm=self)
            pop.set("n_gen", self.n_gen)


        if len(pop) > 0:
            self.evaluator.eval(self.problem, pop, algorithm=self)
        print('\n'*5, 'after initialize evaluator', '\n'*5)
        print('len(self.all_pop_run_X)', len(self.all_pop_run_X))


        # that call is a dummy survival to set attributes that are necessary for the mating selection
        if self.survival:
            pop = self.survival.do(self.problem, pop, len(pop), algorithm=self, n_min_infeas_survive=self.min_infeas_pop_size)

        self.pop, self.off = pop, pop

def determine_y_upon_weights(objective_list, objective_weights):
    collision_activated = np.sum(objective_weights[:3] != 0) > 0
    offroad_activated = (np.abs(objective_weights[3]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    wronglane_activated = (np.abs(objective_weights[4]) > 0) | (
        np.abs(objective_weights[5]) > 0
    )
    red_light_activated = np.abs(objective_weights[-1]) > 0

    y = np.zeros(len(objective_list))
    for i, obj in enumerate(objective_list):
        cond = 0
        if collision_activated:
            cond |= obj[0] > 0.1
        if offroad_activated:
            cond |= obj[-3] == 1
        if wronglane_activated:
            cond |= obj[-2] == 1
        if red_light_activated:
            cond |= obj[-1] == 1
        y[i] = cond

    return y