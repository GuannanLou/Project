# TODO: AutoFuzz Algorithm
# TODO: AutoFuzz Problem

import os
import json
import re
import time
import pathlib
import pickle
import copy
import atexit
import traceback
import math
from datetime import datetime
from distutils.dir_util import copy_tree
import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from multiprocessing import Process, Manager, set_start_method


from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament

from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover

from pymoo.factory import get_crossover, get_mutation, get_termination
from pymoo.util.termination.default import MultiObjectiveDefaultTermination, SingleObjectiveDefaultTermination

from pymoo.model.termination import Termination
from pymoo.model.repair import Repair
from pymoo.model.mating import Mating
from pymoo.model.initialization import Initialization
from pymoo.model.duplicate import NoDuplicateElimination
from pymoo.model.survival import Survival
from pymoo.model.individual import Individual

# disable pymoo optimization warning
from pymoo.configuration import Configuration
Configuration.show_compile_hint = False

from pgd_attack import pgd_attack, train_net, train_regression_net, VanillaDataset
from acquisition import map_acquisition

from customized_utils import rand_real,  make_hierarchical_dir, exit_handler, is_critical_region, if_violate_constraints, filter_critical_regions, encode_fields, remove_fields_not_changing, get_labels_to_encode, customized_fit, customized_standardize, customized_inverse_standardize, decode_fields, encode_bounds, recover_fields_not_changing, process_X, inverse_process_X, calculate_rep_d, select_batch_max_d_greedy, if_violate_constraints_vectorized, is_distinct_vectorized, eliminate_repetitive_vectorized, get_sorted_subfolders, load_data, get_F, set_general_seed, emptyobject, get_job_results, choose_farthest_offs


'''python ga_fuzzing.py 
-p 2021 
-s 8795 
-d 8796 

--n_gen 15 
--pop_size 50 
-r 'town07_front_0' 
-c 'go_straight_town07' 
--algorithm_name nsga2-un 
--has_run_num 700 


--objective_weights -1 1 1 0 0 0 0 0 0 0 
--rank_mode adv_nn 
--warm_up_path <path-to-warm-up-run-folder> 
--warm_up_len 500 
--check_unique_coeff 0 0.1 0.5 
--has_display 0 
--record_every_n_step 5 
--only_run_unique_cases 1

'''


argv = [
    "-p", "2018",
    "-s", "8793",
    "-d", "8794",
    "-r", "town07_front_0",
    "-c", "go_straight_town07",

    "--n_gen", "200",
    "--pop_size", "4",
    "--algorithm_name", "avfuzzer",
    "--has_run_num", "700",
    "--has_display", "0",
    "--n_offsprings", "50",

    "--record_every_n_step", "5",
    "--only_run_unique_cases", "0",

    "--objective_weights", "-1", "1", "1", "0", "0", "0", "0", "0", "0", "0",
    "--check_unique_coeff", "0", "0.1", "0.5"
]

fuzzing_arguments = parse_fuzzing_arguments(argv)

'''
python ga_fuzzing.py 
-p 2018 
-s 8793 
-d 8794 
-r 'town07_front_0' 
-c 'go_straight_town07' 

    --n_gen         200 
    --pop_size      4 
--algorithm_name    avfuzzer 
--has_run_num       700 
--has_display       0 
--n_offsprings      50

--record_every_n_step   5 
--only_run_unique_cases 0 

--objective_weights -1 1 1 0 0 0 0 0 0 0 
--check_unique_coeff 0 0.1 0.5 
'''

def parse_fuzzing_arguments(argv=None):
    # [ego_linear_speed, min_d, d_angle_norm, offroad_d, wronglane_d, dev_dist, is_offroad, is_wrong_lane, is_run_red_light, is_collision]
    default_objective_weights = np.array([-1., 1., 1., 1., 1., -1., 0., 0., 0., 0.])
    default_objectives = np.array([0., 20., 1., 7., 7., 0., 0., 0., 0., 0.])
    default_check_unique_coeff = [0, 0.1, 0.5]


    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("-r", "--route_type", type=str, default='town05_right_0')
    parser.add_argument("-c", "--scenario_type", type=str, default='default')
    parser.add_argument("-m", "--ego_car_model", type=str, default='lbc')
    parser.add_argument('-a','--algorithm_name', type=str, default='nsga2')

    parser.add_argument('-p','--ports', nargs='+', type=int, default=[2003], help='TCP port(s) to listen to (default: 2003)')
    parser.add_argument("-s", "--scheduler_port", type=int, default=8785)
    parser.add_argument("-d", "--dashboard_address", type=int, default=8786)

    parser.add_argument('--simulator', type=str, default='carla')

    parser.add_argument('--random_seed', type=int, default=0)


    # carla specific
    parser.add_argument("--has_display", type=str, default='0')
    parser.add_argument("--debug", type=int, default=1, help="whether using the debug mode: planned paths will be visualized.")
    parser.add_argument('--correct_spawn_locations_after_run', type=int, default=0)

    # carla_op specific
    parser.add_argument('--carla_path', type=str, default="../carla_0911_rss/CarlaUE4.sh")

    # no_simulation specific
    parser.add_argument('--no_simulation_data_path', type=str, default=None)
    parser.add_argument('--objective_labels', type=str, nargs='+', default=[])



    # logistic
    parser.add_argument("--root_folder", type=str, default='carla_lbc/run_results')
    parser.add_argument("--parent_folder", type=str, default='') # will be automatically created
    parser.add_argument("--mean_objectives_across_generations_path", type=str, default='') # will be automatically created
    parser.add_argument("--episode_max_time", type=int, default=60)
    parser.add_argument('--record_every_n_step', type=int, default=2000)
    parser.add_argument('--gpus', type=str, default='0,1')


    # algorithm related
    parser.add_argument("--n_gen", type=int, default=2)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--survival_multiplier", type=int, default=1)
    parser.add_argument("--n_offsprings", type=int, default=300)
    parser.add_argument("--has_run_num", type=int, default=1000)
    parser.add_argument('--sample_multiplier', type=int, default=200)
    parser.add_argument('--mating_max_iterations', type=int, default=200)
    parser.add_argument('--only_run_unique_cases', type=int, default=1)
    parser.add_argument('--consider_interested_bugs', type=int, default=1)

    parser.add_argument("--outer_iterations", type=int, default=3)
    parser.add_argument('--objective_weights', nargs='+', type=float, default=default_objective_weights)
    parser.add_argument('--default_objectives', nargs='+', type=float, default=default_objectives)
    parser.add_argument("--standardize_objective", type=int, default=0)
    parser.add_argument("--normalize_objective", type=int, default=0)
    parser.add_argument('--traj_dist_metric', type=str, default='nearest')



    parser.add_argument('--check_unique_coeff', nargs='+', type=float, default=default_check_unique_coeff)
    parser.add_argument('--use_single_objective', type=int, default=1)
    parser.add_argument('--rank_mode', type=str, default='none')
    parser.add_argument('--ranking_model', type=str, default='nn_pytorch')
    parser.add_argument('--initial_fit_th', type=int, default=100, help='minimum number of instances needed to train a DNN.')
    parser.add_argument('--min_bug_num_to_fit_dnn', type=int, default=10, help='minimum number of bug instances needed to train a DNN.')

    parser.add_argument('--pgd_eps', type=float, default=1.01)
    parser.add_argument('--adv_conf_th', type=float, default=-4)
    parser.add_argument('--attack_stop_conf', type=float, default=0.9)
    parser.add_argument('--use_single_nn', type=int, default=1)

    parser.add_argument('--warm_up_path', type=str, default=None)
    parser.add_argument('--warm_up_len', type=int, default=-1)
    parser.add_argument('--regression_nn_use_running_data', type=int, default=1)

    parser.add_argument('--sample_avoid_ego_position', type=int, default=0)


    parser.add_argument('--uncertainty', type=str, default='')
    parser.add_argument('--model_type', type=str, default='one_output')


    parser.add_argument('--termination_condition', type=str, default='generations')
    parser.add_argument('--max_running_time', type=int, default=3600*24)

    parser.add_argument('--emcmc', type=int, default=0)
    parser.add_argument('--use_unique_bugs', type=int, default=1)
    parser.add_argument('--finish_after_has_run', type=int, default=1)

    fuzzing_arguments = parser.parse_args(argv)

    os.environ['HAS_DISPLAY'] = fuzzing_arguments.has_display
    os.environ['CUDA_VISIBLE_DEVICES'] = fuzzing_arguments.gpus
    fuzzing_arguments.objective_weights = np.array(fuzzing_arguments.objective_weights)
    # ['BNN', 'one_output']
    # BALD and BatchBALD only support BNN
    if fuzzing_arguments.uncertainty.split('_')[0] in ['BALD', 'BatchBALD']:
        fuzzing_arguments.model_type = 'BNN'

    if 'un' in fuzzing_arguments.algorithm_name:
        fuzzing_arguments.use_unique_bugs = 1
    else:
        fuzzing_arguments.use_unique_bugs = 0

    if fuzzing_arguments.algorithm_name in ['nsga2-emcmc', 'nsga2-un-emcmc']:
        fuzzing_arguments.emcmc = 1
    else:
        fuzzing_arguments.emcmc = 0

    return fuzzing_arguments




# if fuzzing_arguments.simulator in ['carla', 'svl']:
#     sys.path.append('..')
#     carla_lbc_root = 'carla_lbc'
#     sys.path.append(carla_lbc_root)
#     sys.path.append(carla_lbc_root+'/leaderboard')
#     sys.path.append(carla_lbc_root+'/leaderboard/team_code')
#     sys.path.append(carla_lbc_root+'/scenario_runner')
#     sys.path.append(carla_lbc_root+'/carla_project')
#     sys.path.append(carla_lbc_root+'/carla_project/src')
#     sys.path.append(carla_lbc_root+'/carla_specific_utils')

#     if fuzzing_arguments.simulator in ['carla']:
#         carla_root = os.path.expanduser('~/Documents/self-driving-cars/carla_0994_no_rss')
#         sys.path.append(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')
#         sys.path.append(carla_root+'/PythonAPI/carla')
#         sys.path.append(carla_root+'/PythonAPI')
#         assert os.path.exists(carla_root+'/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')


class AutoFuzz(NSGA2):
    def __init__(self, dt=False, X=None, F=None, fuzzing_arguments=None, plain_sampling=None, local_mating=None, **kwargs):
        self.dt = dt
        self.X = X
        self.F = F
        self.plain_sampling = plain_sampling

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

        super().__init__(pop_size=self.pop_size, n_offsprings=self.n_offsprings, **kwargs)

        self.plain_initialization = Initialization(self.plain_sampling, individual=Individual(), repair=self.repair, eliminate_duplicates= NoDuplicateElimination())


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

        if self.algorithm_name == 'avfuzzer':
            cur_best_y = [None, 10000]
            if self.cur_gen >= 0:
                # local search
                if 0 <= self.local_gen <= 4:
                    with open('tmp_log.txt', 'a') as f_out:
                        f_out.write(str(self.cur_gen)+' local '+str(self.local_gen)+'\n')

                    cur_pop = self.pop[-self.pop_size:]
                    for p in cur_pop:
                        if p.F < self.local_best_y[1]:
                            self.local_best_y = [p, p.F]
                    if self.local_gen == 4:
                        self.local_gen = -1
                        if self.local_best_y[1] < self.global_best_y[1]:
                            self.global_best_y = self.local_best_y
                        if self.local_best_y[1] < self.best_y_gen[-1][1]:
                            self.best_y_gen[-1] = self.local_best_y
                        # if self.local_best_y[1] < self.restart_best_y[1]:
                        #     self.restart_best_y = self.local_best_y

                        tmp_best_ind = 0
                        tmp_best_y = [None, 10000]
                        for i, p in enumerate(self.pop_before_local):
                            if p.F < tmp_best_y[1]:
                                tmp_best_y = [p, p.F]
                                tmp_best_ind = i

                        self.pop_before_local[tmp_best_ind] = self.local_best_y[0]
                        self.tmp_off, _ = self.mating.do(self.problem, self.pop_before_local, self.n_offsprings, algorithm=self)

                        self.cur_gen += 1
                    else:
                        self.local_gen += 1

                        self.tmp_off, _ = self.local_mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

                # global search
                else:
                    cur_pop = self.pop[-self.pop_size:]
                    for p in cur_pop:
                        if p.F < cur_best_y[1]:
                            cur_best_y = [p, p.F]
                    if cur_best_y[1] < self.global_best_y[1]:
                        self.global_best_y = cur_best_y
                    if len(self.best_y_gen) == self.cur_gen:
                        self.best_y_gen.append(cur_best_y)
                    else:
                        if cur_best_y[1] < self.best_y_gen[-1][1]:
                            self.best_y_gen[-1] = cur_best_y

                    if self.cur_gen - self.restart_gen <= self.minLisGen:
                        if cur_best_y[1] < self.restart_best_y[1]:
                            self.restart_best_y = cur_best_y


                    with open('tmp_log.txt', 'a') as f_out:
                        f_out.write('self.global_best_y: '+ str(self.global_best_y[1])+', cur_best_y[1]: '+str(cur_best_y[1])+', self.restart_best_y[1]: '+str(self.restart_best_y[1])+'\n')

                    normal = True
                    # restart
                    if self.cur_gen - self.restart_gen > 4:
                        last_5_mean = np.mean([v for _, v in self.best_y_gen[-5:]])

                        with open('tmp_log.txt', 'a') as f_out:
                            f_out.write('last_5_mean: '+str(last_5_mean)+', cur_best_y[1]: '+str(cur_best_y[1])+'\n')
                        if cur_best_y[1] >= last_5_mean:
                            with open('tmp_log.txt', 'a') as f_out:
                                f_out.write(str(self.cur_gen)+' restart'+'\n')

                            tmp_off_candidates = self.plain_initialization.do(self.problem, 1000, algorithm=self)
                            tmp_off_candidates_X = np.stack([p.X for p in tmp_off_candidates])
                            chosen_inds = choose_farthest_offs(tmp_off_candidates_X, self.all_pop_run_X, self.pop_size)
                            self.tmp_off = tmp_off_candidates[chosen_inds]
                            self.restart_best_y = [None, 10000]
                            normal = False
                            self.cur_gen += 1
                            self.restart_gen = self.cur_gen


                    # enter local
                    if normal and self.cur_gen - self.restart_gen > self.minLisGen and cur_best_y[1] < self.restart_best_y[1]:
                            with open('tmp_log.txt', 'a') as f_out:
                                f_out.write(str(self.cur_gen)+'enter local'+'\n')
                            self.restart_best_y[1] = cur_best_y[1]
                            self.pop_before_local = copy.deepcopy(self.pop)
                            pop = Population(self.pop_size, individual=Individual())
                            pop.set("X", [self.global_best_y[0].X for _ in range(self.pop_size)])
                            pop.set("F", [self.global_best_y[1] for _ in range(self.pop_size)])
                            self.tmp_off = self.mutation.do(self.problem, pop)

                            self.local_best_y = [None, 10000]
                            self.local_gen = 0
                            normal = False
                            # not increasing cur_gen in this case
                    if normal:
                        with open('tmp_log.txt', 'a') as f_out:
                            f_out.write(str(self.cur_gen)+' normal'+'\n')
                        self.tmp_off, _ = self.mating.do(self.problem, self.pop, self.pop_size, algorithm=self)
                        self.cur_gen += 1
            else:
                # initialization
                self.tmp_off = self.plain_initialization.do(self.problem, self.n_offsprings, algorithm=self)
                self.cur_gen += 1

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(self.tmp_off) == 0 or (not self.problem.call_from_dt and self.problem.fuzzing_arguments.finish_after_has_run and self.problem.has_run >= self.problem.fuzzing_arguments.has_run_num):
            self.termination.force_termination = True
            print("Mating cannot generate new springs, terminate earlier.")
            print('self.tmp_off', len(self.tmp_off), self.tmp_off)
            return
        # if not the desired number of offspring could be created
        elif len(self.tmp_off) < self.n_offsprings:
            if self.verbose:
                print("WARNING: Mating could not produce the required number of (unique) offsprings!")


        # TODO
        # additional step to rank and select self.off after gathering initial population
        if (self.rank_mode == 'none'):
            print("self.rank_mode == 'none'")
            self.off = self.tmp_off[:self.pop_size]

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


        if self.algorithm_name in ['random', 'avfuzzer']:
            print("self.algorithm_name in ['random', 'avfuzzer']")
            self.pop = self.off


    def _initialize(self):
        if self.warm_up_path and ((self.dt and not self.problem.cumulative_info) or (not self.dt)):
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


class MyProblem(Problem):

    def __init__(self, fuzzing_arguments, 
                 sim_specific_arguments, 
                 fuzzing_content, 
                 run_simulation, 
                 dt_arguments):

        self.fuzzing_arguments = fuzzing_arguments
        self.sim_specific_arguments = sim_specific_arguments
        self.fuzzing_content = fuzzing_content
        self.run_simulation = run_simulation
        self.dt_arguments = dt_arguments


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


        self.call_from_dt = dt_arguments.call_from_dt
        self.dt = dt_arguments.dt
        self.estimator = dt_arguments.estimator
        self.critical_unique_leaves = dt_arguments.critical_unique_leaves
        self.cumulative_info = dt_arguments.cumulative_info
        cumulative_info = dt_arguments.cumulative_info

        if cumulative_info:
            self.counter = cumulative_info['counter']
            self.has_run = cumulative_info['has_run']
            self.start_time = cumulative_info['start_time']
            self.time_list = cumulative_info['time_list']
            self.bugs = cumulative_info['bugs']
            self.unique_bugs = cumulative_info['unique_bugs']
            self.interested_unique_bugs = cumulative_info['interested_unique_bugs']
            self.bugs_type_list = cumulative_info['bugs_type_list']
            self.bugs_inds_list = cumulative_info['bugs_inds_list']
            self.bugs_num_list = cumulative_info['bugs_num_list']
            self.unique_bugs_num_list = cumulative_info['unique_bugs_num_list']
            self.has_run_list = cumulative_info['has_run_list']
        else:
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


        self.labels = fuzzing_content.labels
        self.mask = fuzzing_content.mask
        self.parameters_min_bounds = fuzzing_content.parameters_min_bounds
        self.parameters_max_bounds = fuzzing_content.parameters_max_bounds
        self.parameters_distributions = fuzzing_content.parameters_distributions
        self.customized_constraints = fuzzing_content.customized_constraints
        self.customized_center_transforms = fuzzing_content.customized_center_transforms
        xl = [pair[1] for pair in self.parameters_min_bounds.items()]
        xu = [pair[1] for pair in self.parameters_max_bounds.items()]
        n_var = fuzzing_content.n_var

        self.p, self.c, self.th = self.check_unique_coeff
        self.launch_server = True
        self.objectives_list = []
        self.trajectory_vector_list = []
        self.x_list = []
        self.y_list = []
        self.F_list = []

        super().__init__(n_var=n_var, n_obj=4, n_constr=0, xl=xl, xu=xu)


    def _evaluate(self, X, out, *args, **kwargs):
        objective_weights = self.objective_weights
        customized_center_transforms = self.customized_center_transforms

        episode_max_time = self.episode_max_time

        default_objectives = self.fuzzing_arguments.default_objectives
        standardize_objective = self.fuzzing_arguments.standardize_objective
        normalize_objective = self.fuzzing_arguments.normalize_objective
        traj_dist_metric = self.fuzzing_arguments.traj_dist_metric


        all_final_generated_transforms_list = []

        # non-dask subprocess implementation
        # rng = np.random.default_rng(random_seeds[1])

        tmp_run_info_list = []
        x_sublist = []
        objectives_sublist_non_traj = []
        trajectory_vector_sublist = []

        for i in range(X.shape[0]):
            if self.counter == 0:
                launch_server = True
            else:
                launch_server = False
            cur_i = i
            total_i = self.counter

            port = self.ports[0]
            x = X[cur_i]

            manager = Manager()
            return_dict = manager.dict()
            try:
                p = Process(target=fun, args=(self, x, launch_server, self.counter, port, return_dict))
                p.start()
                p.join(240)
                if p.is_alive():
                    print("Function is hanging!")
                    p.terminate()
                    print("Kidding, just terminated!")
                if 'returned_data' in return_dict:
                    objectives, run_info, has_run = return_dict['returned_data']
                else:
                    raise
            except:
                traceback.print_exc()
                objectives, run_info, has_run = default_objectives, None, 0

            print('get job result for', total_i)
            if run_info and 'all_final_generated_transforms' in run_info:
                all_final_generated_transforms_list.append(run_info['all_final_generated_transforms'])

            self.has_run_list.append(has_run)
            self.has_run += has_run

            # record bug
            if run_info and run_info['is_bug']:
                self.bugs.append(X[cur_i].astype(float))
                self.bugs_inds_list.append(total_i)
                self.bugs_type_list.append(run_info['bug_type'])

                self.y_list.append(run_info['bug_type'])
            else:
                self.y_list.append(0)



            self.counter += 1
            tmp_run_info_list.append(run_info)
            x_sublist.append(x)
            objectives_sublist_non_traj.append(objectives)
            if run_info and 'trajectory_vector' in run_info:
                trajectory_vector_sublist.append(run_info['trajectory_vector'])
            else:
                trajectory_vector_sublist.append(None)


        job_results, self.x_list, self.objectives_list, self.trajectory_vector_list = get_job_results(tmp_run_info_list, x_sublist, objectives_sublist_non_traj, trajectory_vector_sublist, self.x_list, self.objectives_list, self.trajectory_vector_list, traj_dist_metric)
        # print('self.objectives_list', self.objectives_list)


        # hack:
        if run_info and 'all_final_generated_transforms' in run_info:
            with open('carla_lbc/tmp_folder/total.pickle', 'wb') as f_out:
                pickle.dump(all_final_generated_transforms_list, f_out)

        # record time elapsed and bug numbers
        time_elapsed = time.time() - self.start_time
        self.time_list.append(time_elapsed)




        current_F = get_F(job_results, self.objectives_list, objective_weights, self.use_single_objective, standardize=standardize_objective, normalize=normalize_objective)

        out["F"] = current_F
        self.F_list.append(current_F)
        print('\n'*3, 'self.F_list', len(self.F_list), self.F_list, '\n'*3)

        print('\n'*10, '+'*100)



        bugs_type_list_tmp = self.bugs_type_list
        bugs_tmp = self.bugs
        bugs_inds_list_tmp = self.bugs_inds_list

        self.unique_bugs, unique_bugs_inds_list, self.interested_unique_bugs, bugcounts = get_unique_bugs(self.x_list, self.objectives_list, self.mask, self.xl, self.xu, self.check_unique_coeff, objective_weights, return_mode='unique_inds_and_interested_and_bugcounts', consider_interested_bugs=1, bugs_type_list=bugs_type_list_tmp, bugs=bugs_tmp, bugs_inds_list=bugs_inds_list_tmp, trajectory_vector_list=self.trajectory_vector_list)


        time_elapsed = time.time() - self.start_time
        num_of_bugs = len(self.bugs)
        num_of_unique_bugs = len(self.unique_bugs)
        num_of_interested_unique_bugs = len(self.interested_unique_bugs)

        self.bugs_num_list.append(num_of_bugs)
        self.unique_bugs_num_list.append(num_of_unique_bugs)
        mean_objectives_this_generation = np.mean(np.array(self.objectives_list[-X.shape[0]:]), axis=0)

        with open(self.fuzzing_arguments.mean_objectives_across_generations_path, 'a') as f_out:

            info_dict = {
                'counter': self.counter,
                'has_run': self.has_run,
                'time_elapsed': time_elapsed,
                'num_of_bugs': num_of_bugs,
                'num_of_unique_bugs': num_of_unique_bugs,
                'num_of_interested_unique_bugs': num_of_interested_unique_bugs,
                'bugcounts and unique bug counts': bugcounts, 'mean_objectives_this_generation': mean_objectives_this_generation.tolist(),
                'current_F': current_F
            }

            f_out.write(str(info_dict))
            f_out.write(';'.join([str(ind) for ind in unique_bugs_inds_list])+' objective_weights : '+str(self.objective_weights)+'\n')
        print(info_dict)
        print('+'*100, '\n'*10)

