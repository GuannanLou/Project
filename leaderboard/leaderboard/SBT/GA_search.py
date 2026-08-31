import os
import sys
import numpy as np
import pandas as pd
import traceback
import subprocess
import datetime
import random
import torch
import time

from pathlib import Path



# from TCP import data
# from leaderboard.leaderboard.SBT import problem

from .simulator_utils import run_carla, kill_carla, run_agent, kill_agent, kill_by_port, get_free_port
from .surrogate_train import train

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize



# GA = os.environ['GA']==True
# LOG = os.environ['LOG']==True
# REGION = int(os.environ.get('REGION', 7))
# SURROGATE = os.environ['SURROGATE']==True
save_surrogate_log = True
AGENT = 'TCP'
ROAD = 'Straight'
seed = int(time.time_ns() % (2**32 - 1))

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# SBT_DATA_ROOT = Path(
#     os.environ.get(
#         "SBT_DATA_ROOT",
#         PROJECT_ROOT.parent / "SBT-data"
#     )
# )

SBT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SBT_DIR.parents[2]
# PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

#SBT_DATA_ROOT = PROJECT_ROOT.parent / "SBT-data"
SBT_DATA_ROOT = DATA_DIR
UNIQUE_FAILURES_FILE = SBT_DIR / "unique_failures.npy"
TEMP_FILE = PROJECT_ROOT / "temp.npz"
OUTPUT_FILE = PROJECT_ROOT / "data" / "output.npz"

#CARLA_PATH = '~/Projects/CARLA_0.9.10/CarlaUE4.sh'
CARLA_PATH = str(Path.home() / "carla" / "CarlaUE4.sh")


class CollisionMultiProblem(Problem):
    def __init__(self, n_var, modules=[]):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         xl=np.zeros(n_var),
                         xu=np.ones(n_var))
        self.current_port = 2000
        self.generations = 0
        self.surrogate_model = None
        self.start = datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
        self.end = (datetime.datetime.now()+datetime.timedelta(days=3)).strftime("%Y-%m-%d|%H:%M:%S")
        self.fail_cases = []
        self.density_bins = np.array([0]*10)

        self.has_diversity = 'diversity' in modules
        self.has_similarity = 'similarity' in modules
        self.has_initpopulation = 'initpopulation' in modules
        self.has_givenpopulation = 'givenpopulation' in modules

        # mutually exclusive options
        if self.has_initpopulation and self.has_givenpopulation:
            print("Warning: both initpopulation and givenpopulation specified in CollisionMultiProblem; using givenpopulation.")
            self.has_initpopulation = False

        self.sampling = AverageSampling if self.has_initpopulation else FloatRandomSampling
        self.sampling = GivenSampling if self.has_givenpopulation else self.sampling

        print('modules:', modules)
        print('has_diversity:', self.has_diversity)
        print('has_similarity:', self.has_similarity)
        print('has_initpopulation:', self.has_initpopulation)

    def _evaluate(self, x, out, *args, **kwargs):
        density = x[:,9+2:].mean(axis=1) # 0 - high density, 1 - low density
        # density = np.array(self.fail_cases)[:,9+2:].mean(axis=1) if np.array(self.fail_cases).shape[0] > 0 else np.zeros(len(x))   # 0 - high density, 1 - low density

        # x[9+2:] = x[9+2:]*0.7+0.3  
        # x[:,9+2:] = x[:,9+2:]*0.7+0.3 # Scale the other vehicle vector to [0.3, 1.0]

        self.generations += 1
        self.current_port = get_free_port(self.current_port, 2000,3000)

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d|%H:%M:%S")

        print("Current generation:", self.generations)
        # if self.generations >= 7 and self.generations % 4 == 1:
        print('problem_evaluate [:10]', x[:10])
        print('problem_evaluate [10:]', x[10:])
        
        # similarity = self._similarity(cosine_similarity, np.array(self.fail_cases), np.array(x))
        similarity = self._similarity(elementwise_similarity, np.array(self.fail_cases), np.array(x))
        diversity = self._diversity(density)

        print("Fitness similarity:", similarity)
        print("Fitness diversity:", diversity)

        if self.generations % 4 == 1 or self.generations <= 7:
        # if self.generations % 4 == 1 or self.generations <= 1:
        # if True:
            print("Train Surrogate Model")
            fitness = []
            while len(fitness) != len(x):
                fitness = execute(x[len(fitness):], self.current_port, formatted_datetime, agent=AGENT, road=ROAD)
                # print(fitness.shape, x.shape)
                # print(fitness)
            self.surrogate_model = train(root  = str(SBT_DATA_ROOT / AGENT), 
                                         start = self.start, 
                                         end   = self.end)


            self.fail_cases += x[fitness==0].tolist()
        else:
            print("Train Surrogate Model")
            self.surrogate_model.eval()
            with torch.no_grad():
                fitness = np.array(
                    self.surrogate_model(
                        torch.tensor(np.array(x).astype(np.float32))
                    )
                ).reshape(-1)
            print("Surrogate model fitness:", fitness)
        
        print("Fitness original:", fitness)

        # if self.fail_cases != []:
        print('-_-_-_-_-')
        print(np.array(self.fail_cases).shape)
        print(np.array(x).shape)
        print('_-_-_-_-_')


        r_similarity = 0.2 if self.has_similarity else 0.0
        r_diversity  = 0.2 if self.has_diversity else 0.0
        r_fitness   = 1-r_similarity-r_diversity

        out['F'] = np.array([r_fitness*np.array(fitness), 
                             r_similarity*np.array(similarity), 
                             r_diversity*np.array(diversity)]).T
        print("Fitness weighted:", out['F'])

        # self.density_bins += np.histogram(density, bins=10, range=(0,1))[0]
        self.density_bins += np.histogram(x[fitness==0], bins=10, range=(0,1))[0]
        print("Density bins:", self.density_bins)
        print(x[fitness==0])

    def _diversity(self, density):
        bin_index = (density//0.1).astype(int)
        visits = self.density_bins[bin_index].copy()
        visits[visits==0] = 1
        score = 1-1/visits**0.5
        return score
        # print(new_density.round(2))
        # print(score.round(2))

    def _similarity(self, simi_func, fail, x):
        print('fail', fail.shape)
        print('x', x.shape)
        if fail.shape[0] == 0:
            return np.array([0]*len(x))
        else:
            # print(2)
            return simi_func(np.array(fail), np.array(x)).max(axis=0)
        # return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))




class CollisionProblem(Problem):
    def __init__(self, n_var, modules=[]):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         xl=np.zeros(n_var),
                         xu=np.ones(n_var))
        self.current_port = 2000
        self.generations = 0
        self.surrogate_model = None
        self.start = datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
        self.end = (datetime.datetime.now()+datetime.timedelta(days=3)).strftime("%Y-%m-%d|%H:%M:%S")
        self.explor_cases = []
        self.fail_cases = []
        # self.unique_fail_cases = []
        self.fail_collision_features = []
        self.density_bins = np.array([0]*10)

        self.has_diversity = 'diversity' in modules
        self.has_similarity = 'similarity' in modules
        self.has_local_similarity = 'local_similarity' in modules
        self.has_collision_similarity = 'collision_similarity' in modules
        self.has_initpopulation = 'initpopulation' in modules
        self.has_givenpopulation = 'givenpopulation' in modules

        # enforce mutual exclusivity rules
        if self.has_initpopulation and self.has_givenpopulation:
            # givenpopulation takes precedence
            print("Warning: both initpopulation and givenpopulation enabled; using givenpopulation and disabling initpopulation")
            self.has_initpopulation = False
        if self.has_similarity and self.has_local_similarity:
            # local similarity preferred
            print("Warning: both similarity and local_similarity enabled; using local_similarity and disabling similarity")
            self.has_similarity = False

        self.sampling = AverageSampling if self.has_initpopulation else FloatRandomSampling
        self.sampling = GivenSampling if self.has_givenpopulation else self.sampling

        # similarity function selection
        # collision similarity has highest priority when present
        if self.has_collision_similarity:
            self.collision_similarity = collision_feature_similarity

        if self.has_local_similarity:
            self.similarity = local_elementwise_similarity
        elif self.has_similarity:
            self.similarity = elementwise_similarity
        else:
            # default no similarity: returns zeros
            self.similarity = lambda fail, x: np.zeros(len(x))

        print('modules:', modules)
        print('has_diversity:', self.has_diversity)
        print('has_similarity:', self.has_similarity)
        print('has_local_similarity:', self.has_local_similarity)
        print('has_collision_similarity:', self.has_collision_similarity)
        print('has_initpopulation:', self.has_initpopulation)
        print('has_givenpopulation:', self.has_givenpopulation)

    def _evaluate(self, x, out, *args, **kwargs):
        density = x[:,9+2:].mean(axis=1) # 0 - high density, 1 - low density
        # density = np.array(self.fail_cases)[:,9+2:].mean(axis=1) if np.array(self.fail_cases).shape[0] > 0 else np.zeros(len(x))   # 0 - high density, 1 - low density

        # x[9+2:] = x[9+2:]*0.7+0.3  
        # x[:,9+2:] = x[:,9+2:]*0.7+0.3 # Scale the other vehicle vector to [0.3, 1.0]

        self.generations += 1
        self.current_port = get_free_port(self.current_port, 2000,3000)

        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d|%H:%M:%S")

        print("Current generation:", self.generations)
        # if self.generations >= 7 and self.generations % 4 == 1:
        print('problem_evaluate [:10]', x[:10])
        print('problem_evaluate [10:]', x[10:])
        
        simulation = True if self.generations % 4 == 1 or self.generations <= 7 else False

        if simulation:
            print("Train Surrogate Model")
            fitness = []
            while len(fitness) != len(x):
                fitness = execute(x[len(fitness):], self.current_port, formatted_datetime, agent=AGENT, road=ROAD)

            # self.end = (datetime.datetime.now()+datetime.timedelta(days=3)).strftime("%Y-%m-%d|%H:%M:%S")

            self.surrogate_model = train(root  = str(SBT_DATA_ROOT / AGENT), 
                                         start = self.start, 
                                         end   = self.end)

            
            # collision_details = get_collision_details(fitness_file)
            # print("Collision Details:", collision_details)
        
        else:
            print("Train Surrogate Model")
            self.surrogate_model.eval()
            with torch.no_grad():
                fitness = np.array(
                    self.surrogate_model(
                        torch.tensor(np.array(x).astype(np.float32))
                    )
                ).reshape(-1)
            print("Surrogate model fitness:", fitness)
        print("Fitness original:", fitness)

        if simulation and self.has_collision_similarity:
            fitness_file = (
                SBT_DATA_ROOT
                / AGENT
                / formatted_datetime
                / "fitness.csv"
            )
            collision_details = get_collision_details(str(fitness_file))
            # collision_details  = get_collision_details('/home/guannan/Projects/SBT-data/{}/{}/fitness.csv'.format(AGENT, formatted_datetime))
            collision_features = collision_details['end_points'][:, :2]
            collision_features = np.concatenate((collision_features, collision_details['speeds'][:, np.newaxis], collision_details['collision_direction'][:, np.newaxis]), axis=1)
            collision_features = self.collision_standardize(collision_features)

            self.collision_similarity = self._similarity(collision_feature_similarity, 
                                    np.array(self.fail_collision_features),
                                    np.array(collision_features))
        else:
            collision_features = None
            self.collision_similarity = 0
            
        print('Collision features:', collision_features)

        # similarity = self._similarity(cosine_similarity, np.array(self.fail_cases), np.array(x))
        # similarity = self._similarity(elementwise_similarity, 
        #                               np.array(self.unique_fail_cases), 
        #                               np.array(x))
        self.local_similarity = self._similarity(
            local_elementwise_similarity,
            np.array(self.explor_cases),
            np.array(x)
        )
        
        diversity = self._diversity(density)
        
        r = 0.3
        print("Collision similarity:", self.collision_similarity)
        print("Fitness similarity:", self.local_similarity)
        print("Fitness diversity:", diversity)
        
        print(collision_features)
        if simulation:
            self.fail_cases += x[fitness==0].tolist()
            self.explor_cases += x.tolist()
        
            if self.has_collision_similarity:
                self.fail_collision_features += collision_features[fitness==0].tolist()
        
        
        print('-_-_-_-_-')
        print('Fail Case', np.array(self.fail_cases).shape)
        print('X Case', np.array(x).shape)
        print('_-_-_-_-_')
        
        
        weight_collision = 0
        weight_scenario = 0
        
        if self.has_similarity and self.has_collision_similarity:
            weight_collision = 0.15
            weight_scenario = 0.15
        
        elif self.has_similarity and not self.has_collision_similarity:
            weight_scenario = 0.3
        
        elif not self.has_similarity and self.has_collision_similarity:
            weight_collision = 0.3
        
        weight_critical = 1 if weight_collision + weight_scenario == 0 else 0.7
        
        out['F'] = (
            weight_critical * np.array(fitness)
            + weight_collision * np.array(self.collision_similarity)
            + weight_scenario * np.array(self.local_similarity)
        )
        
        print("Fitness weighted:", out['F'])
        # self.density_bins += np.histogram(density, bins=10, range=(0,1))[0]
        self.density_bins += np.histogram(x[fitness==0], bins=10, range=(0,1))[0]
        print("Density bins:", self.density_bins)
        print(x[fitness==0])

        print()

    def collision_standardize(self, features):
        features[:, 0] = np.clip(features[:, 0], 25, 115)
        features[:, 1] = np.clip(features[:, 1], 55, 145)
        features[:, 2] = np.clip(features[:, 2], 0, 6)
        features[:, 3] = np.clip(features[:, 3], -180, 270)
        # 标准化位置和速度
        features[:, 0] = (features[:, 0]-25)  / 90  # 假设位置在25-115范围内
        features[:, 1] = (features[:, 1]-55)  / 90  # 假设位置在55-145范围内
        features[:, 2] = (features[:, 2]-0)   / 6   # 假设速度在0-6 m/s范围内
        features[:, 3] = (features[:, 3]+180) / 450 # 将碰撞方向标准化到-180-270范围内
        return features


    def _diversity(self, density):
        bin_index = (density//0.1).astype(int)
        visits = self.density_bins[bin_index].copy()
        visits[visits==0] = 1
        score = 1-1/visits**0.5
        return score


    def _similarity(self, simi_func, fail, x):
        # print(fail.shape,fail.shape[0], fail.shape[0]==0)
        if fail.shape[0] == 0:
            return np.array([0]*len(x))
        else:
            # print(2)
            return simi_func(np.array(fail), np.array(x)).max(axis=0)
        # return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosine_similarity(x, y):
    # x: (n, d), y: (m, d)
    # 先归一化
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    # 相似度矩阵 (n, m)
    sim_matrix = x_norm @ y_norm.T  
    return sim_matrix

def elementwise_similarity(x,y,threshold=0.1):
    similarity = np.zeros((x.shape[0], y.shape[0]))
    for i, s1 in enumerate(x):
        for j, s2 in enumerate(y):
            similarity[i, j] = ((s2-s1)**2 < threshold**2).mean()
    return similarity

def local_elementwise_similarity(x,y,threshold=0.1, a=5):
    mask = np.array([1]*11+([1]*a+[0]*(15-a))*4)
    mask = mask/mask.sum()*71

    similarity = np.zeros((x.shape[0], y.shape[0]))
    for i, s1 in enumerate(x):
        for j, s2 in enumerate(y):
            similarity[i, j] = (((s2-s1)**2 < threshold**2)*mask).mean()
    return similarity

def collision_feature_similarity(x,y,threshold=0.1):
    distance = np.zeros((x.shape[0], y.shape[0]))
    for i, s1 in enumerate(x):
        for j, s2 in enumerate(y):
            distance[i, j] = (((s2-s1)**2).sum())**0.5    
    
    distance /= 2

    similarity = 1/(distance+1)*2-1
    return similarity


class AverageSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # print('SEED: '+ str(seed+int(time.time())))
        # torch.manual_seed(seed+int(time.time()))
        # np.random.seed(seed+int(time.time()))
        # random.seed(seed+int(time.time()))

        print('SEED: '+ str(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        X = np.zeros((n_samples, problem.n_var))
        target_means = np.linspace(0.1, 1.0, num=n_samples)
        for i in range(n_samples):
            X[i, :11] = np.random.uniform(0, 1, 11)  # 前11列一致
            random_60 = self.sample_with_fixed_mean(60, target_means[i], alpha=1) 
            # random_60 = np.random.uniform(20, 21, 60)
            # random_60 = random_60 * target_means[i] / random_60.mean()
            X[i, 11:] = np.clip(random_60, 0, 1)
        return X  # 返回自定义种群

    def sample_with_fixed_mean(self, n, target_mean, alpha=1, max_iter=100):
        if not (0.0 <= target_mean <= 1.0):
            raise ValueError("target_mean must be in [0,1]")

        target_sum = n * target_mean

        y = np.random.beta(alpha, alpha, size=n)

        def f(tau):
            return np.clip(y - tau, 0.001, 1.0).sum() - target_sum

        lo = y.min() - 1.0
        hi = y.max()

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid

        x = np.clip(y - (lo + hi) / 2.0, 0.001, 1.0)

        diff = target_sum - x.sum()
        if abs(diff) > 1e-10:
            free_idx = np.where((x > 0) & (x < 1))[0]
            if len(free_idx) > 0:
                x[free_idx[0]] += diff
                x = np.clip(x, 0.001, 1.0)

        return x

class GivenSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        # X = np.load('/home/guannan/Projects/TCP-Interfuser/leaderboard/leaderboard/SBT/unique_failures.npy', allow_pickle=True)
        X = np.load(str(UNIQUE_FAILURES_FILE), allow_pickle=True)
        return X[:n_samples]  # 返回自定义种群


class GradientMutation(Mutation):

    def __init__(self, e, threshold, step, prob=1.0, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.e = e
        self.threshold = threshold
        self.step = step
        

    def _do(self, problem, X, params=None, **kwargs):
        model = problem.surrogate_model

        # X = X.astype(float)
        # X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)

        # # print("X dtype:", X_tensor.dtype)
        # # print("X_tensor device:", X_tensor.device)
        # # print("Model weight dtype:", next(model.parameters()).dtype)
        # # print("Model device:", next(model.parameters()).device)

        # y_hat = model(X_tensor)
        # X_tensor.retain_grad()
        # y_hat.backward(torch.ones_like(y_hat))
        # gard = X_tensor.grad

        # X_fuzzed = X_tensor.detach().clone()
        # y_fuzzed = y_hat.clone()

        # for i in range(self.step):
        #     mask = y_fuzzed.squeeze() > self.threshold
        #     X_fuzzed[mask] = X_fuzzed[mask] - self.e*gard.sign()[mask]
        #     X_fuzzed[X_fuzzed <= 0 + 1e-3] = 0 + 1e-3 
        #     X_fuzzed[X_fuzzed >= 1 - 1e-3] = 1 - 1e-3

        # return X_fuzzed.numpy()


        # step, threshold, e = 20, 0.05, 0.01

        # X = scenarios[-1].data.astype(np.float32)
        # X = X.astype(float)
        # X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)

        X = X.astype(float)
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        X_fuzzed = X_tensor.detach().clone()

        for _ in range(self.step):
            X_fuzzed.requires_grad = True
            y_hat = model(X_fuzzed)
            X_fuzzed.retain_grad()
            y_hat.backward(torch.ones_like(y_hat))
            gard = X_fuzzed.grad

            y_fuzzed = y_hat.clone()
            y_fuzzed = y_fuzzed.squeeze().detach().numpy()

            # print(y_fuzzed.round(3))
            # print(y_fuzzed.min().round(3), y_fuzzed.max().round(3), y_fuzzed.mean().round(3))

            X_fuzzed.requires_grad = False
            mask = y_fuzzed > self.threshold
            X_fuzzed[mask] = X_fuzzed[mask] - self.e*gard.sign()[mask]
            X_fuzzed[X_fuzzed <= 0 + 1e-3] = 0 + 1e-3 
            X_fuzzed[X_fuzzed >= 1 - 1e-3] = 1 - 1e-3
        
        return X_fuzzed.numpy()


def execute(scenario_vector, current_port, formatted_datetime, agent='InterFuser', road='Curve'):
    # return np.array([0.1]*len(scenario_vector))

    print()
    print('Save Scenario Vector')
    # print('Scenario Vector:', scenario_vector.shape)
    # vector_path = '/home/guannan/Projects/TCP-Interfuser/temp.npz'
    vector_path = str(TEMP_FILE)
    print(vector_path)
    np.savez(vector_path, scenario_vector)

    carla_path = CARLA_PATH

    run_carla(carla_path=carla_path, rander=False, port=current_port)
    run_agent(agent, road, vector_path, formatted_datetime, current_port)

    # kill_carla()
    # kill_agent()
    kill_by_port(current_port)

    # data_root = '/home/guannan/Projects/SBT-data/{}/{}/'.format(agent,formatted_datetime)
    # fitness_file = data_root+'fitness.csv'
    # cirtion_file = data_root+'criterion.csv'

    data_root = SBT_DATA_ROOT / agent / formatted_datetime
    fitness_file = str(data_root / "fitness.csv")
    criterion_file = str(data_root / "criterion.csv")

    
    fitness = get_fitness(criterion_file, fitness_file, col='CollisionTest')
    # out['F'] = get_fitness(cirtion_file, fitness_file, col='CollisionTest', length=len(x))
    
    # collision_details = get_collision_details(fitness_file)
    # print("Collision Details:", collision_details)

    # print("fitness", fitness.shape, fitness.dtype, fitness)
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


def get_collision_details(fitness_file):
    root = fitness_file.replace('fitness.csv','')
    folders = os.listdir(root)
    folders.sort()
    folders = [folder for folder in folders if 'routes_' in folder ]
    start_points = []
    end_points = []
    speeds = []
    for folder in folders:
        real_route  = pd.read_csv(root+folder+'/real_route.csv').to_numpy()
        speeds.append(np.load(root+folder+'/v.npy')[-1])
        start_points.append(real_route[0]) 
        end_points.append(real_route[-1])
        
    # fitness_header = ["DOL","DVE",'DVE-d',"DPD","DSM","DFD"]
    fitness_header = ["DOL","DVE",'DVE-d','DVE-x','DVE-y',"DPD","DSM","DFD"]

    fitness = pd.read_csv(fitness_file,names=fitness_header)
    # fitness['DVE-d'] = fitness['DVE-d'].apply(lambda data: float(data[:-1]))
    fitness['DVE-d'] = fitness['DVE-d'].apply(lambda data: float(data))
    fitness['DVE-x'] = fitness['DVE-x'].apply(lambda data: float(data))
    fitness['DVE-y'] = fitness['DVE-y'].apply(lambda data: float(data[:-1]))
    
    collision_direction = fitness['DVE-d']
    collision_position  = list(zip(fitness['DVE-x'], fitness['DVE-y']))
    # np.array(list(zip(data['c'], data['d'])))


    collision_details = {
        'start_points': np.array(start_points),
        'end_points': np.array(collision_position),
        'speeds': np.array(speeds),
        'collision_direction': np.array(collision_direction)
    }
    return collision_details
    

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
        # fitness_header = ["DOL","DVE",'DVE-d',"DPD","DSM","DFD"]
        fitness_header = ["DOL","DVE",'DVE-d','DVE-x','DVE-y',"DPD","DSM","DFD"]


    if length:
        criterion = pd.read_csv(criterion_dir,names=criterion_header, skiprows=range(total_lines-length))
        fitness = pd.read_csv(fitness_dir,names=fitness_header, skiprows=range(total_lines-length))
    else:
        criterion = pd.read_csv(criterion_dir,names=criterion_header)
        fitness = pd.read_csv(fitness_dir,names=fitness_header)

    if direction:
        fitness['DVE']   = fitness['DVE'].apply(lambda data: float(data[1:]))
        fitness['DVE-d'] = fitness['DVE-d'].apply(lambda data: float(data))
        fitness['DVE-x'] = fitness['DVE-x'].apply(lambda data: float(data))
        fitness['DVE-y'] = fitness['DVE-y'].apply(lambda data: float(data[:-1]))
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
        # result['RouteCompletionTest'][index] = 1
        # result['CollisionTest'][index] = 1

    return result[col].to_numpy()

def NSGA2_search_collision(pop_size = 20, n_offsprings = 1, generations = 50, modules=None):
    print("begin NSGA2_search_collision")

    problem = CollisionMultiProblem(n_var=11+15*4, modules=modules)

    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=problem.sampling(),
        crossover=SBX(prob=0.8, eta=5),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", generations)

    res = minimize(problem,
        algorithm,
        termination,
        seed=seed,
        save_history=False,
        verbose=True)

    X = res.X
    F = res.F
    
    print('X:', X.shape, X)
    print('F:', F.shape, F)
    
    # np.savez('./data/output.npz', X, F)
    np.savez(OUTPUT_FILE, X, F)

    if save_surrogate_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__


def GA_search_collision(pop_size = 20, n_offsprings = 1, generations = 50, modules=None):
    print("begin GA_search_collision")

    problem = CollisionProblem(n_var=11+15*4, modules=modules)

    algorithm = GA(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=problem.sampling(),
        crossover=SBX(prob=0.8, eta=5),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", generations)

    res = minimize(problem,
        algorithm,
        termination,
        seed=seed,
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

def GBGA_search_collision(pop_size = 20, n_offsprings = 1, generations = 50, modules=None):
    print("begin GBGA_search_collision")

    problem = CollisionProblem(n_var=11+15*4, modules=modules)

    algorithm = GA(
        pop_size=pop_size,
        n_offsprings=n_offsprings,
        sampling=problem.sampling(),
        crossover=SBX(prob=0.8, eta=5),
        mutation=GradientMutation(e=0.01, threshold=0.05, step=20, prob=1),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", generations)

    res = minimize(problem,
        algorithm,
        termination,
        seed=seed,
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

def smartrandom_search_collision(pop_size = 20, n_offsprings = 10, generations = 30, modules=None):
    print("begin smartrandom_search_collision")

    problem = CollisionProblem(n_var=11+15*4, modules=modules)

    problem.has_initpopulation = True
    # sampling = problem.sampling()
    sampling = AverageSampling()

    experiment_seed = int(time.time_ns() % (2**32 - 1))
    rng = np.random.default_rng(experiment_seed)
    print(sampling)
        
    for i in range(generations):

        global seed
        seed = int(rng.integers(0, 2**32 - 1))
        
        print("Generation:", i+1)

        # Generate random solutions
        if i == 0:
            X = sampling._do(problem=problem, n_samples=pop_size)
        else:
            X = sampling._do(problem=problem, n_samples=n_offsprings)

        # Evaluate the solutions
        out={'F': None}
        problem._evaluate(X, out=out)

        # Get the fitness values
        F = np.array([out['F']])

        print('X:', X.shape, X)
        print('F:', F.shape, F)


    np.savez('./data/output.npz', X, F)

    if save_surrogate_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

def random_search_collision(pop_size = 20, n_offsprings = 10, generations = 30, modules=None):
    print("begin random_search_collision")

    problem = CollisionProblem(n_var=11+15*4, modules=modules)
        # pop_size     = 2 
        # n_offsprings = 2
        # generations  = 5
    problem.has_initpopulation = False
    sampling = problem.sampling()
    print(sampling)
        
    for i in range(generations):
        print("Generation:", i+1)

        # Generate random solutions
        if i == 0:
            X = sampling._do(problem=problem, n_samples=pop_size)
        else:
            X = sampling._do(problem=problem, n_samples=n_offsprings)

        # Evaluate the solutions
        out={'F': None}
        problem._evaluate(X, out=out)

        # Get the fitness values
        F = np.array([out['F']])

        print('X:', X.shape, X)
        print('F:', F.shape, F)


    np.savez('./data/output.npz', X, F)

    if save_surrogate_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

def given_search_collision(pop_size = 20, n_offsprings = 10, generations = 10, modules=None):
    print("begin given_search_collision")
    # np.random.seed(3)

    problem = CollisionProblem(n_var=11+15*4)

    for _ in range(3):
        # data = np.load('/home/guannan/Projects/TCP-Interfuser/leaderboard/leaderboard/SBT/unique_failures.npy', allow_pickle=True)
        data = np.load(str(UNIQUE_FAILURES_FILE), allow_pickle=True)

        print(data.shape)  # 输出：(30, 71)
        print(data[:, :11])  # 所有行前11列一样
        print(np.mean(data[:, 11:], axis=1))  # 每行后60列的均值约为0.6

        for i in range(data.shape[0]//10+1):
            print(i*10, (i+1)*10)

            X = data[i*10:(i+1)*10]
            # Evaluate the solutions
            out={'F': None}
            problem._evaluate(X, out=out)

            # Get the fitness values
            F = np.array([out['F']])

            print('X:', X.shape, X)
            print('F:', F.shape, F)

    np.savez('./data/output.npz', X, F)

    if save_surrogate_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__




def search_based_testing(setting='random', agent='TCP', line='Straight', modules=None):
    search_algorithms = {
        'random': random_search_collision,
        'smartrandom': smartrandom_search_collision,
        'GA': GA_search_collision,
        'NSGA2': NSGA2_search_collision,
        'GBGA': GBGA_search_collision,
        'given_search_collision': given_search_collision
    }
    if setting in search_algorithms.keys() and agent in ['TCP', 'InterFuser'] and line in ['Curve', 'Straight']:
        global AGENT 
        global ROAD

        AGENT = agent
        ROAD = line
        
        pop_size     = 20 
        n_offsprings = 10
        generations  = 54
        # # generations  = 30

        # pop_size     = 2 
        # n_offsprings = 2
        # generations  = 3
        if setting == 'given_search_collision':
            generations  = 10
        print("pop_size:", pop_size, "n_offsprings:", n_offsprings, "generations:", generations)
        try:
            search_algorithms[setting](
                pop_size     = pop_size, 
                n_offsprings = n_offsprings, 
                generations  = generations,
                modules      = modules
            )
        except Exception as e:
            traceback.print_exc()
    else:
        print('Check Settings')
