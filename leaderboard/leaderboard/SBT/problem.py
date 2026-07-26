import sys
import numpy as np
import joblib
import pandas as pd
import os

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize



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

def get_fitness(criterion_dir, fitness_dir, direction=True):
    folder = criterion_dir.replace('criterion.csv','')
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
        # result['RouteCompletionTest'][index] = 1
        # result['CollisionTest'][index] = 1

    return result

class CustomizedProblem(ElementwiseProblem):
    def __init__(self, fitness_file, cirtion_file, fitness_generator, config):
        super().__init__(n_var=14,
                         n_obj=3,
                         xl=np.zeros(14),
                         xu=np.ones(14))
        self.fitness_file = fitness_file
        self.cirtion_file = cirtion_file
        self.fitness_generator = fitness_generator
        self.config = config


    def _evaluate(self, x, out, *args, **kwargs):
        dict_cirtion_index = {"RouteCompletionTest": 0,   
            "RouteCompletionTest_figure": 1,
            "OutsideRouteLanesTest": 2, 
            "OutsideRouteLanesTest_figure": 3,
            "CollisionTest": 4,         
            "CollisionTest_figure": 5,
            "RunningRedLightTest": 6,   
            "RunningRedLightTest_figure": 7,
            "RunningStopTest": 8,       
            "RunningStopTest_figure": 9,
            "InRouteTest": 10, 
            "InRouteTest_figure": 11,          
            "AgentBlockedTest": 12,
            "AgentBlockedTest_figure": 13,      
            "Timeout": 14}
        
        # x[6] = 0

        self.fitness_generator(x, self.config)
        result = {
            'RouteCompletionTest':0,
            'OutsideRouteLanesTest':0,
            'CollisionTest':0,
            'RunningRedLightTest':0,
            'RunningStopTest':0,
            'InRouteTest':0,
            'AgentBlockedTest':0,
            'Timeout':0
        }
        with open(self.cirtion_file, 'r') as file:
            data = [float(item) for item in file.readlines()[-1].strip().split(',')]
            result['RouteCompletionTest']   = data[dict_cirtion_index["RouteCompletionTest_figure"]]/100
            result['OutsideRouteLanesTest'] = 1-data[dict_cirtion_index["OutsideRouteLanesTest_figure"]]/100
            result['CollisionTest']         = data[dict_cirtion_index["CollisionTest"]]
            result['RunningRedLightTest']   = 1-data[dict_cirtion_index["RunningRedLightTest"]]
            result['RunningStopTest']       = 1-data[dict_cirtion_index["RunningStopTest"]]
            result['InRouteTest']           = 1-data[dict_cirtion_index["InRouteTest"]]
            result['AgentBlockedTest']      = 1-data[dict_cirtion_index["AgentBlockedTest"]]
            result['Timeout']               = 1-data[dict_cirtion_index["Timeout"]]

        with open(self.fitness_file, 'r') as file:
            data = [float(item) for item in file.readlines()[-1].strip().split(',')] 
            result['CollisionTest'] = 0 if result['CollisionTest'] == 1 else min(data[1],2)/2
            
        out['F'] = [
            result['RouteCompletionTest'],
            result['OutsideRouteLanesTest'],
            result['CollisionTest']
        ]

class CollisionProblem(Problem):
    def __init__(self, fitness_file, cirtion_file, fitness_generator, config, n_var):
        super().__init__(n_var=n_var,
                         n_obj=1,
                         xl=np.zeros(n_var),
                         xu=np.ones(n_var))
        self.fitness_file = fitness_file
        self.cirtion_file = cirtion_file
        self.fitness_generator = fitness_generator
        self.config = config


    def _evaluate(self, x, out, *args, **kwargs):
        print('x:', x.shape, x)
        
        for scenario_vec in x:
            self.fitness_generator(scenario_vec, self.config)
        result = get_fitness(self.cirtion_file, self.fitness_file)
       
        # out['F'] = [
        #     result['CollisionTest']
        # ]
        # print(result['CollisionTest'], type(result['CollisionTest']))

        out['F'] = result['CollisionTest'].to_numpy()

        print("out['F']", out['F'], out['F'].dtype, out['F'].shape)
        print()
        print()


class SurrogateProblem(ElementwiseProblem):
    def __init__(self, config, surrogate_path):
        super().__init__(n_var=14,
                         n_obj=3,
                         xl=np.zeros(14),
                         xu=np.ones(14))
        self.config = config
        self.surrogate_path = surrogate_path
        

    def _evaluate(self, x, out, *args, **kwargs):
        # model_path = './tools/models/'
        model_path = './tools/models/regression-Kriging'
        surrogate_models = {"RouteCompletionTest"  : joblib.load(model_path+'-RouteCompletionTest.pkl'), 
                            "CollisionTest"        : joblib.load(model_path+'-CollisionTest.pkl'), 
                            "OutsideRouteLanesTest": joblib.load(model_path+'-OutsideRouteLanesTest.pkl'), 
                            "Timeout"              : joblib.load(model_path+'-Timeout.pkl')}
        # result = [
        #     1-surrogate_models["OutsideRouteLanesTest"].predict([x])[0],
        #     1-surrogate_models["CollisionTest"].predict([x])[0],
        #     1-surrogate_models["Timeout"].predict([x])[0]
        # ]
        result = np.array([
            surrogate_models["OutsideRouteLanesTest"].predict([x])[0],
            surrogate_models["CollisionTest"].predict([x])[0],
            surrogate_models["RouteCompletionTest"].predict([x])[0]
        ])
        result[result>1] = 1
        result[result<0] = 0
        # print(result)

        file = open(self.surrogate_path+'criterion.csv', 'a')
        file.write(','.join([str(item) for item in result])+'\n')
        file.close()

        file = open(self.surrogate_path+'scenario.csv', 'a')
        file.write(','.join([str(item) for item in x])+'\n')
        file.close()

        out['F'] = result

