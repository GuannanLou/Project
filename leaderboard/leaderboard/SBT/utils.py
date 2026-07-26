import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import imageio
import json

import numpy as np

class LargeScenario():
    def __init__(self, path, vec_len=15):
        data = []
        with open(path+'scenario.csv','r') as f:
            for line in f:
                data.append([float(numb) for numb in line.split(',')])
        self.data = np.array(data)

        scenario_header = ["cloudiness",
                           "precipitation",
                           "precipitation_deposits",
                           "wind_intensity",
                           "sun_azimuth_angle",
                           "sun_altitude_angle",
                           "fog_density",
                           "wetness",
                           "fog_falloff",
                           "start_offset",
                           "end_offset"]
        self.weather = pd.DataFrame(self.data[:,:11],columns = scenario_header)
        self.weather_col = self.weather.columns.to_numpy().tolist()
        self.length = self.data.shape[0]
        self.weather_vec = self.data[:,:11]
        self.vehicle_vec = self.data[:,11:].reshape(self.length,4,vec_len)


def get_folder(dir, indexs):
    result = []
    dirs = os.listdir(dir)
    dirs.sort()
    dirs = dirs[3:-1]
    for i in indexs:
        result.append(dirs[i])
    return result

def get_max_gear(case):
    return pd.read_csv(case+'control.csv')['gear'].max()

def generat_gif(input_folder, output_file, extend=10, format='.png', sampling=10):
    file_names = os.listdir(input_folder)
    file_names = sorted([f for f in file_names if f.endswith(format)], key=lambda x: int(x.split('_')[-1].split('.')[0]))
    sampled_file_names = file_names[::sampling]
    file_names = sampled_file_names + [file_names[-1]]*extend

    with imageio.get_writer(output_file, mode='I') as writer:
        for file_name in file_names:
            image = imageio.imread(os.path.join(input_folder, file_name))
            writer.append_data(image)
    print(output_file)

def get_fitness(dir, direction=True):
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


    criterion = pd.read_csv(dir+'criterion.csv',names=criterion_header)
    fitness = pd.read_csv(dir+'fitness.csv',names=fitness_header)

    # if direction:
    #     fitness['DVE'] = fitness['DVE'].apply(lambda data: float(data[1:]))
    #     fitness['DVE-d'] = fitness['DVE-d'].apply(lambda data: float(data[:-1]))
    # else:
    #     fitness['DVE-d'] = np.zeros_like(fitness['DVE'])
    if direction:
        fitness['DVE']   = fitness['DVE'].apply(lambda data: float(data[1:]))
        fitness['DVE-d'] = fitness['DVE-d'].apply(lambda data: float(data))
        fitness['DVE-x'] = fitness['DVE-x'].apply(lambda data: float(data))
        fitness['DVE-y'] = fitness['DVE-y'].apply(lambda data: float(data[:-1]))
    else:
        fitness['DVE-d'] = np.zeros_like(fitness['DVE'])
        fitness['DVE-x'] = np.zeros_like(fitness['DVE'])
        fitness['DVE-y'] = np.zeros_like(fitness['DVE'])

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
    for i, case in enumerate(get_folder(dir, result[result['CollisionTest'] == 0].index.to_numpy())):
        if get_max_gear(dir+case+'/') == 0:
            index = result[result['CollisionTest'] == 0].index.to_numpy()[i]
            change.append(index)
            print('ERROR: Wrong Collision -', dir+case)

    for index in change:
        result.loc[index,'CollisionTest'] = 1
        result.loc[index,'RouteCompletionTest'] = 1
        # result['RouteCompletionTest'][index] = 1
        # result['CollisionTest'][index] = 1
    
    return result

def get_one_fitness(criterion_dir, fitness_dir, col, length=False, direction=True):
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
        fitness['DVE-x'] = np.zeros_like(fitness['DVE'])
        fitness['DVE-y'] = np.zeros_like(fitness['DVE'])
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

def get_scenario(dir):
    scenario_header = ["cloudiness",
                    "precipitation",
                    "precipitation_deposits",
                    "wind_intensity",
                    "sun_azimuth_angle",
                    "sun_altitude_angle",
                    "fog_density",
                    "wetness",
                    "fog_falloff",
                    "vehicle_infront", 
                    "vehicle_opposite", 
                    "vehicle_side",
                    "start_offset",
                    "end_offset"]
    
    return pd.read_csv(dir+'scenario.csv',names=scenario_header)
  


def get_config(dir):
    data = json.load(open(dir+'experiment_config.json'))
    return {'REGION':data['REGION'],
            'ROUTE_FILE':data['ROUTE_FILE']}

def region_to_level(region):
    mapping = {
        'baseline': 'baseline',
        0: 0,
        28: 1,
        21: 2,
        14: 3,
        7: 4
    }
    if region == 'baseline':
        return 'baseline'
    else:
        return mapping[int(region)]
    
def plot_route_compair(ori_route, real_route, section='Curve', width=False, title=''):
    plt.style.use('_mpl-gallery')

    # linewidth need to be changed accodring to the pix/m, but this figure is not find yet
    linewidth = (60,40) if width else (2,2)


    if section=='Curve':
        fig, ax = plt.subplots(figsize=(3, 5))

        ax.plot(ori_route['y'], ori_route['x'], linewidth=linewidth[0], label='Original Path')
        ax.plot(real_route['y'], real_route['x'], linewidth=linewidth[1], label='Real Path', color='orange')

        ax.set(xlim=(85, 145),  xticks=np.arange(90, 150, 10),
               ylim=(15, 115), yticks=np.arange(20, 120, 10))
        # plt.legend()
    elif section=='Straight':
        fig, ax = plt.subplots(figsize=(2, 5))

        ax.plot(ori_route['x'], ori_route['y'], linewidth=linewidth[0], label='Target Route')
        ax.plot(real_route['x'], real_route['y'], linewidth=linewidth[1], label='Real Route', color='orange')

        ax.set(xlim=(-88.5, -81.5),  xticks=np.arange(-88, -81, 1),
               ylim=(-115, 95), yticks=np.arange(-120, 91, 10))
        # plt.legend(loc=2)
    plt.title(title)
    plt.show()
