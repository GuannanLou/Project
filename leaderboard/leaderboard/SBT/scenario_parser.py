import carla
import datetime
import numpy as np

def ego_vehicle_parser(trajectory, ego_vehicle_vec, current_map, offset_range = 50):
    result = []
    for i, location in enumerate(trajectory):

        waypoint = current_map.get_waypoint(location)
        offset = ego_vehicle_vec[i]
        start_direction = waypoint.transform.rotation.yaw % 360

        if start_direction > 315 or start_direction < 45:
            new_location = carla.Location(x = location.x + offset*offset_range - offset_range/2,
                                          y = location.y,
                                          z = location.z)
        elif start_direction > 225:
            new_location = carla.Location(x = location.x,
                                          y = location.y - offset*offset_range + offset_range/2,
                                          z = location.z)
        elif start_direction > 135:
            new_location = carla.Location(x = location.x - offset*offset_range + offset_range/2,
                                          y = location.y,
                                          z = location.z)
        elif start_direction > 45: # Current road is in this direction. It is right.
            new_location = carla.Location(x = location.x,
                                          y = location.y + offset*offset_range - offset_range/2,
                                          z = location.z)
        result.append(new_location)

    return result

def other_vehicle_parser(other_vehicle_vec):
    result = [] 
    for flag in other_vehicle_vec:
        if flag >= 0.5:
            result.append(True)
        else:
            result.append(False)
        
    return result

def lane_parser(distance_vec, vec_len):
    # print(len(distance_vec))
    return distance_vec.reshape((int(len(distance_vec)/vec_len), vec_len))

def weather_parser(weather_vec):
    '''
    Converts a 9-length array of 0-1 numbers to a CARLA weather parameter object.

    Args:
        weather_vec (list): a 9-length array of 0-1 numbers
    
    Returns:
        carla.WeatherParameters
    '''
    c, p, pd, wi, sz, sl, fd, w, ff = weather_vec 
    return carla.WeatherParameters(
        cloudiness              = 0 if c  < 0.5 else (c -0.5)*2*100, 
        precipitation           = 0 if p  < 0.5 else (p -0.5)*2*100, 
        precipitation_deposits  = 0 if pd < 0.5 else (pd-0.5)*2*100, 
        wind_intensity          = 0 if wi < 0.5 else (wi-0.5)*2*100, 
        wetness                 = 0 if w  < 0.5 else (w -0.5)*2*100, 
        sun_azimuth_angle       = 0 if sz < 0.5 else (sz-0.5)*2*360, 
        sun_altitude_angle      = 0 if sl < 0.5 else (sl-0.5)*2*180-90, 
        fog_density             = 0 if fd < 0.5 else (fd-0.5)*2*100, 
        fog_falloff             = 0 if ff < 0.5 else (ff-0.5)*2*5
        # sun_azimuth_angle       = 0 , 
        # sun_altitude_angle      = 75,  
        # fog_density             = 0 , 
        # fog_falloff             = 0 
        # fog_density             = 80, 
    )

def get_next(waypoint, distance_percentage, init_distance=5.5, max_distance=45):
    distance = distance_percentage*max_distance + init_distance
    next_waypoint = waypoint.next(distance)[0]
    return next_waypoint if next_waypoint.road_id == waypoint.road_id else None

def get_previous(waypoint, distance_percentage, init_distance=5.5, max_distance=45):
    distance = distance_percentage*max_distance + init_distance
    previous_waypoint = waypoint.previous(distance)[0]
    return previous_waypoint if previous_waypoint.road_id == waypoint.road_id else None

def get_lane_start(waypoint):
    road_id = waypoint.road_id
    last = waypoint
    while True:
        previous = last.previous(1)[0]
        # print(previous.road_id, previous.transform.location)
        if not previous:
            break
        if previous.road_id != road_id or last.transform.location.distance(previous.transform.location) > 2:
            break
        last = previous
    return last 

def fill_junction(road_waypoint, road_vector):
    road_waypoints = []

    end_junc_waypoint = road_waypoint
    while not end_junc_waypoint.is_junction:
        end_junc_waypoint = end_junc_waypoint.next(1)[0]

    road_waypoints += _get_road_by_junction(end_junc_waypoint.get_junction(), road_waypoint.road_id)

    start_junc_waypoint = road_waypoint
    while not start_junc_waypoint.is_junction:
        start_junc_waypoint = start_junc_waypoint.previous(1)[0]
    
    road_waypoints += _get_road_by_junction(start_junc_waypoint.get_junction(), road_waypoint.road_id)

    # return road_waypoints
    new_vehicles, region = _fill_road(road_waypoint, road_vector)

    print('Avg region:',region)

    for road_waypoint in road_waypoints:
        new_vehicles += _fill_road_region(road_waypoint, region)
    
    print('# of added vehicles:',len(new_vehicles))
    return new_vehicles

def _get_road_by_junction(junction, filted_road):
    roads = [filted_road]
    road_waypoints = []

    lane_in_junc =  junction.get_waypoints(carla.LaneType.Driving)
    for waypoint,_ in lane_in_junc:
        next_waypoint = waypoint
        while next_waypoint.is_junction:
            next_waypoint = next_waypoint.next(1)[0]
        if next_waypoint.road_id not in roads:
            roads.append(next_waypoint.road_id)
            road_waypoints.append(next_waypoint)

        previous_waypoint = waypoint
        while previous_waypoint.is_junction:
            previous_waypoint = previous_waypoint.previous(1)[0]
        if previous_waypoint.road_id not in roads:
            roads.append(previous_waypoint.road_id)
            road_waypoints.append(previous_waypoint)

    return road_waypoints

def _fill_road_region(road_waypoint, region=7):

    waypoints = []
    lanes = [road_waypoint.lane_id]
    road_id = road_waypoint.road_id
    lane_stack = [road_waypoint]
    while lane_stack:
        waypoint = lane_stack.pop()
        if str(waypoint.lane_type) == 'Driving':
            waypoints.append(waypoint)
        next_waypoints = [waypoint.get_left_lane(), waypoint.get_right_lane()]
        for next_waypoint in next_waypoints:
            if next_waypoint:
                if next_waypoint.road_id == road_id:
                    if next_waypoint.lane_id not in lanes:
                        lanes.append(next_waypoint.lane_id)
                        lane_stack.append(next_waypoint)


    new_vehicles = []
    for waypoint in waypoints:
        new_vehicles += _fill_lane_region(waypoint, region)

    return new_vehicles

def _fill_lane_region(lane_waypoint, region=7, padding = 4):
    lane_waypoint = get_lane_start(lane_waypoint)

    new_vehicles = [lane_waypoint]
    max_move = int((region-padding)/2)

    next_waypoint = lane_waypoint
    while (not next_waypoint.is_junction) and next_waypoint.road_id == lane_waypoint.road_id:
        next_waypoint = next_waypoint.next(region)[0]
        new_vehicles.append(next_waypoint)

    new_vehicles = [_random_move_waypoint(waypoint, max_move) for waypoint in new_vehicles]
    new_vehicles = [vehicle for vehicle in new_vehicles if vehicle.road_id == lane_waypoint.road_id]
    return new_vehicles

def _random_move_waypoint(waypoint, max_move):
    move = np.random.randint(-max_move, max_move)
    if move == 0:
        return waypoint
    elif move<0:
        return waypoint.previous(-move)[0]
    else:
        return waypoint.next(move)[0]

def _fill_road(road_waypoint, road_vector):

    waypoints = []
    lanes = [road_waypoint.lane_id]
    road_id = road_waypoint.road_id
    lane_stack = [road_waypoint]
    while lane_stack:
        waypoint = lane_stack.pop()
        if str(waypoint.lane_type) == 'Driving':
            waypoints.append(waypoint)
        next_waypoints = [waypoint.get_left_lane(), waypoint.get_right_lane()]
        for next_waypoint in next_waypoints:
            if next_waypoint:
                if next_waypoint.road_id == road_id:
                    if next_waypoint.lane_id not in lanes:
                        lanes.append(next_waypoint.lane_id)
                        lane_stack.append(next_waypoint)

    new_vehicles = []
    total_region = 0
    for i, waypoint in enumerate(waypoints):
        vehicles, region = _fill_lane(get_lane_start(waypoint), road_vector[i])
        new_vehicles += vehicles
        total_region += region

    return new_vehicles, total_region/(len(new_vehicles)-len(waypoints))

def _fill_lane(lane_waypoint, lane_vector):
    positions = []
    current = lane_waypoint

    current = get_next(current, lane_vector[0], 0, max_distance=45)
    if current != None:
        positions.append(current)

    region = 0
    for distance_percentage in lane_vector[1:]:
        current = get_next(current, distance_percentage, max_distance=45)
        if current == None:
            break
        positions.append(current)
        region += distance_percentage*20 + 5.5

    return positions, region
