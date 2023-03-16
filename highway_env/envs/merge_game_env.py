from re import T
from tkinter import Widget
from turtle import width
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.utils import near_split

START_DIS = 540
END_DIS = 730

class Merge_game_env(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    def __init__(self, avg_speed= -1, min_density=0., max_density=1., 
                 cooperative_prob = 0., mpc_control=True, safe_check=False, use_mpc_cost =False, mpc_nth_position=4):
        self.avg_speed = avg_speed
        self.min_density = min_density
        self.max_density = max_density
        self.have_changed = False
        self.reach_goal = False
        self.config = self.default_config()
        self.config.update({"cooperative_prob": cooperative_prob})
        self.config.update({"mpc_nth_position": mpc_nth_position})
        self.config.update({"mpc_control":mpc_control})
        self.config.update({'safe_check': safe_check})
        self.config.update({'use_mpc_cost': use_mpc_cost})
        super().__init__(self.config)

    def _cost(self, action: int) -> float:
        cost, wrong_action_cost, crash_cost, predict_cost = 0., 0., 0., 0.

        # collision check
        def collision_check(vehicle_1, vehicle_2):
            '''
            input: two vehicles that may collide
            return: collision True of False
            '''
            dt = 0.1
            N_1 = 1
            N_2 = 10
            crashed = False
            risk_velocity = False
            current_position = vehicle_1.position
            # compute the other vehicle future position

            # predict other vehicle position
            v1_future_position_x_end = current_position[0] + vehicle_1.speed * dt * N_1 # 0.1s 
            v1_future_position_x_start = current_position[0] + vehicle_1.speed *dt * N_2 # 1.0s
            # the n-th predicted (x,y) position 
            v2_future_position_x = vehicle_2.future_state[self.config['mpc_nth_position']*4, 0]  
            v2_future_position_y = vehicle_2.future_state[self.config['mpc_nth_position']*4+1, 0]

            # check risk and collision
            x_collision = True if v2_future_position_x > v1_future_position_x_start and v2_future_position_x < v1_future_position_x_end else False
            y_collision = True if abs(current_position[1] - v2_future_position_y) < 2 else False
            # velocity_collision = True if abs(vehicle_1.speed - vehicle_2.future_state[16+2,0]) > 2 else False
            if  y_collision and x_collision:
                crashed = True
            
            risk_velocity = True if abs(vehicle_1.speed - vehicle_2.speed) < 1.5 and abs(current_position[0] - vehicle_2.position[0]) < 5 else False

            return [crashed, risk_velocity]
            
        # wrong action cost 
        if self.vehicle.position[1] < 5:
            if action == 2:
                wrong_action_cost += 0.5

        near_vehicle_distance = 10
        near_vehicle = list()
        for i in range(len(self.road.vehicles)-1):
            if self.road.vehicles[i].crashed and not self.vehicle.crashed:
                crash_cost += 1.
            # find near vehicles
            if self.vehicle.position[0] > 620 and self.vehicle.position[0] < 700:
                if abs(self.road.vehicles[i].position[0] - self.vehicle.position[0]) < near_vehicle_distance:
                    near_vehicle.append(self.road.vehicles[i])

        # predict cost
        if self.config['mpc_control'] and self.config['use_mpc_cost']:
            if self.vehicle.position[1] < 6 and self.vehicle.position[1] > 0 and not self.have_changed:
                for i in range(len(near_vehicle)):
                    result = collision_check(near_vehicle[i], self.vehicle)
                    if action == 0 and result[0]:
                        predict_cost += 0.2
                    elif result[1]:
                        predict_cost += 0.1
                
                if self.vehicle.position[0] > 680 and not self.have_changed:
                    predict_cost += 0.2

            # if predict_cost >= 0.05: 
            #     print('Attention')

        # crashed cost
        if self.vehicle.crashed:
            crash_cost += 1.

        cost = crash_cost + wrong_action_cost + predict_cost
        return cost

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        # r = -0.1    #Time penalty

        changelane_reward, success_reward, velocity_reward, predict_reward = 0, 0, 0, 0

        other_vehicles_position, other_vehicles_velocity = list(), list()
        for i in range(len(self.road.vehicles)-1):
            other_vehicles_position.append(self.road.vehicles[i].position[0])
            other_vehicles_velocity.append(self.road.vehicles[i].velocity[0])
        other_vehicles = dict(position=np.array(other_vehicles_position), velocity = np.array(other_vehicles_velocity))

        # velocity reward
        index = np.where(abs(other_vehicles['position'] - self.vehicle.position[0]) < 100)
        near_vehicle_velocity = other_vehicles['velocity'][index]

        if near_vehicle_velocity.shape[0] != 0:
            np.insert(near_vehicle_velocity, -1, self.vehicle.velocity[0])
            # max_min = near_vehicle_velocity.max() - near_vehicle_velocity.min()
            # if max_min == 0:
            #     max_min = 1
            velocity_reward = 0.1 if abs(self.vehicle.velocity[0] - near_vehicle_velocity.mean()) < near_vehicle_velocity.mean() * 0.2 else -0.5
            # velocity_reward = self.vehicle.velocity[0] - near_vehicle_velocity.mean() / (10*max_min)
        # else:
        #     if self.vehicle.velocity[0] < 10:
        #         velocity_reward = -1

        # merge reward
        merge_lane = self.road.network.get_lane(("b","c",1))
        if self.vehicle.position[0] > merge_lane.start[0] :
            if int(self.vehicle.position[1]) == 0 and not self.vehicle.crashed and not self.have_changed:
                changelane_reward = 5.
                self.have_changed = True
            
        # reach reward
        if self.vehicle.position[0] > END_DIS and not self.reach_goal:
            success_reward = 10. # 5.
            self.reach_goal = True

        r = velocity_reward + changelane_reward + success_reward
        return r

    def _is_terminal(self) -> bool:

        """The episode is over when a collision occurs or when the access ramp has been passed."""
        terminal_end = True
        terminal_crashed = False
        for i in range(len(self.road.vehicles)):
            if self.road.vehicles[i].crashed:
                terminal_crashed = True
            if self.vehicle.position[0] < END_DIS:
                terminal_end = False

        terminal = (self.vehicle.crashed or terminal_crashed) or terminal_end

        return terminal

    def _reset(self) -> None:
        # #high_speed
        if self.avg_speed == -1:
            avg_speed = 30.
            vehicles_density=np.random.uniform(0.4,0.6)
            if np.random.random()<0.5:#by 50 percent switch to low_speed config
                avg_speed = 10.
                vehicles_density=np.random.uniform(0.4,0.6)
        else:
            avg_speed = self.avg_speed
            vehicles_density=np.random.uniform(self.min_density, self.max_density)
        
        # print(vehicles_density)
        
        self.config.update({"vehicles_density": vehicles_density,})
        self.config.update({"avg_speed": avg_speed,})

        self.have_changed = False
        self.reach_goal = False
        self._make_road()
        self._make_vehicles()

    @classmethod
    def default_config(cls) -> dict:
        print("default config")
        config = super().default_config()
        config.update({
            "observation": {
                "type": "GameObservation",
                "vehicles_count": 11,
                "features": ["presence", "x", "y","vx","vy"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-30, 30],
                    "vy": [-30, 30],
                },
                "absolute": False,
                "normalize": True,
                "order": "sorted",
                "clip":True,
                "see_behind":False,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
            },
            "policy_frequency": 2,
            "simulation_frequency": 10,
            'show_trajectories': False,
            'real_time_rendering': False,
            'show_mpc_trajectory': True,
            "show_other_vehicles_predict": True,
            "show_ego_history":False,
            "mpc_nth_position":4
        })
        return config

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [550, 80, 70, 200]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0,StraightLane.DEFAULT_WIDTH]
        line_type = [[c, c], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(1):
            net.add_lane("a", "b", StraightLane([250, y[i]], [sum(ends[:2]), y[i]], speed_limit=27, line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], speed_limit=27, line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], speed_limit=27, line_types=line_type[i]))

        # Merging lane
        y_dis = 25 + y[0]
        center_lkh = [ends[0], y[0]]
        radius_lkh = y_dis
        deg_lkh = 15.1185
        ljk = StraightLane([0, y_dis], [ends[0], y_dis], speed_limit=27, line_types=line_type[0])
        lkh = CircularLane(center_lkh,radius_lkh,np.deg2rad(-270-deg_lkh),np.deg2rad(-270), speed_limit=27, line_types=(LineType.CONTINUOUS,LineType.CONTINUOUS))

        lhe_len = 70.7107
        lhe_start = [ends[0] + radius_lkh * np.sin(np.deg2rad(deg_lkh)), radius_lkh * np.cos(np.deg2rad(deg_lkh)) + center_lkh[1]]
        lhe_end = [lhe_start[0]+lhe_len * np.cos(np.deg2rad(deg_lkh)), lhe_start[1]-lhe_len * np.sin(np.deg2rad(deg_lkh))]
        lhe = StraightLane(lhe_start,lhe_end, speed_limit=27, line_types=line_type[0])
        
        radius_leb = 20
        deg_leb = 15.1185
        center_leb = [sum(ends[:2]), y_dis]
        leb = CircularLane(center_leb,radius_leb,np.deg2rad(-90-deg_leb),np.deg2rad(-90), speed_limit=27, line_types=(LineType.CONTINUOUS,LineType.CONTINUOUS))
        lbc_start = [center_leb[0], center_leb[1] - radius_leb]
        lbc = StraightLane(lbc_start, [sum(ends[:3]),lbc_start[1]], speed_limit=27, line_types=[n, c])
        net.add_lane("j", "k", ljk)
        net.add_lane('k','h',lkh)
        net.add_lane('h','e',lhe)
        net.add_lane('e','b',leb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"], road_ends = END_DIS)
        road.objects.append(Obstacle(road, lbc.position(sum(ends[:3]) - lbc_start[0], 0)))

        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        ego_init_speed = np.random.uniform(17.,27.)
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(START_DIS, 0), speed=ego_init_speed)
        
        ego_vehicle.mpc = self.config['mpc_control']

        speed = np.random.normal(self.config["avg_speed"],1.)
        new_vehicle = other_vehicles_type.create_first(self.road, lane_from="b",lane_to="c",lane_id=0,speed=speed)
        new_vehicle.enable_lane_change = False
        new_vehicle.mpc = False
        self.road.vehicles.append(new_vehicle)

        for _ in range(self.config['observation']["vehicles_count"] - 1):

            lanes = np.arange(1)
            lane_id = self.road.np_random.choice(lanes, size=1).astype(int)[0]
            lane = self.road.network.get_lane(("a", "b", lane_id))
            speed=np.random.normal(self.config["avg_speed"], 3.)
            speed=np.clip(speed, 17., lane.speed_limit)
            cooperative = np.random.uniform()<self.config["cooperative_prob"]
            # new_vehicle = other_vehicles_type.create_random(self.road,
            #                                                 lane_from="a",
            #                                                 lane_to="b",
            #                                                 lane_id=lane_id,
            #                                                 speed=speed,
            #                                                 spacing= 1 / self.config["vehicles_density"],
            #                                                 cooperative = cooperative
            #                                                 )

            
            new_vehicle = other_vehicles_type.create_from_new(new_vehicle, 
                                                              space=1 / self.config["vehicles_density"],
                                                              speed=speed,
                                                              cooperative=cooperative)
            new_vehicle.enable_lane_change = False#np.random.random()<0.5
            new_vehicle.mpc = False
            self.road.vehicles.append(new_vehicle)

        #IMPORTANT: Ego vehicle should be added after others!
        road.append_ego_vehicle(ego_vehicle)
        self.vehicle = ego_vehicle

register(
    id='merge_game_env-v0',
    entry_point='highway_env.envs:Merge_game_env',
    kwargs={'avg_speed' : 22, 
    'min_density' : 0.5,
    'max_density' : 1.0,
    'cooperative_prob': 0,
    'mpc_control': True,
    'safe_check': True,
    'use_mpc_cost':True,
    'mpc_nth_position':4 # 0-9 
    },
)

register(
    id='merge_eval_high_density-v0',
    entry_point='highway_env.envs:Merge_game_env',
    kwargs={'avg_speed' : 22, 
    'min_density' : 0.8,
    'max_density' : 1.0,
    'cooperative_prob': 0,
    'mpc_control': True,
    'safe_check': True,
    'use_mpc_cost':True,
    'mpc_nth_position':4 # 0-9 
    },
)

register(
    id='merge_eval_low_density-v0',
    entry_point='highway_env.envs:Merge_game_env',
    kwargs={'avg_speed' : 22, 
    'min_density' : 0.5,
    'max_density' : 0.7,
    'cooperative_prob': 0,
    'mpc_control': True,
    'safe_check': True,
    'use_mpc_cost':True,
    'mpc_nth_position':4 # 0-9 
    },
)