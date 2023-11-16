# Safe_protect_module
# Author: Yuansj
# Time : 2022/03/08
import math

class safe_check:
    '''
    This script is used in abstract.py
    A rule based module to ensure safety, if ego want to change lane, 
    but there have been a vehicle in the target lane. Then reject the action, choose slower to make sure safety.
    '''
    def __init__(self, road, ego, config, check_type) -> None:
        self.repalce_type = check_type
        self.ego = ego
        self.road = road
        self.all_action = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
        self.dt = 0.1
        self.N_1 = 1
        self.N_2 = 10
        self.predicted_nth = config['mpc_nth_position'] # 0 - 9

    def collision_check(self, action:int) -> bool:
        choose_action = self.all_action[action]
        collision_other, collision_object, wrong_action = False, False, False
        future_state = self.ego.safe_act(choose_action)
        if self.repalce_type == "mpc":
            ego_future_positionx = future_state[self.predicted_nth*4,0]
            ego_future_positiony = future_state[self.predicted_nth*4+1,0]
        elif self.repalce_type == "simple":
             current_ego_head = self.ego.heading
             current_speed = self.ego.speed
             current_x = self.ego.position[0]
             current_y = self.ego.position[1]
             ego_future_positionx = current_x + current_speed * self.predicted_nth * math.cos(current_ego_head)
             ego_future_positiony = current_y + current_speed * self.predicted_nth * math.sin(current_ego_head)

        if self.ego.position[0] < self.road.road_ends and self.ego.position[0] > 630:
            for i in range(len(self.road.vehicles) - 1):
                current_position = self.road.vehicles[i].position
                current_speed = self.road.vehicles[i].speed

                # check collision with other vehicles
                position1 = current_position[0] + current_speed * self.dt * self.N_1
                position2 = current_position[0] + current_speed * self.dt * self.N_2
                
                if ego_future_positionx > position1 and ego_future_positionx < position2 and \
                    abs(ego_future_positiony - current_position[1]) < 2 \
                    and self.ego.position[1] > 2.5:
                # if ego_future_positionx > position1 and ego_future_positionx < position2:
                    collision_other = True
                
                # if abs(self.ego.position[0] - current_position[0]) < 2 and self.ego.position[1] > 2.5:
                #     collision_other = True
                
                # collision with objects
                if ego_future_positionx > self.road.road_ends - 30 and ego_future_positiony > 4.8:
                    collision_object = True

                # wrong action
                if self.ego.position[1] < 2.5 and action == 2:
                    wrong_action = True
        
        unsafe_action = [collision_other, collision_object, wrong_action]
        return unsafe_action
    
    def correct_action(self, unsafe_action, action:int):
        correct_action = action
        if unsafe_action[2] == True:
                correct_action = 1
        if unsafe_action[0] == True:
                correct_action = 4
        if unsafe_action[1] == True:
                correct_action = 4

        return correct_action