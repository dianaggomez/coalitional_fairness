from os import environ
import gym
from gym import spaces
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections import deque

class MARLHighLevelControllerEnv(gym.Env):
    def __init__(self,config=None):
        '''
        Action Space:  0 -> (1,0), 1 -> (0,1), 2 -> (1,1)
        Observation Space: [current queue, no. of AVs remaining, initial no. of AVs]
        '''
        self.config = config
        self.num_agents = 2
        self._agent_ids = set(["agent_{}".format(num) for num in range(1, self.num_agents+1)])
        
        self.action_space = {agent_id: spaces.Discrete(3) for agent_id in self._agent_ids}
        self.observation_space = {agent_id: spaces.Box(low = 0, high = 10, shape= (14,), dtype = np.float64) for agent_id in self._agent_ids}
        
        self.all_queues = self.get_all_queues()
        self.queue_config = 0
        self.left_queue, self.right_queue = self.generate_queue()
        self.orig_left_queue, self.orig_right_queue = self.left_queue.copy(), self.right_queue.copy()
        self.left_vehicle_queue = deque([])
        self.right_vehicle_queue = deque([])

        self.AVs_done, self.Humans_done = False, False
        self.AVs_time, self.Humans_time = 0, 0
        self.time = 0
        self.time_remaining = 10 #each turn a set with two rounds
        
        self.info = {"AVs_cleared_queue": None, "AVs_time": None, "Humans_cleared": None, "humans_time": None}
        self.infos = {agent_id: self.info for agent_id in self._agent_ids}
        self.dones = {agent_id: False for agent_id in self._agent_ids}
        self.dones["__all__"] = False
        
        self.av_baseline_dict = self.get_av_baseline_dict()
            
        self.num_s1 = self.queue_config #human drivers or AV_1
        self.num_s2 = 12 - self.queue_config # AV_2
        self.n_s1, self.n_s2 = self.get_fair_wealth() # population
        self.w_s1, self.w_s2 = [], [] # wealth @ each decision # income share
        self.w_s1_raw, self.w_s2_raw = [], []
        self.fair_reward = 0
        self.current_obs = None
        
        #Parameters to change
        self.fairness = False
        self.alpha = 0
        print("Fairness alpha: ", self.alpha)
        self.training = True
        # self.model = PPO.load("/home/diana/journal/single_agent/train/models/alpha_1/best_success/best_model.zip")
        
        
    def get_all_queues(self): 
        path ='/home/diana/journal/queues_3to10.npy'
        # path = "/home/diana/metadrive/new_test_batch_52.npy"
        all_queues = np.load(path, allow_pickle=True)

        all_queues_list = []
        for i in all_queues:
            all_queues_list.append(i.tolist())

        np.random.shuffle(all_queues_list) #shuffle the list 
        return deque(all_queues_list)
    
    def generate_queue(self): 
        queue = self.all_queues.pop()
        # TODO: revert back after gathering queues
        self.original_queue = queue
        # self.original_queue = ''.join(str(e) for e in queue)
        if len(self.all_queues) == 0:
            self.all_queues = self.get_all_queues()

        self.queue_config = queue.count(1)
        
        left_queue = deque(queue[:len(queue)//2])
        right_queue = deque(queue[len(queue)//2:])

        return left_queue, right_queue
    
    def get_fair_wealth(self):
        n_s1_count = []
        n_s2_count = []
        l_q_temp = deque(self.orig_left_queue)
        r_q_temp = deque(self.orig_right_queue)

        for i in range(6):
            # left = l_q_temp.popleft() #DGG
            left = l_q_temp.pop()
            right = r_q_temp.popleft()
            if left == 2 and right == 2:
                n_s1_count.append(0)
                n_s2_count.append(2)
            elif (left == 1 and right == 2) or (left == 2 and right == 1):
                n_s1_count.append(1)
                n_s2_count.append(1)
            elif left == 1 and right == 1:
                n_s1_count.append(2)
                n_s2_count.append(0)

        # "Fair wealth" at each time step 
        n_s1_init = np.array(n_s1_count).cumsum()
        n_s2_init = np.array(n_s2_count).cumsum()   

        # Normalize the "fair wealth"
        n_s1 = (n_s1_init)/(n_s1_init+n_s2_init)
        n_s2 = (n_s2_init)/(n_s1_init+n_s2_init)

        return n_s1, n_s2
    
    # def get_company_action(self):
    #     # print("Current Obs: ", self.current_obs)
    #     action = self.model.predict(self.current_obs)
    #     return action
    
    def preprocess_actions(self, actions, front_of_queue):
        ''' 
        Choose a single action.
        actions: [Discrete(3),Discrete(3)] -> [av_1, av_2]
        '''
        # print("Actions: ", actions)
        if front_of_queue[0] == front_of_queue[1]:
            # if same coalition, take that coalition's action
            return actions[front_of_queue[0]-1]
        else: #if different coaltions,
             # but same actions, take that action
            if actions[0] == actions[1]:
                return actions[0]
            else: # if conflicting actions, revert to SR
                return 2
        
        
    def process_high_level_action(self, high_level_actions):
        '''
        high-level actions:  0 -> (1,0), 1 -> (0,1), 2 -> (1,1)
        human :1, coalition: 2, empty:0 
        '''
        
        if not (len(self.left_queue) > 0):
            left = 0
            right = self.right_queue[0]
        elif not (len(self.right_queue) > 0):
            right = 0
            left = self.left_queue[-1]
        else:
            left = self.left_queue[-1]
            right = self.right_queue[0]

        front_of_queue = (left, right)
        
        high_level_action = self.preprocess_actions(high_level_actions, front_of_queue)
        
        if high_level_action == 0:
            high_level_action = (1,0)
        elif high_level_action == 1:
            high_level_action = (0,1)
        elif high_level_action == 2:
            high_level_action = (1,1)
 
        if high_level_action == (0, 1):
            # print(front_of_queue)
            if front_of_queue == (1,1) or front_of_queue == (0,1):
                num_of_vehicles =  self.num_of_vehicles_to_exit(high_level_action, coalition=1)
            elif front_of_queue == (1,2) or front_of_queue == (0,2):
                high_level_action = (0,1)
                num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=2)
            elif front_of_queue == (1,0): #right side empty they should still exit
                high_level_action = (1,0)
                num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=1)
            elif front_of_queue == (2,1):
                # TODO: try the case in which you allow the other vehicle to stop
                high_level_action = (1,1) 
                num_of_vehicles = np.array([1,1])
            elif front_of_queue == (2,0):
                high_level_action = (1,0)
                num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=2)
            else:# proceed to Social Rule
                high_level_action = (1,1)
                num_of_vehicles = np.array([1,1])

        elif high_level_action == (1,0):
            if front_of_queue == (1,1) or front_of_queue == (1,0):
                num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=1)
            elif front_of_queue == (2,1) or front_of_queue == (2,0):
                high_level_action = (1,0)
                num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=2)
            elif front_of_queue == (0,1): # left side empty w/S2
                high_level_action = (0,1)
                num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=1)
            elif front_of_queue == (0,2): # left side empty
                high_level_action = (0,1)
                num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=2)
            else: #proceed to Social Rule
                high_level_action = (1,1)
                num_of_vehicles = np.array([1,1])

        elif high_level_action == (1,1):
            if front_of_queue == (2,0) or front_of_queue == (1,0):
                high_level_action = (1,0)
                if front_of_queue == (2,0):
                    num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=2)
                else:
                    num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=1)
            elif front_of_queue == (0,2) or front_of_queue == (0,1):
                high_level_action = (0,1)
                if front_of_queue == (0,2):
                    num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=2)
                else:
                    num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action, coalition=1)
            else:
                num_of_vehicles = np.array([1,1])

        self.HLA = high_level_action
        # print("hla: ", high_level_action)
        # print("No. of vehicles: ", num_of_vehicles)
        return high_level_action, num_of_vehicles
    
    def num_of_vehicles_to_exit(self, high_level_action, coalition):
        '''
        Calculate the number of vehicles that should exit according to a 
        high level action. Up to 3 AVs in consecutive order may exit.
        '''
        counter = 0
        if high_level_action == (0,1):
            if len(self.right_queue) > 0:
                for i in range(min(len(self.right_queue),3)):
                    if self.right_queue[i] == coalition:
                        counter+=1
                    else:
                        break
            num_of_vehicles = np.array([0,counter])
        elif high_level_action == (1,0):
            lq_copy = self.left_queue.copy()
            lq_copy.reverse()
            if len(lq_copy) > 0:
                for i in range(min(len(lq_copy),3)):
                    if lq_copy[i] == coalition:
                        counter+=1
                    else:
                        break
            num_of_vehicles = np.array([counter,0])
        return num_of_vehicles
    
    def update_queue(self, side, num_of_vehicles):
        w_s1, w_s2 = 0, 0
        for i in range(num_of_vehicles):
            if side is "right":
                v = self.right_queue.popleft()
            else:
                v = self.left_queue.pop() 
            # For Theil Index, check which vehicle exited
            if v == 1:
                w_s1 +=1 
                self.num_s1 -= 1 #human drivers
            if v == 2:
                w_s2 +=1
                self.num_s2 -= 1 # AVs
        return w_s1, w_s2 
        
    def take_step(self, high_level_action):
        HLA = []
        #TODO: Check action, you may have to unpack it once more.
        for agent, action in high_level_action:
            HLA.append(action)
            
        hla, num_of_vehicles = self.process_high_level_action(HLA)
        
        if hla == (0,1):
            w_s1, w_s2  = self.update_queue("right", num_of_vehicles[1])
            return w_s1, w_s2  
        
        if hla == (1,0):
            w_s1, w_s2  = self.update_queue("left", num_of_vehicles[0])
            return w_s1, w_s2
        
        if hla == (1,1):
            w_s1, w_s2  = self.update_queue("right", num_of_vehicles[1])
            w_s1_, w_s2_  = self.update_queue("left", num_of_vehicles[0])
            w_s1+= w_s1_
            w_s2+= w_s2_
            return w_s1, w_s2
       
    def update_wealth(self, w_s1, w_s2):
        if self.w_s1 == [] and self.w_s2 == []:
            self.w_s1_raw.append(w_s1), self.w_s2_raw.append(w_s2)
            # normalize
            if w_s1 == 0 and w_s2 == 0:
                self.w_s1.append(0), self.w_s2.append(0)
            else:
                self.w_s1.append(w_s1/(w_s1+w_s2)), self.w_s2.append(w_s2/(w_s1+w_s2))
        else:
            self.w_s1_raw.append(self.w_s1_raw[-1]+w_s1)
            self.w_s2_raw.append(self.w_s2_raw[-1]+w_s2)
            # normalize
            if self.w_s1_raw[-1] == 0 and self.w_s2_raw[-1] == 0:
                self.w_s1.append(0), self.w_s2.append(0)
            else:
                self.w_s1.append(self.w_s1_raw[-1]/(self.w_s1_raw[-1]+self.w_s2_raw[-1]))
                self.w_s2.append(self.w_s2_raw[-1]/(self.w_s1_raw[-1]+self.w_s2_raw[-1]))

    def update_fair_wealth(self):
        if len(np.array(self.w_s1)) > len(self.n_s1):
            # concatenate last value to "fair wealth"
            self.n_s1 = np.concatenate((self.n_s1, np.array([self.n_s1[-1]])), axis=None)
            self.n_s2 = np.concatenate((self.n_s2, np.array([self.n_s2[-1]])), axis=None)

    def calculate_rf(self):
        if len(np.array(self.w_s1)) < len(self.n_s1):
            w_s1, w_s2 = self.w_s1[-1], self.w_s2[-1]
            n_s1, n_s2 = self.n_s1[len(self.w_s1) - 1], self.n_s2[len(self.w_s2) - 1]
        else:
            w_s1, w_s2 = self.w_s1[-1], self.w_s2[-1]
            n_s1, n_s2 = self.n_s1[-1], self.n_s2[-1]

        # Each coalitions contribution to the Theil Index
        if (w_s1 == 0 and n_s1 == 0) or (w_s2 == 0 and n_s2 == 0): # no unfairness
            T_s1 , T_s2 = 0, 0
        elif (w_s1 == 0 and n_s1 != 0) or (w_s2!= 0 and n_s2 == 0): # extreme unfairness to coalition 1
            T_s1 , T_s2 = 0, 1
        elif (w_s2 == 0 and n_s2 != 0) or (w_s1!= 0 and n_s1 == 0): # extreme unfairness to coalition 2
            T_s1 , T_s2 = 1, 0
        elif (w_s1 == 0 and w_s2 == 0):
            T_s1 , T_s2 = 1, 0
        else:
            T_s1 = w_s1*np.log(w_s1/n_s1)
            T_s2 = w_s2*np.log(w_s2/n_s2)

        # THEIL INDEX
        T = T_s1 + T_s2
        # return self.alpha*((12-self.queue_config)/12)*T
        return self.alpha*T
    
    def get_reward(self, w_s1, w_s2, coalition):
        current_vehices = list(self.left_queue).count(coalition) + list(self.right_queue).count(coalition)
        r = (1 - (current_vehices/list(self.original_queue.count(coalition)))**0.5) - 0.1
        # ======== Theil Index Reward Function ========
        if self.fairness:
            self.update_wealth(w_s1, w_s2)
            self.update_fair_wealth()
            r_f = self.calculate_rf()
        else:
            r_f = 0   
        self.fair_reward += r_f
        reward = r - r_f
        return reward
    
    def done(self):

        AV_done = list(self.left_queue).count(1) + list(self.right_queue).count(1) == 0 # AV_1
        human_done = list(self.left_queue).count(2) + list(self.right_queue).count(2) == 0 # AV_2
        
        if not self.training:
            # DONE FUNCTION FOR TIME DATA 
            # TODO: Only record time for successes (i.e. AVs exit first)
            if AV_done and not self.AVs_done:
                # print("AVs are out!")
                self.AVs_done = True
                self.info["AVs_time"] = self.time * 0.2
                self.info["fair_reward"] = self.fair_reward
            if human_done and not self.Humans_done:
                self.Humans_done = True
                self.info["humans_time"] = self.time * 0.2
            if AV_done and human_done:
                self.info["cleared_queue"] = True
                return True
            return False
        else:
            # DONE FUNCTION FOR TRAINING
            if human_done: # Terminate: if humans clear the queue
                self.Humans_time = self.time
                self.Humans_done = True
                self.info["humans_time"] = self.Humans_time*0.2
                self.info["AVs_cleared_queue"] = False
                self.info["Humans_cleared"] = True
                self.dones['agent_1'] = True
                return True
            if AV_done: # Terminate: if AVs clear the queue
                self.AVs_time = self.time
                self.info["AVs_time"] = self.AVs_time*0.2
                self.info["AVs_cleared_queue"] = True
                self.info["fair_reward"] = self.fair_reward
                self.info["Humans_cleared"] = False
                self.dones['agent_2'] = True
                return True
            if self.time_remaining == 0:  # Terminate: if time elapsed
                self.info["AVs_cleared_queue"] = False
                self.dones["__all__"] = True # the env terminated
                return True
            return False

    def step(self, action):
        # av_2 = self.get_company_action()[0]
        # w_s1, w_s2 = self.take_step([action, av_2])
        w_s1, w_s2 = self.take_step(action)
        self.time_remaining -= 1
        self.time += 1
        
        # d = self.done()
        self.done()
        # i = self.info
        i = self.infos
        r =  {"agent_{}".format(num): self.get_reward(w_s1, w_s2, num) for num in range(1, self.num_agents+1)}
        o = self.get_observation()
        
        return o, r, d, i

    def reset(self):
        self.left_queue, self.right_queue = self.generate_queue()
        self.orig_left_queue, self.orig_right_queue = self.left_queue.copy(), self.right_queue.copy()
        self.left_vehicle_queue = deque([])
        self.right_vehicle_queue = deque([])

        self.AVs_done, self.Humans_done = False, False
        self.AVs_time, self.Humans_time = 0, 0
        self.time = 0
        self.time_remaining = 10 #each turn a set with two rounds
        self.info = {"AVs_cleared_queue": None, "AVs_time": None, "Humans_cleared": None, "humans_time": None}
        self.infos = {agent_id: self.info for agent_id in self._agent_ids}
        self.dones = {agent_id: False for agent_id in self._agent_ids}
        self.dones["__all__"] = False
        
        self.num_s1 = self.queue_config #human drivers
        self.num_s2 = 12 - self.queue_config # AVs
        self.n_s1, self.n_s2 = self.get_fair_wealth() # population
        self.w_s1, self.w_s2 = [], [] # wealth @ each decision # income share
        self.w_s1_raw, self.w_s2_raw = [], []
        
        self.fair_reward = 0
        
        obs = self.get_observation()
        return obs

    def get_observation(self):
        left_queue_copy = list(self.left_queue.copy())
        right_queue_copy = list(self.right_queue.copy())

        # gather the number of vehicles on each side  
        left_empty = 6 - len(left_queue_copy)
        right_empty = 6 - len(right_queue_copy)
        
        queue = [0]*left_empty + left_queue_copy + right_queue_copy + [0]*right_empty
        
        # add the number of vehicles
        # Observation for AV_1
        num_of_vehicles_remaining_1 = left_queue_copy.count(1) + right_queue_copy.count(1)
        obs = queue + [num_of_vehicles_remaining_1] + [self.queue_config]
        obs = np.array(obs)
        # print(obs)
        # Observation for AV_2
        num_of_vehicles_remaining_2 = left_queue_copy.count(2) + right_queue_copy.count(2)
        self.current_obs = np.array(queue + [num_of_vehicles_remaining_2] + [12-self.queue_config])
        
        observations = {'agent_1': obs, 'agent_2': self.current_obs}
        return observations
    
    def get_av_baseline_dict(self):
        path = "/home/diana/metadrive/av_baseline.npy"
        av_baseline = np.load(path)
        keys, vals = av_baseline[:,0], av_baseline[:,1]
    
        av_baseline_dict = {}
        for k,v in zip(keys, vals):
            av_baseline_dict[k] = float(v)
        return av_baseline_dict

    def visualize_queue(self):
        string = "|"
        lq = self.left_queue.copy()
        # lq.reverse()
        for vehicle in lq:
            string += str(vehicle) + "|"
        string += "*|"
        for vehicle in self.right_queue:
            string += str(vehicle) +"|"
        print(string)
        
    def render(self):
        pass  

    def close(self):
        pass

    def seed(self):
        pass

if __name__ == "__main__":
    # Following SR policy
    env = MARLHighLevelControllerEnv()
    env.reset()
    actions = env.action_space.sample()
    print(actions)
    o, r, done, info = env.step(actions) 
    print("Observations: ", o)
    print("Rewards: ", r)
    # action = 2 #env.action_space.sample()

    # time_to_exit = {}
    # time_step = 0.2

    # sr_outcome = []
    
    # path ='/home/diana/journal/queues/queues_3to10.npy'
    # qs = np.load(path)
    # print("Number of queues: ", len(qs))
    
    # for i in range(len(qs)):
    #     # if i%25 == 0: print(i)
    #     done = False
    #     # print("initial queue")
    #     # env.visualize_queue()
    #     while done == False:
    #         # action = env.action_space.sample()
    #         # print("Sampled Actions: ", action)
    #         o, r, done, info = env.step(action) 
    #         # env.visualize_queue() 
            
    #     # time_to_exit[env.original_queue] = env.AVs_time*0.2
    #     sr_outcome.append(np.append(env.original_queue, 
    #                                [env.queue_config, not info["AVs_cleared_queue"], info["AVs_cleared_queue"]]
    #                                )
    #                      )
    #     env.reset()
    
    # sr_outcome = np.array(sr_outcome)
    # print("queues recorded: " , len(sr_outcome))
    # np.save("/home/diana/journal/train/models/alpha_0/sr_outcome.npy", sr_outcome)    
    # # queues = np.array(list(time_to_exit.keys())) 
    # # print("No of queues: ", len(queues))
    # # time = np.array(list(time_to_exit.values())) 

    # # time_data = np.column_stack((queues, time))

    # # print(time_data)


    # # np.save("av_baseline.npy", time_data)
