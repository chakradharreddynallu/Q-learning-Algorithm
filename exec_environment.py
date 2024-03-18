"""
    This code communicates with the coppeliaSim software and simulates shaking a container to mix objects of different color 

    Install dependencies:
    https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
    
    MacOS: coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
    Ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
"""

import sys
# Change to the path of your ZMQ python API
sys.path.append('/app/zmq/')
import numpy as np
from zmqRemoteApi import RemoteAPIClient
import time

class Simulation():
    def __init__(self, sim_port = 23000):
        self.sim_port = sim_port
        self.directions = ['Up','Down','Left','Right']
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost',port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)  
        
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()
    
    def getObjectHandles(self):
        self.tableHandle=self.sim.getObject('/Table')
        self.boxHandle=self.sim.getObject('/Table/Box')
    
    def dropObjects(self):
        self.blocks = 18
        frictionCube=0.06
        frictionCup=0.8
        blockLength=0.016
        massOfBlock=14.375e-03
        
        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript,self.tableHandle)
        self.client.step()
        retInts,retFloats,retStrings=self.sim.callScriptFunction('setNumberOfBlocks',self.scriptHandle,[self.blocks],[massOfBlock,blockLength,frictionCube,frictionCup],['cylinder'])
        
        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue=self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break
    
    def getObjectsInBoxHandles(self):
        self.object_shapes_handles=[]
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            # get the starting position of source
            obj_position = self.sim.getObjectPosition(obj_handle,self.sim.handle_world)
            obj_position = np.array(obj_position) #- np.array(box_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step
    
    def action(self,direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5
        
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir*span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()

def calculate_reward(state):
    sum_=0
    for i in state:
        if i=='1':
            sum_+=1
        else:
            sum_+=-1
    return(sum_)

def find_quadrant(object_x, object_y, center_x, center_y):
    relative_x = object_x
    relative_y = object_y
    # finding the quadrant using object and box positions
    if relative_x > center_x and relative_y > center_y:
        return 'Q1'
    elif relative_x < center_x and relative_y > center_y:
        return 'Q2'
    elif relative_x < center_x and relative_y < center_y:
        return 'Q3'
    elif relative_x > center_x and relative_y < center_y:
        return 'Q4'
def findstate(quadrant):
    blue_quadrant=[0,0,0,0]
    red_quadrant=[0,0,0,0]
    for i in range(9):
        if quadrant[i]=="Q1" :
            blue_quadrant[0]+=1
        elif quadrant[i]=="Q2":
            blue_quadrant[1]+=1
        elif quadrant[i]=="Q3":
            blue_quadrant[2]+=1
        elif quadrant[i]=="Q4":
            blue_quadrant[3]+=1
    for i in range(9,18):
        if quadrant[i]=="Q1" :
            red_quadrant[0]+=1
        elif quadrant[i]=="Q2":
            red_quadrant[1]+=1
        elif quadrant[i]=="Q3":
            red_quadrant[2]+=1
        elif quadrant[i]=="Q4":
            red_quadrant[3]+=1
    state=''
    for i in range(4):
        if red_quadrant[i]==blue_quadrant[i]:
            state+='1'
        else:
            state+='0'
    return(state)

def train_agent():
    dict_action={0:"Up",1:"Down",2:"Left",3:"Right"}
    #need to select the state
    steps=10
    global alpha         
    global gamma         
    global epsilon       
    global max_exploration_rate
    global min_exploration_rate 
    global exploration_decay_rate
    global Q_table
    for episode in range(200):
        print(f'Running episode: {episode + 1}')
        episode_reward=[]
        env = Simulation()
        boxposition=env.sim.getObjectPosition(env.boxHandle,env.sim.handle_world)
        positions = env.getObjectsPositions()
        quadrant=[find_quadrant(obj[0], obj[1], boxposition[0], boxposition[1]) for obj in positions]
        state=findstate(quadrant)
        for _ in range(steps):
            random=np.random.uniform(0, 1)
            if random < epsilon:                       #selecting random action
                action = np.random.choice(num_actions)  # Exploration is performed in this step
            else:
                action = np.argmax(Q_table[int(state,2)])      # Exploitation is performed in this step
            env.action(direction = dict_action[action])
            boxposition=env.sim.getObjectPosition(env.boxHandle,env.sim.handle_world)
            positions = env.getObjectsPositions()
            quadrant=[find_quadrant(obj[0], obj[1], boxposition[0], boxposition[1]) for obj in positions]
            new_state=findstate(quadrant)
            reward=calculate_reward(new_state)
            episode_reward.append(reward)
            old_value = Q_table[int(state, 2), action]
            next_max = np.max(Q_table[int(new_state, 2)])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            Q_table[int(state, 2), action] = new_value
            state = new_state
            epsilon = min_exploration_rate +(max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
            # print(Q_table)
        env.stopSim()
    with open('reward.txt', 'w') as file:
        for i in episode_reward:
            file.write(f"{i}\n")
    np.savetxt('q_table.txt', Q_table, fmt='%f')   #saving the Q table in q_table.txt file  


def test_agent():
    Q_table = np.loadtxt("Q_table.txt", delimiter=" ")   #loading the Q table from the text file
    # print("Qtable laoded is ",Q_table)
    dict_action={0:"Up",1:"Down",2:"Left",3:"Right"}
    counter=0
    steps=30
    allepisode_how_long=[]
    for episode in range(10):
        # Initializing the episode 
        env = Simulation()
        how_long=0
        boxposition=env.sim.getObjectPosition(env.boxHandle,env.sim.handle_world)
        positions = env.getObjectsPositions()
        quadrant=[find_quadrant(obj[0], obj[1], boxposition[0], boxposition[1]) for obj in positions]
        state=findstate(quadrant)

        for _ in range(steps):
            action = np.argmax(Q_table[int(state, 2), :])        #chosing action based on Q table
            env.action(direction = dict_action[action])
            boxposition=env.sim.getObjectPosition(env.boxHandle,env.sim.handle_world)
            positions = env.getObjectsPositions()
            quadrant=[find_quadrant(obj[0], obj[1], boxposition[0], boxposition[1]) for obj in positions]
            new_state=findstate(quadrant)
            how_long+=1
            if new_state=="1111":
                counter+=1
                env.stopSim()
                allepisode_how_long.append(how_long)
                break
            state = new_state
        allepisode_how_long.append("failed")
        env.stopSim()
    with open('Qlearning_mixed.txt', 'w') as file:
        file.write(f"number times the objects were mixed successfully in 10 trials if you choose action based on Q-learning {counter}\n")
        for i in range(len(allepisode_how_long)):
            if isinstance(allepisode_how_long[i], int):
                file.write(f" episode {i} took {allepisode_how_long[i]} steps to mix properly\n")
            elif isinstance(allepisode_how_long[i], str):
                file.write(f" episode {i} failed to mix properly\n")
    return(counter)

def test_agent_random():
    randomcounter=0
    steps=10
    dict_action={0:"Up",1:"Down",2:"Left",3:"Right"}
    allepisode_how_long=[]
    for episode in range(10):
        env = Simulation()
        boxposition=env.sim.getObjectPosition(env.boxHandle,env.sim.handle_world)
        positions = env.getObjectsPositions()
        quadrant=[find_quadrant(obj[0], obj[1], boxposition[0], boxposition[1]) for obj in positions]
        state=findstate(quadrant)
        how_long=0
        for _ in range(steps):
            action = np.random.choice(num_actions)  #Selecting the random action
            env.action(direction = dict_action[action])
            boxposition=env.sim.getObjectPosition(env.boxHandle,env.sim.handle_world)
            positions = env.getObjectsPositions()
            quadrant=[find_quadrant(obj[0], obj[1], boxposition[0], boxposition[1]) for obj in positions]
            new_state=findstate(quadrant)
            how_long+=1
            if new_state=="1111":
                randomcounter+=1
                env.stopSim()
                allepisode_how_long.append(how_long)
                break
            state = new_state
        allepisode_how_long.append("failed")
        env.stopSim()
    with open('random_mixed.txt', 'w') as file:
        file.write(f"number times the objects were mixed successfully in 10 trials if we choose action randomly {randomcounter}\n")
        for i in range(len(allepisode_how_long)):
            if isinstance(allepisode_how_long[i], int):
                file.write(f" episode {i} took {allepisode_how_long[i]} steps to mix properly\n")
            elif isinstance(allepisode_how_long[i], str):
                file.write(f" episode {i} is failed to mix properly\n")
    return(randomcounter)

epsilon = 1.0        #Exploration rate
gamma = 0.9         #Discount factor
alpha = 0.5         #Learning rate
min_exploration_rate = 0.01
max_exploration_rate = 1.0
exploration_decay_rate = 0.1

# Q-Table Initialization
num_states = 16    # 2 power 4 objects
num_actions = 4    # four actions can be performed  such as Up, Down, Left, Right
Q_table = np.zeros((num_states, num_actions))


def main():
    train_agent()
    test_agent()
    test_agent_random()

if __name__ == '__main__':
    
    main()
