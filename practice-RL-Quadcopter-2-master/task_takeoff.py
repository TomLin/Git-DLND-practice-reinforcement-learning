import numpy as np
from physics_sim import PhysicsSim

class TaskTakeOff():
    '''Task(environment) that defines the goal and provides feedback to the agent'''
    
    def __init__(self, init_pose=None, init_velocities=None,\
                 init_angle_velocities=None, runtime=5., target_pos=None):
        '''initialize a task object
        Params
        ======
        init_pose               : initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
        init_velocities         : initial velocities of the quadcopter in (x,y,z) dimensions
        init_angle_velocities   : initial radians/second for each of the three Euler angles
        runtime                 : time limit for each episode
        target_pos              : target/goal (x,y,z) position for the agent
        
        Footnote
        ========
        adjust action_low and action_high to shrink the continuous action space
        
        '''

        # simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 410
        self.action_high = 420
        self.action_size = 4

        # goal
        self.target_pos = target_pos if target_pos is not None else np.array([0.,0.,10.])

    def get_reward(self):
        '''use current position of simulation to return reward
        
        Footnote:
        ========
        1. include vertical velocity as part of the reward, encouraging the copter to fly vertically
        2. penalize crash
        3. reward clipping to range (-1,1)
        4. or use np.tanh() to squeeze reward in range (-1,1)

        '''
        # version 01 - udacity task template
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum() + 0.3*(self.sim.v[2])
        #reward = np.clip(reward, -1,1)
        
        # version 02 - udacity takeoff task template
        # reward = zero for matching target z, -ve as you go farther, upto -20
        #reward = -min(abs(self.target_pos[2] - self.sim.pose[2]), 20.) 
        
        #if self.sim.pose[2] > self.target_pos[2]:
        #    reward += 1. # bonus reward
        #elif self.sim.done and self.sim.time < self.sim.runtime:
        #    reward -= 1. # crash before timeout
        
        # version 03 - use np.tanh()
        reward = 3.0*self.sim.v[2] - 0.025*abs(self.sim.v[:2]).sum() + 1.5*self.sim.pose[2] - 0.25*(abs(self.sim.pose[:2]).sum())
        reward = np.tanh(reward)
        
        return reward

    def step(self, rotor_speeds):
        '''use action to obtain next state, reward, done
        
        Footnote
        ========
        self.sim.next_timestep()    : take in action inputs and update the sim.pose to new position,
                                      accumulate the system run-time, return flag of done status
        
        '''

        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the simulation position and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all) # total state_size: action_repeat * simulation's position
        return next_state, reward, done

    def reset(self):
        '''reset the simulation to start a new episode'''
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state



