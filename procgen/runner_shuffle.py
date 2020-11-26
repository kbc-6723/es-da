import numpy as np
from baselines.common.runners import AbstractEnvRunner
import random
class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner
    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, load_path):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
        self.load_path = load_path
        
        self.obs1 = env.reset()
        self.obs2 = env.reset()
        self.obs3 = env.reset()
        self.obs4 = env.reset()
        self.obs5 = env.reset()
        self.new_obs = env.reset()
        self.nsteps = 128
    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_dones = [],[],[],[]
        mb_states = self.states
        epinfos = []
        # For n in range number of steps
        
        for _ in range(self.nsteps):
            
            actions = self.model.step1(self.obs1, S=self.states, M=self.dones)
            mb_obs.append(self.obs1.copy())
            self.obs1[:], rewards, self.dones, infos = self.env.step(actions)
            
            mb_actions.append(actions)
            mb_dones.append(self.dones)
                 
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            
        for _ in range(self.nsteps):
            
            actions = self.model.step2(self.obs2, S=self.states, M=self.dones)
            mb_obs.append(self.obs2.copy())
            self.obs2[:], rewards, self.dones, infos = self.env.step(actions)

            mb_actions.append(actions)
            mb_dones.append(self.dones)
                 
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)    
        for _ in range(self.nsteps):
        
            actions = self.model.step3(self.obs3, S=self.states, M=self.dones)
            mb_obs.append(self.obs3.copy())
            self.obs3[:], rewards, self.dones, infos = self.env.step(actions)
          
            mb_actions.append(actions)
            mb_dones.append(self.dones)
                 
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        for _ in range(self.nsteps):
            actions = self.model.step4(self.obs4, S=self.states, M=self.dones)
            mb_obs.append(self.obs5.copy())
            self.obs5[:], rewards, self.dones, infos = self.env.step(actions)
        
            mb_actions.append(actions)
            mb_dones.append(self.dones)
           
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        for _ in range(self.nsteps):
            actions = self.model.step5(self.obs5, S=self.states, M=self.dones)
            mb_obs.append(self.obs5.copy())
            self.obs5[:], rewards, self.dones, infos = self.env.step(actions)
            
            mb_actions.append(actions)
            mb_dones.append(self.dones)
                 
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards/2)
        for _ in range(self.nsteps):
            actions = self.model.target_step(self.obs, S=self.states, M=self.dones)
            mb_obs.append(self.obs.copy())
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
          
            mb_actions.append(actions)
            mb_dones.append(self.dones)
          
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        
       
        return (*map(sf01, (mb_obs, mb_dones, mb_actions)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()


    def run_eval(self):
        # Here, we init the lists that will contain the mb of experiences
    
        mb_states = self.states
        epinfos = []
        
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            
            actions = self.model.new_step(self.new_obs, S=self.states, M=self.dones)
            self.new_obs, rewards, self.dones, infos = self.env.step(actions)
            #_ , new_rewards, _ , new_infos = self.env.step(new_actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            
        return epinfos
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
