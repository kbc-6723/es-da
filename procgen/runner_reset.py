import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=nsteps)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma
  
    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        
        # For n in range number of steps
        for k in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
  
            
            if k > 0:
                ind = []
                for i in range(self.nenv):
                    if self.dones[i] and (rewards[i] <= 0):
                        ind.append(i)
                mb_obs = np.concatenate((mb_obs,np.delete(self.obs, ind ,0)),0)
            else:
                mb_obs = self.obs[:]  
          
            
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
           
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)       

        return (mb_obs,mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()


    def run_eval(self):
        # Here, we init the lists that will contain the mb of experiences
    
        mb_states = self.states
        epinfos = []
        
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, self.states, neglogpacs = self.model.new_step(self.obs, S=self.states, M=self.dones)
            
            #new_actions, new_values, self.new_states, new_neglogpacs = self.model.new_step(self.obs, S=self.states, M=self.dones)
            
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
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


