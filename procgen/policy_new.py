import tensorflow as tf
from procgen import tf_util
from baselines.a2c.utils import fc
from procgen.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder

import gym


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent(latent,init_scale=0.01)
        self.softout = tf.nn.softmax(self.pi)
        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)
        
    def get_softmax(self, ob, **extra_feed):
        return self._evaluate([self.softout, self.pi], ob,**extra_feed)
    
    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

class PolicyWithValue_distill(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, observations2, latent, latent2, estimate_q=False, vf_latent=None, vf_latent2=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.X2 = observations2
        self.state = tf.constant([])
        
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent
        vf_latent2 = vf_latent2 if vf_latent2 is not None else latent2
        
        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        vf_latent2 = tf.layers.flatten(vf_latent2)
        latent2 = tf.layers.flatten(latent2)
        
        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)
        self.aug_pdtype = make_pdtype(env.action_space)
        self.pd, self.pi = self.pdtype.pdfromlatent(latent,init_scale=0.01)
        self.aug_pd, self.aug_pi = self.pdtype.pdfromlatent(latent2,init_scale=0.01)
        self.softout = tf.nn.softmax(self.pi)
        self.aug_softout = tf.nn.softmax(self.aug_pi)
        # Take an action
        self.action = self.pd.sample()
        self.aug_action = self.aug_pd.sample()
        # Calculate the neg log of our probability
        self.aug_neglogp = self.pd.neglogp(self.aug_action)
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]
            self.vf2 = fc(vf_latent2, 'vf2', 1)
            self.aug_vf = self.vf2[:,0]
    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)
        
    def get_softmax(self, ob, **extra_feed):
        return self._evaluate([self.softout, self.pi], ob,**extra_feed)
    
    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


class PolicyWithValue_new(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, estimate_q=False, vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent

        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)

        self.pd, self.pi = self.pdtype.pdfromlatent_sample(latent,'new_pi', init_scale=0.01)# reset
        self.softout = tf.nn.softmax(self.pi)
        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'new_q', env.action_space.n)
            self.vf = self.q
        else:
            self.vf = fc(vf_latent, 'new_vf', 1)
            self.vf = self.vf[:,0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)
        
    def get_softmax(self, ob, **extra_feed):
        return self._evaluate([self.softout, self.pi], ob,**extra_feed)
    
    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)
       
class PolicyWithValue_reset(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, new_latent, estimate_q=False, new_vf_latent=None,vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.new_state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent
        new_vf_latent = new_vf_latent if new_vf_latent is not None else new_latent
        
        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)


        new_vf_latent = tf.layers.flatten(new_vf_latent)
        new_latent = tf.layers.flatten(new_latent)
        
        
        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)
        self.new_pdtype = make_pdtype(env.action_space)
        self.pd, self.pi = self.pdtype.pdfromlatent(latent, init_scale=0.01) 
        self.new_pd, self.new_pi = self.new_pdtype.pdfromlatent(new_latent, init_scale=0.01)# reset
        self.new_softout = tf.nn.softmax(self.new_pi)
        # Take an action
        self.action = self.pd.sample()
        self.new_action = self.new_pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.new_neglogp = self.pd.neglogp(self.new_action)
        
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.new_q=fc(new_vf_latent, 'q', env.action_space.n)
            
            self.vf = self.q
            self.new_vf = self.new_q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]
            self.new_vf = fc(new_vf_latent, 'vf', 1)
            self.new_vf = self.new_vf[:,0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)
    
    def get_softmax(self, ob, **extra_feed):
        return self._evaluate([self.new_softout, self.new_pi], ob,**extra_feed)
        
    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp
        
    def new_step(self, observation, **extra_feed):
        n_a, n_v, n_state, n_neglogp = self._evaluate([self.new_action, self.new_vf, self.new_state, self.new_neglogp], observation, **extra_feed)
        if n_state.size == 0:
            n_state = None
        return n_a, n_v, n_state, n_neglogp
    
    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def new_value(self, ob, *args, **kwargs):
        return self._evaluate(self.new_vf, ob, *args, **kwargs)
        
    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

class PolicyWithValue_sample(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, new_latent, estimate_q=False, new_vf_latent=None,vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.new_state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent
        new_vf_latent = new_vf_latent if new_vf_latent is not None else new_latent
        
        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)


        new_vf_latent = tf.layers.flatten(new_vf_latent)
        new_latent = tf.layers.flatten(new_latent)
        
        
        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)
        self.new_pdtype = make_pdtype(env.action_space)
        self.pd, self.pi = self.pdtype.pdfromlatent_sample(latent,'pi', init_scale=0.01)
        self.new_pd, self.new_pi = self.new_pdtype.pdfromlatent_sample(new_latent,'new_pi', init_scale=0.01)# reset
        
        # Take an action
        self.action = self.pd.sample()
        self.new_action = self.new_pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.new_neglogp = self.new_pd.neglogp(self.new_action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.new_q=fc(new_vf_latent, 'new_q', env.action_space.n)
            self.vf = self.q
            self.new_vf = self.new_q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]
            self.new_vf = fc(new_vf_latent, 'new_vf', 1)
            self.new_vf = self.new_vf[:,0]
            
    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp
        
    def new_step(self, observation, **extra_feed):
        n_a, n_v, n_state, n_neglogp = self._evaluate([self.new_action, self.new_vf, self.new_state, self.new_neglogp], observation, **extra_feed)
        if n_state.size == 0:
            n_state = None
        return n_a, n_v, n_state, n_neglogp
    
    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def new_value(self, ob, *args, **kwargs):
        return self._evaluate(self.new_vf, ob, *args, **kwargs)
    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

class PolicyWithValue_aug(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, observations2, latent, new_latent, estimate_q=False, new_vf_latent=None,vf_latent=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.X2 = observations2
        self.state = tf.constant([])
        self.new_state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent
        new_vf_latent = new_vf_latent if new_vf_latent is not None else new_latent
        
        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)


        new_vf_latent = tf.layers.flatten(new_vf_latent)
        new_latent = tf.layers.flatten(new_latent)
        
        
        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)
        self.new_pdtype = make_pdtype(env.action_space)
        self.pd, self.pi = self.pdtype.pdfromlatent_sample(latent,'pi', init_scale=0.01)
        self.new_pd, self.new_pi = self.new_pdtype.pdfromlatent_sample(new_latent,'new_pi', init_scale=0.01)# reset
        
        # Take an action
        self.action = self.pd.sample()
        self.new_action = self.new_pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)
        self.new_neglogp = self.new_pd.neglogp(self.new_action)
        self.sess = sess or tf.get_default_session()

        if estimate_q:
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.q = fc(vf_latent, 'q', env.action_space.n)
            self.new_q=fc(new_vf_latent, 'new_q', env.action_space.n)
            self.vf = self.q
            self.new_vf = self.new_q
        else:
            self.vf = fc(vf_latent, 'vf', 1)
            self.vf = self.vf[:,0]
            self.new_vf = fc(new_vf_latent, 'new_vf', 1)
            self.new_vf = self.new_vf[:,0]
            
    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)
    def _evaluate2(self, variables, observation2, **extra_feed):
        sess = self.sess
        feed_dict = {self.X2: adjust_shape(self.X2, observation2)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)
    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp
        
    def new_step(self, observation2, **extra_feed):
        n_a, n_v, n_state, n_neglogp = self._evaluate2([self.new_action, self.new_vf, self.new_state, self.new_neglogp], observation2 , **extra_feed)
        if n_state.size == 0:
            n_state = None
        return n_a, n_v, n_state, n_neglogp
    
    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def new_value(self, ob, *args, **kwargs):
        return self._evaluate2(self.new_vf, ob, *args, **kwargs)
    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)


class PolicyWithValue_multiple_sample(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self, env, observations, latent, latent1, latent2, latent3, latent4, latent5, new_latent,target_latent, estimate_q=False, new_vf_latent=None,vf_latent=None,target_vf_latent=None,vf_latent1=None, vf_latent2=None,vf_latent3=None,vf_latent4=None, vf_latent5=None, sess=None, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.state = tf.constant([])
        self.target_state = tf.constant([])
        self.new_state = tf.constant([])
        self.initial_state = None
        self.__dict__.update(tensors)

        vf_latent = vf_latent if vf_latent is not None else latent
        vf_latent1 = vf_latent1 if vf_latent1 is not None else latent1
        vf_latent2 = vf_latent2 if vf_latent2 is not None else latent2
        vf_latent3 = vf_latent3 if vf_latent3 is not None else latent3
        vf_latent4 = vf_latent4 if vf_latent4 is not None else latent4
        vf_latent5 = vf_latent5 if vf_latent5 is not None else latent5
        new_vf_latent = new_vf_latent if new_vf_latent is not None else new_latent
        target_vf_latent = target_vf_latent if target_vf_latent is not None else target_latent
        
        
        vf_latent = tf.layers.flatten(vf_latent)
        latent = tf.layers.flatten(latent)

        vf_latent1 = tf.layers.flatten(vf_latent1)
        latent1 = tf.layers.flatten(latent1)
      
        vf_latent2 = tf.layers.flatten(vf_latent2)
        latent2 = tf.layers.flatten(latent2)

        vf_latent3 = tf.layers.flatten(vf_latent3)
        latent3 = tf.layers.flatten(latent3)

        vf_latent4 = tf.layers.flatten(vf_latent4)
        latent4 = tf.layers.flatten(latent4)
        
        vf_latent5 = tf.layers.flatten(vf_latent5)
        latent5 = tf.layers.flatten(latent5)

        new_vf_latent = tf.layers.flatten(new_vf_latent)
        new_latent = tf.layers.flatten(new_latent)
        
        target_vf_latent = tf.layers.flatten(target_vf_latent)
        target_latent = tf.layers.flatten(target_latent)
        
        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(env.action_space)
        
        self.pd, self.pi = self.pdtype.pdfromlatent_sample(latent,'pi', init_scale=0.01)
        self.pd1, self.pi1 = self.pdtype.pdfromlatent_sample(latent1,'sample1_pi', init_scale=0.01)
        self.pd2, self.pi2 = self.pdtype.pdfromlatent_sample(latent2,'sample2_pi', init_scale=0.01)
        self.pd3, self.pi3 = self.pdtype.pdfromlatent_sample(latent3,'sample3_pi', init_scale=0.01)
        self.pd4, self.pi4 = self.pdtype.pdfromlatent_sample(latent4,'sample4_pi', init_scale=0.01)
        self.pd5, self.pi5 = self.pdtype.pdfromlatent_sample(latent5,'sample5_pi', init_scale=0.01)
        self.target_pd, self.target_pi = self.pdtype.pdfromlatent_sample(target_latent,'target_pi', init_scale=0.01)
        self.new_pd, self.new_pi = self.pdtype.pdfromlatent_sample(new_latent,'new_pi', init_scale=0.01)# reset
        self.new_softout = tf.nn.softmax(self.new_pi)
        # Take an action
        self.action = self.pd.sample()
        self.action1 = self.pd1.sample()
        self.action2 = self.pd2.sample()
        self.action3 = self.pd3.sample()
        self.action4 = self.pd4.sample()
        self.action5 = self.pd5.sample()
        self.new_action = self.new_pd.sample()
        self.target_action = self.target_pd.sample()

        # Calculate the neg log of our probability
        
        self.sess = sess or tf.get_default_session()

        
        self.vf = fc(vf_latent, 'vf', 1)
        self.vf = self.vf[:,0]
        self.vf1 = fc(vf_latent1, 'sample1_vf', 1)
        self.vf1 = self.vf1[:,0]
        self.vf2 = fc(vf_latent2, 'sample2_vf', 1)
        self.vf2 = self.vf2[:,0]
        self.vf3 = fc(vf_latent3, 'sample3_vf', 1)
        self.vf3 = self.vf3[:,0]
        self.vf4 = fc(vf_latent4, 'sample4_vf', 1)
        self.vf4 = self.vf4[:,0]
        self.vf5 = fc(vf_latent5, 'sample5_vf', 1)
        self.vf5 = self.vf5[:,0]
        self.new_vf = fc(new_vf_latent, 'new_vf', 1)
        self.new_vf = self.new_vf[:,0]
        self.target_vf = fc(target_vf_latent, 'target_vf', 1)
        self.target_vf = self.target_vf[:,0]
            
    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)
    def get_softmax(self, ob, **extra_feed):
        return self._evaluate([self.new_softout, self.new_pi], ob,**extra_feed)
    def step(self, observation, **extra_feed):
        """
        Compute next action(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        (action, value estimate, next state, negative log likelihood of the action under current policy parameters) tuple
        """

        a, v, state, neglogp = self._evaluate([self.action, self.vf, self.state, self.neglogp], observation, **extra_feed)
        if state.size == 0:
            state = None
        return a, v, state, neglogp
    
    def step1(self, observation, **extra_feed):
        a1 = self._evaluate(self.action1, observation, **extra_feed)
        return a1
        
    def step2(self, observation, **extra_feed):
        a2 = self._evaluate(self.action2, observation, **extra_feed)
        return a2
    
    def step3(self, observation, **extra_feed):
        a3 = self._evaluate(self.action3, observation, **extra_feed)
        return a3
        
    def step4(self, observation, **extra_feed):
        a4 = self._evaluate(self.action4, observation, **extra_feed)
        return a4   
          
    def step5(self, observation, **extra_feed):
        a5 = self._evaluate(self.action5, observation, **extra_feed)
        return a5
    
    def new_step(self, observation, **extra_feed):
        n_a = self._evaluate(self.new_action, observation, **extra_feed)
       
        return n_a
    def target_step(self, observation, **extra_feed):
        t_a = self._evaluate(self.target_action, observation, **extra_feed)
    
        return t_a
    
    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def new_value(self, ob, *args, **kwargs):
        return self._evaluate(self.new_vf, ob, *args, **kwargs)
    def target_value(self, ob, *args, **kwargs):
        return self._evaluate(self.target_vf, ob, *args, **kwargs)    
    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)



def build_policy(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn

def build_policy_distill(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        X2 = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X
            encoded_x2 = X2
        encoded_x = encode_observation(ob_space, encoded_x)
        encoded_x2 = encode_observation(ob_space, encoded_x2)
        
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            policy_latent2 = policy_network(encoded_x2)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent
                policy_latent2, recurrent_tensors2 = policy_latent2
                
                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)
                    
                if recurrent_tensors2 is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent2, recurrent_tensors2 = policy_network(encoded_x2, nenv)
                    extra_tensors.update(recurrent_tensors2)



        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
            vf_latent2 = policy_latent2
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)
                vf_latent2 = _v_net(encoded_x2)

        policy = PolicyWithValue_distill(
            env=env,
            observations=X,
            observations2=X2,
            latent=policy_latent,
            latent2 = policy_latent2,
            vf_latent=vf_latent,
            vf_latent2=vf_latent2,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn

def build_policy_aug(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        X2 = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X
            encoded_x2 = X2

        encoded_x = encode_observation(ob_space, encoded_x)
        encoded_x2 = encode_observation(ob_space, encoded_x2)
        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)
                    
        with tf.variable_scope('new_impala', reuse=tf.AUTO_REUSE):
            new_policy_latent = policy_network(encoded_x2)
            if isinstance(new_policy_latent, tuple):
                new_policy_latent, new_recurrent_tensors = new_policy_latent

                if new_recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    new_policy_latent, new_recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(new_recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
            new_vf_latent = new_policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network     
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)
                
            with tf.variable_scope('new_vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                new_vf_latent = _v_net(encoded_x2)


        policy = PolicyWithValue_aug(
            env=env,
            observations=X,
            observations2=X2,
            latent=policy_latent,
            new_latent=new_policy_latent,
            vf_latent=vf_latent,
            new_vf_latent=new_vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def build_policy_new(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('new_impala', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)


        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)

        policy = PolicyWithValue_new(
            env=env,
            observations=X,
            latent=policy_latent,
            vf_latent=vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn
def build_policy_reset(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)

        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)

        with tf.variable_scope('new_impala', reuse=tf.AUTO_REUSE):
            new_policy_latent = policy_network(encoded_x)
            if isinstance(new_policy_latent, tuple):
                new_policy_latent, new_recurrent_tensors = new_policy_latent

                if new_recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    new_policy_latent, new_recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(new_recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
            new_vf_latent = new_policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network     
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)
            
            with tf.variable_scope('new_vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                new_vf_latent = _v_net(encoded_x)


        policy = PolicyWithValue_reset(
            env=env,
            observations=X,
            latent=policy_latent,
            new_latent=new_policy_latent,
            vf_latent=vf_latent,
            new_vf_latent=new_vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn


def build_policy_sample(env, policy_network, value_network=None,  normalize_observations=False, estimate_q=False, **policy_kwargs):
    if isinstance(policy_network, str):
        network_type = policy_network
        policy_network = get_network_builder(network_type)(**policy_kwargs)

    def policy_fn(nbatch=None, nsteps=None, sess=None, observ_placeholder=None):
        ob_space = env.observation_space

        X = observ_placeholder if observ_placeholder is not None else observation_placeholder(ob_space, batch_size=nbatch)
        
        extra_tensors = {}

        if normalize_observations and X.dtype == tf.float32:
            encoded_x, rms = _normalize_clip_observation(X)
            extra_tensors['rms'] = rms
        else:
            encoded_x = X

        encoded_x = encode_observation(ob_space, encoded_x)

        with tf.variable_scope('impala', reuse=tf.AUTO_REUSE):
            policy_latent = policy_network(encoded_x)
            if isinstance(policy_latent, tuple):
                policy_latent, recurrent_tensors = policy_latent

                if recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    policy_latent, recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(recurrent_tensors)
                    
        with tf.variable_scope('new_impala', reuse=tf.AUTO_REUSE):
            new_policy_latent = policy_network(encoded_x)
            if isinstance(new_policy_latent, tuple):
                new_policy_latent, new_recurrent_tensors = new_policy_latent

                if new_recurrent_tensors is not None:
                    # recurrent architecture, need a few more steps
                    nenv = nbatch // nsteps
                    assert nenv > 0, 'Bad input for recurrent policy: batch size {} smaller than nsteps {}'.format(nbatch, nsteps)
                    new_policy_latent, new_recurrent_tensors = policy_network(encoded_x, nenv)
                    extra_tensors.update(new_recurrent_tensors)

        _v_net = value_network

        if _v_net is None or _v_net == 'shared':
            vf_latent = policy_latent
            new_vf_latent = new_policy_latent
        else:
            if _v_net == 'copy':
                _v_net = policy_network     
            else:
                assert callable(_v_net)

            with tf.variable_scope('vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                vf_latent = _v_net(encoded_x)
                
            with tf.variable_scope('new_vf', reuse=tf.AUTO_REUSE):
                # TODO recurrent architectures are not supported with value_network=copy yet
                new_vf_latent = _v_net(encoded_x)


        policy = PolicyWithValue_sample(
            env=env,
            observations=X,
            latent=policy_latent,
            new_latent=new_policy_latent,
            vf_latent=vf_latent,
            new_vf_latent=new_vf_latent,
            sess=sess,
            estimate_q=estimate_q,
            **extra_tensors
        )
        return policy

    return policy_fn

def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms

