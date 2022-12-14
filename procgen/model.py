from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import functools
import numpy as np
from procgen.tf_util import get_session, save_variables, load_variables 
from procgen.tf_util import initialize
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
from mpi4py import MPI
from baselines.common.mpi_util import sync_from_root
from collections import deque
from procgen.PCGrad_tf import PCGrad, PDTGrad, PDTGrad_InDA
'''
try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None
'''

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


def kl(a,b):
    a0 = a - tf.reduce_max(a, axis=-1, keepdims=True)
    a1 = b - tf.reduce_max(b, axis=-1, keepdims=True)
    ea0 = tf.exp(a0)
    ea1 = tf.exp(a1)
    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
    z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,stu_policy = None, args = None,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
       
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        
        if comm is not None and comm.Get_size() > 1:
                self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        
        if args.use_pcgrad:
            self.trainer = PCGrad(self.trainer)
        elif args.use_pdtgrad:
            self.trainer = PDTGrad(self.trainer)    

        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actorcc
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        
        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        ratio_loss = tf.reduce_mean(tf.minimum(ratio, tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        
        if args.use_rand_FM:
            fm_loss = tf.losses.mean_squared_error(labels=tf.stop_gradient(train_model.latent2), predictions=train_model.latent)
            loss += fm_loss * 0.0002
        if args.use_drac:
            pi_loss = tf.reduce_mean(train_model.aug_pd.kl(train_model.pd))
            vd_loss = tf.reduce_mean(tf.square(vpred-train_model.aug_vf))
            if args.use_pcgrad or args.use_pdtgrad:
                losses = [loss, (pi_loss + vd_loss) * 0.1]
                # losses = [loss, pi_loss + vd_loss]
            elif args.only_distill:
                loss = (pi_loss + vd_loss) * 0.1
            else:
                loss += (pi_loss + vd_loss) * 0.1
            
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        ppo_params = tf.trainable_variables('ppo2_model')
        
        # 2. Build our trainer
        
        # 3. Calculate the gradients
        
        if args.use_pcgrad or args.use_pdtgrad:
            grads_and_var = self.trainer.compute_gradients(losses, ppo_params)
        else:
            grads_and_var = self.trainer.compute_gradients(loss, ppo_params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            #_, policy_grad_norm = tf.clip_by_global_norm(policy_grads, max_grad_norm)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
        
        
        self.grads = grads
        self.grads_norm = _grad_norm
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        if args.use_rand_FM:
            self.loss_names = ['loss', 'policy_loss', 'value_loss','fm_loss','policy_entropy', 'approxkl', 'clipfrac', 'grad']
            self.stats_list = [loss, pg_loss, vf_loss, fm_loss, entropy, approxkl, clipfrac, _grad_norm]
        elif args.use_drac:
            self.loss_names = ['loss', 'policy_loss', 'value_loss','pi_loss','vd_loss','policy_entropy', 'approxkl', 'clipfrac', 'grad']
            self.stats_list = [loss, pg_loss, vf_loss, pi_loss, vd_loss, entropy, approxkl, clipfrac, _grad_norm]
        else:    
            self.loss_names = ['loss', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'grad']
            self.stats_list = [loss, pg_loss, vf_loss, entropy, approxkl, clipfrac, _grad_norm]
                
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = functools.partial(save_variables, sess=sess)
       
        self.load = functools.partial(load_variables, sess=sess)
        self.conv_out = act_model.conv_out
        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
    def train_drac(self, lr, cliprange, obs, aug_obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.train_model.augmented_X : aug_obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]


    
    
class ExDA(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train, eval_load = False , args = None,
                nsteps, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD
        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
   
        self.LR = LR = tf.placeholder(tf.float32, [])
        self.P = P = train_model.pdtype.param_placeholder([None])
        self.V = V = tf.placeholder(tf.float32, [None])
        if comm is not None and comm.Get_size() > 1:
                self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR)
        
       
        pd_loss = tf.reduce_mean(kl(P, train_model.pi))
        distill_loss = pd_loss
        
        if args.use_vdf:
            self.use_vdf = True
            vd_loss = tf.reduce_mean(tf.square(V-train_model.vf)) # value distance loss
            distill_loss += vd_loss
        else:
            self.use_vdf = False
        total_params = tf.trainable_variables('ppo2_model')

        distill_grads_and_var = self.trainer.compute_gradients(distill_loss, total_params)        

        self.total_params = total_params
        self._distill_train_op = self.trainer.apply_gradients(distill_grads_and_var)
        self.step = act_model.step
        self.pi = train_model.policy
        self.value = train_model.value
        self.distill_loss_names = ['distill_loss', 'pd_loss']
        self.distill_stats_list = [distill_loss, pd_loss]
        if args.use_vdf:
            self.distill_stats_list = [distill_loss, pd_loss, vd_loss]
            self.distill_loss_names = ['distill_loss', 'pd_loss', 'vd_loss']

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)
        
        self.initial_state = act_model.initial_state
        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101
        self.conv_out = act_model.conv_out
    def distill(self,lr, obs, policy, value = None):
        
        td_map = {
            self.train_model.X : obs,
            self.P : policy,
            self.LR : lr,
        }
        if self.use_vdf:
            td_map[self.V] = value
        
        return self.sess.run(
            self.distill_stats_list + [self._distill_train_op], # reset
            td_map
        )[:-1]
    
    def initialize(self):
        return self.sess.run(tf.variables_initializer(self.total_params))
    
  


class InDA(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,stu_policy = None, args = None,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)
            
            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
       
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        
        if comm is not None and comm.Get_size() > 1:
                self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
                self.distill_trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            self.distill_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)

        if args.use_pdtgrad:
            self.distill_trainer = PDTGrad_InDA(self.distill_trainer)

        self.P = P = train_model.pdtype.param_placeholder([None])
        self.V = V = train_model.pdtype.sample_placeholder([None])
        self.L = L = tf.placeholder(tf.float32, [None] + [256])
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        
        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        ratio_loss = tf.reduce_mean(tf.minimum(ratio, tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        
        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        total_params = tf.trainable_variables('ppo2_model')
        self.G = G = tf.placeholder(tf.float32, shape = (626256))
        grads_and_var = self.trainer.compute_gradients(loss, total_params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            #_, policy_grad_norm = tf.clip_by_global_norm(policy_grads, max_grad_norm)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

          
  
        vd_loss = tf.losses.mean_squared_error(V,train_model.vf)
        pd_loss = tf.reduce_mean(kl(P, train_model.pi))
        distill_loss = pd_loss + vd_loss
  
        if args.use_pdtgrad:
            distill_grads_and_var = self.distill_trainer.compute_gradients(distill_loss, G, total_params)   
        else:
            distill_grads_and_var = self.distill_trainer.compute_gradients(distill_loss, total_params)        
        
        self._distill_train_op = self.distill_trainer.apply_gradients(distill_grads_and_var)
        self.grads = grads
        self.grads_norm = _grad_norm
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)         
        if args.use_pdtgrad:
            self.loss_names = ['loss', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'grad_norm']
            self.stats_list = [loss, pg_loss, vf_loss, entropy, approxkl, clipfrac, _grad_norm, grads]
        else:
            self.loss_names = ['loss', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'grad_norm']
            self.stats_list = [loss, pg_loss, vf_loss, entropy, approxkl, clipfrac, _grad_norm]

        self.step = act_model.step
       
        self.distill_stats_list = [distill_loss, pd_loss, vd_loss]
        self.distill_loss_names = ['distill_loss', 'pd_loss', 'vd_loss']
     
        self.train_model = train_model
        self.act_model = act_model
        self.value = act_model.value
        self.vf = train_model.value
        self.pi = train_model.policy
       
        self.initial_state = act_model.initial_state
        self.save = functools.partial(save_variables, sess=sess)
        self.load = self.load = functools.partial(load_variables, sess=sess)
        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101
    
    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
        
    def distill(self,lr, obs, policy, value):
        
        td_map = {
            self.train_model.X : obs,
            self.P : policy,
            self.V : value,
            self.LR : lr,
        }
        
        return self.sess.run(
            self.distill_stats_list + [self._distill_train_op], # reset
            td_map
        )[:-1]
    
    def distill_pdt(self, lr, train_grads, obs,  policy, value):

        td_map = {
            self.train_model.X : obs,
            self.P : policy,
            self.V : value,
            self.LR : lr,
            self.G : train_grads,
        }

        return self.sess.run(
            self.distill_stats_list + [self._distill_train_op], # reset
            td_map
        )[:-1]
 
class UCB_InDA(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space,num_aug_types,ucb_window_length,ucb_exploration_coef,aug_list, nbatch_act, nbatch_train,stu_policy = None, args = None,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)
            
            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
       
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        
        if comm is not None and comm.Get_size() > 1:
                self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
                self.distill_trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
            self.distill_trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        self.P = P = train_model.pdtype.param_placeholder([None])
        self.V = V = train_model.pdtype.sample_placeholder([None])
        self.L = L = tf.placeholder(tf.float32, [None] + [256])
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actor
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        
        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        ratio_loss = tf.reduce_mean(tf.minimum(ratio, tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        
        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        total_params = tf.trainable_variables('ppo2_model')
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        
    
        # 2. Build our trainer
        
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, total_params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            #_, policy_grad_norm = tf.clip_by_global_norm(policy_grads, max_grad_norm)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
       
        #vd_loss = tf.reduce_mean(tf.square(train_model.new_vf-tf.stop_gradient(train_model.vf))) # value distance loss
         
        
        vd_loss = tf.losses.mean_squared_error(V,train_model.vf)
        pd_loss = tf.reduce_mean(kl(P, train_model.pi))
        distill_loss = pd_loss + vd_loss
  
        
             
        distill_grads_and_var = self.distill_trainer.compute_gradients(distill_loss, total_params)        
        
        
        self.ucb_exploration_coef = ucb_exploration_coef
        self.ucb_window_length = ucb_window_length

        self.num_aug_types = num_aug_types
        self.total_num = 1
        self.num_action = np.array([1.] * self.num_aug_types) 
        self.qval_action = np.array([0.] * self.num_aug_types) 

        self.expl_action = np.array([0.] * self.num_aug_types)  
        self.ucb_action = np.array([0.] * self.num_aug_types) 
    
        
        if args.swucb:
            self.arm_return = deque(maxlen=args.sliding_window)
            self.delta_return_action = np.zeros(num_aug_types)
        elif args.ducb:
            self.delta_return_action = np.zeros(num_aug_types)
        else:
            self.delta_return_action = []
            for i in range(num_aug_types):
                self.delta_return_action.append(deque(maxlen=ucb_window_length))
        self.round = 0
        self.aug_list = aug_list
        self._distill_train_op = self.distill_trainer.apply_gradients(distill_grads_and_var)
        self.grads = grads
        self.grads_norm = _grad_norm
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)         
        self.loss_names = ['loss', 'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac', 'grad']
        self.stats_list = [loss, pg_loss, vf_loss, entropy, approxkl, clipfrac, _grad_norm]
        self.step = act_model.step
       
        self.distill_stats_list = [distill_loss, pd_loss, vd_loss]
        self.distill_loss_names = ['distill_loss', 'pd_loss', 'vd_loss']
        self.df_type = args.df_type    
        self.train_model = train_model
        self.act_model = act_model
        self.value = act_model.value
        self.vf = train_model.value
        self.pi = train_model.policy
       
        self.initial_state = act_model.initial_state
        self.save = functools.partial(save_variables, sess=sess)
        self.load = self.load = functools.partial(load_variables, sess=sess)
        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101
    
    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]
        
    def distill(self,lr, obs, policy, value):
        
        td_map = {
            self.train_model.X : obs,
            self.P : policy,
            self.V : value,
            self.LR : lr,
        }
        
        return self.sess.run(
            self.distill_stats_list + [self._distill_train_op], # reset
            td_map
        )[:-1]
        
    def select_ucb_aug(self):
        for i in range(self.num_aug_types):
            self.expl_action[i] = self.ucb_exploration_coef * \
                np.sqrt( np.log(self.total_num)  / self.num_action[i])
            self.ucb_action[i] = self.qval_action[i] + self.expl_action[i]
        ucb_aug_id = np.argmax(self.ucb_action)
        
        print(self.ucb_exploration_coef)
        print(self.expl_action)
        print(self.qval_action)
        print(self.ucb_action)
        return ucb_aug_id, self.aug_list[ucb_aug_id]

    def update_ucb_values(self, current_aug_id, delta_return):
        self.total_num += 1
        self.num_action[current_aug_id] += 1
        self.delta_return_action[current_aug_id].append(delta_return)
        self.qval_action[current_aug_id] = np.mean(self.delta_return_action[current_aug_id])
        if self.total_num < self.num_aug_types * self.ucb_window_length * 10:
            min_select = np.argmin(self.qval_action)
            max_select = np.argmax(self.qval_action)
            self.ucb_exploration_coef = 1.01 * (self.qval_action[max_select]- self.qval_action[min_select]) / (np.sqrt(np.log(self.total_num)) * np.maximum((1/np.sqrt(self.num_action[min_select]) -
             1/np.sqrt(self.num_action[max_select])),(1/np.sqrt(self.ucb_window_length - 1) -
             1/np.sqrt(self.ucb_window_length)) )) 
    '''
    def update_ducb_values(self, current_aug_id, delta_return):
        self.round += 1
        discount_factor = 1 - 1 / (4 * np.sqrt(self.round))
        self.total_num *= discount_factor
        self.num_action *= discount_factor
        self.delta_return_action *= discount_factor
        self.total_num += 1
        self.num_action[current_aug_id] += 1
        self.delta_return_action[current_aug_id] += delta_return
        self.qval_action[current_aug_id] = self.delta_return_action[current_aug_id] / self.num_action[current_aug_id]

    def update_swucb_values(self, current_aug_id, delta_return):
        sw_size = 4 * np.sqrt(2* np.log(self.round))
        if len(self.arm_return) == sw_size:
            pop_aug_id, pop_aug_return = self.arm_return.popleft()
            self.num_action[pop_aug_id] -= 1
            self.delta_return_action[pop_aug_id] -= pop_aug_return 
            self.qval_action[pop_aug_id] = self.delta_return_action[pop_aug_id] / self.num_action[pop_aug_id]
        else: 
            self.total_num += 1
        self.num_action[current_aug_id] += 1
        self.delta_return_action[current_aug_id] += delta_return
        self.qval_action[current_aug_id] = self.delta_return_action[current_aug_id] / self.num_action[current_aug_id]
    '''

class UCBDrAC(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,stu_policy = None, args = None,
                nsteps, ent_coef, vf_coef, max_grad_norm, mpi_rank_weight=1, ucb_coef = 1.5, comm=None, microbatch_size=None):
        self.sess = sess = get_session()

        if MPI is not None and comm is None:
            comm = MPI.COMM_WORLD

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # CREATE OUR TWO MODELS
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # CREATE THE PLACEHOLDERS
       
        self.LR = LR = tf.placeholder(tf.float32, [])
        # Cliprange
        self.CLIPRANGE = CLIPRANGE = tf.placeholder(tf.float32, [])
        
        if comm is not None and comm.Get_size() > 1:
                self.trainer = MpiAdamOptimizer(comm, learning_rate=LR, mpi_rank_weight=mpi_rank_weight, epsilon=1e-5)
        else:
            self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        
        self.aug_list = [ 'rccj', None, 'crop']
        self.A = A = train_model.pdtype.sample_placeholder([None])
        self.ADV = ADV = tf.placeholder(tf.float32, [None])
        self.R = R = tf.placeholder(tf.float32, [None])
        # Keep track of old actorcc
        self.OLDNEGLOGPAC = OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        self.OLDVPRED = OLDVPRED = tf.placeholder(tf.float32, [None])
        
        neglogpac = train_model.pd.neglogp(A)

        # Calculate the entropy
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # CALCULATE THE LOSS
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped value
        vf_losses1 = tf.square(vpred - R)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - R)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -ADV * ratio

        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        ratio_loss = tf.reduce_mean(tf.minimum(ratio, tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        
        
        pi_loss = tf.reduce_mean(train_model.aug_pd.kl(train_model.pd))
        vd_loss = tf.reduce_mean(tf.square(vpred-train_model.aug_vf))
        
        loss += (pi_loss + vd_loss) * 0.1
           
        # UPDATE THE PARAMETERS USING LOSS
        # 1. Get the model parameters
        ppo_params = tf.trainable_variables('ppo2_model')
        
        # 2. Build our trainer
        
        # 3. Calculate the gradients
        
       
        grads_and_var = self.trainer.compute_gradients(loss, ppo_params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            #_, policy_grad_norm = tf.clip_by_global_norm(policy_grads, max_grad_norm)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
        self.ucb_exploration_coef = ucb_coef
      

        self.num_aug_types = 3
        self.ucb_window_length = 10
        self.total_num = 1
        self.num_action = np.array([1.] * self.num_aug_types) 
        self.qval_action = np.array([0.] * self.num_aug_types) 

        self.expl_action = np.array([0.] * self.num_aug_types)  
        self.ucb_action = np.array([0.] * self.num_aug_types) 
    
        self.return_action = []
        for i in range(self.num_aug_types):
            self.return_action.append(deque(maxlen=self.ucb_window_length))
        self.grads = grads
        self.grads_norm = _grad_norm
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        
        self.loss_names = ['loss', 'policy_loss', 'value_loss','pi_loss','vd_loss','policy_entropy', 'approxkl', 'clipfrac', 'grad']
        self.stats_list = [loss, pg_loss, vf_loss, pi_loss, vd_loss, entropy, approxkl, clipfrac, _grad_norm]
                
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = functools.partial(save_variables, sess=sess)
       
        self.load = functools.partial(load_variables, sess=sess)
        self.conv_out = act_model.conv_out
        initialize()
        global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        if MPI is not None:
            sync_from_root(sess, global_variables, comm=comm) #pylint: disable=E1101

    def train(self, lr, cliprange, obs, aug_obs, returns, masks, actions, values, neglogpacs, states=None):
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.train_model.X : obs,
            self.train_model.augmented_X : aug_obs,
            self.A : actions,
            self.ADV : advs,
            self.R : returns,
            self.LR : lr,
            self.CLIPRANGE : cliprange,
            self.OLDNEGLOGPAC : neglogpacs,
            self.OLDVPRED : values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(
            self.stats_list + [self._train_op],
            td_map
        )[:-1]

    def select_ucb_aug(self):
        for i in range(3):
            self.expl_action[i] = self.ucb_exploration_coef * \
                np.sqrt(np.log(self.total_num) / self.num_action[i])
            self.ucb_action[i] = self.qval_action[i] + self.expl_action[i]
        ucb_aug_id = np.argmax(self.ucb_action)
        print(self.qval_action, self.expl_action, self.ucb_action)
        
        return ucb_aug_id, self.aug_list[ucb_aug_id]

    def update_ucb_values(self, current_aug_id, returns):
        self.total_num += 1
        self.num_action[current_aug_id] += 1
        self.return_action[current_aug_id].append(returns)
        self.qval_action[current_aug_id] = np.mean(self.return_action[current_aug_id])
