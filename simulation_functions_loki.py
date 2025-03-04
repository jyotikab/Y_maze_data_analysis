import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from bayesian_belief_model_p_ext import update_bayesian_belief
import pdb

class Simulation:
	def __init__(self, model, learning_rates, drift_start, bound_start, sp_start,tr_start, experimental_parameters, condition_id,conflict_id,volatility_id,session):
		print("In Simulator")
		self.model = model
		#self.reward_code = reward_code
		self.conflict_id = conflict_id
		#self.run_n = run_n
		self.condition_id = condition_id
		self.volatility_id = volatility_id
		self.session = session

		self.data_type = ""
		self.learning_rates = learning_rates
		expParam = pd.read_csv(experimental_parameters)
		columns = expParam.columns
		#self.expParam = expParam.rename(columns=(dict(zip(columns, ['r_t0', 'r_t1', 'cp', 'obs_cp', 'epoch_number','reward_p_t0', 'iti', 'epoch_trial', 'epoch_length','seed','conflict','condition','p_id_solution','Block_change','lambda','action_history','optimal']))))
		#self.expParam = expParam.rename(columns=(dict(zip(columns, ['r_t0', 'r_t1', 'cp',  'epoch_number','reward_p_t0', 'seed','conflict','condition','p_id_solution','Block_change','lambda','action_history','optimal']))))
		self.expParam = expParam
		self.reward_diff = np.transpose(np.array(self.expParam.r_t0.values - self.expParam.r_t1.values))
		self.reward_targets = np.transpose(np.array([self.expParam.r_t0.values, self.expParam.r_t1.values]))
		self.expParam['reward_p_t1'] = 1 - self.expParam.reward_p_t0
		self.p_targets = np.transpose(np.array([self.expParam.reward_p_t0, self.expParam.reward_p_t1]))
		self.action_history = self.expParam.action_history
		self.optimal = self.expParam.optimal
		self.state =  np.zeros((len(self.action_history),9))
		self.state_variable_names = ["B","signed_B_diff","B_diff","learning_rate","RPE","CPP","MC","epoch_length","sF (variance)"] 
		self.block = self.expParam.p_id_solution
		self.nTrials = len(self.reward_targets)
		self.timebound = 0.7
		self.dt = 0.001
		self.nTimeSteps = np.int(self.timebound / self.dt)
		self.a_baseline = 1
		self.a_start = 1
		#self.a_current = np.zeros(self.nTrials) + self.a_start
		#self.z_baseline = self.a_baseline / 2
		#self.z_start = self.z_baseline
		#self.z_current = np.zeros(self.nTrials) + self.z_start
		#self.base_evidence = 0.5 * self.a_current
		self.lower_bound_baseline = 0
		self.v_start = 0.01
		self.v_diff_current = np.zeros(self.nTrials) + self.v_start
		self.v_diff_mu = 0.8
		#self.tr_start = 0.25
		#self.tr = np.zeros(self.nTrials) + self.tr_start
		#self.trace = np.nan * np.ones((self.nTrials, self.nTimeSteps))
		#self.end_timestep = np.zeros(self.nTrials)
		self.si = 0.1
		self.randProb = np.random.random((self.nTrials, self.nTimeSteps))
		self.dx = self.si * np.sqrt(self.dt)
		self.reaction_times = np.nan * np.ones(self.nTrials)
		#self.choices = np.zeros_like((self.reaction_times), dtype=(np.int8))
		self.H = self.expParam[(self.expParam.cp == 1)].shape[0] / self.expParam.shape[0]
		#self.sN = np.ones_like(self.reaction_times)
		self.sN = np.zeros_like(self.reaction_times)+0.05
		#self.sN = np.ones_like(self.reaction_times)
		#self.sN = np.zeros_like(self.nTrials)+0.05
		self.low = self.reward_targets.min()
		self.up = self.reward_targets.max()
		self.high = self.up - self.low
		self.nChoices = self.reward_targets.shape[1]
		self.B = np.zeros([self.nTrials, self.nChoices])
		self.lr = np.zeros([self.nTrials])
		self.signed_B_diff = np.zeros_like(self.lr)
		self.B_diff = np.zeros_like(self.lr)
		self.rpe = np.zeros_like(self.B)
		self.CPP = np.zeros_like(self.lr)
		self.MC = np.zeros_like(self.lr) + 0.5
		self.epoch_length = np.zeros_like(self.lr) + 1
		self.sF = np.zeros_like(self.lr)
		self.ideal_B = np.zeros((self.nTrials))
		#self.optimal = np.zeros((self.nTrials))
	def calc_B_CPP(self):
		self.choices = self.action_history
		#pdb.set_trace()
		for t in np.arange(0,self.nTrials):
			self.B, self.signed_B_diff, self.B_diff, self.lr, self.rpe, self.CPP, self.MC, self.epoch_length, self.sF, optimal = update_bayesian_belief(t=t, nTrials=(self.nTrials), prob_reward_targets=(self.reward_targets), H=(self.H), sN=(self.sN), low=(self.low), up=(self.up), high=(self.high), choices=(self.choices), B=(self.B), signed_B_diff=(self.signed_B_diff), B_diff=(self.B_diff), lr=(self.lr), rpe=(self.rpe), CPP=(self.CPP), MC=(self.MC), epoch_length=(self.epoch_length), sF=(self.sF),data_type=(self.data_type))
			#pdb.set_trace()
			#self.state[t,:]	= [self.B,self.signed_B_diff,self.B_diff,self.lr,self.rpe,self.CPP,self.MC,self.epoch_length,self.sF]
			#self.optimal[t] = optimal
			self.corrTarget = np.argmax(self.p_targets, 1)
			self.choiceAcc = np.ones([self.nTrials]) * np.nan
			self.choiceAcc = self.corrTarget == self.choices
			if self.corrTarget[t] == 1:
				sub_opt = 0
			elif self.corrTarget[t] == 0:
				sub_opt = 1
			self.ideal_B[t] = self.B[t,self.corrTarget[t]] - self.B[t,sub_opt]

 



