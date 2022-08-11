from scipy.stats.distributions import uniform, norm,binom
import random
import numpy as np
import pdb

def update_bayesian_belief(H,nTrials, prob_reward_targets, sN,choices):
    B = np.zeros([self.nTrials, self.nChoices])
    lr = np.zeros([self.nTrials])
    signed_B_diff = np.zeros_like(self.lr)
    B_diff = np.zeros_like(self.lr)
    rpe = np.zeros_like(self.B)
    CPP = np.zeros_like(self.lr)
    MC = np.zeros_like(self.lr) + 0.5
    epoch_length = np.zeros_like(self.lr) + 1
    sF = np.zeros_like(self.lr)
    ideal_B = np.zeros((self.nTrials))

    for t in np.arange(0,nTrials):
        #pdb.set_trace()
        #unifpdf = lambda x_t: uniform(low, high).pdf(x_t)
        binompdf = lambda x: binom(1,0.5).pmf(int(x))
        #normpdf = lambda mu, sig, x_t: norm(mu, sig).pdf(x_t)
        normpdf = lambda mu, sig, x_t: norm(mu, sig).cdf(x_t)
        #pdb.set_trace()	
        e_r = 0.5
            
        choice = choices[t]
        if  choice == 0:
             nonchoice = 1
        elif choice == 1:
             nonchoice = 0
        #elif np.isnan(choice) == True:
        #     return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]			
        #pdb.set_trace()
        if choice == np.argmax(prob_reward_targets[t]):
            optimal = 1
        else:
            optimal = 0
        rpe[t, choice] =  prob_reward_targets[t,  choice] -  B[t,  choice]
        sF[ t] = np.sqrt( sN[t]**2 + ( sN[t]**2 * (1- MC[ t]))/( MC[ t]))
        #u_val = unifpdf(prob_reward_targets[t,choice])
        u_val = binompdf(prob_reward_targets[t,choice])
        n_val = normpdf( B[ t,choice], sF[ t], prob_reward_targets[t,choice])
        print("n_val",n_val)
        CPP[ t] = (u_val*H)/((u_val*H) + (n_val*(1-H)))
        #print("CPP[t]", CPP[t])
        lr[ t] =  CPP[ t] + (1- MC[ t])*(1- CPP[ t]) # LR should be high after a CPP. It is not the case here. Fig 3, Vaghi et al 2013


         # Next trial calculations
        if t < (nTrials-1):
            B[ t+1, choice] =  B[ t, choice] + lr[ t]*rpe[ t,choice]


            #instead, for values, decay to average val for both targets ((3+0)/2)
            #B[ t+1,  nonchoice] =  B[ t, nonchoice]*(1-CPP[t])+CPP[t]*1.5
            B[ t+1,  nonchoice] =  B[ t, nonchoice]*(1-CPP[t])+CPP[t]*e_r

            signed_B_diff[ t+1] = B[ t,1] - B[ t,0]
            B_diff[ t+1] = B[ t,choice] - B[ t,nonchoice]

            term1 =  CPP[ t]* sN[t]**2
            term2 = (1- CPP[ t])*(1- MC[ t])* sN[t]**2 
            term3 =  CPP[ t]*(1- CPP[ t])*(rpe[ t,choice]*( MC[ t]))**2
            MC[ t+1] = 1 - ((term1+term2+term3)/(term1+term2+term3 +  sN[t]**2))
            epoch_length[ t+1] = (epoch_length[ t] + 1)*(1- CPP[ t]) +  CPP[ t]

    return   

