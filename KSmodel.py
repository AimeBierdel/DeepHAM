import numpy as np
import torch
import math





class KSmodel:
    def __init__(self,paramvector, Pi, exoState):
        self.params = paramvector
        self.Pi = Pi
        self.Ergo = np.linalg.matrix_power(Pi,100)[0,:] ## Ergodic distribution 
        self.exoState = exoState

    ### method to create a sample of nSamples economies with nAgents agents. The exogenous state is drawn from the ergodic distribution
    def Sample(self,nSamples, nAgents,initk):
        #Create sample of economies 
        sample = torch.zeros((nSamples,nAgents,3))
        sample.requires_grad = False
        #initial capital for all samples 
        sample[:,:,2] = initk 


        # Assign macro state to each economy randomly drawing from ergodic distribution
        probaZ= sum(self.Ergo[0:2])
        draws = torch.rand(nSamples)
        for i in range(0,nSamples):
            Z = 0 if draws[i]<probaZ else 1
            #assign macro state to each sample economy
            sample[i,:,0] = self.exoState[ Z ]
            #assign employment state to each agent in each sample economy
            n1 = nAgents - math.ceil(nAgents* self.Ergo[Z*2])
            sample[i,np.random.choice(range(0,nAgents), n1,replace=False ),1] = torch.tensor([1.])

        return sample 
    ## Utility function 
    def U(self,c): 
        return torch.log(c)



   
    # returns the marginal individual state transition matrices for the two current macro states and
    # The macro state transition matrix
    def getMarginals(self):
        Pi = self.Pi
        Pi0 =  torch.tensor([ [Pi[0,0] + Pi[0,2] , Pi[0,1] + Pi[0,3]] , [Pi[1,0] + Pi[1,2] , Pi[1,1] + Pi[1,3]]  ])
        Pi1 = torch.tensor([ [Pi[2,0] + Pi[2,2] , Pi[2,1] + Pi[2,3]] , [Pi[3,0] + Pi[3,2] , Pi[3,1] + Pi[3,3]]  ])
        PiZ = torch.zeros((2,2))
        temp0 = Pi[0,:] + Pi[2,:] 
        temp1 = Pi[1,:] + Pi[3,:] 
        PiZ[0,0] = temp0[0]+temp0[2]
        PiZ[0,1] = temp0[1] + temp0[3]
        PiZ[1,0] = temp1[0] + temp1[2]
        PiZ[1,1] = temp1[1] + temp1[3]
        return [Pi0 , Pi1] , PiZ
    ### Function that computes the macroeconomic states K,L,r,w and tau from the state X and indices of individual assets and z. Specific to one model
    def MacroState(self,economy):
        nAgents = economy.size(0)
        Z = economy[0,0]
        K = torch.max(torch.sum( economy,0)[2]/nAgents , torch.tensor([1e-8]))
        L = (torch.sum(economy, 0)[1] + 0.00000001)*self.params[2]/nAgents
        r = Z* self.params[0]* (K/L)**(self.params[0]-1)
        w = Z * ( 1- self.params[0])*((K/L)**self.params[0])
        tau = self.params[3]*(1-L)/(self.params[2] * L)

        
        return K,L , r,w , tau

  

    ## Compute next state 
    def stateForward(self, economy, pol, withCons=True ):

        # last period Z and nAgents 
        nAgents = economy.size(0)
        Zind = 0 if economy[0,0] == self.exoState[0] else 1

        # Create output vector

        res = torch.zeros(nAgents,economy.size(1))

        #current macro state and number of agents


         # draw this period's aggregate and individual states from marginals
        PiVec , PiZ= self.getMarginals()
        
        Zdraw = torch.rand(1)
        zdraw = torch.rand(nAgents)
        Z = 0 if Zdraw < PiZ[Zind, 0] else 1 
        res[:, 0] = self.exoState[ Z ]
        Piz = PiVec[Z]
        res[:,1] =  torch.tensor([ 0. if zdraw[i] < Piz[int(economy[i,1]),0] else 1.  for i in range(0,nAgents)])


        # First compute macroeconomic state of the current economy 

        K,L,r,w,tau = self.MacroState(economy)
        
       
        # Compute next period's capital. We detach gradient from macro aggregates as they don't internalize the impact
        budget = (1+r - self.params[1])*economy[:,2] + ((1 - tau)*self.params[2]*economy[:,1] + self.params[3]*(torch.ones(nAgents) - economy[:,1])     )*w
       
        res[:,2] = budget*(torch.ones(nAgents)- torch.squeeze(pol))
        if withCons:
            return res , budget * torch.squeeze(pol)
        else: 
            return res

    def GMoment(self,Economy,Basis):
        nAgents = Economy.size(0)
        return torch.div(torch.matmul(torch.ones(nAgents),Basis(Economy)) , torch.tensor([nAgents*1.]))

 

  




