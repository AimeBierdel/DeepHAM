
import torch 
import torch.nn as nn 
from matplotlib.pyplot import plot, show, title
import numpy as np
import Aux 
import Objects
import KSmodel
import gc
from collections import defaultdict
from gc import get_objects, set_debug



torch.autograd.set_detect_anomaly(True)


#########################################
##### Macro Model Parameters ############
#########################################

Zstates = torch.tensor([0.99,1.01])

alpha = 0.36 
delta = 0.25
lbar = 1/0.9 
b = 0.15
beta = 0.99



modelparams = torch.tensor([alpha,delta,lbar,b,beta,Zstates[0],Zstates[1] ])

Pi = torch.tensor([ [0.525 , 0.35 , 0.03125,0.09375 ],[ 0.038889 , 0.836111 , 0.002083,0.122917],[0.09375,0.03125,0.291667,0.583333],[0.009115,0.115885,0.024306,0.850694] ])


nIndStates = 1 # individual states : assets and idiosyncratic shock
nstates = 4  # individual capital, individual z, macro Z, generalized moment 


####################################
### DeepHAM procedure parameters ###
####################################
###### Number of iterations / agents 

# parameters for the simulation of the stationary distribution
nAgents=10
nEconomies =50


Nk = 30 #Nb of loops for the whole procedure 
Tsimul = 100 # Number of periods for simulation of stationary distribution
Tvalue = 20 # Number of periods for simulation of expectations
Tpol = 5 # Number of plays before value for policy function optimization
Nv = 100 #Number of optimization loops for value fct is Nv/Nb
Np = 100 # Number of optimization loops for policy is Np/Nb
Nb =10 #number of elements per batch

#######################################
###### Neural networks parameters #####
#######################################

NbLayersPol = 3 
NbLayersVal = 3
NbLayersGM = 3
hiddenSizePol= 10
hiddenSizeVal = 10 
hiddenSizeGM=10



###############################
### Objects initialization ####
###############################

## Create macroeconomic model 

macroEngine = KSmodel.KSmodel(modelparams,Pi,Zstates) 

#### Create and initialize neural networks : one for policy and one for value function, and one for generalized moment

GMModel = Objects.GMomentNet(nIndStates,hiddenSizeGM,NbLayersGM, 1,lastActivFunc=None)
# I use softmax with two outputs because sigmoid behaves super badly, it returns a gradient of zero for anything
# out of the -sqrt(3), sqrt(3) range so there is almost never learning. 
# Instead I use the first output of a two outputs softmax (ain't dumb if it works ?)
PolModel = Objects.FittingNet(nstates,hiddenSizePol,NbLayersPol,2,"softmax")
ValModel = Objects.FittingNet(nstates,hiddenSizeVal,NbLayersVal,1,lastActivFunc=None)

val = Objects.ValueModel(GMModel,ValModel,macroEngine)
pol = Objects.PolicyModel(GMModel,PolModel,macroEngine)

# define loss function 

lossFct= nn.MSELoss()

###################################
#### Main loop for optimization ###
###################################

for i in range(0,Nk):
    # Define the optimizers with 
    #decaying learning rate
    learning_rate = 0.1 * ((Nk - 1 - i)/(Nk-1))**3+  1e-6 * (i/(Nk-1))**3
    valOptimizer = torch.optim.SGD(val.parameters(), lr=learning_rate)
    polOptimizer = torch.optim.SGD(pol.parameters(), lr=learning_rate)
  
    ## Main loop : number of epochs
    print('iteration '+ str( i ))
    ## Prepare the stationnary distribution which is assumed to exist. We do this by iterating the optimization of households according to our
    ## Policy functions for a number of periods. 
    ## Then put the stationary distribution into a dataloader to sample from it
    if i == 0 :
        DistDataset = Objects.StatDataset(Aux.MCstationaryDistribution(pol,Tsimul, nAgents,nEconomies,macroEngine).detach())
    else :
        DistDataset.update(Aux.MCstationaryDistribution(pol,Tsimul, nAgents,nEconomies,macroEngine).detach())

    # Create the two samplers
    ValueSampler , PolicySampler = Aux.initLoaders(DistDataset,Nv,Np,Nb)
 
    ### draw a sample of size Nb
    for data in ValueSampler:
       # Initialize vector of expected predicted and actual values
        Vi = torch.zeros(data.size(1))
        for k in range(0,Nb):
            ### Compute Vi for all agents i in the economy by simulating it forward
            Vsim = Aux.SimulateValue(pol,Tvalue, data[k,:,:],macroEngine).detach()
            # Add to expected value 
            Vi = Vi +  torch.div(Vsim , torch.tensor([Nb*1.]))
        LossHist = 1e8 
        count = 0
         # we can parametrize the number of optimization steps on each batch, either using a count or the percentage
        # change of the loss function
        while (LossHist > 1e-4) and ( count < 10 ):
            Vpred = torch.zeros(data.size(1))
            # For each economy in a batch 
            for k in range(0,Nb): 

                ### Compute Value from model and add it to compute the expected value predicted by the neural network
                Vpred = Vpred + torch.div(torch.squeeze(val(data[k,:,:])), torch.tensor([Nb*1.]))
            ### Compute the Loss
            Loss = lossFct(Vi, Vpred)
            ### Backpropagation to compute gradient      
            Loss.backward()
            ### Clip the gradient 
            torch.nn.utils.clip_grad_norm_(val.parameters(), 5.)
            ### Update value function parameters 
            valOptimizer.step()
            ### Empty gradients 
            valOptimizer.zero_grad()
            polOptimizer.zero_grad()
            LossHist = Loss.item()
            del Vpred 
            count = count + 1
        
           
        print(Loss)
        del Loss
        del Vsim
        del Vi 
     
       
    gc.collect()
    
    # Now we optimize the policy function 
    print("Optimizing Policy")
    for data in PolicySampler:
        #initialize expected value of agent 1 over simulated paths
       
        epochloss = 1e6
        prev = 1.
        count = 0
        # we can parametrize the number of optimization steps on each batch, either using a count or the percentage
        # change of the loss function
        while (abs((epochloss - prev)/prev) > 1e-8 )and ( count < 10 ):
            loss = torch.zeros(1)
            for k in range(0,Nb):
                x = torch.div(Aux.SimulatePolicy(pol,val,Tpol, data[k,:,:],macroEngine),torch.tensor([Nb*1.]))
                ### Compute utility for all agents in the economy by simulating it forward
                # and add the utility of the path to agent 1's value. All done within SimulatePolicy function from Aux
                loss =  loss - x
            
                # Backward propagation
            
            loss.backward()
            prev = epochloss
            epochloss = loss.item()
    
            
                ### Update value function parameters 
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.)
            polOptimizer.step()

            
                ### Empty gradients
            polOptimizer.zero_grad()
            valOptimizer.zero_grad()
            
            
            count = count+1
            
            del loss
        print(epochloss)
    
    gc.collect()


### Some plots 

with torch.no_grad():
        testTensor = torch.tensor([ i*1. for i in range(20,61) ]).view((41,1))
        result = GMModel(testTensor).detach()        
        plot(testTensor,result)
        title('Generalized moment Q(a)')
        show()

        x= torch.tensor([ [1.01, 1., 40. , 0. ] for i in range(20,61)] )
        x[:,3] = torch.squeeze(result)
        result2 = ValModel(x)
        plot(result,result2)
        title('Value function for Z=1 , z=1, a = 40 as fct of Q')
        show()




    

        

   

               

