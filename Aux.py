import torch
from torch.utils.data import  DataLoader, RandomSampler

### Simulation of the stationary distribution using monte-carlo method 
## Inputs : Policy function, number of agents and economies in samples, number of periods for forward simulation
def MCstationaryDistribution(policy ,T, nAgents, nEconomies, macroEngine):


    # Initialize sample of size N
    sample = macroEngine.Sample(nEconomies,nAgents,torch.tensor([40.])) 
    with torch.no_grad():
    # simulate for T periods
        for t in range(0,T):
            
            # for each economy compute next period's state
            for i in range(0,nEconomies):
                pol = policy(sample[i,:,:])
                sample[i,:,:] = macroEngine.stateForward( sample[i,:,:], pol, withCons=False ).detach()

    return sample 


def SimulateValue(policy, T, Economy,macroEngine):
    sample = Economy
    nAgents = Economy.size(0)
    welfare = torch.zeros(nAgents)
    with torch.no_grad():
        for t in range(0,T):
            pol = policy(sample)
            sample , consumption = macroEngine.stateForward( sample, pol )
            for i in range(0,nAgents):
                welfare = welfare + torch.pow(macroEngine.params[4],t)*macroEngine.U(consumption +  torch.tensor([1e-8]).expand(consumption.size()))
    return welfare

def SimulatePolicy(policy,value,T,Economy,macroEngine):

    sample = Economy
    nAgents = Economy.size(0)
    PathUtility = torch.zeros(1)
    for t in range(0,T):
        # compute policy taking only the gradient with respect to the consumption of the first agent
        pol = policy.forwardIndex(sample,0)
        #compute next period state
        sample , c= macroEngine.stateForward( sample, pol )
        PathUtility = PathUtility +  torch.pow(macroEngine.params[4],t)*macroEngine.U(c[0]+ torch.tensor([1e-8]))
        
    #add value of terminal state after  T
    value.val.eval()
    PathUtility = PathUtility + torch.squeeze(value(sample))[0]
    value.val.train()
    return PathUtility
        




def initLoaders(dataset,Nv,Np,Nb):
    samplerValue = RandomSampler(dataset, replacement=True, num_samples=Nv)
    samplerPolicy = RandomSampler(dataset, replacement=True, num_samples=Np)

    # create the two loaders

    # We will train on batches of size Nb
    dataloaderValue = DataLoader(dataset = dataset, sampler = samplerValue,batch_size=Nb, num_workers=0)
    dataloaderPolicy = DataLoader(dataset = dataset,sampler = samplerPolicy, batch_size=Nb,  num_workers=0)

    return dataloaderValue, dataloaderPolicy

