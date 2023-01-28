# DeepHAM\

The algorithm launches from the "Rep.py" file. 

One should first input all the parameters wanted for both the neural networks and macroeconomic model. 

The code works in a modular manner : 

_ To make the replication exercise as transparent as possible, Rep.py features the main loop that is described in the paper in pseudo-code. 
This allows to check easily. 

_ The file "Objects.py" has the definition of the neural networks for both policy and value function as well as the dataset subclass we will
be using for our dataloader. The dataset object will contain the computed stationary distribution computed at every epoch. 

_ The file Aux.py contains three functions : One that computes the stationary distribution given a policy function and a macroeconomic model
One that iterates the policy function forward to obtain an empirical counterpart to the value function 
And One that mixes the iteration of the poliy function with the addition of the value of the terminal state after T iterations. 

_ All of these functions only rely on one object to function : The object KSModel defined in KSModel.py. It could be replaced with any 
other macroeconomic model as it only requires three functions : 
Utility U(x) 
stateForward(policy, economy) which computes the economy's next state given current state and policy function
GMoment( economy, basis) which computes and assembles the generalized moment (here done by averaging the generalized moments as in the paper
but it could be built differently). 

One can create another model with these same functions and the DeepHam method will be computable using the same code.

To run the project, simply run rep.py ! 

To anyone who is trying the code, it is still work in progress :) 
