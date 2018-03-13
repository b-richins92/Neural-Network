%Raymond Canchola
%
%Data to calculate how changing weights effect output
%%

%The programm calls activation_func beginning with small -> big weights
%Then, the program will change the input weights to anaylze how
%inputs effect outputs.


%First: Small Weights
input = [1 2 3 4];
weight = [.1 .2 .3 .4]/100000;
activation_func(input,weight)
%Ouput = 0.5000

%Second: Medum Weights
input = [1 2 3 4];
weight = [.1 .2 .3 .4]/10;
activation_func(input,weight)
%Ouput = 0.5744

%Third: Large Weights
input = [1 2 3 4];
weight = [10 20 30 40];
activation_func(input,weight)
%Ouput =  1

%Fourth:Small Inputs
input = [1 2 3 4];
weight = [.1 .2 .3 .4]/100;
activation_func(input,weight)
%Ouput = 0.5075

%Fifth:Large Inputs
input = [100 200 300 400];
weight = [.1 .2 .3 .4]/100;
activation_func(input,weight)
%Ouput = 0.9526

%Fifth:Large Inputs
input = [10000 20000 300000 400000];
weight = [.1 .2 .3 .4]/100;
activation_func(input,weight)
%Ouput = 1

%Anaylsis:  By increasing the weights from small to large and leaving the 
%           input stable, the output increases to 1. The same analogy can be
%           applied to the inputs.
%
%Functions: Different functions can be applied as the activated function
%           such as F(net) = (tanh(net) + 1)/2 and the relu activation function:
%           F(net) = max(0,net). Different functions increase and decrease
%           accucracy.   
