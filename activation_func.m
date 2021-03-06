%Raymond & Leigh
%
%Activation function
%%
% Part iii
%The following function takes in Input=(O_1,...,O_n) and assigned
%weights(W1,...,Wn) to find the output('neuron')

function output = activation_func(input,weight,bias)
     
     net = sum(input*weight'-bias);
     
     %Calling Sigmoid function
     output = sigmoid(net);
end
