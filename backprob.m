clear all;

%Variables and activation function
train_rate = 0.05; 

prompt = 'Enter number of layers: ';
nLayers = input(prompt);
 

%Prompt user to select number of neurons per hidden layers
prompt = 'Enter number of neurons per hidden layer: ';
nNeurons = input(prompt);     
 

%Prompt user to select digit
prompt = 'Select digit to evaluate: ';
digit = input(prompt);                   %Digit Value
 

%Prompt user to select bias
prompt = 'Select the bias: ';
bias_num = double(input(prompt));
 

% digit = 3;
% nLayers = 2;
% nNeurons = 200;
% bias_num = 1;
% train_rate = 0.05; 


[traindata, testdata] = getdata(digit); % Input digit from user to train data and test data

inputdata = traindata;
nInputs = size(inputdata, 2); % number of features (number of inputs)
target = zeros(10,1);
target(digit+1) = 1;                 
 

%random_weights and bias for layers
weights = cell(nLayers, 1);
biases = cell(nLayers,1);

delta = cell(nLayers+2, 1); %Creating empty cell for inner to loop through matrix of each layer

weights{1} = 10^-4*rand(nNeurons, nInputs); % first weight between input data and first layer
biases{1} = bias_num*ones(nNeurons,1);
%Each layer will have same amount of outputs
for i = 1:nLayers-1
    weights{i+1} = 10^-4*rand(nNeurons, nNeurons);
    biases{i+1} = bias_num*ones(nNeurons,1);
end

weights{nLayers+1} = 10^-4*rand(10, nNeurons);
biases{nLayers+1} = bias_num*ones(10,1);


for row = 1:200 % for entire set put size(inputdata,2)
    a = cell(nLayers + 2,1);
    a{1} = double(inputdata(row,:)');
    
    for i = 1:nLayers+1
        a{i+1} = sigmoid(weights{i}*a{i} + biases{i});
    end
    
    err = abs(target - a{nLayers+2});
    delta{nLayers+2} = a{nLayers+2}.*(ones(10,1)-a{nLayers+2}).*err;
    
    for i = nLayers+1:-1:1
        dW = zeros(size(weights{i}));
        for q = 1:size(weights{i},1)
            for p = 1:size(weights{i},2)
                dW(q,p) = train_rate*delta{i+1}(q)*a{i}(p);
            end
        end
        weights{i} = weights{i} + dW;
        delta{i} = weights{i}'*delta{i+1}.*(a{i}.*(ones(size(a{i}))-a{i}));
    end
end

%% Test 

inputdata = testdata;

for row = 1:size(inputdata,1)
    a = cell(nLayers + 2,1);
    a{1} = double(inputdata(row,:)');
    
    for i = 1:nLayers+1
        a{i+1} = arrayfun(f, weights{i}*a{i} + biases{i});
    end
    [~, pred(row)] = min(a{end});
end

fprintf('Classification Error: %.2f\n', mean(digit ~= (pred -1)))
