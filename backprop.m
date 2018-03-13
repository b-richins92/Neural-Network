f=@sigmoid;
%first you must download and run the code "initializeweights" then use that data for the back prop code 
a1=double(test8(1,:)'); %inputting our first test row for the number 8
a2=W1*a1 + b1;
a2=arrayfun(f,a2);
a3=W2*a2 + b2;
a3=arrayfun(f,a3);
a4=W3*a3 + b3;
a4=arrayfun(f,a4);  % output which we will use to compare to target

tar=zeros(1,10)';
tar(9,1)=1;      % our target is 0's in all rows with a one in the 9th row corresponding to number 8 (bc list starts w/ 0)
error=abs(tar-a4);    
d4=a4.*(ones(10,1)-a4).*error; %delta for output layer
trc=.05;        %training rate coefficient
dW3=trc*d4*a3'; %change of W3
W3=W3+dW3;
d3=W3'*d4.*(a3.*(ones(25,1)-a3)); %I believe there is a typo on the formula at the end of part 6. I switched dk and wk' so the 
%dimensions will work for matrix multiplication and so that dj will have the correct amount of entries. If anyone has a better solution please let me know.
dW2=trc*d3*a2';
W2=W2+dW2;
d2=W2'*d3.*(a2.*(ones(200,1)-a2));
dW1=trc*d2*a1';
W1=W1+dW1;
function y =sigmoid(x)
y=1/(1+exp(-x));
end