f=@sigmoid;

%first you must download and run the code "initializeweights" then use that data for the back prop code 
for k=1:200       %random batch of 200 samples
p=randi(5421);
 %5421 is the lowest number of rows that any of the training digits has;
t=5;
if t==1
    tr=train0;
elseif t==2
        tr=train1;
elseif  t==3
        tr=train2;
elseif  t==4
        tr=train3;
elseif  t==5
        tr=train4;
elseif  t==6
        tr=train5;
elseif  t==7
        tr=train6;
elseif  t==8
        tr=train7;
elseif  t==9
        tr=train8;
elseif  t==10
        tr=train9;
end
a1=double(tr(p,:)'); %inputting our randomly test row
a2=W1*a1 + b1;
a2=arrayfun(f,a2);
a3=W2*a2 + b2;
a3=arrayfun(f,a3);
a4=W3*a3 + b3;
a4=arrayfun(f,a4);  % output which we will use to compare to target

tar=zeros(1,10)';
tar(t,1)=1;      % our target is 0's in all rows with a one in the t'th row corresponding to the correct answer
error=abs(tar-a4);    
d4=a4.*(ones(10,1)-a4).*error; %delta for output layer
trc=.1;        %training rate coefficient
for q=1:10
    for p=1:25
        dW3(q,p)=trc*d4(q,1)*a3(p,1);
    end
end
W3=W3+dW3;
d3=W3'*d4.*(a3.*(ones(25,1)-a3));
for q=1:25
    for p=1:200
        dW2(q,p)=trc*d3(q,1)*a2(p,1);
    end
end
W2=dW2+W2;
d2=W2'*d3.*(a2.*(ones(200,1)-a2));
for q=1:200
    for p=1:784
        dW1(q,p)=trc*d2(q,1)*a1(p,1);
    end
end
W1=dW1+W1;
end
function y =sigmoid(x)
y=1/(1+exp(-x));
end