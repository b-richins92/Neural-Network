f=@sigmoid;
W=rand(392,784);
b=rand(392,1);
W1=rand(196,392);
b1=rand(196,1);
W2=rand(98,196);
b2=rand(98,1);
W3=rand(49,98);
b3=rand(49,1);
W4=rand(20,49);
b4=rand(20,1);
W5=rand(10,20);
b5=rand(10,1);


t8=double(test8(1,:)');
t8=W*t8 + b;
t8=arrayfun(f,t8);
t8=W1*t8 + b1;
t8=arrayfun(f,t8);
t8=W2*t8 + b2;
t8=arrayfun(f,t8);
t8=W3*t8 + b3;
t8=arrayfun(f,t8);
t8=W4*t8 + b4;
t8=arrayfun(f,t8);
t8=W5*t8 + b5;
t8=arrayfun(f,t8)

out=zeros(1,10)';
out(10,1)=1;
error=out-t8
function y =sigmoid(x)
y=1/(1+exp(-x));
end