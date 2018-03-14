function [train, test] = getdata(t)
load('mnist_all.mat');
if t==1
    train=train0;
    test = test0;
elseif t==2
        train=train1;
        test = test1;
elseif  t==3
        train=train2;
        test = test2;
elseif  t==4
        train=train3;
        test = test3;
elseif  t==5
        train=train4;
        test = test4;
elseif  t==6
        train=train5;
        test = test5;
elseif  t==7
        train=train6;
        test = test6;
elseif  t==8
        train=train7;
        test = test7;
elseif  t==9
        train=train8;
        test = test8;
elseif  t==10
        train=train9;
        test = test9;
end