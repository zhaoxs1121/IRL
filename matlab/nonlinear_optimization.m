% main function that optimizes the cost_func. 
clear;
clc;

A = [0,0,0,-1,1]; % constraint 'h_st < h_go'
b = 0;
lb = [0 0 20 10 1];
ub = [1 0.1 60 100 20];
x0 = [0.1, 0.005, 30, 60, 10];
options = optimoptions(@fmincon,'OptimalityTolerance',1e-12,'FunctionTolerance',1e-12);
x = fmincon(@(x)cost_func(x),x0,A,b,[],[],lb,ub,[],options);
% x = [0.48604124931269,0.015934574721651,40.197731381234,61.827498299547,5.9530242372604];
[cost0,e_var,e_mean]=show_cost(x);
fprintf('x=[%f,%f,%f,%f,%f]\n',x)
fprintf('cost=%f,e_mean=%f,e_var=%f\n',cost0,e_mean,e_var)

x = fmincon(@(x)linear_cost_func(x),[0.01,0.15,1],[],[],[],[],[0,0,0],[1,1,10]);
% x = [0.015344,0.487301,1.429661];
[cost0,e_var,e_mean]=linear_show_cost(x);
fprintf('x_linear=[%f,%f,%f]\n',x)
fprintf('cost=%f,e_mean=%f,e_var=%f\n',cost0,e_mean,e_var)