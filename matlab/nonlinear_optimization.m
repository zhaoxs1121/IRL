% main function that optimizes the cost_func. 
clear;
clc;

% A = [0,0,0,-1,1]; % constraint 'h_st < h_go'
% lb = [100 0.001 25 30 1];
% ub = [500 0.1 37.28 200 25];
% x0 = [300, 0.01, 32, 60, 10];
% % x0 = [319.373201,0.003135,32.357479,44.436407,9.860932];
% options = optimoptions(@fmincon,'OptimalityTolerance',1e-12,'FunctionTolerance',1e-12);
% x = fmincon(@(x)cost_func(x),x0,A,0,[],[],lb,ub,[],options);
% [cost0,e_var,e_mean]=show_cost(x);
% fprintf('x=[%f,%f,%f,%f,%f]\n',x)
% fprintf('cost=%f,e_mean=%f,e_var=%f\n',cost0,e_mean,e_var)
% %% cost=15.097743,e_mean=0.000098,e_var=0.406204

x = fmincon(@(x)linear_cost_func(x),[0.01,0.15,1,0],[],[],[],[],[0,0,0,0],[1,1,10,100]);
% x = fmincon(@(x)linear_cost_func(x),[0.01,0.15,1],[],[],[],[],[0,0,0],[1,1,10]);
% x = [0.015344,0.487301,1.429661];
[cost0,e_var,e_mean]=linear_show_cost(x);
fprintf('x_linear=[%f,%f,%f,%f]\n',x)
fprintf('cost=%f,e_mean=%f,e_var=%f\n',cost0,e_mean,e_var)
%% cost=16.210985,e_mean=0.000164,e_var=0.274038