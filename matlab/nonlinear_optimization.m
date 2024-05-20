% main function that optimizes the cost_func. 
clear;
clc;

% A = [0,0,0,-1,1]; % constraint 'h_st < h_go'
% lb = [100 0.001 25 30 1];
% ub = [500 0.1 37.28 200 25];
% x0 = [300, 0.01, 32, 60, 10];
% options = optimoptions(@fmincon,'OptimalityTolerance',1e-12,'FunctionTolerance',1e-12);
% x = fmincon(@(x)cost_func(x),x0,A,0,[],[],lb,ub,[],options);
% [cost0,e_var,e_mean]=show_cost(x);
% fprintf('x=[%f,%f,%f,%f,%f]\n',x)
% fprintf('cost=%f,e_mean=%f,e_var=%f\n',cost0,e_mean,e_var)
% 
% alpha = x(1);
% beta = x(2);
% v_max = x(3);
% h_go = x(4);
% h_st = x(5);
% h = pi^(3/2)*(h_go - h_st)/(10*v_max)
% r = (h_go + h_st)/2 - v_max*h/2

%% before including spacing error in the cost
%% cost=15.097743,e_mean=0.000098,e_var=0.406204
%% x = [319.373201,0.003135,32.357479,44.436407,9.860932];
%% h = 0.5950; r = 17.5223;

%% after including spacing error in the cost
%% x=[315.926958,0.004892,27.763091,30.000012,10.710852]
%% cost=15.112697,e_mean=0.000119,e_var=0.398548
%% h = 0.3972; r = 15.0363


%% linear system

x = fmincon(@(x)linear_cost_func(x),[0.01,0.15,1],[],[],[],[],[0,0,0],[1,1,10]);
% x = [0.016163,0.584277,1.030583];
[cost0,e_var,e_mean]=linear_show_cost(x);
fprintf('x_linear=[%f,%f,%f]\n',x)
fprintf('cost=%f,e_mean=%f,e_var=%f\n',cost0,e_mean,e_var)

%% including 'r' in the dynamic 
% x = fmincon(@(x)linear_cost_func(x),[0.01,0.15,1,10],[],[],[],[],[0,0,0,0],[1,10,10,100]);
% fprintf('x_linear=[%f,%f,%f,%f]\n',x)
% [cost0,e_var,e_mean]=linear_show_cost(x);
% fprintf('cost=%f,e_mean=%f,e_var=%f\n',cost0,e_mean,e_var)

%% cost=17.994266,e_mean=0.000013,e_var=0.374512