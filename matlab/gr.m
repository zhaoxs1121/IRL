clear;
clc;

% main function

% parameters calculated in python
% kp= 0.006354857329713133;
% kd= 0.16748344692513145;
% h= 0.8903920819008573;
kp= 0.015344;
kd= 0.487301;
h= 1.429661;
x_opt(1) = kp;
x_opt(2) = kd;
x_opt(3) = h;

% % use these parameters to estimate q1, q2 in matrix C
% Q = ga(@(x)idare_12_01(x,x_opt), 3, [],[],[],[],[0,0,0]);
% 
% % the solution of idare
% [X,K] = idare_x(Q,x_opt);

% q1*q2-q3^2;

% QP
P = readmatrix('../data_inter/P.csv');
P_ = readmatrix('../data_inter/P_.csv');
q = readmatrix('../data_inter/q.csv');
G = readmatrix('../data_inter/G.csv');
[m,~] = size(P);

% 2n without W
% n = m/2;
% x = quadprog(2*P,q,G,zeros(3*n,1),[ones(1,n) zeros(1,n)],0);

% 3n witH W
options = optimoptions(@quadprog,'MaxIterations',500);
n = m/3;
% x = quadprog(2*P,q,G,zeros(4*n,1),[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0],[],[],[],options); %%%%old
% x = quadprog(2*P,q,G,[zeros(2*n,1);-0.05*ones(n,1);zeros(n,1)],[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0],[],[],[],options); %%%%old

% 2n+1 witH W and single c
n = (m-1)/2;
x = quadprog(2*P,q,G,[zeros(2*n,1);-0.005;zeros(n,1)],[ones(1,n) zeros(1,n+1);zeros(1,n+1) ones(1,n)],[0;0],[],[],[]); %%%%old

% x = quadprog(2*P1,q1,G1,zeros(n,1),[ones(1,n)],0); %%%%origin
% x = quadprog(2*P,q,G,zeros(1.5*n,1),[ones(1,n/2) zeros(1,n/2)],0); %%%%old
% x = quadprog(2*P,q,G,zeros(4*n,1),[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0]); %%%%old
% x = quadprog(2*P2,q2,V2,zeros(n,1),G2,zeros(n,1)); %%%%new

save('../data_inter/solution.mat',"x")

S1 = x.'*P_*x;
S2 = x.'*P*x;