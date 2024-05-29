clear;
clc;

% main function

% parameters in dynamic
% kp= 0.006354857329713133;
% kd= 0.16748344692513145;
% h= 0.8903920819008573;
% kp= 0.016163;
% kd= 0.584277;
% h= 1.030583;
% x_opt(1) = kp;
% x_opt(2) = kd;
% x_opt(3) = h;
% 
% % use these parameters to estimate q1, q2 in matrix C
% % Q = ga(@(x)idare_12_01(x,x_opt), 3, [],[],[],[],[0,0,0]);
% Q = fmincon(@(x)idare_12_01(x,x_opt),[0,0,0]);
% 
% % the solution of idare
% [X,K] = idare_x(Q,x_opt);
% 
% Ts = 0.04;
% R = 2.5;
% A = [1 Ts; 0 1];
% B = [-Ts*(h+Ts); -Ts];
% D = [Ts; 1];
% q1 = Q(1);
% q2 = Q(2);
% q3 = Q(3);
% Qx = [q1 q3;q3 q2];
% q1*q2-q3^2
% [~,S,~] = dlqr(A,B,Qx,R);
% S

% kernel
P = readmatrix('../data_inter/P.csv');
P_ = readmatrix('../data_inter/P_.csv');
q = readmatrix('../data_inter/q.csv');
G = readmatrix('../data_inter/G.csv');
xx = readmatrix('../data_inter/xx.csv');
[m,~] = size(P);


% 2n without W
% n = m/2;
% x = quadprog(2*P,q,G,zeros(3*n,1),[ones(1,n) zeros(1,n)],0);

% 3n witH W
% options = optimoptions(@quadprog,'MaxIterations',500);
% n = m/3;
% x = quadprog(2*P,q,G,zeros(4*n,1),[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0],[],[],[],options); %%%%old
% x = quadprog(2*P,q,G,[zeros(2*n,1);-0.05*ones(n,1);zeros(n,1)],[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0],[],[],[],options); %%%%old

% 2n+1 witH W and single c
options = optimoptions(@quadprog,'MaxIterations',500);
n = (m-1)/2;
con = 0.1;
x = quadprog(2*P,q,G,[zeros(n+1,1);-con*xx;-0*xx],[ones(1,n) zeros(1,n+1);zeros(1,n+1) ones(1,n)],[0;0],[],[],[],options); %%%%old
% x = quadprog(2*P1,q1,G1,zeros(n,1),[ones(1,n)],0); %%%%origin
% x = quadprog(2*P,q,G,zeros(1.5*n,1),[ones(1,n/2) zeros(1,n/2)],0); %%%%old
% x = quadprog(2*P,q,G,zeros(4*n,1),[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0]); %%%%old

save('../data_inter/solution.mat',"x")
% 
% S1 = x.'*P_*x;
% S2 = x.'*P*x;


% quadratic
H = readmatrix('../data_inter/P_quad.csv');
f = readmatrix('../data_inter/q_quad.csv');
A = readmatrix('../data_inter/A_quad.csv');
b = readmatrix('../data_inter/B_quad.csv');

x_quad = quadprog(2*H,f,A,b);
save('../data_inter/solution_quad.mat',"x_quad")