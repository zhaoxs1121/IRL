clear;
clc;

% main function

% old dynamic parameters used in the paper
kp= 0.006354857329713133;
kd= 0.16748344692513145;
h= 0.8903920819008573;

% new dynamic parameters
kp= 0.017258;
kd= 0.617991;
h= 0.913008;

Ts = 0.04;
R = 2.5;
A = [1 Ts; 0 1];
B = [-Ts*(h+Ts); -Ts];
D = [Ts; 1];

x_opt(1) = kp;
x_opt(2) = kd;
x_opt(3) = h;
x_opt(4) = R;

% use these parameters to estimate q1, q2 in matrix C
% Q = ga(@(x)idare_12_01(x,x_opt), 3, [],[],[],[],[0,0,0]);
Q = fmincon(@(x)idare_12_01(x,x_opt),[0,0,0]);

% the solution of idare
[X,K] = idare_x(Q,x_opt);

Qx = [Q(1) Q(3);Q(3) Q(2)];
Qx
% q1*q2-q3^2
[~,S,~] = dlqr(A,B,Qx,R);
S
x_optimal = [S(1,1) S(2,1) S(2,2)];
save('../data_inter/solution_optimal.mat',"x_optimal")

% %% kernel
% H = readmatrix('../data_inter/P.csv');
% % P_ = readmatrix('../data_inter/P_.csv');
% f = readmatrix('../data_inter/q.csv');
% A = readmatrix('../data_inter/G.csv');
% b = readmatrix('../data_inter/h.csv');
% % xx = readmatrix('../data_inter/xx.csv');
% [m,~] = size(H);
% 
% % 2n without W
% % n = m/2;
% % x = quadprog(2*P,q,G,zeros(3*n,1),[ones(1,n) zeros(1,n)],0);
% 
% % 3n witH W
% % options = optimoptions(@quadprog,'MaxIterations',500);
% % n = m/3;
% % x = quadprog(2*P,q,G,zeros(4*n,1),[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0],[],[],[],options); %%%%old
% % x = quadprog(2*P,q,G,[zeros(2*n,1);-0.05*ones(n,1);zeros(n,1)],[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0],[],[],[],options); %%%%old
% 
% % 2n+1 witH W and single c
% % options = optimoptions(@quadprog,'MaxIterations',500);
% % n = (m-1)/2;
% % con = 0.1;
% % x = quadprog(2*P,q,G,[zeros(n+1,1);-con*xx;-0*xx],[ones(1,n) zeros(1,n+1);zeros(1,n+1) ones(1,n)],[0;0],[],[],[],options); %%%%old
% % x = quadprog(2*P1,q1,G1,zeros(n,1),[ones(1,n)],0); %%%%origin
% % x = quadprog(2*P,q,G,zeros(1.5*n,1),[ones(1,n/2) zeros(1,n/2)],0); %%%%old
% % x = quadprog(2*P,q,G,zeros(4*n,1),[ones(1,n) zeros(1,2*n);zeros(1,2*n) ones(1,n)],[0;0]); %%%%old
% 
% % new structure same with quad
% n = (m-1)/2;
% x = quadprog(2*H,f,A,b,[ones(1,n) zeros(1,n+1);zeros(1,n+1) ones(1,n)],[0;0]);
% 
% save('../data_inter/solution.mat',"x")
% 
% % S1 = x.'*P_*x;
% % S2 = x.'*P*x;
% 
% 
% %% quadratic
% H_quad = readmatrix('../data_inter/P_quad.csv');
% f_quad = readmatrix('../data_inter/q_quad.csv');
% A_quad = readmatrix('../data_inter/A_quad.csv');
% b_quad = readmatrix('../data_inter/B_quad.csv');
% 
% x_quad = quadprog(2*H_quad,f_quad,A_quad,b_quad); % ,[ones(1,3) zeros(1,4)],0
% save('../data_inter/solution_quad.mat',"x_quad")