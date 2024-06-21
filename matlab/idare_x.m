function [X,K]=idare_x(x,x_opt)
q1 = x(1,1);
q2 = x(1,2);
q3 = x(1,3);
dt = 0.04;
r = x_opt(4);
h = x_opt(3);

A = [1 dt; 0 1];
B = [-dt*(h+dt); -dt];
Q = [q1 q3;q3 q2];
R = r;

[X,K,~] = idare(A,B,Q,R,[],[]);
end