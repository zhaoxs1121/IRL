function min=idare_12_01(x,x_opt)
[~,K]=idare_x(x,x_opt);
q1 = x(1,1);
q2 = x(1,2);
q3 = x(1,3);
kp = x_opt(1);
kd = x_opt(2);
c = 1;
d = 1;

%c = 0.2  min = 1000 d =1


if isempty(K)
    min = 100;
else
    min = 1000*(-K(1,1)-kp)^2 + (-K(1,2)-kd)^2 + 10*c*(tanh(-d*(q1*q2-q3^2))+1);
end
end
