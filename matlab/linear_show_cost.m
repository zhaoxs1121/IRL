function [cost,e_var,e_mean]=linear_show_cost(X)
kp = X(1,1);
kd = X(1,2);
h = X(1,3);
% r = X(1,4);

count = 0;
fast = 0;
dt = 0.04;

cost = 0;
e_mean = 0;
e_var = 0;

Data_arrays = readmatrix('../data_inter/Data_arrays.csv');
[m,~] = size(Data_arrays);

for j = 1:m 
    if Data_arrays(j,1)==0 
        slow = fast;
        fast = j;
        count = count + 1;

        x = Data_arrays(slow+1:fast-1,1);
        v = Data_arrays(slow+1:fast-1,2);
        a = Data_arrays(slow+1:fast-1,3);
        p_x = Data_arrays(slow+1:fast-1,4);
        p_v = Data_arrays(slow+1:fast-1,5);
        p_l = Data_arrays(slow+1,6);

        v_reg = v(1);
        x_reg = x(1);
        e = zeros(fast-slow-2,1);

        for i = 1:fast-slow-2
            % cost calculation
            s = p_x(i) - x_reg - p_l;
            nu = p_v(i) - v_reg;
            u_reg = kp * (s - h * v_reg) + kd * nu;
%             u_reg = kp * (s - h * v_reg - r) + kd * nu;
            v_reg = v_reg + dt * u_reg;
            x_reg = x_reg + dt * v_reg;

            cost = cost + (x_reg - x(i+1))^2;

            % error calculation 
            s_error = p_x(i) - x(i) - p_l;
            nu_error = p_v(i) - v(i);
            u_error = kp * (s_error - h * v_reg) + kd * nu_error;
%             u_error = kp * (s_error - h * v(i) - r) + kd * nu_error;
            e(i) = a(i) - u_error;
        end
        e_mean = e_mean + sum(e);
        e_var = e_var + sum(e.*e);
    end
end
e_mean = e_mean^2/(m-count);
e_var = e_var/(m-count);
cost = cost/(m-count);
end








