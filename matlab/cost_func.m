function min=cost_func(X)
alpha = X(1,1);
beta = X(1,2);
v_max = X(1,3);
h_go = X(1,4);
h_st = X(1,5);

h = pi^(3/2)*(h_go - h_st)/(10*v_max);
r = (h_go + h_st)/2 - v_max*h/2;

fast = 0;
dt = 0.04;

cost = 0;
e_mean = 0;
e_var = 0;
sp_mean = 0;

% data is exported from python, with shape of (n,6). 6 columns correspond
% to x/v/a of the ego vehicle, x/v/length of the preceeding vehicle. 
% Different pairs are seperated by a zero row. 
Data_arrays = readmatrix('../data_inter/Data_arrays.csv');
[m,~] = size(Data_arrays);

for j = 1:m 
    if Data_arrays(j,1)==0 % data of different pairs are seperated by a zero row
        % fast and slow pointers to extract each pair. Duration of each
        % pair is (slow+1,fast-1), with length = fast-slow-1
        slow = fast;
        fast = j;

        x = Data_arrays(slow+1:fast-1,1);
        v = Data_arrays(slow+1:fast-1,2);
        a = Data_arrays(slow+1:fast-1,3);
        p_x = Data_arrays(slow+1:fast-1,4);
        p_v = Data_arrays(slow+1:fast-1,5);
        p_l = Data_arrays(slow+1,6);

        v_reg = v(1);
        x_reg = x(1);
        e = zeros(fast-slow-2,1);

        for i = 1:fast-slow-2 % here the right bound is smaller than length because in line 47 the information of next frame is used 
            % cost calculation
            s = p_x(i) - x_reg - p_l;
            nu = p_v(i) - v_reg;
            u_reg = alpha * nu / s^2 + beta * (OV(s,v_max,h_go,h_st) - v_reg);
            v_reg = v_reg + dt * u_reg;
            x_reg = x_reg + dt * v_reg;

            cost = cost + (x_reg - x(i+1))^2;

            % error calculation 
            nu_error = p_v(i) - v(i);
            s_error = p_x(i) - x(i) - p_l;
            u_error = alpha * nu_error / s_error^2 + beta * (OV(s_error,v_max,h_go,h_st) - v(i));
            e(i) = a(i) - u_error;

            % spacing error
            sp_mean = sp_mean + s_error - h * v(i) -r;
        end
        e_mean = e_mean + sum(e);
        e_var = e_var + sum(e.*e);
    end
end
min = cost + e_var + e_mean^2 + sp_mean^2;
% min = cost + e_var + e_mean^2;
end








