function result = OV(x1,x2,x3,x4)
s = x1;
v_max = x2;
h_go = x3;
h_st = x4;

result = 0.5 * v_max * (1 + erf(10*(s - (h_go + h_st)/2) / (pi * (h_go - h_st + 0.001))));
end