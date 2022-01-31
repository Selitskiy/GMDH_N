function M = w_series2_rescale(Mn, Min, Max)
    M = Mn * (Max - Min) + Min;
end