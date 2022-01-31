function w_series2_err_graph(Y2, Yh2)
    f = figure();
    Z2(:, :) = (Y2(1, :, :) - Yh2(1, :, :)) ./ Yh2(1, :, :);
    sp = surf(Z2, 'FaceColor', 'interp');
    xlabel('Session')
    ylabel('Observation')
    zlabel('Error')
end