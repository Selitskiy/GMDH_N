function w_seriesv_ser_graph(M, Y2, l_whole, l_sess, m_in, n_out, k_tob, n_sess)
    % Re-format sessions back into through array
    M2 = M;
    for i = 1:n_sess
        for j = 1:k_tob
            idx = i*l_sess - m_in + (j-1)*n_out + 1;
            M2(idx+m_in:idx+m_in+n_out-1) = Y2(1:n_out, j, i);
        end
    end

    f = figure();
    lp = plot(1:l_whole, M2, 'r:', 1:l_whole, M, 'b','LineWidth', 2);
    %title('WSE Main Index Plot')
    xlabel('Days')
    ylabel('Index value')
    legend('prediction', 'observation')
end