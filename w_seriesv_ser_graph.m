function w_seriesv_ser_graph(M, l_m, Y2, l_y, l_sess, m_in, n_out, k_tob, n_sess, k_start)
    % Re-format sessions back into through array
    M2 = M;
    for i = 1:n_sess
        for j = 1:k_tob
            idx = i*l_sess - m_in + (j-1)*n_out + 1;
            M2(idx+m_in:idx+m_in+n_out-1) = Y2(1:n_out, j, i);
        end
    end

    if(k_start==0)
        k_start=1;
    end

    f = figure();
    lp = plot(k_start:l_y, M2(k_start:l_y), 'r:', k_start:l_m, M(k_start:l_m), 'b','LineWidth', 2);
    %title('WSE Main Index Plot')
    xlabel('Days')
    ylabel('Index value')
    legend('prediction', 'observation')
end