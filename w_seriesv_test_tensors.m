function [X2, Y2, Yh2, B, k_tob] = w_seriesv_test_tensors(M, m_in, n_out, l_sess, l_test, n_sess, norm_fl, m_ine, n_oute)
    %% Test regression ANN
    k_tob = ceil(l_test/n_out);

    X2 = ones([m_ine, k_tob, n_sess]);
    Y2 = zeros([n_oute, k_tob, n_sess]);
    Yh2 = zeros([n_oute, k_tob, n_sess]);
    B = zeros([2, k_tob, n_sess]);

    % Re-format test input into session tensor
    for i = 1:n_sess
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = i*l_sess - m_in + (j-1)*n_out + 1;

            Mx = M(idx:idx+m_in-1);
            Vx = Mx(1:m_in-1) - Mx(2:m_in);
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series2_scale(Mx, B(1,j,i), B(2,j,i));
                Vx = Mx(1:m_in-1) - Mx(2:m_in);
            end
            X2(1:m_in, j, i) = Mx(:);
            X2(m_in+1:m_ine, j, i) = Vx(:);

            My = M(idx+m_in:idx+m_in+n_out-1);
            Vy = My(1:n_out-1) - My(2:n_out);
            if(norm_fl)
                My = w_series2_scale(My, B(1,j,i), B(2,j,i));
                Vy = My(1:n_out-1) - My(2:n_out);
            end
            Yh2(1:n_out, j, i) = My(:);
            Yh2(n_out+1:n_oute, j, i) = Vy(:);
        end
    end
end