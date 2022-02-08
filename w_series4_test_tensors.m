function [X2, Y2, Yh2, B, k_tob] = w_series4_test_tensors(M, m_in, n_out, l_sess, l_test, n_sess, sess_off, offset, norm_fl, k_tob)
    %% Test regression ANN
    if(k_tob == 0)
        k_tob = ceil(l_sess/n_out);
    end

    X2 = ones([m_in, k_tob, n_sess-sess_off]);
    Y2 = zeros([n_out, k_tob, n_sess-sess_off]);
    Yh2 = zeros([n_out, k_tob, n_sess-sess_off]);
    B = zeros([2, k_tob, n_sess-sess_off]);

    % Re-format test input into session tensor
    for i = 1:n_sess-sess_off
        for j = 1:k_tob
            % extract and scale observation sequence
            idx = (i+sess_off)*l_sess + (j-1)*n_out + 1 + offset;

            Mx = M(idx:idx+m_in-1);
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series2_scale(Mx, B(1,j,i), B(2,j,i));
            end
            X2(1:m_in, j, i) = Mx(:);

            My = M(idx+m_in:idx+m_in+n_out-1);
            if(norm_fl)
                My = w_series2_scale(My, B(1,j,i), B(2,j,i));
            end
            Yh2(1:n_out, j, i) = My(:);
        end
    end
end