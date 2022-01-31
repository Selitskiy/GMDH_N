function [X, Y, B, k_ob, m_ine, n_oute] = w_seriesv_train_tensors(M, m_in, n_out, l_sess, n_sess, norm_fl)

    % Number of observations in a session (training label(sequence) does
    % not touch test period
    k_ob = l_sess - m_in - n_out + 1;

    m_ine = 2*m_in-1;
    n_oute = 2*n_out-1;

    % Re-format input into session tensor
    % ('ones' (not 'zeros') for X are for bias 'trick'
    X = zeros([m_ine, k_ob, n_sess]);
    Y = zeros([n_oute, k_ob, n_sess]);
    B = zeros([2, k_ob, n_sess]);

    for i = 1:n_sess
        for j = 1:k_ob
            % extract and scale observation sequence
            idx = (i-1)*l_sess + j;
            
            Mx = M(idx:idx+m_in-1);
            Vx = Mx(1:m_in-1) - Mx(2:m_in);
            % scale bounds over observation span
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series2_scale(Mx, B(1,j,i), B(2,j,i));
                Vx = Mx(1:m_in-1) - Mx(2:m_in);
            end
            X(1:m_in, j, i) = Mx(:);
            X(m_in+1:m_ine, j, i) = Vx(:);


            My = M(idx+m_in:idx+m_in+n_out-1);
            Vy = My(1:n_out-1) - My(2:n_out);
            if(norm_fl)
                My = w_series2_scale(My, B(1,j,i), B(2,j,i));
                Vy = My(1:n_out-1) - My(2:n_out);
            end
            Y(1:n_out, j, i) = My(:);
            Y(n_out+1:n_oute, j, i) = Vy(:);

        end
    end
end