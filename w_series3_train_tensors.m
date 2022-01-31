function [X, Y, B, k_ob] = w_series3_train_tensors(M, m_in, n_out, l_sess, n_sess, norm_fl)

    % Number of observations in a session (training label(sequence) does
    % not touch test period
    k_ob = l_sess - m_in - n_out + 1;        

    % Re-format input into session tensor
    % ('ones' (not 'zeros') for X are for bias 'trick'
    X = zeros([m_in, k_ob, n_sess]);
    Y = zeros([n_out, k_ob, n_sess]);
    B = zeros([2, k_ob, n_sess]);

    for i = 1:n_sess
        for j = 1:k_ob
            % extract and scale observation sequence
            idx = (i-1)*l_sess + j;
            
            Mx = M(idx:idx+m_in-1);
            % scale bounds over observation span
            [B(1,j,i), B(2,j,i)] = bounds(Mx);
            if(norm_fl)
                Mx = w_series2_scale(Mx, B(1,j,i), B(2,j,i));
            end
            X(1:m_in, j, i) = Mx(:);

            My = M(idx+m_in:idx+m_in+n_out-1);
            if(norm_fl)
                My = w_series2_scale(My, B(1,j,i), B(2,j,i));
            end
            Y(:, j, i) = My(:);
        end
    end
end