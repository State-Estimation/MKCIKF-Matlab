clc;
clear;

%设置陀螺仪噪声
gyro_noise_std = 0.002*ones(3,1); 
gyro_bias_noise_std = 0.001*ones(3,1);

%设置加速度计噪声
accel_noise_std = 0.04*ones(3,1); 
accel_bias_noise_std = 0.001*ones(3,1);

%设置接触噪声和编码器噪声
contact_noise_std = 0.09*ones(3,1);
encoder_noise_std = deg2rad(0.5)*ones(3,1);

base_pose_std = [0.01*ones(3,1); 0.01*ones(3,1)]; 
base_velocity_std = 0.1*ones(3,1);
contact_position_std = 1.0*ones(3,1);
gyro_bias_std = 0.01*ones(3,1);
accel_bias_std = 0.1*ones(3,1);
forward_kinematics_std = 0.03*ones(3,1);

%初始化协方差矩阵
Qg = diag(gyro_noise_std.^2);          %角速度协方差矩阵
Qbg = diag(gyro_bias_noise_std.^2);    %角速度偏差协方差矩阵
Qa = diag(accel_noise_std.^2);         %加速度协方差矩阵
Qba = diag(accel_bias_noise_std.^2);   %加速度偏差协方差矩阵
Qco = diag(contact_noise_std.^2);      %接触协方差矩阵
Qe = diag(encoder_noise_std.^2);       %编码器协方差矩阵
Np = diag(forward_kinematics_std.^2);  %前向运动学协方差矩阵

%载入数据
load('data\ground_truth_data.mat');
load('data\imu_data.mat');
load('data\leg_data.mat');

%提取imu测量的加速度，角速度信息
acce_ = transpose(imu_data{:, 2:4});
gyro_ = transpose(imu_data{:, 5:7});

%提取位置、速度、方向的真实信息
position_true = transpose(ground_truth_data{:, 2:4});
velocity_true = transpose(ground_truth_data{:, 9:11});
R_true = transpose(ground_truth_data{:, 5:8});

%初始方向、位置、速度、imu偏差、重力加速度
a_R = quaternionToRotationMatrix(ground_truth_data{1, 5:8}); %获得初始姿态矩阵
p_ = transpose((ground_truth_data{1, 2:4}))+[0;0;0.355];   %初始位置
v_ = [0; 0; 0];
R_ = a_R;
ba_ = [1; 1; 1] * 3.552e-04;
bg_ = [1; 1; 1] * 1.655e-05;
gravity = [0; 0; -9.8];

%设置采样间隔和采样时间
dt = 0.001;
end_time =65 ;
time = 0:dt:end_time;
num_steps = length(time);

%存储滤波计算的位置、速度、方向信息
p_history = zeros(3, num_steps);
v_history = zeros(3, num_steps);
R_history = zeros(3, num_steps);
o_history = zeros(3, num_steps);
q_history = zeros(4, num_steps);
o_true = zeros(3, num_steps);
iter_history = zeros(1, num_steps);
ac_history =  zeros(3, num_steps);
Cx_history =  zeros(27, num_steps);
Cy_history =  zeros(6, num_steps);
%获得采样时间内的位置、速度、方向的真实信息
position_true_resampled = position_true(:, 1:num_steps);
velocity_true_resampled = velocity_true(:, 1:num_steps);
R_true_resampled = R_true(:, 1:num_steps);

%获得测量的以body为坐标系的四足位置、速度以及四足与地面的接触信息
d0_measure_pos = transpose(leg_data{1:num_steps, 2:4});
d1_measure_pos = transpose(leg_data{1:num_steps, 12:14});
d2_measure_pos = transpose(leg_data{1:num_steps, 22:24});
d3_measure_pos = transpose(leg_data{1:num_steps, 32:34});

d0_measure_speed = transpose(leg_data{1:num_steps, 5:7});
d1_measure_speed = transpose(leg_data{1:num_steps, 15:17});
d2_measure_speed = transpose(leg_data{1:num_steps, 25:27});
d3_measure_speed = transpose(leg_data{1:num_steps, 35:37});

contact_0 = transpose(leg_data{1:num_steps, 11});
contact_1 = transpose(leg_data{1:num_steps, 21});
contact_2 = transpose(leg_data{1:num_steps, 31});
contact_3 = transpose(leg_data{1:num_steps, 41});

%初始四足位置
d0 = [0; 0; 0];
d1 = [0; 0; 0];
d2 = [0; 0; 0];
d3 = [0; 0; 0];

%初始状态转移协方差矩阵
P = blkdiag(diag(base_pose_std(1:3).^2), ...
                  diag(base_velocity_std.^2), ...
                  diag(base_pose_std(4:6).^2), ...
                  diag(contact_position_std.^2), ...
                  diag(contact_position_std.^2), ...
                  diag(contact_position_std.^2), ...
                  diag(contact_position_std.^2), ...
                  diag(gyro_bias_std.^2),...
                  diag(accel_bias_std.^2));

%进行滤波
for i = 1:num_steps

    %第一次滤波时使用初始条件构建状态并进行预测和更新
    if i == 1
        [X, theta] = ConstructState(R_, v_, p_, d0, d1, d2, d3, bg_, ba_);
        [X_pred, theta_pred, P_pred] = PredictState(X, theta, P, gyro_, acce_, dt, gravity, i, contact_0, contact_1, contact_2, contact_3, Qg, Qa, Qco, Qbg, Qba);
        [X_update, theta_update, P_update,out] = UpdateState_forwardkinematics(X_pred, theta_pred, P_pred, contact_0, contact_1, contact_2, contact_3, d0_measure_pos, d1_measure_pos, d2_measure_pos, d3_measure_pos, i, Qe, Np);
        
        %将滤波结果存储在数组中
        p_history(:, i) = X_update(1:3, 5) + [0;0;0];
        v_history(:, i) = X_update(1:3, 4);
        o_history(:, i) = RotationMatrixToEulerAngles(X_update(1:3, 1:3));
        q_history(:, i) = rotationMatrixToQuaternion(X_update(1:3, 1:3));
        o_true(:, i) = QuaternionToEulerAngles(R_true_resampled(1:4, i));
        iter_history(:, i)= out.iter;
        Cx_history(:, i)=out.Cx;
        Cy_history(:, i)=out.Cy;
    %之后每次即使用上次更新过程后得到的状态作为本次滤波的初始状态
    else
        X = X_update;
        theta = theta_update;
        P = P_update;
        [X_pred, theta_pred, P_pred] = PredictState(X, theta, P, gyro_, acce_, dt, gravity, i, contact_0, contact_1, contact_2, contact_3, Qg, Qa, Qco, Qbg, Qba);
        [X_update, theta_update, P_update,out] = UpdateState_forwardkinematics(X_pred, theta_pred, P_pred, contact_0, contact_1, contact_2, contact_3, d0_measure_pos, d1_measure_pos, d2_measure_pos, d3_measure_pos, i, Qe, Np);
        
        %将滤波结果存储在数组中
        p_history(:, i) = X_update(1:3, 5) + [0;0;0];
        v_history(:, i) = X_update(1:3, 4);
        o_history(:, i) = RotationMatrixToEulerAngles(X_update(1:3, 1:3));
        q_history(:, i) = rotationMatrixToQuaternion(X_update(1:3, 1:3)); 
        o_true(:, i) = QuaternionToEulerAngles(R_true_resampled(1:4, i));
        iter_history(:, i)= out.iter;
        Cx_history(:, i)=out.Cx;
        Cy_history(:, i)=out.Cy;
    end

end
for i = 1:num_steps
    position_true_resampled(3,i) = position_true_resampled(3,i)+0.355;
end

%%画出滤波结果和真实结果的位置、方向、二维轨迹对比图
plot_results(time, p_history, position_true_resampled, o_history, o_true);
%plot_error(time,Cx_history,Cy_history,iter_history);
%需要打印的数据
outcome_iekf_mc = transpose([time;p_history(:,:);q_history(:,:)]);
% 将 result 保存为 MAT 文件
save('IEKF_MC.mat', 'outcome_iekf_mc');
% 将 outcome_eskf 保存为 TXT 文件
writematrix(outcome_iekf_mc, 'IEKF_MC', 'Delimiter', ' ')

outcome_gt = transpose([time;position_true_resampled(:,:);R_true_resampled(:,:)]);
save('GroundTruth.mat', 'outcome_gt');
% 将 outcome_eskf 保存为 TXT 文件
writematrix(outcome_gt, 'GroundTruth', 'Delimiter', ' ')

%状态的预测过程
function [X_pred, theta_pred, P_pred] = PredictState(X, theta, P, gyro_, acce_, dt, gravity, i, contact_0, contact_1, contact_2, contact_3, Qg, Qa, Qco, Qbg, Qba)

    [R_, v_, p_, d0, d1, d2, d3, bg_, ba_] = SeperateState(X, theta);
    R_pred = R_ * expm(skew((gyro_(1:3, i) - bg_) * dt));
    v_pred = v_ + gravity * dt + R_ * (acce_(1:3, i) - ba_) * dt;
    p_pred = p_ + v_ * dt + 0.5 * gravity * dt * dt+ 0.5 * R_ * (acce_(1:3, i) - ba_) * dt * dt;
    bg_pred = bg_;
    ba_pred = ba_;
    Ac = [     zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3);
          skew(gravity), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3);
               zeros(3),   eye(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3);
               zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3);
               zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3);
               zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3);
               zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3), zeros(3)]; 
    Ac = blkdiag(Ac, zeros(6));
    Ac(1:end-6, end-5:end) = [         -R_, zeros(3);
                              -skew(v_)*R_,      -R_;
                              -skew(p_)*R_, zeros(3);
                              -skew(d0)*R_, zeros(3);
                              -skew(d1)*R_, zeros(3);
                              -skew(d2)*R_, zeros(3);
                              -skew(d3)*R_, zeros(3)];
    Ak = eye(size(Ac)) + Ac * dt;
    Lc = blkdiag(Adjoint(X), eye(6));
    Qc = blkdiag(Qg, Qa, zeros(3), (Qco+(1e4*eye(3).*(1-contact_0(i)))), (Qco+(1e4*eye(3).*(1-contact_1(i)))), (Qco+(1e4*eye(3).*(1-contact_2(i)))), (Qco+(1e4*eye(3).*(1-contact_3(i)))), Qbg, Qba);
     Qk = Ak * Lc * Qc * Lc' * Ak' * dt; %原论文
    %Qk = Qc;
    P_pred =  Ak * P * Ak' + Qk;
    [X_pred, theta_pred] =  ConstructState(R_pred, v_pred, p_pred, d0, d1, d2, d3, bg_pred, ba_pred);

end


%状态的更新过程
function [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI)
    
    num = 4;
    cnt = 4;
    bp =  chol(P_pred, 'lower');
    br =  chol(N, 'lower');
    n=length(P_pred);
    m=length(N);
     X_Cell = repmat({X_pred}, 1, length(Y)/size(X_pred, 1));
    Z = blkdiag(X_Cell{:}) * Y - b;
    new_xyz=[10,10,10];
    new_leg = [new_xyz,new_xyz,new_xyz,new_xyz];
    sigma_p=[10*ones(9,1);new_leg';10*ones(6,1)];
    sigma_r=[10*ones(m,1)];
    
     while(num>0)
        if(num == cnt)
            x_=zeros(n,1); 
            x_tlast=zeros(n,1); 
        else
             x_tlast=x_t; 
        end
        num=num-1;
        dp= bp\x_;
        z_=Z(1:m);
        %z_=nonzeros(Z);
        dr= br\z_;
        wp= bp\x_tlast;
        wr= inv(br)*H*x_tlast;
        ep=dp-wp;
        er=dr-wr;
        %  P_ and R
        Cx=diag(exp(-ep.*ep ./(2*sigma_p.*sigma_p)));
        Cy=diag(exp(-er.*er./(2*sigma_r.*sigma_r)));
        for kk=1:n
            if(Cx(kk,kk)<0.0001)
                Cx(kk,kk)=0.0001;
            end   
        end
        for kk=1:m
            if(Cy(kk,kk)<0.0001)
                Cy(kk,kk)=0.0001;
            end
        end
        R_1=br/Cy*br';
        P_1=bp/Cx*bp';
        S = H * P_1 * H' + R_1;
        K_1 = (P_1 * H')/S;
        x_t = K_1*PI*Z;
        xe=norm(x_t-x_tlast)/(norm(x_tlast)+0.001);
        % stored data for inspectatio
        if(xe<0.01)
            break
        end
    end
        K=K_1;
        delta=x_t;
        dX = exp_(delta(1:end-6));
        dtheta = delta(end-5:end);
        X_update = dX * X_pred;
        theta_update = dtheta + theta_pred;
        I = eye(size(P_pred));
        P_update = (I - K*H) * P_pred * (I - K*H)' + K * R_1 * K';
        out.iter=cnt-num;
        out.Cx=diag(Cx);
        out.Cy=diag(Cy);

end


%根据四足与地面的接触情况来使用不同的观测方程
function [X_update, theta_update, P_update,out] = UpdateState_forwardkinematics(X_pred, theta_pred, P_pred, contact_0, contact_1, contact_2, contact_3, d0_measure_pos, d1_measure_pos, d2_measure_pos, d3_measure_pos, i, Qe, Np)
    
    [R_pred, v_pred, p_pred, d0, d1, d2, d3, bg_pred, ba_pred] = SeperateState(X_pred, theta_pred);

    %四条腿同时与地面接触
    if contact_0(i) == 1 && contact_1(i) == 1 && contact_2(i) == 1 && contact_3(i) == 1

%注意观测变量不能写成这种形式，会导致观测误差Z = X * Y - b错误，最后结果发散
%         Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
%              d1_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
%              d2_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
%              d3_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0];


        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
             d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0;
             d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0;
             d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];       
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3), zeros(3), zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np, zeros(3), zeros(3);
             zeros(3), zeros(3), R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), zeros(3), zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,27);
              zeros(3,9), eye(3), zeros(3,6), zeros(3,18);
              zeros(3,18), eye(3), zeros(3,6), zeros(3,9);
              zeros(3,27), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);


    %只有三条腿同时与地面接触
    elseif contact_0(i) == 0 && contact_1(i) == 1 && contact_2(i) == 1 && contact_3(i) == 1

        Y = [d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0;
             d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0;
             d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3), zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,18);
              zeros(3,9), eye(3), zeros(3,6), zeros(3,9);
              zeros(3,18), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 1 && contact_1(i) == 0 && contact_2(i) == 1 && contact_3(i) == 1

        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
             d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0;
             d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3), zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,18);
              zeros(3,9), eye(3), zeros(3,6), zeros(3,9);
              zeros(3,18), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 1 && contact_1(i) == 1 && contact_2(i) == 0 && contact_3(i) == 1

        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
             d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0;
             d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3), zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,18);
              zeros(3,9), eye(3), zeros(3,6), zeros(3,9);
              zeros(3,18), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 1 && contact_1(i) == 1 && contact_2(i) == 1 && contact_3(i) == 0

        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
             d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0;
             d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3), zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,18);
              zeros(3,9), eye(3), zeros(3,6), zeros(3,9);
              zeros(3,18), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);


    %只有两条腿同时与地面接触
    elseif contact_0(i) == 0 && contact_1(i) == 0 && contact_2(i) == 1 && contact_3(i) == 1

        Y = [d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0;
             d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,9);
              zeros(3,9), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 0 && contact_1(i) == 1 && contact_2(i) == 0 && contact_3(i) == 1

        Y = [d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0;
             d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,9);
              zeros(3,9), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 0 && contact_1(i) == 1 && contact_2(i) == 1 && contact_3(i) == 0

        Y = [d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0;
             d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,9);
              zeros(3,9), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 1 && contact_1(i) == 0 && contact_2(i) == 0 && contact_3(i) == 1

        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
             d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,9);
              zeros(3,9), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 1 && contact_1(i) == 0 && contact_2(i) == 1 && contact_3(i) == 0

        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
             d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,9);
              zeros(3,9), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 1 && contact_1(i) == 1 && contact_2(i) == 0 && contact_3(i) == 0

        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0;
             d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6);
             zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6)];
        N = [R_pred*Qe*R_pred' + Np, zeros(3);
             zeros(3), R_pred*Qe*R_pred' + Np];
        PI = [eye(3), zeros(3,6), zeros(3,9);
              zeros(3,9), eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    
    %只有一条腿与地面接触
    elseif contact_0(i) == 1 && contact_1(i) == 0 && contact_2(i) == 0 && contact_3(i) == 0

        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6)];
        N = R_pred*Qe*R_pred' + Np;
        PI = [eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 0 && contact_1(i) == 1 && contact_2(i) == 0 && contact_3(i) == 0 

        Y = [d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6)];
        N = R_pred*Qe*R_pred' + Np;
        PI = [eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 0 && contact_1(i) == 0 && contact_2(i) == 1 && contact_3(i) == 0

        Y = [d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6)];
        N = R_pred*Qe*R_pred' + Np;
        PI = [eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 0 && contact_1(i) == 0 && contact_2(i) == 0 && contact_3(i) == 1

        Y = [d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = R_pred*Qe*R_pred' + Np;
        PI = [eye(3), zeros(3,6)];
        [X_update, theta_update, P_update,out] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);


    end
end


%对状态进行分离，提取其中的各个状态变量
function [R_, v_, p_, d0, d1, d2, d3, bg_, ba_] = SeperateState(X, theta)

    R_ = X(1:3, 1:3);
    v_ = X(1:3, 4);
    p_ = X(1:3, 5);
    d0 = X(1:3, 6);
    d1 = X(1:3, 7);
    d2 = X(1:3, 8);
    d3 = X(1:3, 9);
    bg_ = theta(1:3);
    ba_ = theta(4:6);

end


%利用状态变量构建状态
function [X, theta] = ConstructState(R_, v_, p_, d0, d1, d2, d3, bg_, ba_)
  
    X = [R_, v_, p_, d0, d1, d2, d3;
        0, 0, 0, 1, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 1, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 1, 0, 0, 0;
        0, 0, 0, 0, 0, 0, 1, 0, 0;
        0, 0, 0, 0, 0, 0, 0, 1, 0;
        0, 0, 0, 0, 0, 0, 0, 0, 1];
    theta = [bg_; ba_];

end


%欧拉角转旋转矩阵
function rotation_matrix = EulerAnglesToRotationMatrix(euler_angles)

    roll = euler_angles(1);%x
    pitch = euler_angles(2);%y
    yaw = euler_angles(3);%z
    Rx = [1, 0, 0; 0, cos(roll), -sin(roll); 0, sin(roll), cos(roll)];
    Ry = [cos(pitch), 0, sin(pitch); 0, 1, 0; -sin(pitch), 0, cos(pitch)];
    Rz = [cos(yaw), -sin(yaw), 0; sin(yaw), cos(yaw), 0; 0, 0, 1];
    rotation_matrix = Rx * Ry * Rz;

end


%反对称矩阵
function A = skew(v)

    A = [0, -v(3), v(2);
        v(3), 0, -v(1);
        -v(2), v(1), 0];

end


%指数映射
function exp_result = exp_(v)

    N = (length(v)-3) / 3;
    Lg = zeros(N + 3);
    Lg(1:3, :) = [skew(v(1:3)), reshape(v(4:end), 3, [])];
    exp_result = expm(Lg);

end


%伴随矩阵运算
function Adjx = Adjoint(X)

    N = size(X, 2) - 3;
    R = X(1:3, 1:3);
    R_cell = repmat({R}, 1, N + 1);
    Adjx = blkdiag(R_cell{:});
    for i = 1:N
        Adjx(1 + 3*i:3 + 3*i, 1:3) = skew(X(1:3, 3 + i))*R;
    end

end


%旋转矩阵转欧拉角
function euler_angles = RotationMatrixToEulerAngles(rotation_matrix)

    euler_angles = transpose(rotm2eul(rotation_matrix, 'ZYX'));
    euler_angles = rad2deg(euler_angles);

end


%四元数转欧拉角
function euler_angles = QuaternionToEulerAngles(quaternion)
    
    x = quaternion(1);
    y = quaternion(2);
    z = quaternion(3);
    w = quaternion(4);
    norm = sqrt(x^2 + y^2 + z^2 + w^2);
    x = x / norm;
    y = y / norm;
    z = z / norm;
    w = w / norm;
    % Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z);
    cosr_cosp = 1 - 2 * (x * x + y * y);
    roll = atan2(sinr_cosp, cosr_cosp);
    
    % Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x);
    if abs(sinp) >= 1
        pitch = sign(sinp) * pi / 2; % use 90 degrees if out of range
    else
        pitch = asin(sinp);
    end
    
    % Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y);
    cosy_cosp = 1 - 2 * (y * y + z * z);
    yaw = atan2(siny_cosp, cosy_cosp);
    
    euler_angles = transpose([roll, pitch, yaw]);
  
    % 将弧度转为角度
    euler_angles = rad2deg(euler_angles);

end


%四元数转旋转矩阵
function rotation_matrix = QuaternionToRotationMatrix(quaternion)

    x = quaternion(1);
    y = quaternion(2);
    z = quaternion(3);
    w = quaternion(4);
    norm = sqrt(x^2 + y^2 + z^2 + w^2);
    x = x / norm;
    y = y / norm;
    z = z / norm;
    w = w / norm;
    rotation_matrix = [1 - 2*y^2 - 2*z^2, 2*x*y - 2*z*w, 2*x*z + 2*y*w;
                       2*x*y + 2*z*w, 1 - 2*x^2 - 2*z^2, 2*y*z - 2*x*w;
                       2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x^2 - 2*y^2];

end

%% 四元数转R矩阵
function R = quaternionToRotationMatrix(quaternion)
    %QUATERNIONTOROTATIONMATRIX Converts quaternion to rotation matrix.
    % Input:
    %   x, y, z, w - Quaternion components
    % Output:
    %   R - 3x3 rotation matrix
    x = quaternion(1);
    y = quaternion(2);
    z = quaternion(3);
    w = quaternion(4);
    % Normalize the quaternion
    norm_q = sqrt(x^2 + y^2 + z^2 + w^2);
    x = x / norm_q;
    y = y / norm_q;
    z = z / norm_q;
    w = w / norm_q;
    
    % Compute the rotation matrix elements
    R = [1 - 2*y^2 - 2*z^2, 2*x*y - 2*z*w, 2*x*z + 2*y*w;
         2*x*y + 2*z*w, 1 - 2*x^2 - 2*z^2, 2*y*z - 2*x*w;
         2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x^2 - 2*y^2];
end

%% R矩阵转四元数
function quaternion = rotationMatrixToQuaternion(R)
    %ROTATIONMATRIXTOQUATERNION Converts rotation matrix to quaternion.
    % Input:
    %   R - 3x3 rotation matrix
    % Output:
    %   x, y, z, w - Quaternion components
    
    % Ensure the matrix is orthogonal and its determinant is 1
    if abs(det(R) - 1) > 1e-6
        error('Input matrix is not a valid rotation matrix');
    end
    
    % Compute the trace of the matrix
    trace_R = trace(R);
    
    if trace_R > 0
        S = sqrt(trace_R + 1.0) * 2; % S = 4 * w
        w = 0.25 * S;
        x = (R(3,2) - R(2,3)) / S;
        y = (R(1,3) - R(3,1)) / S;
        z = (R(2,1) - R(1,2)) / S;
    elseif (R(1,1) > R(2,2)) && (R(1,1) > R(3,3))
        S = sqrt(1.0 + R(1,1) - R(2,2) - R(3,3)) * 2; % S = 4 * x
        w = (R(3,2) - R(2,3)) / S;
        x = 0.25 * S;
        y = (R(1,2) + R(2,1)) / S;
        z = (R(1,3) + R(3,1)) / S;
    elseif R(2,2) > R(3,3)
        S = sqrt(1.0 + R(2,2) - R(1,1) - R(3,3)) * 2; % S = 4 * y
        w = (R(1,3) - R(3,1)) / S;
        x = (R(1,2) + R(2,1)) / S;
        y = 0.25 * S;
        z = (R(2,3) + R(3,2)) / S;
    else
        S = sqrt(1.0 + R(3,3) - R(1,1) - R(2,2)) * 2; % S = 4 * z
        w = (R(2,1) - R(1,2)) / S;
        x = (R(1,3) + R(3,1)) / S;
        y = (R(2,3) + R(3,2)) / S;
        z = 0.25 * S;
    end
    quaternion = [x, y, z, w];
end

%画出滤波结果和真实结果的位置、方向、二维轨迹对比图
function plot_results(time, p_history, position_true_resampled, o_history, o_true)

    %位置对比图
    figure;
    subplot(3, 1, 1);
    plot(time, p_history(1, :), 'b', time, position_true_resampled(1, :), 'r');
    title('Position - X');
    legend('Computed', 'True');

    subplot(3, 1, 2);
    plot(time, p_history(2, :), 'b', time, position_true_resampled(2, :), 'r');
    title('Position - Y');
    legend('Computed', 'True');    

    subplot(3, 1, 3);
    plot(time, p_history(3, :), 'b', time, position_true_resampled(3, :), 'r');
    title('Position - Z');
    legend('Computed', 'True');

    %方向对比图
    figure;
    subplot(3, 1, 1);
    plot(time, o_history(3, :), 'b', time, o_true(1, :), 'r');
    title('Orientation - Roll');
    legend('Computed', 'True');

    subplot(3, 1, 2);
    plot(time, o_history(2, :), 'b', time, o_true(2, :), 'r');
    title('Orientation - Pitch');
    legend('Computed', 'True');

    subplot(3, 1, 3);
    plot(time, o_history(1, :), 'b', time, o_true(3, :), 'r');
    title('Orientation - Yaw');
    legend('Computed', 'True');

    %二维轨迹对比图
    figure;
    plot(position_true_resampled(1, :), position_true_resampled(2, :), 'r-', 'LineWidth', 1.5);
    hold on;
    plot(p_history(1, :), p_history(2, :), 'b', 'LineWidth', 1.5);
    title('2D Trajectory - XY Plane');
    xlabel('X Position');
    ylabel('Y Position');
    legend('True', 'Computed');
    hold off;


end

function plot_error(time,Cx_history,Cy_history,iter_history)

    figure;
    subplot(4, 1, 1);
    plot(time, Cx_history(10:12, :),  'LineWidth', 0.5);
    title('leg o');
    legend('Column 1', 'Column 2', 'Column 3'); % 为每一列数据添加图例
    xlabel('Time');
    ylabel('Value');
    
    subplot(4, 1, 2);
    plot(time, Cx_history(13:15, :),  'LineWidth', 0.5);
    title('leg 1');
    legend('Column 1', 'Column 2', 'Column 3'); % 为每一列数据添加图例
    
    subplot(4, 1, 3);
    plot(time, Cx_history(16:18, :),  'LineWidth', 0.5);
    title('leg 2');
    legend('Column 1', 'Column 2', 'Column 3'); % 为每一列数据添加图例
    
    subplot(4, 1, 4);
    plot(time, Cx_history(19:21, :),  'LineWidth', 0.5);
    title('leg 3');
    legend('Column 1', 'Column 2', 'Column 3'); % 为每一列数据添加图例 
    
    figure;
    hold on;
    plot(time, Cx_history(1:6, :),  'LineWidth', 0.5);
    title('errory');
    xlabel('Time');
    ylabel(' error cy');
    hold off;
    
     figure;
    hold on;
    plot(time, iter_history(1, :), 'r', 'LineWidth', 0.5);
    title('iter times');
    xlabel('Time');
    ylabel(' iter');
    hold off;
end

