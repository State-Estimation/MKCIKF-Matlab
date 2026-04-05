clc;
clear;

%设置陀螺仪噪声
gyro_noise_std = 0.002*ones(3,1); 
gyro_bias_noise_std = 0.001*ones(3,1);
accel_noise_std = 0.04*ones(3,1); 
accel_bias_noise_std = 0.001*ones(3,1);

%设置接触噪声和编码器噪声
contact_noise_std = 0.05*ones(3,1);
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

%提取位置、方向的真实信息
position_true = transpose(ground_truth_data{:, 2:4});
R_true = transpose(ground_truth_data{:, 5:8});

%初始方向、位置、速度、imu偏差、重力加速度
a_R = [-1,0,0;0,-1,0;0,0,1];
p_ = transpose((ground_truth_data{1, 2:4}))+[0;0;0.355];
%p_ = [0; 0; 0];
v_ = [0; 0; 0];
R_ = QuaternionToRotationMatrix(ground_truth_data{1, 5:8});
%R_=[1,0,0;0,1,0;0,0,1];
ba_ = [1; 1; 1] * 3.552e-04 * 0;
bg_ = [1; 1; 1] * 1.655e-05 * 0;
gravity = [0; 0; -9.8];

%设置采样间隔和采样时间
%11000/900
% 确定传感器时间向量
dt = 1/769;      % 采样时间间隔
end_time = 11000*dt;    %采样时间为8s
time = 0:dt:end_time; %从第1个数据开始
% 确定真值采样时间
dt_true = 1/62;      % 采样时间间隔
end_time = 900*dt_true;    %采样时间为8s
time_true = 0:dt_true:end_time; %从第1个数据开始
% 初始化数组以存储每个时间步的结果
num_steps = length(time);
num_steps_true = length(time_true);

%存储滤波计算的位置、速度、方向信息
p_history = zeros(3, num_steps);
v_history = zeros(3, num_steps);
R_history = zeros(3, 3, num_steps);
o_history = zeros(3, num_steps);
q_history = zeros(4, num_steps);
o_true = zeros(3, num_steps_true);

%获得采样时间内的位置、速度、方向的真实信息
position_true_resampled = position_true(:, 1:num_steps_true);
R_true_resampled = R_true(:, 1:num_steps_true);

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


for i = 1:num_steps_true
    o_true(:, i) =  QuaternionToEulerAngles(transpose( R_true_resampled(:,i) ));
end


%进行滤波
for i = 1:num_steps


    %第一次滤波时使用初始条件构建状态并进行预测和更新
    if i == 1
        [X, theta] = ConstructState(R_, v_, p_, d0, d1, d2, d3, bg_, ba_);
        [X_pred, theta_pred, P_pred] = PredictState(X, theta, P, gyro_, acce_, dt, gravity, i, contact_0, contact_1, contact_2, contact_3, Qg, Qa, Qco, Qbg, Qba);
        [X_update, theta_update, P_update] = UpdateState_forwardkinematics(X_pred, theta_pred, P_pred, contact_0, contact_1, contact_2, contact_3, d0_measure_pos, d1_measure_pos, d2_measure_pos, d3_measure_pos, i, Qe, Np);
        
        %将滤波结果存储在数组中
        p_history(:, i) = a_R * (X_update(1:3, 5) + [0;0;0]) ;
        v_history(:, i) = X_update(1:3, 4);
        o_history(:, i) = a_R * RotationMatrixToEulerAngles(X_update(1:3, 1:3));
        q_history(:, i) =  rotationMatrixToQuaternion(X_update(1:3, 1:3));

    %之后每次即使用上次更新过程后得到的状态作为本次滤波的初始状态
    else
        X = X_update;
        theta = theta_update;
        P = P_update;
        [X_pred, theta_pred, P_pred] = PredictState(X, theta, P, gyro_, acce_, dt, gravity, i, contact_0, contact_1, contact_2, contact_3, Qg, Qa, Qco, Qbg, Qba);
        [X_update, theta_update, P_update] = UpdateState_forwardkinematics(X_pred, theta_pred, P_pred, contact_0, contact_1, contact_2, contact_3, d0_measure_pos, d1_measure_pos, d2_measure_pos, d3_measure_pos, i, Qe, Np);
        
        %将滤波结果存储在数组中
        p_history(:, i) = a_R * (X_update(1:3, 5) + [0;0;0]);
        v_history(:, i) = X_update(1:3, 4);
        o_history(:, i) = a_R * RotationMatrixToEulerAngles(X_update(1:3, 1:3));
        q_history(:, i) = rotationMatrixToQuaternion(X_update(1:3, 1:3));

    end

end


for i = 1:num_steps_true
    position_true_resampled(3,i) = position_true_resampled(3,i)+0.355;
end

%%画出滤波结果和真实结果的位置、方向、二维轨迹对比图
%plot_results(time, p_history, o_history);
plot_results(time, time_true, p_history, position_true_resampled, o_history, o_true);

%需要打印的数据
outcome_iekf = transpose([time;p_history(:,:);q_history(:,:)]);
% 将 outcome_only_imu 保存为 MAT 文件
save('IEKF.mat', 'outcome_iekf');

% 将 outcome_eskf 保存为 TXT 文件
writematrix(outcome_iekf, 'IEKF', 'Delimiter', ' ')

outcome_gt = transpose([time_true;position_true_resampled(:,:);R_true_resampled(:,:)]);
save('GroundTruth.mat', 'outcome_gt');
% 将 outcome_eskf 保存为 TXT 文件
writematrix(outcome_gt, 'GroundTruth', 'Delimiter', ' ')

%状态的预测过程
function [X_pred, theta_pred, P_pred] = PredictState(X, theta, P, gyro_, acce_, dt, gravity, i, contact_0, contact_1, contact_2, contact_3, Qg, Qa, Qco, Qbg, Qba)

    [R_, v_, p_, d0, d1, d2, d3, bg_, ba_] = SeperateState(X, theta);
%     R_pred = R_ * expm(skew((gyro_(1:3, i) - bg_) * dt));
    rotationMatrix = eulerAnglesToRotationMatrix((gyro_(1:3, i) - bg_) * dt);  % 欧拉角转旋转矩阵
    R_pred = R_ * rotationMatrix;% 假设欧拉角顺序为 ZYX
    v_pred = v_ + gravity * dt + R_ * (acce_(1:3, i) - ba_) * dt;
    p_pred = p_ + v_ * dt + 0.5 * gravity * dt * dt + 0.5 * R_ * (acce_(1:3, i) - ba_) * dt * dt;
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
    Qk =  Ak * Lc * Qc * Lc' * Ak' * dt;
    P_pred =  Ak * P * Ak' + Qk;
    [X_pred, theta_pred] =  ConstructState(R_pred, v_pred, p_pred, d0, d1, d2, d3, bg_pred, ba_pred);

end


%状态的更新过程
function [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI)

    S = H * P_pred *H' + N;
    K = (P_pred * H') / S;
    X_Cell = repmat({X_pred}, 1, length(Y)/size(X_pred, 1));
    Z = blkdiag(X_Cell{:}) * Y - b;
    delta = K * PI * Z;
    dX = exp(delta(1:end-6));
    dtheta = delta(end-5:end);
    X_update = dX * X_pred;
    theta_update = dtheta + theta_pred;
    I = eye(size(P_pred));
    P_update = (I - K*H) * P_pred * (I - K*H)' + K * N * K';

end


%根据四足与地面的接触情况来使用不同的观测方程
function [X_update, theta_update, P_update] = UpdateState_forwardkinematics(X_pred, theta_pred, P_pred, contact_0, contact_1, contact_2, contact_3, d0_measure_pos, d1_measure_pos, d2_measure_pos, d3_measure_pos, i, Qe, Np)
    
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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);


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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);


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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

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
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    
    %只有一条腿与地面接触
    elseif contact_0(i) == 1 && contact_1(i) == 0 && contact_2(i) == 0 && contact_3(i) == 0

        Y = [d0_measure_pos(1:3, i); 0; 1; -1; 0; 0; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), eye(3), zeros(3), zeros(3), zeros(3), zeros(3, 6)];
        N = R_pred*Qe*R_pred' + Np;
        PI = [eye(3), zeros(3,6)];
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 0 && contact_1(i) == 1 && contact_2(i) == 0 && contact_3(i) == 0 

        Y = [d1_measure_pos(1:3, i); 0; 1; 0; -1; 0; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), eye(3), zeros(3), zeros(3), zeros(3, 6)];
        N = R_pred*Qe*R_pred' + Np;
        PI = [eye(3), zeros(3,6)];
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 0 && contact_1(i) == 0 && contact_2(i) == 1 && contact_3(i) == 0

        Y = [d2_measure_pos(1:3, i); 0; 1; 0; 0; -1; 0];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), eye(3), zeros(3), zeros(3, 6)];
        N = R_pred*Qe*R_pred' + Np;
        PI = [eye(3), zeros(3,6)];
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);

    elseif contact_0(i) == 0 && contact_1(i) == 0 && contact_2(i) == 0 && contact_3(i) == 1

        Y = [d3_measure_pos(1:3, i); 0; 1; 0; 0; 0; -1];
        b = zeros(size(Y));
        H = [zeros(3), zeros(3), -eye(3), zeros(3), zeros(3), zeros(3), eye(3), zeros(3, 6)];
        N = R_pred*Qe*R_pred' + Np;
        PI = [eye(3), zeros(3,6)];
        [X_update, theta_update, P_update] = UpdateState(X_pred, theta_pred, P_pred, Y, b, H, N, PI);


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
function exp_result = exp(v)

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
    
    %把弧度以roll pitch yaw 的顺序发布
    yaw = euler_angles(1);
    pitch = euler_angles(2);
    roll = euler_angles(3);
    euler_angles = [roll;pitch;yaw];
    
    % 将弧度转为角度
    euler_angles = rad2deg(euler_angles);

end

%% 欧拉角函数
function R = eulerAnglesToRotationMatrix(angles)
    % angles 是一个包含绕X轴、Y轴和Z轴的欧拉角的 1x3 向量

    % 提取欧拉角
    roll = angles(1); %绕x轴旋转
    pitch = angles(2);   %绕y轴旋转
    yaw = angles(3);  %绕z轴旋转

    % 构建绕X轴的旋转矩阵
    Rx = [1, 0, 0; 0, cos(roll), -sin(roll); 0, sin(roll), cos(roll)];

    % 构建绕Y轴的旋转矩阵
    Ry = [cos(pitch), 0, sin(pitch); 0, 1, 0; -sin(pitch), 0, cos(pitch)];
    
    % 构建绕Z轴的旋转矩阵
    Rz = [cos(yaw), -sin(yaw), 0; sin(yaw), cos(yaw), 0; 0, 0, 1];

    % 构建最终的旋转矩阵
    R = Rz * Ry * Rx;
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
function plot_results(time, time_true, p_history, position_true_resampled, o_history, o_true)

    %位置对比图
    figure;
    subplot(3, 1, 1);
    plot(time, p_history(1, :), 'b', time_true, position_true_resampled(1, :), 'r');
    title('Position - X');
    legend('Computed', 'True');

    subplot(3, 1, 2);
    plot(time, p_history(2, :), 'b', time_true, position_true_resampled(2, :), 'r');
    title('Position - Y');
    legend('Computed', 'True');    

    subplot(3, 1, 3);
    plot(time, p_history(3, :), 'b', time_true, position_true_resampled(3, :), 'r');
    title('Position - Z');
    legend('Computed', 'True');

    %方向对比图
    figure;
    subplot(3, 1, 1);
    plot(time, o_history(1, :), 'b', time_true, o_true(1, :), 'r');
    title('Orientation - Roll');
    legend('Computed', 'True');

    subplot(3, 1, 2);
    plot(time, o_history(2, :), 'b', time_true, o_true(2, :), 'r');
    title('Orientation - Pitch');
    legend('Computed', 'True');

    subplot(3, 1, 3);
    plot(time, o_history(3, :), 'b', time_true, o_true(3, :), 'r');
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









