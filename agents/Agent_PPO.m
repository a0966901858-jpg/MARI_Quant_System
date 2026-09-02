classdef Agent_PPO < handle
    % =========================================================================
    % 類別：Agent_PPO (CIO 總管三軌代理人 - 核心骨幹)
    % 升級：Phase 14.20 (★ 修正動作空間雜訊後即時正規化、對齊環境因果與策略梯度)
    % 職責：接收 5 維宏觀狀態，輸出 [0,1] 有界資金配置權重，並透過 PPO 更新網路
    % =========================================================================
    
    properties
        Actor               % 策略網路 (Actor)：負責根據狀態給出動作 (輸出 3 維 Logits)
        Critic              % 價值網路 (Critic)：負責評估當前狀態的價值 (輸出 1 維 Baseline Value)
        
        % Adam 優化器狀態快取：儲存梯度的一階動量 (AvgG) 與二階動量 (AvgSG)
        AvgG_Actor = []; AvgSG_Actor = [];
        AvgG_Critic = []; AvgSG_Critic = [];
        Iter = 0;           % 全域訓練迭代次數計數器
        
        % ★ PPO 與強化學習核心超參數
        ActionStd = 0.30;   % 動作標準差，用於建構高斯分佈與探索
        PPO_Epochs = 3;     % 每次收集資料後重複更新的次數
        PPO_Clip = 0.2;     % PPO 截斷閾值 (epsilon)
        EntropyCoeff = 0.05;% 熵係數 (Entropy Bonus)，鼓勵有效探索
    end
    
    methods
        % 建構子：初始化 PPO 代理人的神經網路與狀態
        function obj = Agent_PPO()
            % 1. 建立 Actor 網路 (多層感知機 MLP 架構)
            act_layers = [
                featureInputLayer(5, 'Name', 'state_in')      % 輸入層：接收 5 維宏觀市場狀態
                fullyConnectedLayer(64, 'Name', 'fc1')        % 隱藏層 1：64 個神經元
                reluLayer('Name', 'relu1')                    % ReLU 激勵函數
                fullyConnectedLayer(32, 'Name', 'fc2')        % 隱藏層 2：32 個神經元
                reluLayer('Name', 'relu2')                    % ReLU 激勵函數
                fullyConnectedLayer(3, 'Name', 'action_out')  % 輸出層：輸出 3 維動作 Logits
            ];
            obj.Actor = dlnetwork(act_layers);
            
            % 2. 建立 Critic 網路 (多層感知機 MLP 架構)
            crit_layers = [
                featureInputLayer(5, 'Name', 'state_in')      % 輸入層：接收 5 維宏觀狀態
                fullyConnectedLayer(64, 'Name', 'fc1')        % 隱藏層 1：64 個神經元
                reluLayer('Name', 'relu1')                    % ReLU 激勵函數
                fullyConnectedLayer(32, 'Name', 'fc2')        % 隱藏層 2：32 個神經元
                reluLayer('Name', 'relu2')                    % ReLU 激勵函數
                fullyConnectedLayer(1, 'Name', 'value_out')   % 輸出層：輸出 1 維 Baseline 價值
            ];
            obj.Critic = dlnetwork(crit_layers);
            
            % 強制留在 CPU 以啟用 Intel MKL 多核矩陣加速
            fprintf(' 🧠 [Agent_PPO] 實例化完成。已鎖定純 CPU 運算模式，準備釋放多核算力。\n');
        end
        
        % 取得動作 (推論階段 / 採樣階段)
        function [actions, value] = get_actions(obj, state, noise_std)
            % 將傳入矩陣轉為 dlarray，並賦予 'CB' 標籤
            dl_state = dlarray(state, 'CB');
            
            % Actor 預測動作分佈參數 (Logits)，Critic 預估當前狀態價值
            logits = predict(obj.Actor, dl_state);
            val_raw = predict(obj.Critic, dl_state);
            
            % 將無界的 Logits 轉換為 [0,1] 的有界資金權重
            mu_dl = obj.apply_activation(logits);
            
            % 提取純數值矩陣
            actions = extractdata(mu_dl);
            value = extractdata(val_raw);
            
            % 探索階段加上高斯雜訊
            if noise_std > 0
                actions = actions + noise_std * randn(size(actions), 'single');
                actions = max(0, min(1, actions));
                
                % ★ 計畫書修正 (問題 7-1 🟠 P1)：雜訊後立即重新正規化時空專家權重
                % 確保經驗回放池中的動作與環境物理執行的動作完全同一，消除非單射轉換的梯度噪聲
                ts_sum = actions(1, :) + actions(2, :);
                zero_mask = (ts_sum == 0);
                actions(1, zero_mask) = 0.5;
                actions(2, zero_mask) = 0.5;
                actions(1:2, ~zero_mask) = actions(1:2, ~zero_mask) ./ ts_sum(~zero_mask);
            end
        end
        
        % 更新神經網路權重 (訓練階段)
        function loss_val = update_weights(obj, states, actions, rewards, lr)
            dl_s = dlarray(states, 'CB');
            dl_a = dlarray(actions, 'CB');
            dl_r = dlarray(rewards, 'CB');
            
            % 1. 計算舊策略基礎資料
            v_old_dl = predict(obj.Critic, dl_s);
            v_old = extractdata(v_old_dl);
            
            % 計算標準化優勢函數 Advantage
            adv = extractdata(dl_r) - v_old;
            adv = (adv - mean(adv, 'all')) ./ (std(adv, 0, 'all') + 1e-8);
            
            % 取得歷史狀態在舊策略下的輸出
            logits_old = predict(obj.Actor, dl_s);
            mu_old_dl = obj.apply_activation(logits_old);
            mu_old = extractdata(mu_old_dl);
            
            % 計算舊策略對數機率
            log_p_old = sum(-0.5 * ((extractdata(dl_a) - mu_old) ./ obj.ActionStd).^2, 1);
            
            dl_adv = dlarray(adv, 'CB');
            dl_log_p_old = dlarray(log_p_old, 'CB');
            
            loss_sum = 0;
            
            % 2. 執行 PPO 多次循環更新
            for ppo_ep = 1:obj.PPO_Epochs
                obj.Iter = obj.Iter + 1;
                
                % 進入自動微分環境計算梯度
                [lossA, lossC, gradA, gradC] = dlfeval(@obj.ppo_loss, obj.Actor, obj.Critic, ...
                    dl_s, dl_a, dl_r, dl_adv, dl_log_p_old, obj.ActionStd, obj.PPO_Clip, obj.EntropyCoeff);
                
                % 梯度裁剪防爆
                gradA = dlupdate(@(g) obj.clip_gradient(g, 1.0), gradA);
                gradC = dlupdate(@(g) obj.clip_gradient(g, 1.0), gradC);
                
                % Adam 更新參數
                [obj.Actor, obj.AvgG_Actor, obj.AvgSG_Actor] = adamupdate(...
                    obj.Actor, gradA, obj.AvgG_Actor, obj.AvgSG_Actor, obj.Iter, lr);
                [obj.Critic, obj.AvgG_Critic, obj.AvgSG_Critic] = adamupdate(...
                    obj.Critic, gradC, obj.AvgG_Critic, obj.AvgSG_Critic, obj.Iter, lr);
                
                loss_sum = loss_sum + extractdata(lossA) + extractdata(lossC);
            end
            
            loss_val = loss_sum / obj.PPO_Epochs;
        end
    end
    
    methods (Static)
        % 特製激勵函數：前兩維 Softmax (時序/空間分配)，第三維 Sigmoid (現金比例)
        function mu = apply_activation(logits)
            logits_ts = logits(1:2, :);
            logits_ts = logits_ts - max(logits_ts, [], 1);
            exp_ts = exp(logits_ts);
            mu_ts = exp_ts ./ (sum(exp_ts, 1) + 1e-8);
            
            mu_cash = 1 ./ (1 + exp(-logits(3, :)));
            mu = cat(1, mu_ts, mu_cash);
        end
        
        % PPO 核心損失函數
        function [lossA, lossC, gradA, gradC] = ppo_loss(actor, critic, s, a_target, r, adv, log_p_old, sigma, clip_val, ent_coeff)
            % s 保留 'CB' 標籤直接餵給 forward
            v_pred = forward(critic, s);
            v_pred_flat = stripdims(v_pred);
            r_flat = stripdims(r);
            lossC = 0.5 * mean((r_flat - v_pred_flat).^2, 'all');
            
            logits = forward(actor, s);
            logits_flat = stripdims(logits);
            mu = Agent_PPO.apply_activation(logits_flat);
            
            a_flat = stripdims(a_target);
            adv_flat = stripdims(adv);
            log_p_old_flat = stripdims(log_p_old);
            
            % 計算新策略對數機率與機率比率
            log_p_new = sum(-0.5 * ((a_flat - mu) ./ sigma).^2, 1);
            ratio = exp(log_p_new - log_p_old_flat);
            
            % PPO Clipped Surrogate Loss
            surr1 = ratio .* adv_flat;
            surr2 = min(max(ratio, 1 - clip_val), 1 + clip_val) .* adv_flat;
            actor_loss = -mean(min(surr1, surr2), 'all');
            
            % 計算資訊熵懲罰
            mu_ts = mu(1:2, :);
            ent_ts = -sum(mu_ts .* log(mu_ts + 1e-8), 1);
            
            mu_cash = mu(3, :);
            ent_cash = -mu_cash .* log(mu_cash + 1e-8) - (1 - mu_cash) .* log(1 - mu_cash + 1e-8);
            
            entropy_bonus = mean(ent_ts + ent_cash, 'all');
            lossA = actor_loss - ent_coeff * entropy_bonus;
            
            % 自動微分求導
            gradA = dlgradient(lossA, actor.Learnables);
            gradC = dlgradient(lossC, critic.Learnables);
        end
        
        % 梯度裁剪函數
        function g_clipped = clip_gradient(g, threshold)
            g_norm = sqrt(sum(g.^2, 'all') + 1e-8);
            if g_norm > threshold
                g_clipped = g .* (threshold / g_norm);
            else
                g_clipped = g;
            end
        end
    end
end