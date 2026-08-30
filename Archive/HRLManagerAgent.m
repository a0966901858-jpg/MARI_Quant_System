classdef HRLManagerAgent < handle
    % =========================================================================
    % 類別：HRLManagerAgent (CIO 總管強化學習代理人)
    % 升級：Phase 14.11 (★ 修復 LSTM 失憶悖論、梯度外洩防禦、Sigmoid 動作邊界)
    % 職責：搭配 MATLAB 官方 RL 工具箱進行概念驗證（PoC）。但在驗證成功後，
    % 為了實作底層的並列運算（Parfor）以及客製化 PPO 損失函數，
    % 我將架構推翻重構，開發了原生的 Agent_PPO.m取代此程式
    % =========================================================================
    
    properties
        Actor       % 策略網路 (輸出動作)
        Critic      % 價值網路 (評估狀態)
        
        % Adam 優化器狀態
        AvgG_Actor = []; AvgSG_Actor = [];
        AvgG_Critic = []; AvgSG_Critic = [];
        Iter = 0;
    end
    
    methods
        function obj = HRLManagerAgent()
            % -----------------------------------------------------
            % 1. 建立 Actor 網路 (輸入: 5維特徵, 輸出: 3維連續機率)
            % -----------------------------------------------------
            % ★ 核心修復：改用 featureInputLayer，並加上 sigmoidLayer 確保輸出在 [0,1]
            act_layers = [
                featureInputLayer(5, 'Name', 'state_in')
                fullyConnectedLayer(64, 'Name', 'fc1'); reluLayer('Name', 'relu1')
                fullyConnectedLayer(32, 'Name', 'fc2'); reluLayer('Name', 'relu2')
                fullyConnectedLayer(3, 'Name', 'action_out')
                sigmoidLayer('Name', 'sigmoid_bound') 
            ];
            obj.Actor = dlnetwork(act_layers);
            
            % -----------------------------------------------------
            % 2. 建立 Critic 網路 (輸入: 5維特徵, 輸出: 1維狀態價值)
            % -----------------------------------------------------
            crit_layers = [
                featureInputLayer(5, 'Name', 'state_in')
                fullyConnectedLayer(64, 'Name', 'fc1'); reluLayer('Name', 'relu1')
                fullyConnectedLayer(32, 'Name', 'fc2'); reluLayer('Name', 'relu2')
                fullyConnectedLayer(1, 'Name', 'value_out')
            ];
            obj.Critic = dlnetwork(crit_layers);
            
            if canUseGPU()
                obj.Actor = dlupdate(@gpuArray, obj.Actor);
                obj.Critic = dlupdate(@gpuArray, obj.Critic);
            end
            
            fprintf(' ⚙️ [HRLManagerAgent] 初始化完成。MLP 雙頭架構與梯度屏障已建立。\n');
        end
        
        % =========================================================
        % 前向推論 (取得投資決策)
        % =========================================================
        function [actions, value] = get_actions(obj, state, noise_std)
            % 封裝為深度學習張量 (C:特徵維度=5, B:批次=1)
            dl_state = dlarray(state, 'CB');
            if canUseGPU()
                dl_state = gpuArray(dl_state);
            end
            
            % 推論
            acts_raw = predict(obj.Actor, dl_state);
            val_raw = predict(obj.Critic, dl_state);
            
            actions = extractdata(acts_raw);
            value = extractdata(val_raw);
            
            % 探索雜訊 (Exploration Noise)
            if noise_std > 0
                actions = actions + noise_std * randn(size(actions), 'single');
                % ★ 核心修復：加上雜訊後，強制截斷以符合環境物理界線
                actions = max(min(actions, 1.0), 0.0); 
            end
        end
        
        % =========================================================
        % 代理人學習 (Actor-Critic 權重更新)
        % =========================================================
        function update_weights(obj, states, actions, rewards, lr)
            dl_s = dlarray(states, 'CB');
            dl_a = dlarray(actions, 'CB');
            dl_r = dlarray(rewards, 'CB');
            
            if canUseGPU()
                dl_s = gpuArray(dl_s);
                dl_a = gpuArray(dl_a);
                dl_r = gpuArray(dl_r);
            end
            
            obj.Iter = obj.Iter + 1;
            
            [lossA, lossC, gradA, gradC] = dlfeval(@obj.ac_loss, obj.Actor, obj.Critic, dl_s, dl_a, dl_r);
            
            % 梯度裁剪防爆 (Gradient Clipping)
            gradA = dlupdate(@(g) obj.clip_gradient(g, 1.0), gradA);
            gradC = dlupdate(@(g) obj.clip_gradient(g, 1.0), gradC);
            
            % Adam 參數更新
            [obj.Actor, obj.AvgG_Actor, obj.AvgSG_Actor] = adamupdate(...
                obj.Actor, gradA, obj.AvgG_Actor, obj.AvgSG_Actor, obj.Iter, lr);
                
            [obj.Critic, obj.AvgG_Critic, obj.AvgSG_Critic] = adamupdate(...
                obj.Critic, gradC, obj.AvgG_Critic, obj.AvgSG_Critic, obj.Iter, lr);
        end
    end
    
    methods (Static)
        % =========================================================
        % 損失函數 (Actor-Critic Loss)
        % =========================================================
        function [lossA, lossC, gradA, gradC] = ac_loss(actor, critic, s, a_target, r)
            % 1. Critic Loss (TD Error / MSE)
            v_pred = forward(critic, s);
            lossC = mean((r - v_pred).^2, 'all');
            
            % 2. ★ 核心修復：強制剝離 Critic 計算圖 (Detach)
            % 將 v_pred 從 dlarray 轉為普通數值，阻斷 Actor 梯度流回 Critic
            v_detached = extractdata(v_pred);
            advantage = r - v_detached;
            
            % 3. Actor Loss (Advantage Weighted Regression 風格)
            a_pred = forward(actor, s);
            % 若 Advantage 為正，縮小 a_pred 與 a_target 的距離
            % 由於 a_pred 被 sigmoid 限制在 [0,1]，就算 Advantage 為負，也不會引發梯度爆炸
            lossA = mean(sum((a_target - a_pred).^2, 1) .* advantage, 'all');
            
            % 計算對網路可學習參數的梯度
            gradA = dlgradient(lossA, actor.Learnables);
            gradC = dlgradient(lossC, critic.Learnables);
        end
        
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