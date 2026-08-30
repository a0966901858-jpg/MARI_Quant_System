classdef HRLEnvironment < rl.env.MATLABEnvironment
    % =========================================================================
    % 模組：HRLEnvironment.m (MARI 強化學習總管標準互動環境)
    % 升級：Phase 14.15 (使用「標準物件導向OOP」嚴格定義狀態空間與動作空間)
    % 職責：（現已無用）提供標準化的 Step/Reset 介面，測試單一個標準代理人，或者想把環境抽離出來跑 MATLAB 內建的演算法（如 DDPG）
    % =========================================================================
    
    properties
        Prices          
        Volumes         
        Expert          
        P_time          
        P_space         
        P_crash         
        Dates           
        
        NumTickers      
        NumDays         
        SpyIdx          
        
        Guard_Crash = 0.85;     
        Frict_Mask  = 0.0050;   
        BaseFee     = 0.0005;   
        Top_K       = 20;       
        InitialAUM  = 1000000;  
        
        CurrentStep     
        EpisodeStart    
        EpisodeLength = 252     
        PrevAssets              
        PrevCash                
        PortValue               
    end
    
    methods
        function obj = HRLEnvironment(prices, volumes, expert, p_time, p_space, p_crash, dates, config)
            obsInfo = rlNumericSpec([5 1]);
            obsInfo.Name = 'Macro_And_Risk_State';
            
            actInfo = rlNumericSpec([3 1], 'LowerLimit', 0, 'UpperLimit', 1);
            actInfo.Name = 'CIO_Resource_Allocation';
            
            obj = obj@rl.env.MATLABEnvironment(obsInfo, actInfo);
            
            obj.Prices  = prices;
            obj.Volumes = volumes; 
            obj.Expert  = expert;
            obj.P_time  = p_time;
            obj.P_space = p_space;
            obj.P_crash = p_crash;
            obj.Dates   = dates;
            
            obj.NumTickers = config.NumTickers;
            obj.NumDays    = length(dates);
            
            spy_idx = find(strcmp(config.IdxTickers, 'SPY'));
            if isempty(spy_idx)
                error('❌ 致命錯誤：環境初始化失敗，找不到 SPY 基準標的。');
            end
            obj.SpyIdx = spy_idx;
            
            obj.Guard_Crash = config.Guardrail_CrashProb;
            obj.Top_K       = config.Top_K_Assets;
            obj.Frict_Mask  = config.MoE_FrictionMask;
            
            fprintf(' ⚙️ [HRLEnvironment] 初始化完成。時空因果律已鎖定，Amihud 模型上膛。\n');
        end
        
        function [InitialObservation, LoggedSignals] = reset(obj)
            min_start = 253; 
            max_start = obj.NumDays - obj.EpisodeLength - 1;
            
            if max_start <= min_start
                obj.EpisodeStart = min_start;
            else
                obj.EpisodeStart = randi([min_start, max_start]);
            end
            
            obj.CurrentStep = obj.EpisodeStart;
            
            obj.PrevAssets = zeros(obj.NumTickers, 1, 'single');
            obj.PrevCash   = 1.0;
            obj.PortValue  = obj.InitialAUM; 
            
            InitialObservation = obj.getObservation();
            LoggedSignals = [];
        end
        
        function [Observation, Reward, IsDone, LoggedSignals] = step(obj, Action)
            t = obj.CurrentStep;
            
            % ---------------------------------------------------------
            % 1. 解析網路動作
            % ---------------------------------------------------------
            w_time_raw  = Action(1);
            w_space_raw = Action(2);
            target_cash = Action(3);
            
            sum_experts = w_time_raw + w_space_raw;
            if sum_experts == 0
                w_time = 0.5; w_space = 0.5;
            else
                w_time  = w_time_raw / sum_experts;
                w_space = w_space_raw / sum_experts;
            end
            
            % ---------------------------------------------------------
            % 2. ★ 核心修復：使用 T-1 到 T 的真實歷史計算當下權重漂移
            % ---------------------------------------------------------
            ret_t = (obj.Prices(t, :)' - obj.Prices(t-1, :)') ./ obj.Prices(t-1, :)';
            ret_t(isnan(ret_t) | isinf(ret_t)) = 0;
            
            drift_weights = obj.PrevAssets .* (1 + ret_t);
            drift_cash    = obj.PrevCash;
            
            port_val_multiplier = sum(drift_weights) + drift_cash;
            
            w_drift    = drift_weights / port_val_multiplier;
            cash_drift = drift_cash / port_val_multiplier;
            
            % 更新調倉前的實體帳戶淨值 (T 日收盤結算價值)
            obj.PortValue = obj.PortValue * port_val_multiplier;
            
            % ---------------------------------------------------------
            % 3. 停牌鎖死防護 (使用 T 日觀測)
            % ---------------------------------------------------------
            halted_mask = (obj.Volumes(t, :)' <= 0) | isnan(obj.Prices(t, :)');
            
            locked_weights = w_drift;
            locked_weights(~halted_mask) = 0; 
            
            locked_sum = sum(locked_weights); 
            available_cap = 1.0 - locked_sum;
            
            % ---------------------------------------------------------
            % 4. GBDT 護欄與可用資金分配
            % ---------------------------------------------------------
            if obj.P_crash(t) > obj.Guard_Crash
                target_cash = 1.0; 
                w_time = 0.0;
                w_space = 0.0;
            end
            
            actual_w_cash = min(target_cash, available_cap);
            rem_cap_for_assets = available_cap - actual_w_cash;
            
            % ---------------------------------------------------------
            % 5. 生成底層微觀標的目標權重
            % ---------------------------------------------------------
            comb_p = obj.P_time(t, :)' * w_time + obj.P_space(t, :)' * w_space;
            active_mask = obj.Expert(t, :)';
            comb_p = comb_p .* active_mask;
            
            comb_p(halted_mask) = 0;
            
            if sum(comb_p > 0) > obj.Top_K
                [~, sort_idx] = sort(comb_p, 'descend');
                threshold_val = comb_p(sort_idx(obj.Top_K));
                comb_p(comb_p < threshold_val) = 0;
            end
            
            if sum(comb_p) > 0
                comb_p = comb_p ./ sum(comb_p);
            else
                comb_p = zeros(obj.NumTickers, 1, 'single');
            end
            
            asset_weights = (comb_p .* rem_cap_for_assets) + locked_weights;
            
            delta_w = asset_weights - w_drift;
            ignore_mask = abs(delta_w) < obj.Frict_Mask;
            asset_weights(ignore_mask) = w_drift(ignore_mask);
            
            final_active_sum = sum(asset_weights);
            final_total_cap  = final_active_sum + actual_w_cash;
            asset_weights    = asset_weights / final_total_cap;
            actual_w_cash    = actual_w_cash / final_total_cap;
            
            % ---------------------------------------------------------
            % 6. ★ 核心修復：基於 T 日流動性估算 Amihud 衝擊成本
            % ---------------------------------------------------------
            current_obs = obj.getObservation();
            annual_vol = current_obs(3);
            if isnan(annual_vol) || isinf(annual_vol), annual_vol = 0.15; end
            daily_vol = annual_vol / sqrt(252);
            
            Q_dollars = abs(asset_weights - w_drift) * obj.PortValue;
            V_dollars = max((obj.Volumes(t, :)' .* obj.Prices(t, :)'), 10000);
            
            impact_tc_rate = zeros(obj.NumTickers, 1, 'single');
            traded_idx = (Q_dollars > 0) & ~halted_mask;
            impact_tc_rate(traded_idx) = daily_vol * sqrt(Q_dollars(traded_idx) ./ V_dollars(traded_idx));
            
            total_tc_rate = obj.BaseFee + impact_tc_rate;
            frict_cost_dollars = sum(Q_dollars .* total_tc_rate);
            frict_cost_pct     = frict_cost_dollars / obj.PortValue;
            
            % 扣除手續費後的 T 日真實淨值
            obj.PortValue = obj.PortValue - frict_cost_dollars;
            
            % ---------------------------------------------------------
            % 7. ★ 核心修復：計算新投資組合在 T 到 T+1 的真實表現
            % ---------------------------------------------------------
            ret_t1 = (obj.Prices(t+1, :)' - obj.Prices(t, :)') ./ obj.Prices(t, :)';
            ret_t1(isnan(ret_t1) | isinf(ret_t1)) = 0;
            
            % 將摩擦成本算入這一步的淨報酬損耗中
            port_ret = sum(asset_weights .* ret_t1) - frict_cost_pct;
            spy_ret = ret_t1(obj.SpyIdx);
            
            excess_ret = port_ret - spy_ret;
            
            if excess_ret >= 0
                Reward = excess_ret * 100.0; 
            else
                Reward = -(abs(excess_ret) * 100.0)^2; 
            end
            
            if obj.PortValue <= (obj.InitialAUM * 0.5)
                Reward = Reward - 50; 
            end
            
            % ---------------------------------------------------------
            % 8. 狀態推進
            % ---------------------------------------------------------
            obj.PrevAssets  = asset_weights;
            obj.PrevCash    = actual_w_cash;
            obj.CurrentStep = obj.CurrentStep + 1;
            
            Observation = obj.getObservation();
            
            steps_taken = obj.CurrentStep - obj.EpisodeStart;
            IsDone = (steps_taken >= obj.EpisodeLength) || ...
                     (obj.PortValue <= (obj.InitialAUM * 0.2)) || ...
                     (obj.CurrentStep >= obj.NumDays - 1);
            
            LoggedSignals = [];
        end
        
        function obs = getObservation(obj)
            t = obj.CurrentStep;
            start_idx = max(1, t - 251);
            spy_history = obj.Prices(start_idx:t, obj.SpyIdx);
            spy_rets = [0; diff(spy_history) ./ spy_history(1:end-1)];
            spy_rets(isnan(spy_rets) | isinf(spy_rets)) = 0;
            
            p_crash_val = obj.P_crash(t);
            
            spy_ret20 = 0;
            if t >= 21
                spy_ret20 = (obj.Prices(t, obj.SpyIdx) - obj.Prices(t-20, obj.SpyIdx)) / obj.Prices(t-20, obj.SpyIdx);
            end
            
            roll_vol = std(spy_rets(max(1, end-19):end)) * sqrt(252);
            if isnan(roll_vol), roll_vol = 0; end
            
            cum_ret = cumprod(1 + spy_rets);
            running_max = cummax(cum_ret);
            drawdowns = (cum_ret - running_max) ./ running_max;
            roll_mdd = min(drawdowns);
            if isempty(roll_mdd) || isnan(roll_mdd), roll_mdd = 0; end
            
            obs = single([p_crash_val; spy_ret20; roll_vol; abs(roll_mdd); obj.PrevCash]);
        end
    end
end