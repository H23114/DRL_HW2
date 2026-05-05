# 專案協作對話紀錄

以下為本次專案開發過程中的完整對話整理紀錄，依照主題與時間順序收錄，作為開發與報告撰寫的輔助文件。

## 1. 專案初始化階段

### 使用者需求

- 建立乾淨、可執行的 Python 專案骨架
- 使用 `numpy`、`matplotlib`
- 建立 `src/`、`results/`、`report/`
- 完成 `env.py`、`agents.py`、`train.py`、`plot.py`、`README.md`
- 程式需可執行，且有清楚註解

### 完成內容

- 檢查既有專案骨架
- 補強 `README.md`
- 新增 `report/README.md`
- 建立可用的 `.venv_local`
- 安裝 `numpy`、`matplotlib`
- 驗證 `src.train` 可正常執行

## 2. Cliff Walking 環境實作

### 使用者需求

- 實作 `4 x 12` Cliff Walking
- 起點左下、終點右下、底部中間為 cliff
- 支援 `reset()`、`step(action)`、state/index 轉換
- 加入最終路徑視覺化輔助方法

### 完成內容

- 在 `src/env.py` 中實作：
  - `CliffWalkingEnv`
  - `StepResult`
  - `state_to_index()`
  - `index_to_state()`
  - `render_policy()`
  - `render_path()`
  - `rollout_greedy_path()`
- 確認 reward、cliff 行為與終止條件正確

## 3. Q-learning 與 SARSA agent

### 使用者需求

- 完成 `QLearningAgent`
- 完成 `SARSAAgent`
- 共用 epsilon-greedy 與 Q-table 邏輯
- 保持程式結構整潔、避免重複

### 完成內容

- 在 `src/agents.py` 中建立：
  - `AgentConfig`
  - `BaseAgent`
  - `QLearningAgent`
  - `SARSAAgent`
- 實作正確更新公式
- 加入 random tie-breaking
- 保留 `SarsaAgent` 相容別名

## 4. 訓練流程與結果輸出

### 使用者需求

- 建立 Q-learning / SARSA 的完整訓練流程
- 記錄每回合 reward
- 回傳 Q-table 與最終 greedy path
- 主程式可直接執行
- 將結果輸出到 `results/`

### 完成內容

- 在 `src/train.py` 中建立：
  - `train_q_learning()`
  - `train_sarsa()`
  - `evaluate_greedy_path()`
- 輸出：
  - reward CSV
  - Q-table CSV
  - path / policy 文字檔
  - summary JSON

## 5. 繪圖與 path 視覺化

### 使用者需求

- reward curve
- moving average
- final path 文字圖與圖片
- 圖表輸出到 `results/`

### 完成內容

- 在 `src/plot.py` 中實作：
  - `moving_average()`
  - `plot_reward_curves()`
  - `plot_path_map()`
- 輸出：
  - `reward_curve.png`
  - `q_learning_path.png`
  - `sarsa_path.png`

## 6. 穩定性分析

### 使用者需求

- 比較訓練波動程度
- 實作 moving average 與 rolling std
- 可重複實驗多次後取平均
- 產出穩定性圖表與文字分析

### 完成內容

- 在 `src/train.py` 中加入：
  - `StabilityAnalysisResult`
  - `run_stability_analysis()`
  - `estimate_convergence_episode()`
  - `build_stability_report()`
- 在 `src/plot.py` 中加入：
  - `rolling_std()`
  - `plot_stability_analysis()`
- 輸出：
  - `stability_analysis.png`
  - `stability_metrics.csv`
  - `stability_analysis.txt`

## 7. Exploration 影響分析

### 使用者需求

- 比較 `epsilon = 0.01 / 0.1 / 0.2`
- 比較收斂速度、最終 reward、路徑風格
- 產出可重複執行的實驗函式
- 額外輸出報告草稿文字

### 完成內容

- 在 `src/train.py` 中加入：
  - `ExplorationSettingResult`
  - `ExplorationExperimentResult`
  - `run_exploration_experiment()`
  - `save_exploration_summary_csv()`
  - `build_exploration_report()`
- 在 `src/plot.py` 中加入：
  - `plot_exploration_analysis()`
  - `plot_exploration_summary()`
- 輸出：
  - `q_learning_exploration.png`
  - `sarsa_exploration.png`
  - `exploration_summary.png`
  - `exploration_summary.csv`
  - `exploration_analysis_draft.txt`

## 8. 報告撰寫

### 使用者需求

- 依據實驗結果撰寫繁體中文報告初稿
- 結構需符合作業要求
- 可直接寫入 `report/report.md`
- 要能預留結果圖表說明位置

### 完成內容

- 先根據短版驗證結果撰寫初稿
- 後續再以正式實驗結果重寫為最終版
- 加入指定結論要求：
  - 哪一種方法收斂較快
  - 哪一種方法較穩定
  - 在何種情境下應選擇 Q-learning 或 SARSA

## 9. Code Review 與修正

### 使用者需求

- 檢查：
  - Q-learning / SARSA 更新公式
  - Cliff Walking 是否正確
  - reward 與終止條件
  - epsilon-greedy
  - 路徑視覺化
  - 圖表與輸出檔案
  - 公平比較設計

### 發現與修正

- 發現原本 Q-learning 與 SARSA 比較時使用不同 seed 偏移，造成不公平比較
- 已修正為相同 seed schedule 做 paired comparison

## 10. 目錄清理

### 使用者需求

- 清理沒有用到的舊檔與暫存物

### 完成內容

- 刪除：
  - 舊的 `.venv`
  - `src/__pycache__/`
  - 舊版結果檔

## 11. 正式實驗與最終定稿

### 使用者需求

- 跑正式實驗
- 將內容改成最終繳交定案

### 完成內容

- 執行：

```powershell
.venv_local\Scripts\python.exe -m src.train --episodes 500 --analysis-runs 10 --seed 42
```

- 更新 `results/`
- 重寫 `report/report.md`
- 整理 `README.md`

## 12. 最終專案狀態

目前專案已包含：

- 完整可執行的 Cliff Walking 實作
- Q-learning 與 SARSA agent
- 訓練流程
- reward / stability / exploration 分析
- 最終路徑與圖表輸出
- 繁體中文正式報告
- 專案協作紀錄

本檔案作為開發紀錄保存於專案中，供繳交或日後回顧使用。
