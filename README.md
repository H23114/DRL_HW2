# DRL HW4 正確網址 https://github.com/H23114/DRL_HW4.git
不好意思，作業繳交後發現連結一直跳轉不正確，只好將網址附上，謝謝助教！

本專案為深度強化學習作業二的最終繳交版本，主題為比較 Q-learning 與 SARSA 在 Cliff Walking / Gridworld 環境中的學習表現、穩定性與 exploration 影響。

## 專案內容

- `src/env.py`：Cliff Walking 環境
- `src/agents.py`：Q-learning 與 SARSA agent
- `src/train.py`：訓練流程、穩定性分析、exploration 分析
- `src/plot.py`：繪圖與結果輸出
- `results/`：正式實驗輸出結果
- `report/report.md`：作業報告初稿 / 最終版
- `report/conversation_log.md`：本次專案協作對話紀錄

## 執行環境

- Python 3.10 以上
- `numpy`
- `matplotlib`

安裝套件：

```powershell
pip install -r requirements.txt
```

本專案目前使用的本地虛擬環境為：

```powershell
.venv_local\Scripts\python.exe
```

## 執行方式

### 1. 一般執行

```powershell
.venv_local\Scripts\python.exe -m src.train
```

### 2. 正式實驗設定

本次正式報告採用以下指令：

```powershell
.venv_local\Scripts\python.exe -m src.train --episodes 500 --analysis-runs 10 --seed 42
```

## 主要輸出結果

執行完成後，結果會輸出到 `results/`，包含：

- `reward_curve.png`
- `stability_analysis.png`
- `q_learning_path.png`
- `sarsa_path.png`
- `q_learning_exploration.png`
- `sarsa_exploration.png`
- `exploration_summary.png`
- `summary.json`
- `stability_analysis.txt`
- `exploration_analysis_draft.txt`
- `q_learning_rewards.csv`
- `sarsa_rewards.csv`

## 作業重點摘要

- Q-learning 為 off-policy，較容易學到理論上的最短路徑，但策略通常更冒險
- SARSA 為 on-policy，會把 exploration 風險直接學進更新，因此通常較保守且較穩定
- 在本次正式實驗中，Q-learning 的 greedy path 較短，但 SARSA 的平均 reward 與穩定性較佳

## 報告檔案

- 正式報告：`report/report.md`
- 對話紀錄：`report/conversation_log.md`
