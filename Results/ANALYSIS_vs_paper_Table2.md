# Analysis: Implementation vs Paper Table 2 (Section 4.2)

## 1. Are the results the same?

**No.** Your implementation’s metrics are **not** the same as Table 2 in the paper. They are consistently and often much **higher** than the paper’s SelfGNN row.

| Dataset   | Metric   | Paper (Table 2) | Your Results | Difference   |
|-----------|----------|------------------|--------------|--------------|
| Amazon    | HR@10    | 0.391            | **0.6556**  | +67%         |
| Amazon    | NDCG@10  | 0.240            | **0.4195**  | +75%         |
| Amazon    | HR@20    | 0.487            | **0.7695**  | +58%         |
| Amazon    | NDCG@20  | 0.264            | **0.4484**  | +70%         |
| Gowalla   | HR@10    | 0.637            | **0.8570**  | +34%         |
| Gowalla   | NDCG@10  | 0.437            | **0.6160**  | +41%         |
| Gowalla   | HR@20    | 0.745            | **0.9372**  | +26%         |
| Gowalla   | NDCG@20  | 0.465            | **0.6366**  | +37%         |
| Movielens | HR@10    | 0.239            | **0.4779**  | +100%        |
| Movielens | NDCG@10  | 0.128            | **0.2655**  | +107%        |
| Movielens | HR@20    | 0.341            | **0.7116**  | +109%        |
| Movielens | NDCG@20  | 0.154            | **0.3246**  | +111%        |
| Yelp      | HR@10    | 0.365            | **0.7209**  | +97%         |
| Yelp      | NDCG@10  | 0.201            | **0.4308**  | +114%        |
| Yelp      | HR@20    | 0.509            | **0.8761**  | +72%         |
| Yelp      | NDCG@20  | 0.237            | **0.4703**  | +98%         |

So **amazon, gowalla, movielens, yelp in your Results are not the same as in Table 2**; your numbers are higher across the board.

---

## 2. Why they differ: evaluation protocol

### 2.1 Paper protocol (Section 4.1.2)

- **Split:** Last interaction = test, penultimate = validation, rest = training.
- **Test users:** Randomly sample **10,000** users as test users.
- **Per test user:** Sample **999 negative** items (no interaction) and rank **1 positive + 999 negatives = 1000** items.
- **Metrics:** HR@N and NDCG@N over this ranking.

### 2.2 Your implementation

| Aspect            | Paper        | Your code                    | File / location              |
|------------------|-------------|------------------------------|------------------------------|
| **Candidates per user** | **1000** (1 + 999 neg) | **100** (1 + 99 neg) | `Params.py`: `--testSize`, default **100** |
| **Negatives per user**  | 999         | `testSize - 1` = **99**      | `model.py` `sample_test_batch`: `args.testSize - 1` |
| **Test users**         | 10,000 (random sample) | All users with `tstInt != None` | `DataHandler.py`: `tstUsrs = np.argwhere(tstStat != False)` |

So the main protocol difference is:

- You rank **100** items per user instead of **1000**.
- With 10× fewer candidates, it is much easier for the positive to land in the top-10 or top-20, so **HR@10, HR@20, NDCG@10, NDCG@20 will be systematically higher** than in the paper. This alone can explain most of the gap.

### 2.3 Test sequence: possible leakage

In `model.py`, `sample_test_batch`:

- When `args.test` is True, the sequence fed to the model is  
  `posset = self.handler.sequence[bat_ids[i]]`  
  i.e. the **full** user sequence, which typically **includes the test (last) item**.
- The paper’s protocol implies that for evaluation the model should only see **history (e.g. before the last interaction)** and predict the last. Feeding the full sequence including the test item can cause **test leakage** and further inflate metrics. This should be checked and, if the data indeed includes the last item in `sequence`, fixed to use `sequence[:-1]` at test time.

---

## 3. Is the implementation “correct” relative to the paper?

- **Model and training:** The codebase (short-term GNN, GRU over intervals, multi-head attention, SSL, etc.) is aligned with the SelfGNN design; no obvious high-level deviation was checked in this analysis.
- **Evaluation:** The implementation is **not** correct with respect to the paper’s **evaluation protocol**:
  - It does **not** use 999 negatives (1000 items per user) as in Table 2.
  - It uses 99 negatives (100 items per user), and possibly all users with a test interaction instead of a fixed 10k test set.
  - Test input may include the target item (leakage).

So the **reported Results/ numbers cannot be considered the same as Table 2** and are not directly comparable.

---

## 4. What to change to match the paper (Table 2)

1. **Use 1000 candidates per test user (1 positive + 999 negatives).**
   - Set `--testSize 1000` (or add a separate “eval pool size” and use 1000 for reporting).
   - Ensure your data pipeline (e.g. `test_dict` or negative sampling) provides **999** negative item IDs per test user so that each evaluation instance ranks exactly 1000 items.

2. **Test users.**
   - If the paper uses exactly 10,000 randomly sampled test users, replicate that: e.g. fix a random seed and sample 10,000 users from those with at least one interaction (or from the same pool as in the paper) and evaluate only on them when reporting Table 2–style results.

3. **No test leakage.**
   - At test time, feed the model only **history** (e.g. `sequence[:-1]`), not the full sequence including the target. Confirm how `self.sequence` and `tstInt` are built in your data prep (whether the last element is the test item) and adjust so the test input never contains the target item.

4. **Data and preprocessing.**
   - Use the same 5-core and same train/val/test split as the paper (last = test, second-to-last = val). Your current split is determined by `trn_mat_time`, `tst_int`, and `sequence`; verify they match the paper’s description.

After these changes, re-run evaluation and compare again to Table 2 (Section 4.2). Metrics should drop and can then be meaningfully compared to the paper’s SelfGNN row.

---

## 5. Summary

| Question | Answer |
|----------|--------|
| Are Results/ (amazon, gowalla, movielens, yelp) the same as Table 2? | **No.** Your metrics are much higher. |
| Is the implementation correct? | **Model/training:** plausibly aligned. **Evaluation:** **no** — different pool size (100 vs 1000), possibly different test set size, and possible test leakage. |
| Main cause of the gap | Using **100** ranked items per user instead of **1000** (999 negatives + 1 positive) as in the paper. |
| What to do | Set test pool to 1000 items (999 neg + 1 pos), fix test-user set and no-leakage input, then re-evaluate and compare to Table 2. |

---

## 6. Updates applied (to match paper)

- **Params:** `--testSize` default set to **1000** (paper: 999 neg + 1 pos). `--save_path` defaults to `--data` (e.g. `--data amazon` → save_path=amazon). `--data` default set to `amazon`.
- **Paths:** Model and history now save under **Results/&lt;data&gt;/**: `Results/amazon/model.pt`, `Results/amazon/history.his`. Metrics and plots: `Results/amazon/amazon_metrics.txt`, `Results/amazon/amazon_loss.png`, etc.
- **Test leakage:** At test time the input sequence is now **history only** (`sequence[:-1]`), so the target item is not fed to the model.
- **999 negatives:** When `test_dict` is missing or has fewer than 999 negatives, the code fills the rest by sampling from items the user did not interact with (no positive in the candidate set).

python main --data amazon --load_model amazon