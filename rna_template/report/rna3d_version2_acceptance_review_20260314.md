# RNA3DB Pipeline v2 验收报告

审查日期：2026-03-14  
审查对象：

- 实现报告: [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_version2.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_version2.md)
- 前一轮验收: [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline_acceptance_review_20260314.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline_acceptance_review_20260314.md)
- 代码:
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh)

## 结论

不是“全部通过”。

本轮 v2 的确修掉了两项实质问题：

- cross-template index 的相对路径前缀已经不再硬编码成 `templates/`
- `test_rna3d_e2e.sh` 的环境激活比上一轮更鲁棒，当前机器上能成功激活 `/inspire/.../conda/envs/protenix`

但仍然没有达到 `rna3d_version2.md` 声称的“已修复并验证”强度，原因有三条：

- E2E 仍然没有覆盖 production query source，它还是用 catalog 自己做 query
- 你新增的 “no cross-template -> fallback to self-templates” 逻辑没有真正回退到 self-index，只会产生空 index
- 当前机器上无法复现报告里声称的 MMseqs2/GPU 验证，因为激活后的环境里没有 `mmseqs`

我的最终判断：

> v2 是继续前进了一步，但 `rna3d_version2.md` 仍然存在过度结论，不能判定为 fully accepted。

---

## Findings

### 1. 高严重度：E2E 仍然没有验证 production query source，`rna3d_version2.md` 对 Bug #2 的结论过强

`test_rna3d_e2e.sh` 已经移除了 `--no_exclude_self`，这点是对的；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L136](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L136) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L147](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L147)。

但这份 E2E 仍然显式写着：

- `Use catalog as both database and query for this test.`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L136](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L136)。

而真正 production 主路径在 [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh) 里仍然是：

- `TRAINING_SEQ_JSON="${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"`
- `--training_sequences "${TRAINING_SEQ_JSON}"`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L184](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L184) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L203](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L203)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L223](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L223) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L234](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L234)。

所以这一轮 E2E 最多只能说明：

- `catalog -> self-exclusion enabled search -> cross build` 这条测试路径被加强了

它仍然不能说明：

- `training_sequences JSON -> mmseqs2 -> cross-template -> index -> train` 这条 production query source 路径已经被端到端验证

因此 `rna3d_version2.md` 里对 Bug #2 “已修复并验证”的表述仍然过强。

### 2. 高严重度：新增的 fallback 逻辑没有真正回退到 self-template index，而是会静默生成空 index

你新增了这段逻辑：

- 如果 `N_CROSS == 0`，就把 `CROSS_TEMPLATE_DIR="${TEST_TEMPLATE_DIR}"`，并打印 “Falling back to self-templates for training.”

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L176](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L176) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L180](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L180)。

但 Step 5 仍然调用的是：

- `03_mmseqs2_search.py --strategy mmseqs2 --skip_search`

它会对已有 `search_results.json` 调用 `build_cross_index()`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L183](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L183) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L194](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L194)。

如果 `search_results.json` 本身是空的，换目录变量没有用，`build_cross_index()` 还是会产出空 index。

我在当前机器上实际跑了缩小版：

```bash
bash test_rna3d_e2e.sh --num_test 6 --skip_training --no_arena
```

得到的真实结果是：

- `Queries with search hits: 0`
- `Cross-templates built: 0`
- 进入 fallback 分支
- 最终 `Sequences in index: 0`
- 但脚本仍打印 `PIPELINE TEST PASSED (training skipped)`

这说明当前 fallback 逻辑并没有做到报告里说的 “fallback to self-templates for training”。  
它只是把目录变量改成 self-template 目录，但没有改：

- index 构建策略
- search_results 内容
- 或切换到 `build_self_index()`

这是一个真实的功能缺口，不只是文档问题。

### 3. 中严重度：`rna3d_version2.md` 声称的 MMseqs2 / GPU 验证在当前机器上无法复现

本轮环境激活修复是有效的。缩小版脚本现在能在当前机器上成功激活：

- `/inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/protenix`

但同一个环境里我实际检查得到：

- `which python` 正常
- `mmseqs: command not found`

所以在当前机器上，脚本会在 Step 3 进入：

- `MMseqs2 is not available. Install it or activate the conda env ...`

随后 search 返回空结果，cross-template 数量为 0。

这意味着 `rna3d_version2.md` 中这些结论我无法在当前工作区复现：

- `MMseqs2 18.8cc5c`
- `11/30 queries hit`
- `Cross-templates built: 11`
- `10 steps on H800`

我不能据此判定这些结果一定是假的，但至少在当前机器上，这份 “已验证” 结论不具备可复现性。

### 4. 中严重度：index 路径修复是真的，这一项通过

`build_self_index()` 和 `build_cross_index()` 现在都改成了根据 `template_dir` 的目录名动态推导相对路径前缀；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L678](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L678) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L748](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L678)。

我做了函数级验证：

- 当 `template_dir = .../cross_templates` 时，`build_cross_index()` 产出 `cross_templates/q1_template.npz`
- 当 `template_dir = .../templates` 时，`build_self_index()` 产出 `templates/a_A_template.npz`

所以上一轮“index 静默回指 self-template 目录”的核心 bug，这一轮已经修掉了。

### 5. 低严重度：环境激活修复是真的，这一项通过

`test_rna3d_e2e.sh` 现在确实支持多路径探测：

- `conda activate protenix`
- `conda activate "${PROJECT_ROOT}/conda/envs/protenix"`
- `conda activate "${PROJECT_ROOT}/conda/envs/r1126_rna"`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L60](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L60) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L70](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L60)。

当前机器上这条修复是生效的。

---

## 通过项

- `build_cross_index()` / `build_self_index()` 的动态相对路径修复：通过
- `test_rna3d_e2e.sh` 移除 `--no_exclude_self`：通过
- `test_rna3d_e2e.sh` 的 conda 环境激活鲁棒性：通过
- Python/Bash 语法检查：通过

## 未通过项

- E2E 覆盖 production query source：未通过
- `zero cross-template` 时真正 fallback 到 self-index：未通过
- 当前机器上复现 `rna3d_version2.md` 的 MMseqs2/GPU 验证结论：未通过

---

## 本次实际验证

我实际完成了这些检查：

- `python3 -m py_compile` 通过：
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)
- `bash -n` 通过：
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh)
- 函数级验证：
  - `build_cross_index()` 对 `cross_templates/` 产出正确相对路径
  - `build_self_index()` 对 `templates/` 产出正确相对路径
- 实跑缩小版：
  - `bash test_rna3d_e2e.sh --num_test 6 --skip_training --no_arena`
  - 结果：环境激活成功，但 `mmseqs` 不存在，search=0，cross_templates=0，最终 index=0

---

## 最终判定

如果标准是“修掉上一轮最明显的索引路径 bug 和环境激活 bug”，v2 通过。

如果标准是“`rna3d_version2.md` 中宣称的修复和验证已经全部成立”，v2 不通过。

最关键的残留问题是：

1. E2E 仍未覆盖 production query source
2. zero-hit fallback 逻辑是假的，最后还是空 index
3. 当前机器上无法复现报告中的 MMseqs2 / GPU 验证结果
