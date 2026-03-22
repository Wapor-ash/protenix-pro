# Cross-Only Pipeline 验收报告

审查日期：2026-03-14  
目标：

- 删除 `cross-template miss -> self-template` 的回退逻辑
- 删除用户可选的 `self-template` 入口，避免误用和数据泄露
- 在 `conda activate /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/protenix` 环境中验证 cross-only 行为

审查文件：

- [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
- [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)
- [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py)
- [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh)
- [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh)
- [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_small_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_small_e2e.sh)

## 结论

核心目标基本达成：

- `self-template` 的用户入口已经被移除
- `cross-template miss -> self-template` 的回退逻辑已经删除
- 当前代码在 search miss 时会直接失败，不会偷偷使用 self-template

但 full cross-only E2E 在你指定的 conda 环境里目前**不能跑通搜索阶段**，原因不是回退到 self-template，而是：

- 环境里没有 `mmseqs` 可执行文件

所以结论要分开说：

- `是否还会采用 self-template`：我验证到**不会**
- `cross-only full pipeline 在该 conda 环境里是否成功`：**当前不能成功，因为缺 `mmseqs`**

---

## Findings

### 1. 已通过：`02_build_rna_templates.py` 不再暴露 self 模式

CLI 现在只允许：

- `--mode {cross}`

见 [`02_build_rna_templates.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L333) 到 [`02_build_rna_templates.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L346)。

而且原来的 `build_self_template()` 已经从文件中删除，当前主流程只会消费 `search_results.json` 来建 cross-template。

我在你指定 conda 环境里跑了：

```bash
python rna_template/scripts/02_build_rna_templates.py --help
```

实际输出确认只剩：

```text
--mode {cross}
Template building mode. Only cross-template mode is supported.
```

### 2. 已通过：`03_mmseqs2_search.py` 不再暴露 self strategy，也不再允许 `--no_exclude_self`

CLI 现在只允许：

- `--strategy {mmseqs2,pairwise}`

见 [`03_mmseqs2_search.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L866) 到 [`03_mmseqs2_search.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L870)。

同时 `--no_exclude_self` 已经从 argparse 里移除，搜索逻辑现在把：

- `exclude_self = True`

写死在主流程里；见 [`03_mmseqs2_search.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L1012) 到 [`03_mmseqs2_search.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L1036)。

我实际跑了：

```bash
python rna_template/scripts/03_mmseqs2_search.py --help
```

help 里已经没有：

- `self`
- `no_exclude_self`

### 3. 已通过：legacy pairwise 脚本也去掉了 self strategy

旧脚本 [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py) 现在只允许：

- `--strategy {pairwise}`

见 [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L256) 到 [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L268)。

它也不再暴露 self-index 入口，也不再允许 `--no_exclude_self`。

### 4. 已通过：`run_pipeline.sh` 已改成 cross-only，显式拒绝 `--strategy self`

当前默认：

- `STRATEGY="mmseqs2"`

见 [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L46)。

而且脚本会在主逻辑入口直接拒绝非 `mmseqs2`：

- `Only 'mmseqs2' cross-template mode is allowed.`

见 [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L152) 到 [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L155)。

我实际跑了：

```bash
bash run_pipeline.sh --strategy self --skip_catalog --skip_search --skip_build --output_dir .../rna_database
```

真实结果是：

```text
ERROR: Unsupported strategy 'self'. Only 'mmseqs2' cross-template mode is allowed.
```

说明 self 路径已经不能从主 pipeline 误触发。

### 5. 已通过：`test_rna3d_e2e.sh` 的 self-template 构建和 fallback 已删除

当前 `test_rna3d_e2e.sh` 的主链路已经变成：

- catalog
- MMseqs2 search
- cross-template build
- index
- NPZ validation
- train

见 [`test_rna3d_e2e.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L132) 到 [`test_rna3d_e2e.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L270)。

关键点：

- 不再 build self-template
- search miss 时不再切回 self-template
- `N_SEARCH == 0` 时直接失败退出
- `N_CROSS == 0` 时直接失败退出

其中最关键的两处是：

- [`test_rna3d_e2e.sh#L154`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L154) 到 [`test_rna3d_e2e.sh#L157`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L157)
- [`test_rna3d_e2e.sh#L173`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L173) 到 [`test_rna3d_e2e.sh#L176`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L173)

也就是说，当前脚本已经是：

> cross hit 才继续；cross miss 就失败  
> 不会再偷偷变成 self-template

### 6. 已通过：`test_small_e2e.sh` 不再是 self smoke test，而是转发到 cross-only E2E

[`test_small_e2e.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_small_e2e.sh) 现在只是一个 wrapper：

- 直接执行 `test_rna3d_e2e.sh --num_test 10`

这避免了旧的 self-template 小测试继续留在仓库里被误用。

### 7. 未通过：在指定 conda 环境里，full cross-only E2E 当前不能跑通，因为没有 `mmseqs`

我实际在你指定环境里跑了：

```bash
bash rna_template/scripts/test_rna3d_e2e.sh --num_test 6 --skip_training --no_arena
```

脚本现在的行为是：

1. 成功激活 conda 环境
2. 成功提取 catalog
3. 到 MMseqs2 search 时失败
4. 由于是 cross-only，不回退 self-template，直接退出

关键输出是：

```text
Activated conda env: /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/protenix
...
MMseqs2 is not available.
Queries with search hits: 0
FAIL: No cross-template hits found. Cross-only test stops here by design.
```

我还单独检查了该环境：

```bash
conda activate /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/protenix
which python
which mmseqs
```

结果是：

- `python` 存在
- `mmseqs` 不在 PATH 上

所以现在不是“偷偷走了 self-template”，而是：

- **根本没法完成 cross search**

### 8. 已通过：当提供一个明确的 non-self hit 时，cross-template 构建成功，且产物只包含外部模板

为了验证“当前代码会不会擅自用 self-template”，我在你的 conda 环境里构造了一份 synthetic `search_results.json`：

- query: `1ddy_G_G`
- template: `1et4_A`

然后运行：

```bash
python rna_template/scripts/02_build_rna_templates.py \
  --catalog .../rna_catalog.json \
  --pdb_rna_dir .../rna3db-mmcifs \
  --output_dir .../cross_templates_synth \
  --mode cross \
  --search_results .../synthetic_search_results.json \
  --max_templates 1
```

结果：

- `Done: 1 templates built, 0 failures`

我继续检查产物 [`1ddy_G_G_template.npz`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database_e2e_test/cross_templates_synth/1ddy_G_G_template.npz) 的 `template_names`，真实值是：

```text
['1et4_A.cif:A']
```

这里没有任何 self-template 名称，说明：

- 只要输入 hit 是 non-self
- 当前 cross builder 就会老老实实只建 cross-template
- 不会擅自回退成 query 自己的 template

---

## 实际验证摘要

我实际完成了这些验证：

- `python -m py_compile` 通过：
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py)
- `bash -n` 通过：
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_small_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_small_e2e.sh)
- CLI 帮助检查：
  - `02_build_rna_templates.py --mode {cross}`
  - `03_mmseqs2_search.py --strategy {mmseqs2,pairwise}`
  - `03_search_and_index.py --strategy {pairwise}`
- 主 pipeline 对 `--strategy self` 的拒绝：通过
- 指定 conda 环境里的 cross-only E2E：能启动，但因缺 `mmseqs` 在 search 阶段失败
- synthetic non-self hit 的 cross build：通过，且产物没有 self-template

---

## 最终回答

### 1. `fallback to miss -> self-template` 还存在吗？

不存在了。当前 cross miss 会直接失败，不会再回退到 self-template。

### 2. 现在代码里还会不会偷偷采用 self-template？

按我这次实际检查的用户入口和构建行为：

- **不会**

因为：

- self CLI 入口已经移除
- self fallback 已删除
- non-self synthetic hit 产物也没有混入 self-template

### 3. 在 `conda activate /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/protenix` 里，cross 成功吗？

当前不能确认 full cross search 成功，因为该环境里没有 `mmseqs`，search 阶段直接失败。

更准确地说：

- cross-only 逻辑本身是对的
- 但 full MMseqs2 search 在这个环境里目前跑不起来

如果你要下一步，我建议直接做一件事：

1. 先把 `mmseqs` 装进这个 conda 环境或把它加到 PATH  
2. 我再替你跑一次缩小版 cross-only E2E，并给出最终“cross hit/无 self-template”的实测报告
