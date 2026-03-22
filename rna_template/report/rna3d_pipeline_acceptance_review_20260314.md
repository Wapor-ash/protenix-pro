# RNA3DB Pipeline 验收报告

审查日期：2026-03-14  
审查对象：

- 问题单: [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/pipe_prob.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/pipe_prob.md)
- 实现报告: [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md)
- 代码:
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh)

## 结论

不是“全部完成验收通过”。

更准确的结论是：

- `Bug #1/#3/#4/#6` 的代码层修复基本已经落地，静态检查和小范围函数级验证都能支持。
- `Bug #2` 的报告结论仍然不能通过验收。新的 `test_rna3d_e2e.sh` 虽然新增了 `cross_templates` 构建步骤，但重建出来的 index 仍然指向 `templates/`，训练时实际吃到的还是 self-template 目录，不是 `cross_templates/`。
- 现有 `rna3d_pipeline.md` 中 “All have been verified, fixed, and tested” 的表述过强。

我的最终判断：

> 主干修复完成度高于之前，但整份 `rna3d_pipeline.md` 还不能算“全部修复并验收通过”。

---

## Findings

### 1. 高严重度：E2E 脚本虽然新建了 `cross_templates/`，但 index 仍然硬编码回 `templates/`，所以 cross-template 没有真正接入训练消费路径

`test_rna3d_e2e.sh` 现在确实新增了 cross-template 构建步骤，把产物写到 `CROSS_TEMPLATE_DIR="${TEST_DIR}/cross_templates"`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L147](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L147) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L169](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L169)。

但 `03_mmseqs2_search.py` 的 `build_cross_index()` 仍然把相对路径硬编码成：

- `templates/<npz_name>`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L722](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L722) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L729](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L729)。

这在生产主 pipeline 中问题不大，因为 production 的 cross-template 就写在 `rna_database/templates/`。但在你新的 E2E 测试里，cross-template 被写在 `cross_templates/`，于是出现了静默错连：

1. Step 4 真实生成的是 `cross_templates/*.npz`
2. Step 5 用 `--template_dir "${CROSS_TEMPLATE_DIR}"` 检查文件存在
3. 但写进 index 的仍是 `templates/*.npz`
4. 训练时 `template_database_dir` 传的是 `${TEST_DIR}`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L346](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L346) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L349](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L349)
5. 运行时实际解析到的是 `${TEST_DIR}/templates/*.npz`

我对当前工作区已有产物做了核对：

- 索引文件 [`rna_template_index.json`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database_e2e_test/rna_template_index.json) 中记录的路径是 `templates/...`
- 但 cross-template 真正落在 [`cross_templates`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database_e2e_test/cross_templates)
- 同名文件在 [`templates`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database_e2e_test/templates) 和 [`cross_templates`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database_e2e_test/cross_templates) 都存在，因此这个错误不会报“文件不存在”，而是更危险的“静默指错目录”

我抽查了同名文件 `1ddy_A_A_template.npz`：

- [`templates/1ddy_A_A_template.npz`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database_e2e_test/templates/1ddy_A_A_template.npz) 的 `template_names` 只有一个真实模板名
- [`cross_templates/1ddy_A_A_template.npz`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database_e2e_test/cross_templates/1ddy_A_A_template.npz) 的 `template_names` 里有两个真实模板名

这说明新的 E2E 虽然“构建了 cross-template 文件”，但训练索引仍然回到了 self-template 目录。  
因此 `rna3d_pipeline.md` 对 Bug #2 “now truly fixed” 的结论不成立。

### 2. 高严重度：新的 E2E 仍然没有验证 production query source，也没有验证 non-self cross-template 行为

Step 3 的搜索仍然写着：

> `# Use catalog as both database and query for this test`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L127](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L127)。

而且仍然显式传了：

- `--no_exclude_self`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L128](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L128) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L137](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L137)。

这意味着当前测试脚本依然没有证明两件 production 关键行为：

- 它没有验证真实 `--training_sequences rna_sequence_to_pdb_chains.json` 这条 query 来源
- 它没有验证“排掉 self 后仍然能找到非自身模板”

我对当前工作区已有 search artifact 做了抽查，[`search_results.json`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database_e2e_test/search_results.json) 里确实还能看到 query 和同 base PDB template 并存的情况，这和 `--no_exclude_self` 的当前写法一致。

所以 Bug #2 不是“完全没修”，而是：

- `新增了 cross-template build`
- 但还没有达到 `rna3d_pipeline.md` 所声称的那种完整验证强度

### 3. 中严重度：`test_rna3d_e2e.sh` 在当前机器上不能直接运行，报告里 “verified/tested” 不能在当前环境复现

脚本仍然硬编码：

- `conda activate protenix`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L60](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L60) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L61](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L61)。

我实际尝试运行缩小版：

- `bash test_rna3d_e2e.sh --num_test 6 --skip_training --no_arena`

脚本在环境激活阶段直接失败：

- `EnvironmentNameNotFound: Could not find conda environment: protenix`

当前机器上的 `conda info --envs` 只有：

- `base`
- `/inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/r1126_rna`

所以就“当前工作区可复现性”而言，这个测试脚本还不能算验收通过。  
这条不一定说明核心逻辑错了，但它说明 `rna3d_pipeline.md` 里的 “verified/tested” 至少在当前机器上无法直接复现。

### 4. 中严重度：`Bug #1/#9` 的核心修复是真的，报告这部分基本成立

`03_mmseqs2_search.py` 已加入 `extract_base_pdb_id()`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L84](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L84) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L100](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L100)。

MMseqs2 路径现在会：

- 给 target 记录 `base_pdb_id`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L407](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L407) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L423](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L423)
- 给 query 侧也统一抽 base PDB；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L425](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L425) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L435](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L425)
- 解析结果时按 base PDB 做 self-exclusion；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L313](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L313) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L321](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L313)

`pairwise_search()` 也同步修了；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L616](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L616) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L644](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L616)。

我用一个合成的 `convertalis` 结果做了函数级验证：

- query `4tna`
- target `4tna_A`
- `exclude_self=True`

`parse_mmseqs2_results()` 返回空结果 `{}`，说明这条修复在代码层确实生效。

### 5. 中严重度：`Bug #3` 的 `--pdb_list` 修复已经落地，报告这部分成立

现在 `main()` 在加载 `training_seqs` 后会显式消费 `args.pdb_list`，并按 `extract_base_pdb_id(k)` 过滤；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L1045](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L1045) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L1065](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L1045)。

我用一个合成训练 JSON 做了 spot check：

- 原始 query IDs：`1ehz`, `1f7y`, `4tna`
- `pdb_list` 只保留 `4tna` 和 `1f7y`
- 过滤后得到：`1f7y`, `4tna`

这部分修复验收通过。

### 6. 中严重度：`Bug #4` 的 `release_date_cutoff` 修复已经落地，报告这部分基本成立

现在脚本已新增：

- `--rna3db_metadata`
- `filter_catalog_by_release_date()`
- 主流程里的 cutoff 校验和 catalog 过滤

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L743](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L743) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L805](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L743)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L986](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L986) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L1031](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L986)。

`run_pipeline.sh` 也已经把参数接进来了，并在未显式指定 metadata 时默认回落到 RNA3DB 的 `filter.json`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L57](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L57) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L58](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L57)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L115](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L115) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L122](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L115)。

我用真实 RNA3DB `filter.json` 做了小样本验证：

- `6n5s_A` 会被 `2023-01-01` cutoff 保留
- `8t2p_A`、`8wmn_O` 会被移除

函数输出和 `filter.json` 里的真实 `release_date` 一致。  
这部分修复验收通过。

### 7. 低严重度：`Bug #6` 的 `cif_path` 修复已经写进 builder，静态实现成立

`find_cif_path()` 现在支持 `catalog_cif_path` 优先；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L83](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L83) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L112](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L83)。

`build_self_template()` 和 `build_cross_template()` 也都已经把 catalog 路径传进去了；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L201](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L201) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L204](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L201)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L317](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L317) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L328](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L317)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L517](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L517) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L529](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L517)。

这部分我主要做了静态核对和语法检查，没有在当前 base 环境下做函数级 import 运行，因为该脚本依赖 `Bio`，而当前默认 Python 环境没有这个包。  
不过从代码路径上看，这条修复已经实现。

---

## 验证摘要

本次验收我实际完成了这些检查：

- `python -m py_compile` 通过：
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)
- `bash -n` 通过：
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh)
- 函数级 spot checks：
  - `extract_base_pdb_id()` / self-exclusion：通过
  - `--pdb_list` 过滤：通过
  - `release_date_cutoff` + `filter.json`：通过
- 产物级检查：
  - 现有 E2E index 路径仍指向 `templates/`
  - 同名 self/cross `.npz` 内容不同，证明“索引没有真正指向 cross 产物”
- 运行脚本验证：
  - `test_rna3d_e2e.sh --num_test 6 --skip_training --no_arena` 在当前机器上因 `conda activate protenix` 失败，无法直接复现报告中的 “verified/tested”

---

## 最终判定

### 已通过

- Self-exclusion ID 归一化修复
- `--pdb_list` 在 search 路径中的生效
- `release_date_cutoff` 的真实实现
- `cif_path` 优先使用的 builder 修复

### 未通过

- `rna3d_pipeline.md` 中关于 Bug #2 “已真正修复并完整验证”的结论

### 原因

新的 E2E 虽然新增了 cross-template 构建，但：

- index 仍然回指 `templates/`
- 测试仍使用 catalog 作为 query source
- 测试仍显式允许 self-hit
- 脚本在当前机器上还不能直接起跑

所以这份实现报告更准确的表述应该是：

> 多数底层修复已完成，但 E2E 验收仍未完全闭环，`Bug #2` 不能判定为 fully fixed。
