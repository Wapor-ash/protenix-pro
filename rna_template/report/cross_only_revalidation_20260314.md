# RNA3D Cross-Only 复查报告

审查日期：2026-03-14

审查对象：

- 前一轮验收报告：`/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_version2_acceptance_review_20260314.md`
- 当前测试脚本：`/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh`
- 当前主 pipeline：`/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh`
- 当前 search/build 脚本：
  - `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py`
  - `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py`

## 结论

本轮结论分三条：

1. 前一轮报告的第一个 bug 还没有真正修复。
2. 第二个 bug 已经解决，现在不会再 `cross-template miss -> self-template`。
3. 你指定的 conda 环境现在已经能跑通 cross-only E2E：`mmseqs` 可用，search 有命中，cross-template 能成功构建，index/NPZ 校验通过，而且产物没有混入 self-template。

## Findings

### 1. 高严重度：前一轮的第一个 bug 仍未修复，E2E 还是没有覆盖 production query source

前一轮报告的第一个 bug 是：

- `test_rna3d_e2e.sh` 没有验证 production 的 query source
- 它用的是 `catalog -> query`
- 不是 `training_sequences JSON -> query`

这个问题现在仍然存在。

证据：

- `test_rna3d_e2e.sh` 里仍然明确写着：
  - `# Use catalog as both database and query for this test.`
  - 位置：`test_rna3d_e2e.sh:115`

- 同一个脚本在 Step 2 调用：
  - `03_mmseqs2_search.py --catalog "${TEST_CATALOG}" ...`
  - 没有传 `--training_sequences`
  - 位置：`test_rna3d_e2e.sh:116-124`

- `03_mmseqs2_search.py` 的主逻辑是：
  - 如果传了 `--training_sequences`，就用 `load_training_sequences_from_json()`
  - 否则退回 `load_training_sequences_from_catalog()`
  - 位置：`03_mmseqs2_search.py:988-995`

- production 主路径 `run_pipeline.sh` 仍然使用：
  - `TRAINING_SEQ_JSON="${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"`
  - 然后传 `--training_sequences "${TRAINING_SEQ_JSON}"`
  - 位置：`run_pipeline.sh:162-181` 和 `run_pipeline.sh:201-212`

所以当前事实是：

- production 路径：`training_sequences JSON -> mmseqs2 -> cross build -> index`
- 测试路径：`catalog -> mmseqs2 -> cross build -> index`

这两条路径不是同一条。

这也是前一轮第一个 bug 为什么还没消失的原因。

### 2. 第一个 bug 应该怎么修

正确修法不是改 search/builder 内核，而是改 E2E 测试入口。

要让 `test_rna3d_e2e.sh` 真正覆盖 production query source，至少需要做到：

1. 在测试目录里准备一个小型 `training_sequences JSON`
   - 格式要和 production 一致：`{sequence: [pdb_id, ...]}`
   - 数据源应该来自你真实训练侧会用到的 `rna_sequence_to_pdb_chains.json` 子集，或者按同样 schema 生成一个 test subset

2. Step 2 和 Step 4 都显式传：
   - `--training_sequences "${TEST_TRAINING_SEQ_JSON}"`

3. 测试断言要从“catalog 数量”改成“training query 数量 / hit 数量 / build 数量”
   - 否则这个测试依旧是在验证数据库 catalog，不是在验证训练 query source

4. 如果你还想保留 “catalog as query” 这条路径，也应该把它拆成另一份测试
   - 名字应明确写成 `catalog_smoke_test`
   - 不能继续把它当作 production E2E

一句话概括：

- 第一个 bug 的修复点在 `test_rna3d_e2e.sh`
- 不是在 `02_build_rna_templates.py`
- 也不是在 `03_mmseqs2_search.py` 的核心搜索逻辑

