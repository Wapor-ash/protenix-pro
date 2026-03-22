# RNA3DB Template Pipeline Code Review

审查日期：2026-03-14  
审查范围：

- Prompt: [/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt](/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt)
- 现有实现报告: [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md)
- 主要脚本:
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh)
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh)
- 运行时消费端:
  - [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py)

## 结论

这套 RNA3DB 接入不是“完全没做”，但也没有达到 prompt 想要的“已严谨跑通的 production 级全链路”。

已经成立的部分：

- `rna3db-mmcifs` 确实被接成 template source，且默认会递归扫到 `train_set` 和 `test_set`。
- Arena refine 确实被接进了 batch template builder，而不是只停留在 `run_arena_and_template.sh`。
- MMseqs2 搜索脚本也确实接进了主 pipeline。

没有成立的部分：

- RNA3DB 的 ID 体系和训练集 query ID 体系没有被统一好，导致最关键的 self-exclusion / 同源排除逻辑实际失效。
- 现有 `rna3d_pipeline.md` 对“E2E 已验证”“若干 code review 问题已修复”的表述明显过强。
- 数据架构虽然能跑，但保留的 provenance 太少，后续做去泄漏、审计、分库、回溯都会吃亏。

我的判断：

> 这是一个“能搭起来、能产出部分结果的原型集成”，不是一个已经可信完成的 RNA3DB production template pipeline。

---

## Findings

### 1. 高风险：RNA3DB catalog ID 和训练 query ID 不在同一命名空间，导致 self-exclusion 实际失效

`01_extract_rna_catalog.py` 的 catalog key 直接使用文件 stem，例如 `1ddy_A`、`1jgp_1`，并没有保留成纯 PDB ID；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L131](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L131) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L145](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L145)。

但 `03_mmseqs2_search.py` 从训练 JSON 读 query 时，仍然把 query key 设成纯 `pdb_id.lower()`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L717](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L717) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L735](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L735)。随后 self-exclusion 又用 `query_id.split("_")[0]` 提取 query PDB；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L403](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L403) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L410](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L410)，匹配时直接和 target `pdb_id` 做字符串相等比较；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L297](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L297) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L301](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L301)。

这在 RNA3DB 接入后变成：

- query：`4tna`
- target：`4tna_A`

字符串不相等，所以“排除自身”不会触发。

我对当前工作区的实际数据做了核对：

- 当前训练 JSON 有 `3460` 个 query ID，全部是不带下划线的纯 PDB ID。
- 当前 RNA3DB catalog 有 `13128` 个 key，全部带下划线。
- 训练 ID 和 catalog key 的 `exact match = 0`
- 但有 `2211` 个训练 ID 能在 catalog 中找到 `prefix match`，说明它们确实对应同一个基础 PDB，只是命名空间不一致。

这不是小问题。它直接影响：

- `exclude_self`
- 同 PDB 去重
- 任何按 PDB 粒度做的数据泄漏控制
- 任何你以为“已经排掉自身命中”的搜索统计

`pairwise_search()` 里也有同样的问题，因为它同样用 `query_id.split("_")[0]` 和 `db_pdb_id` 做直接比较；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L613](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L613) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L618](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L618)。

### 2. 高风险：`rna3d_pipeline.md` 里“E2E 已验证 full MMseqs2 + cross-template + training”的结论不成立

现有报告声称：

- `test_rna3d_e2e.sh` “validates full MMseqs2 + cross-template + training path”；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L180](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L180) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L186](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L186)
- 脚本步骤里也写了 Step 4 “Build cross-templates from search results”；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L7](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L7) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L14](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L14)

但脚本实际做的是：

1. 提 catalog；[/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L75](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L75) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L93](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L93)
2. 只 build `--mode self` 的模板；[/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L107](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L107) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L113](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L113)
3. 做一次 MMseqs2 search，但直接 `--no_exclude_self`；[/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L127](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L127) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L137](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L137)
4. 然后直接拿 Step 2 生成的 self-template `.npz` 做 NPZ 检查和训练；[/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L147](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L147) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L212](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L212)

脚本里根本没有调用：

- `02_build_rna_templates.py --mode cross`

所以它没有验证：

- `MMseqs2 search_results -> cross-template NPZ -> index -> featurizer -> train`

它验证的是：

- `self-template NPZ -> index -> training smoke test`

这和 prompt 第 5、6 条想要的验证强度不一致；见 [/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L29](/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L29) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L31](/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L31)。

### 3. 高风险：`--pdb_list` 在 MMseqs2 search 路径里没有真正生效，报告里“已修复”不成立

`run_pipeline.sh` 在 mmseqs2 模式确实把 `${PDB_LIST_ARGS}` 传给了 `03_mmseqs2_search.py`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L177](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L177) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L189](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L189)。

但 `03_mmseqs2_search.py` 的搜索入口并没有使用 `args.pdb_list` 去过滤 query 或 database。它只在 fallback loader 中支持 `training_pdb_list`，而不是 `pdb_list`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L738](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L738) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L760](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L760)，主分支加载训练序列时只看 `args.training_sequences` 或 `args.training_pdb_list`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L933](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L933) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L941](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L941)。

`args.pdb_list` 只是被 argparse 定义了；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L909](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L909) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L913](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L913)，后面没有消费。

所以当前真实行为是：

- `--pdb_list` 会限制 catalog 提取 / self-build 阶段
- 但不会限制 mmseqs2 search 的 query 集

这与 `rna3d_pipeline.md` 中 “Issue #4: `--pdb_list` not passed to search | Fixed” 的说法冲突；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L180](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L180) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L185](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L185)。

### 4. 高风险：报告宣称“泄漏护栏已修复”，但 `release_date_cutoff` 仍是空壳参数

`03_mmseqs2_search.py` 里确实定义了 `--release_date_cutoff`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L895](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L895) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L901](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L901)，但整个脚本没有任何地方实际使用这个参数做过滤。

因此这类护栏现在都没有真正实现：

- release date cutoff
- query date window
- duplicate/near-duplicate filtering
- 与 protein template 对齐的更严格 anti-leakage 逻辑

`rna3d_pipeline.md` 把 “No data leakage safeguards” 标成 “Fixed” 不成立；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L178](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L178) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L186](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L186)。

在 RNA3DB 被当作大型 template DB、而且 prompt 明确要求跑到 finetune 场景时，这个缺口不是“以后再补”的小事，而是当前结论可靠性的上限。

### 5. 中风险：数据架构虽然能用，但过度扁平化，丢掉了 RNA3DB 最关键的 provenance

`01_extract_rna_catalog.py` 最终只保留：

- `chain_id`
- `sequence`
- `num_residues`
- `cif_path`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L93](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L93) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L97](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L97)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L141](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L141) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L144](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L141)。

但 RNA3DB 原始架构里还有：

- `train_set` / `test_set`
- `component_*`
- sequence-cluster representative
- cluster membership
- filter/cluster/split provenance

这些信息全都在 catalog 扁平化时丢掉了。

这会带来几个工程后果：

- 以后没法在 template DB 层按 RNA3DB split / component 做审计
- 不能方便地排查“这个模板来自 train_set 还是 test_set”
- 不能方便做 family/component 级别去重或抽样
- 一旦想做更严谨的 benchmark 隔离，需要重新回溯文件路径或重建元数据

这不一定会立刻导致脚本跑不通，但它说明“数据架构是否正确”这个问题上，当前设计更像“先把文件堆成可搜集合”，不是“为长期训练和审计设计好的 template DB schema”。

### 6. 中风险：builder 记录了 `cif_path`，但实际仍然二次递归查文件，架构上偏脆弱

`01_extract_rna_catalog.py` 已经把精确 `cif_path` 记录进 catalog；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L141](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L141) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L144](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L144)。

但 `02_build_rna_templates.py` 仍然不用这份路径，而是每次靠 `find_cif_path()` 再递归搜一次；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L83](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L83) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L103](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L103)，self 和 cross 两条路径都是这样；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L192](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L192) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L205](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L205)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L306](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L306) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L323](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L323)。

在当前这一版 RNA3DB 数据里，这不一定立刻出错，因为 catalog key 正好等于文件 stem。但它有两个明显问题：

- catalog 和 builder 维护了两套“定位源结构”的机制，语义重复
- 一旦将来接多 release、混目录、软链接或别的命名风格，二次 glob 的非确定性会比直接用 `cif_path` 更脆弱

所以这不是当前最严重的 bug，但属于明显的架构欠账。

### 7. 中风险：运行时索引是一对多，但 featurizer 只消费第一份可用模板文件

`03_mmseqs2_search.py` 的 index 设计允许一个 sequence 对应多个 `.npz` 路径；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L652](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L652) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L710](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L710)。

但 `RNATemplateFeaturizer` 在运行时只会遍历到第一份能成功 load 的 `.npz` 就停；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L279](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L279) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L295](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L295)。

这意味着：

- 如果你想让“相同 RNA sequence 在 RNA3DB 里对应的多份模板”都在运行时被看到，现在并没有真正发生
- 只有 cross-template 模式下“一个 query 对应一个已经 stack 好的 `.npz`”时，这个限制才相对不敏感

因此把 RNA3DB 变大，并不自动等于运行时有效模板多样性真的变大。

### 8. 中风险：E2E 训练 smoke test 的 train/test 配置本身有泄漏，不应被当作效果证据

`test_rna3d_e2e.sh` 训练阶段把同一个 `VAL_PDB_LIST` 同时喂给 train 和 test；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L251](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L251) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L256](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L256)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L307](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L307) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L316](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_rna3d_e2e.sh#L316)。

所以报告里任何基于这个脚本的指标，例如 “Result: PASSED”“loss 正常”“rna_lddt/mean” 等，只能算 smoke test 级别信号，不能当成模板搜索策略或 RNA3DB 接入质量已经被验证的证据；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L227](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L227) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L237](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_pipeline.md#L237)。

### 9. 低风险：ID 语义已经开始变形，后续很容易继续长歪

当前 RNA3DB catalog key 本身就可能带链信息，例如 `1jgp_1`。MMseqs2 flatten 后再拼一次 chain，会得到 query/database id 类似：

- `1jgp_1_1`

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L390](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L390) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L398](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L390)。

这虽然还能跑，但语义已经不清晰了：

- 第一段是基础 PDB？
- 第二段是 RNA3DB file stem 的后缀？
- 第三段才是真正 chain？

一旦后面还要接 release cutoff、family 去重、chain-level audit，这种 ID 体系会持续放大复杂度。

---

## 实际上做对了什么

下面这些是我确认“确实做了”的，不应该和上面的缺陷混为一谈：

### 1. RNA3DB 确实被当作 template source 接进来了

`run_pipeline.sh` 默认 `PDB_RNA_DIR` 已经切到 RNA3DB；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L29](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L29) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L42](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L42)。

`01_extract_rna_catalog.py` 也确实支持从根目录递归扫描 `**/*.cif`；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L167](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L167) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L174](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L174)。

这满足了 prompt 里“train 和 test 都要用”的基础要求；见 [/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L16](/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L16) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L21](/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L21)。

### 2. Arena refine 真的接进了 batch builder

`02_build_rna_templates.py` 里有真实可用的 `run_arena_refine()`，而不是只在单条脚本 `run_arena_and_template.sh` 存在；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L110](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L110) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L170](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L170)。

而且 self / cross 两条模板构建路径都会在启用时调用它；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L199](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L199) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L209](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L199)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L314](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L314) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L326](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L314)。

### 3. 主 pipeline 顺序修正是对的

`run_pipeline.sh` 的 mmseqs2 路径现在是：

1. 先 search
2. 再 build cross-template
3. 最后 rebuild index

见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L164](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L164) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L223](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L223)。

这一点比之前 pairwise 版本确实更干净。

---

## 与 prompt 的对照

### Prompt 想要的

见 [/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L19](/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L19) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L35](/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-3d.txt#L35)：

1. 用 RNA3DB 建 template DB
2. 把 Arena refine 补进 batch pipeline
3. 对 training data 实现 MMseqs2 template search
4. 修掉已有 code review 的问题
5. 让一个 val RNA 真正走到“找到 template -> 生成 feature -> 送进 datapipeline/protenix”
6. 跑通全部 pipeline

### 当前实现的完成度

1. RNA3DB 作为 template source：`基本完成`
2. Arena 接入 batch builder：`完成`
3. MMseqs2 搜索脚本接入：`完成`
4. code review 问题修复：`部分完成，且报告夸大`
5. val RNA 真正走 cross-template 路径到训练：`没有被当前 E2E 脚本证明`
6. 全 pipeline production 级跑通：`没有被当前证据证明`

---

## 发散思维：可能的漏洞和不及格点

下面这些不一定全部已经爆炸，但都属于“这套架构再往前推进时最容易踩雷”的点。

### 1. 自身命中泄漏会被误当成“MMseqs2 检索效果很好”

因为 ID 空间不统一，很多同 PDB 模板并没有被真正排除。最危险的情况不是脚本报错，而是日志很好看、命中率很高、训练也能跑，但你以为是 template search 在工作，实际上是 self-hit 或近 self-hit 在抬结果。

### 2. 把 RNA3DB 的 `test_set` 也并入 template DB，本身就要求更强的 provenance 和隔离规则

prompt 明确说 train/test 都要用。这个选择不是不可以，但代价是：

- 你必须清楚知道每个 template 来自哪一边
- 你必须明确定义“哪些评估允许看到 test_set template”

当前 catalog schema 没保留 split provenance，后面要补这个边界会很痛。

### 3. 以后如果换数据库版本或混多个 release，`find_cif_path()` 这种二次 glob 容易变成隐性 nondeterminism

现在单库单版本问题不大，但只要目录里同时存在多个同 stem 文件，谁先被 glob 到、哪个版本被当模板，就会变成环境相关行为。

### 4. query/database ID 已经带有“伪 PDB + chain + 再拼 chain”的味道，后续接近蛋白 template 的治理会越来越难

尤其是未来如果还要加：

- release date
- duplicate family filtering
- cluster-level filtering
- component-level audit

继续在字符串上 `split("_")` 会非常脆。

### 5. E2E 脚本现在更像 smoke test，不像 acceptance test

一个真正的 acceptance test 至少要证明：

- search 产出的 `search_results.json` 真正被 `--mode cross` 消费
- 训练里加载的是 cross-template 生成的 `.npz`
- self-exclusion 是开着且生效的
- 至少有一个 query 没有命中自身却命中了外部 template

当前脚本没有做到这些。

### 6. RNA3DB 变大之后，不代表运行时有效模板就真的变多

因为 featurizer 只消费第一份可用 `.npz`。如果不改变消费端逻辑，大库的很多冗余模板只是离线 index 变长，而不是模型真的看到了更多候选。

---

## 最终判断

如果标准是“做出一个能跑的 RNA3DB 原型接入”，当前实现及格。

如果标准是“做出一个和 protein template 同等级别、可以自信宣称已经跑通并有防泄漏保证的全链路 template pipeline”，当前实现不及格。

最关键的原因不是 Arena、也不是 MMseqs2 本身，而是：

- ID 体系没有统一
- 反泄漏护栏没有落地
- 验证脚本没有验证真正的主路径
- 现有实现报告把这些问题说轻了

## 附：本次审查的核心结论一句话版

> RNA3DB 已经“接进来了”，但还没有“被正确治理好并被严格验证过”。
