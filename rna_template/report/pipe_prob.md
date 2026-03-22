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

___

注意针对时间cutffoff 修复：
请参考 /inspire/ssd/project/sais-bio/public/ash_proj/data/RNA3D/rna3db-jsons 这个文件里面有记录每个pdb 的time cutoff， 把用时间作为选项，然后修复这个功能实现


### 6. 中风险：builder 记录了 `cif_path`，但实际仍然二次递归查文件，架构上偏脆弱

`01_extract_rna_catalog.py` 已经把精确 `cif_path` 记录进 catalog；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L141](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L141) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L144](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py#L144)。

但 `02_build_rna_templates.py` 仍然不用这份路径，而是每次靠 `find_cif_path()` 再递归搜一次；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L83](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L83) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L103](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L103)，self 和 cross 两条路径都是这样；见 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L192](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L192) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L205](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L205)，以及 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L306](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L306) 到 [/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L323](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L323)。

在当前这一版 RNA3DB 数据里，这不一定立刻出错，因为 catalog key 正好等于文件 stem。但它有两个明显问题：

- catalog 和 builder 维护了两套“定位源结构”的机制，语义重复
- 一旦将来接多 release、混目录、软链接或别的命名风格，二次 glob 的非确定性会比直接用 `cif_path` 更脆弱

所以这不是当前最严重的 bug，但属于明显的架构欠账。


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

## 附：本次审查的核心结论一句话版

> RNA3DB 已经“接进来了”，但还没有“被正确治理好并被严格验证过”。
