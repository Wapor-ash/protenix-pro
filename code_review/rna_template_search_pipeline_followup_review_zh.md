# RNA Template Search Pipeline 跟进审查报告

**审查日期**: 2026-03-14  
**审查范围**:
- `prompts/rna_template/run_pipeline.txt`
- `code_review/rna_template_search_pipeline_report_zh.md`
- `rna_template/scripts/*.py`
- `rna_template/scripts/*.sh`
- `protenix/data/rna_template/rna_template_featurizer.py`
- `protenix/data/pipeline/dataset.py`
- `protenix/data/inference/infer_dataloader.py`
- `protenix/data/template/*`（用于和 protein template 管线对比）

---

## 1. 结论

当前实现不是“完全未做”，也不是“已经按 production 目标完整完成”，而是：

- **已完成**:
  - RNA template 的离线目录提取、`.npz` 构建、index 构建脚本已经存在。
  - `self-template` 路径已经形成闭环。
  - 训练与 inference 的 RNA template 特征接入已经打通。
  - 当前工作区里存在 `rna_database_test/` 与 `output/rna_template_e2e_test/` 产物，说明至少做过一次小规模闭环验证。

- **未完成/未达到 prompt 原始目标的部分**:
  - `pairwise search` 这条“生产用”主路径**没有达到 protein template 同等级别的安全性与严谨性**。
  - 现有小规模 E2E 验证**只覆盖 self-template**，没有覆盖你真正想要的 `pairwise search + cross-template` 主路径。
  - 现有报告对“已验证”的表述偏强，容易让人误判为“完整生产管线已验证”。

我的判断是：

> **目标完成度：部分完成。**  
> 如果把目标定义成“搭一个能演示 RNA template 全链路的原型”，基本成立。  
> 如果把目标定义成“对标 protein template 的可训练、可验证、可防泄漏的 production pipeline”，目前还没有完成。

---

## 2. 主要发现

### 2.1 高风险: `pairwise` 搜索链路缺少 protein template 已有的防泄漏护栏

**RNA 当前实现**

- `rna_template/scripts/03_search_and_index.py:104-175`
- 这里只做了：
  - 全局 pairwise identity
  - 长度比例过滤
  - `exclude_self`（按 PDB ID 排除自身）

**Protein 现有实现**

- `protenix/data/template/template_featurizer.py:268-304`
- `protenix/data/template/template_featurizer.py:353-380`
- `protenix/data/template/template_featurizer.py:447-450`
- `protenix/data/template/template_utils.py:352-357`
- `protenix/data/template/template_utils.py:907-928`

protein template 现成包含这些护栏：

- release date cutoff
- query release date 向前回退窗口
- duplicate / large subsequence rejection
- hit sequence 去重
- train 阶段 template dropout / shuffle top-k

而 RNA `pairwise` 管线没有任何对应实现。

**影响**

- 训练和评估都存在明显 data leakage 风险。
- 即便 `exclude_self=True`，仍然可能命中“同序列近重复”“发布日期晚于 query 的泄漏模板”“几乎同构的重复条目”。
- 这使得它和 protein template 的可靠性不在一个级别上。

**结论**

这不是风格差异，是**功能级缺口**。  
如果报告把当前 `pairwise` 路径描述成“生产用”，这个结论不成立。

---

### 2.2 高风险: 现有 E2E 测试并没有验证 `pairwise search + cross-template` 主路径

**证据**

- `rna_template/scripts/test_small_e2e.sh:127-132`

脚本硬编码：

```bash
--strategy self
```

也就是说，当前 E2E 验证覆盖的是：

- catalog 提取
- self-template `.npz` 生成
- self index 构建
- self-template 注入训练

它**没有验证**：

- `03_search_and_index.py --strategy pairwise`
- `02_build_rna_templates.py --mode cross`
- `pairwise search_results -> cross-template npz -> index -> train` 的完整闭环

**与现有报告的冲突**

- `code_review/rna_template_search_pipeline_report_zh.md:5`
- `code_review/rna_template_search_pipeline_report_zh.md:50-59`
- `code_review/rna_template_search_pipeline_report_zh.md:315`

现有报告把 `pairwise` 路径描述成“生产用”，并把整体表述成“E2E 管线验证通过”。  
从脚本与现有验证方式看，这个结论最多只能支持：

> `self-template` 验证路径通过。

不能外推成：

> `pairwise search` 主路径已被端到端验证。

---

### 2.3 中风险: 测试脚本把同一份 PDB 列表同时当作 train 和 eval，验证指标有泄漏

**证据**

- `rna_template/scripts/test_small_e2e.sh:299-303`
- `rna_template/scripts/test_small_e2e.sh:304-311`

同一个 `TEST_PDB_LIST` 同时传给：

- `data.train_sets`
- `data.test_sets`

这意味着：

- 训练集和验证集并不独立
- 日志里的 eval 指标只能说明“脚本能跑到 eval”
- 不能说明模板搜索策略或训练设置具有泛化意义

**影响**

如果报告把 `rna_lddt/mean`、`complex_lddt/mean` 这类数值当作“有效验证指标”，结论会偏乐观。  
这些数值在当前脚本里更适合被解释为：

> smoke test 期间的程序运行指标

而不是模型效果证据。

---

### 2.4 中风险: `run_pipeline.sh` 的 `--pdb_list` 在 `pairwise` 模式下没有真正限制 query 集

**证据**

- `rna_template/scripts/run_pipeline.sh:96-100`
- `rna_template/scripts/run_pipeline.sh:141-149`
- `rna_template/scripts/run_pipeline.sh:177-185`
- `rna_template/scripts/03_search_and_index.py:291-299`

`03_search_and_index.py` 明明支持 `--training_pdb_list`，但 `run_pipeline.sh` 在 `pairwise` 模式下并没有把 `PDB_LIST` 透传进去，而是固定用：

```bash
--training_sequences "${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"
```

结果是：

- `--pdb_list` 只限制了 catalog / self-build 输入
- 没有限制 pairwise search 的 query 集

**影响**

- 用户以为自己在“对指定小集合做 pairwise 验证”，实际脚本可能在对全量训练序列做搜索。
- 这会影响调试、耗时预期、以及产物规模。
- 脚本接口语义和真实行为不一致。

---

### 2.5 中风险: `pairwise` 模式在模板尚未生成前就先写 index，存在中间态不一致

**证据**

- `rna_template/scripts/run_pipeline.sh:141-149`
- `rna_template/scripts/run_pipeline.sh:151-158`
- `rna_template/scripts/run_pipeline.sh:175-185`
- `rna_template/scripts/03_search_and_index.py:226-240`

`pairwise` 路径的顺序是：

1. 先运行 `03_search_and_index.py`
2. 它会写 `search_results.json`
3. 它还会立即尝试构建 `rna_template_index.json`
4. 但这时 cross-template `.npz` 还没生成
5. 然后才运行 `02_build_rna_templates.py --mode cross`
6. 最后 Step 3 再重建 index

**影响**

- 如果管线在 Step 2 search 之后中断，磁盘上会留下一个空的或不完整的 index。
- 这会让“目录存在但不可用”的状态更难排查。
- 当前逻辑依赖“必须完整跑到 Step 3 才得到可信 index”。

这不是致命 bug，但属于明显的工程脆弱点。

---

### 2.6 中风险: `pairwise` 搜索实现仍然是原型级 O(Q×DB) 全扫描，不适合作为生产主路径

**证据**

- `rna_template/scripts/03_search_and_index.py:16-22`
- `rna_template/scripts/03_search_and_index.py:61-63`
- `rna_template/scripts/03_search_and_index.py:131`

脚本自己已经把这段搜索实现标注成：

- “Replace for production”
- “Future options: nhmmer, cmscan, BLAST”

而当前实现是 Python 层全量双循环：

- 对每个 query 遍历整个 database
- 每个候选再做 `PairwiseAligner` 全局比对

**影响**

- 在小样本 smoke test 上可用。
- 在全量数据库上，耗时与扩展性都很弱。
- 作为“原型接口”没问题，但作为“生产搜索后端”结论过强。

---

### 2.7 低风险: index 支持“同序列多个 npz 路径”，但 featurizer 只消费第一个有效文件

**证据**

- `rna_template/scripts/03_search_and_index.py:201-204`
- `rna_template/scripts/03_search_and_index.py:231-233`
- `protenix/data/rna_template/rna_template_featurizer.py:279-294`

index 的格式是：

```json
sequence -> [path1, path2, path3, ...]
```

但 featurizer 的实际行为是：

- 遍历这些路径
- 只加载第一个成功的 `.npz`
- 成功后立刻 `break`

**影响**

- 多路径 index 的语义没有被真正利用。
- 对 self-template 模式来说问题不大，只是冗余。
- 对 pairwise 模式来说，这会限制“同一 query sequence 可挂多个 npz 包”的设计价值。

这更像设计不一致，不是立即致命 bug。

---

## 3. 当前已经做成的部分

这次审查里，以下内容我认为已经做成：

### 3.1 `self-template` 原型闭环已经成立

当前工作区可见：

- `rna_database_test/rna_catalog.json`
- `rna_database_test/rna_template_index.json`
- `rna_database_test/templates/*.npz`

我本地读取到的产物状态：

- catalog: 9 个 structure
- templates: 13 个 `.npz`
- index: 12 个 sequence

这说明：

- Step 1 提取
- Step 2 self `.npz`
- Step 3 self index

至少有一次成功运行产物留在工作区。

### 3.2 训练/推理接线已经打通

**训练侧**

- `protenix/data/pipeline/dataset.py:1090-1128`
- `protenix/data/pipeline/dataset.py:568-575`

**推理侧**

- `protenix/data/inference/infer_dataloader.py:195-219`
- `protenix/data/inference/infer_dataloader.py:305-313`

说明：

- RNA template featurizer 在 train / inference 两端都能构造
- 特征也都会并入 `features_dict`

### 3.3 当前代码状态下，self 路径的小规模训练日志是能跑完 10 step 的

我核对了现有日志：

- `output/rna_template_e2e_test/training_test2.log`

其中可见：

- step 9 train metrics
- step 9 eval metrics
- `Finished training after 10 steps`

所以，当前代码状态下：

> “self-template smoke test 可以跑到训练结束”

这个结论成立。

---

## 4. 和 Protein Template 的关键差异

### 4.1 Protein template 是“在线检索 + query-aware 过滤”

protein template 当前是：

- 依据 query sequence / query release date 做动态过滤
- 过滤未来模板
- 过滤大重复 / 近重复
- 对命中序列去重
- train 阶段带 template dropout / top-k shuffle

核心参考：

- `protenix/data/template/template_featurizer.py:268-304`
- `protenix/data/template/template_featurizer.py:353-380`
- `protenix/data/template/template_featurizer.py:447-450`
- `protenix/data/template/template_utils.py:338-380`
- `protenix/data/template/template_utils.py:907-928`

### 4.2 RNA template 当前是“离线序列索引 + 直接加载”

RNA template 当前是：

- 预先把 `.npz` 做好
- 用 `sequence -> npz path` 的 index 做查找
- 不带 query release date
- 不带 duplicate/subsequence 过滤
- 不带 train-time dropout

核心参考：

- `rna_template/scripts/03_search_and_index.py:104-175`
- `protenix/data/rna_template/rna_template_featurizer.py:197-211`
- `protenix/data/rna_template/rna_template_featurizer.py:274-300`

### 4.3 因此，两者现在不是“同等级实现”

更准确的说法应该是：

- protein template: 生产级检索与过滤体系
- RNA template: 已经接好模型入口的离线原型管线

这个差异必须写清楚，否则会高估 RNA search pipeline 的完成度。

---

## 5. 对现有报告 `rna_template_search_pipeline_report_zh.md` 的复核意见

### 5.1 可以保留的结论

- 脚本文件确实已经补齐。
- self-template 验证路径确实存在。
- 训练与 inference 接口确实已经接通。
- 现有工作区里也确实有小规模跑过的测试产物与日志。

### 5.2 需要收紧表述的结论

以下说法建议改成更保守的口径：

- “生产用”
- “E2E 管线验证通过”
- “完整全流程已确认可工作”

更准确的写法应该是：

> 已完成 self-template 原型闭环与训练 smoke test；  
> pairwise search / cross-template 路径已具备脚本原型，但尚未完成与 protein template 同等级别的防泄漏设计和专项 E2E 验证。

---

## 6. 审查方法与核验结果

本次没有修改任何源码，只做了审查与核验。

### 6.1 静态核验

- `python3 -m py_compile`:
  - `01_extract_rna_catalog.py`
  - `02_build_rna_templates.py`
  - `03_search_and_index.py`
  - `rna_template_common.py`
  - `rna_template_featurizer.py`
  - 结果：通过

- `bash -n`:
  - `run_pipeline.sh`
  - `test_small_e2e.sh`
  - 结果：通过

### 6.2 产物核验

读取了现有目录：

- `rna_database_test/`
- `output/rna_template_e2e_test/`

确认：

- self-template 测试产物存在
- 当前代码状态下存在一次 10-step 训练完成日志

---

## 7. 最终判断

### 7.1 如果你的目标是“先把 RNA template 全链路原型搭起来”

我的判断：**基本达成。**

因为你已经有：

- catalog 提取
- template 构建
- index 构建
- train / inference 接线
- self-template smoke test

### 7.2 如果你的目标是“把 pairwise template search 做到可对标 protein template 的完整 pipeline”

我的判断：**尚未达成。**

卡点主要在：

- 缺少 protein template 级别的 anti-leakage 护栏
- pairwise 主路径没有单独做 E2E 验证
- 测试口径仍然偏 smoke-test，不是严谨验证
- 脚本接口与实际行为仍有偏差（`--pdb_list` 在 pairwise 下不完整生效）

---

## 8. 建议优先级

如果后续继续补这条线，建议优先级如下：

1. 先补 `pairwise` 路径的专项 E2E 测试，不再只测 `self`
2. 对齐 protein template 的 release-date / duplicate / subsequence 护栏
3. 修正 `run_pipeline.sh` 在 `pairwise` 下对 `--pdb_list` 的透传逻辑
4. 再考虑把 placeholder pairwise 检索替换成更可扩展的搜索后端

---

## 9. 一句话结论

**现在这套 RNA template search pipeline 已经是可演示、可 smoke-test 的原型；但还不是与 protein template 同等级、可放心宣称“生产可用”的完整实现。**
