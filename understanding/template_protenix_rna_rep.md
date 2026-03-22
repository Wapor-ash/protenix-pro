# RNA Template 注入 Protenix 的 Template Feature 设计报告

## 执行摘要

本报告的核心结论是：在未指定 Protenix 具体输入 API/shape 的前提下，最稳妥、可快速落地且能适配多种“AF3/Protenix 风格”网络的方案，是**优先复用 Protenix 现有 TemplateEmbedder 的输入键与张量语义**，把 RNA template 表达成与其兼容的 5 组模板特征：`template_aatype`、`template_distogram(39 bins)`、`template_pseudo_beta_mask`、`template_unit_vector(3D)`、`template_backbone_frame_mask`，从而无需修改主干 Pairformer/扩散模块即可注入模板信息。该接口在 Protenix 中被明确实现为“Algorithm 16 in AF3”的 TemplateEmbedder，并在 `single_template_forward()` 里将上述特征拼接后线性投影、再过一个小型 PairformerStack 得到模板对偶嵌入。citeturn47view0

实现上，关键不是“把 RNA 当蛋白质算 pseudo-beta/主链框架”，而是**重新定义 RNA 的锚点与局部框架**：把“pseudo-beta”改为“每个核苷酸的代表点（建议优先 base center，退化用 C4' 或 C1'）”，把“backbone frame”改为由 RNA 的糖-磷骨架与糖环原子构成的局部正交基（如以 C4' 为原点，使用 `P−C4'` 与 `C1'−C4'` 定义局部 x/y 轴），再据此生成 distogram 与 unit-vector。RNA 的二级结构（碱基配对与堆叠）不必立刻扩展模型输入维度，也能通过模板几何特征“隐式”注入；但若要系统性覆盖 RNA 特有信息（非规范碱基对、堆叠网络、糖-磷骨架扭转、修饰与离子/配体位点），建议并行提供更富表达的“RNA 图模板”或“RNA 二级结构约束通道”，并通过并行/融合策略与现有模块对接（见后文三种设计方案）。citeturn49search8turn49search9turn47view0

需要强调的工程现实是：Protenix 当前的模板检索与特征组装**显式只对蛋白链启用**，RNA 链会走“无模板/空模板特征”的分支；因此你要么（A）在数据管线中扩展模板支持到 RNA_CHAIN，要么（B）绕过其模板检索逻辑，直接把 RNA 模板特征张量注入到模型的 `input_feature_dict`（或等价输入）。Protenix 的代码与示例 JSON 已清楚反映：蛋白链输入支持 `templatesPath`，而 RNA 示例仅包含 RNA MSA 路径、不含模板路径。citeturn25view3turn30view1turn19view0turn21view0turn39search0

## 目标与约束

你的目标可以形式化为：给定一个 RNA query（单链或多链，可能与蛋白/配体形成复合体），你希望把一组 RNA 模板结构（来自 PDB/mmCIF 等）转成**可被 Protenix（AF3-like）消费的 template feature 张量**，从而在训练或推理中为 Pairformer/扩散模块提供更强的几何先验，提升 RNA 3D 构象或复合体构象的准确度与稳定性。citeturn47view0turn50search0

你给定的关键约束是：

第一，你已熟悉蛋白质模板特征的常见计算范式（结构检索→序列比对→映射到 query 坐标系→生成 distogram/orientation/掩码等），但 RNA 在结构语义上显著不同：核苷酸的关键相互作用高度依赖碱基配对与堆叠网络、糖-磷骨架构象与可能的化学修饰。citeturn49search8turn49search9

第二，未指定 Protenix 的输入 API/shape。为满足“无特定约束、可适配多种输入 shape”的要求，本报告采用**“规范化模板张量接口（canonical template interface）”**的写法：先给出与 Protenix 现有 TemplateEmbedder 兼容的最小接口（可零改模型），再给出两种更富表达但需要并行/融合的扩展接口（可适配其它 AF3-like 或自研网络的输入 shape）。citeturn47view0

第三，必须基于高可信来源（PDBx/mmCIF、DSSR/3DNA、原始论文与官方代码、Rfam/RNAcentral）。因此后续的“张量如何定义/如何计算”会以这些工具与标准的输出为基准。citeturn50search0turn49search8turn49search3turn49search2

工程约束方面，你需要面对一个事实：Protenix 在数据侧确实区分了链类型（蛋白、RNA、DNA 等），并定义了 RNA_CHAIN 的类型字符串 `polyribonucleotide`，以及 RNA/DNA 的模板/msa 字符到 ID 的映射；但其模板特征提取流程在关键处写死“仅蛋白链启用模板”，导致 RNA 链默认得到“全掩码的空模板特征”。这既是限制也是机会：你可以以最小侵入方式复用模板嵌入器，把 RNA 模板特征填进去。citeturn28view0turn25view3turn30view1

## 系统性文献与代码调研

为了设计“RNA template tensor”，需要把现有工作按“信息形态”拆开：哪些方法产生**二级结构/配对矩阵**，哪些方法产生**3D 几何分布（distogram/orientation）**，哪些工具能从已知 3D（模板）中抽取**配对/堆叠/几何参数**，以及哪些体系支持**复合体（蛋白–核酸）**。下表按这一思路，对你点名的方法与紧密相关的近年路线做对照（以论文与官方代码为主）。

| 方法/工具 | 主要任务 | 输入信息形态 | 输出/可直接转为 template feature 的信息 | 与“模板化特征”关系 |
|---|---|---|---|---|
| entity["organization","RNAstructure","rna secondary structure pkg"] | RNA/DNA 二级结构预测与分析 | 序列；能给出配对概率等分析能力 | 二级结构（碱基配对集合/概率）的图结构，可作为 pairwise 约束通道或模板先验 | 更像“从序列到二级结构”的先验生成器；可把其输出作为 template-like 2D map（bp 概率图）citeturn48search0turn48search4turn48search12 |
| SPOT-RNA | 二级结构（含非嵌套/非规范配对）预测 | 序列（深度学习） | 预测碱基配对关系（可编码为 NxN pairing map，亦可派生 contact-like features） | 可用作“无真实模板时”的软模板：用 predicted pairing map 充当 template pair feature 的一部分citeturn48search1turn48search9 |
| RNAformer | 二级结构预测（Transformer/轴向注意力/回收） | 序列 | 预测邻接矩阵/配对矩阵（NxN），可直接做 pairwise map | 同样可提供“模板化的 2D 图先验”（bp adjacency）citeturn48search2turn48search10 |
| trRosettaRNA | RNA 3D 结构预测（以几何分布约束驱动建模的路线） | RNA MSA/共变异等（论文体系） | 典型会输出距离/接触分布（distogram/距离图）并用于 3D 建模（trRosetta 风格） | 其“距离图/几何分布”与蛋白模板 distogram 同构：可直接映射到 template_distogram/距离约束通道citeturn0search3 |
| entity["organization","RhoFold+","rna 3d prediction 2024"] | RNA 3D 结构预测（LM + MSA + Transformer） | RNA 语言模型嵌入 + MSA 特征 | 产生可迭代精炼的结构表征；强调 RNA-FM embedding 与 MSA 特征结合 | 提示“模板化特征”可与 LM/MSA 并行：template 是另一条几何先验支路citeturn48search18 |
| entity["organization","RoseTTAFoldNA","protein-na structure prediction"] | 蛋白–核酸复合体 3D 预测 | 蛋白与核酸的序列/MSA 等（单一网络） | 直接输出蛋白–DNA/蛋白–RNA 复合体 3D 模型与置信度 | 对你最重要的启示：核酸与蛋白共处一网时，核酸几何与相互作用约束必须被显式建模；template feature 可作为“外部几何先验”注入citeturn48search3 |
| entity["organization","DSSR","nucleic acid structure annotator"]（entity["organization","3DNA","nucleic acid analysis suite"]生态） | 从 3D 结构自动注释 RNA/DNA（配对、堆叠、修饰等） | PDBx/mmCIF/PDB 坐标 | 输出碱基配对、非规范配对、堆叠、螺旋等结构要素（可结构化为 pairwise map、edge list、residue annotations） | 这是“真模板→模板特征”的主力抽取器：可从模板结构直接产出 bp/stacking/局部几何与掩码citeturn49search8turn49search12turn49search24 |
| entity["organization","MC-Annotate","rna 3d annotation tool"] | RNA 3D 注释（含堆叠网络等） | 3D 坐标 | 可识别序列、注释成分等（常用于配对/堆叠网络抽取） | 与 DSSR 可互补：不同工具对配对/堆叠判据不同，可用于鲁棒性/一致性估计citeturn49search9turn49search13 |
| entity["organization","RNAcentral","non-coding RNA database"] | 非编码 RNA 序列资源枢纽 | 多来源序列与注释 | 提供大规模 RNA 序列集合与检索入口 | 可作为“模板搜库/同源聚类/家族归并”的序列来源之一citeturn49search2turn49search10 |
| entity["organization","Rfam","rna families database"] | RNA 家族、MSA 与 covariance model | 家族 MSA、协方差模型 | 家族级 MSA+CM，可用于同源检索与结构保守性建模 | 对“RNA 模板检索与比对”尤其关键：RNA 一阶序列弱保守时，CM 能提供更可靠的比对与同源搜寻citeturn49search7turn49search11turn49search3 |

把这些工作映射到“template feature”语境时，一个实用的分解是：

- **真模板（来自结构库）→结构注释与几何张量**：主要靠 DSSR/3DNA、MC-Annotate 一类工具把 PDBx/mmCIF 里的 3D 坐标转成可学习的符号/几何对象（配对类型、堆叠、局部框架、修饰等）。citeturn49search8turn49search9turn50search0  
- **无真模板时的“软模板”**：SPOT-RNA、RNAformer、RNAstructure 一类方法输出的二级结构/配对概率矩阵，本质上就是一种 pairwise 结构先验，可以按“模板化特征”的方式注入网络（作为额外 pair map、或作为约束 embedder 的输入）。citeturn48search1turn48search2turn48search0  
- **3D 约束图路线**（trRosettaRNA 等）：输出 distogram/距离图、接触图等 2D 几何分布，与蛋白模板 distogram 同构，是“模板特征张量化”的天然参照。citeturn0search3  

最后，与你的目标直接对齐的一条“代码实锤”是 Protenix 自己对模板的张量接口定义：TemplateEmbedder 在前向中明确读取 `template_distogram(39)`、`template_pseudo_beta_mask(1)`、`template_unit_vector(3)`、`template_backbone_frame_mask(1)`，再加上把 `template_aatype` one-hot 后扩展成 i/j 两份（各 32 维），拼接后得到 108 维 pairwise 模板特征，再嵌入并与 query pair embedding 融合。citeturn47view0

## 对比蛋白质 template 特征与 RNA 的差异

从“如何把模板变成网络能用的张量”角度，蛋白与 RNA 的差异不是“有没有 distogram”这么简单，而是“**最关键的结构因子**”不同。下面按 Protenix/AF3-like 模板接口里出现的关键字段，逐项说明在 RNA 中哪些可直接复用、哪些必须重写、哪些 RNA 特有信息很难用现接口表达。

首先明确 Protenix 当前模板接口的“最小集合”：  
`template_aatype [T,N]`、`template_distogram [T,N,N,39]`、`template_pseudo_beta_mask [T,N,N]`、`template_unit_vector [T,N,N,3]`、`template_backbone_frame_mask [T,N,N]`。citeturn47view0turn26view2turn32view2

| 蛋白模板特征语义（以 Protenix 为例） | 在 RNA 中的可用性 | 需要怎样改写/替代 | 备注 |
|---|---|---|---|
| `template_aatype`：残基类型（32 类 one-hot） | **可复用** | 用核苷酸字母映射到 Protenix 的 residue-id 体系：RNA 的 A/G/C/U 在该体系里有对应索引；非标准碱基可映射为“UNK_NUCLEIC”一类 | Protenix 常量中给出了 RNA/DNA 字符到 ID 的映射与 RNA_CHAIN 类型字符串，说明框架层面已预留核酸类型位。citeturn28view0 |
| `template_distogram`：基于 pseudo-beta 的距离分桶（39 bins） | **可复用但语义要重定义** | 把 pseudo-beta 替换为“核苷酸代表点”（推荐 base center；备选 C4'/C1'），再用相同 bin 设置生成 39 桶距离 one-hot | Protenix 模板 distogram 的 bin 参数与 39 桶在代码中写死，并被 TemplateEmbedder 当作 39 维输入。citeturn47view0turn25view2 |
| `template_unit_vector`：在局部主链框架中表达的方向信息（3 维） | **可复用但必须重写局部框架定义** | RNA 没有蛋白 N-CA-C 的主链三原子；需定义 RNA 局部框架，例如以 C4' 为原点，用 `(P−C4')` 与 `(C1'−C4')` 做 Gram–Schmidt 得到正交基，再把 `r_ij` 投到该基并归一化 | Protenix 的蛋白 unit vector 计算显式依赖蛋白骨架原子索引（C/CA/N）与刚体组定义，因此不能直接用于 RNA。citeturn32view1 |
| `template_backbone_frame_mask`：局部框架可用性掩码 | **可复用（掩码机制本身）** | 将“框架可用”定义为 RNA 选定的三原子是否齐全（比如 P/C4'/C1' 齐全），并在 pairwise 上取外积或按实现生成 `[N,N]` mask | TemplateEmbedder 会把该 mask 当作 1 通道特征并与 multichain/pair_mask 相乘。citeturn47view0 |
| **RNA 特有但现接口缺失的信息**：碱基配对类型、堆叠网络、糖-磷骨架扭转、非规范配对、修饰、金属/配体位点等 | **现接口难以充分表达** | 需要额外设计通道（方案二/三），或通过模板几何“间接编码”一部分（例如配对/堆叠会在 base-center 距离与相对方向中留下强信号） | DSSR/3DNA 能直接注释非规范碱基对、修饰核苷酸、堆叠等，是构建这些通道的首选抽取器。citeturn49search8turn49search12 |

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["RNA base pairing non-canonical Leontis Westhof diagram","RNA base stacking 3D structure illustration","DSSR 3DNA RNA annotation base pair stacking example"],"num_per_query":1}

特别值得你在设计时“强制包含”的 RNA 结构信息（无论你最终选择哪种张量方案）是：

- **碱基配对网络**（包含 canonical 与 non-canonical）：它直接决定二级结构主骨架，是 RNA 结构的“主约束图”。DSSR 与 RNAview 等工具都围绕“识别并分类碱基对”展开。citeturn49search8turn49search17  
- **堆叠网络**：RNA 二级结构的茎区稳定性、许多三级折叠的拓扑都强依赖堆叠。MC-Annotate 被明确用于计算 base-stacking network。citeturn49search13turn49search9  
- **修饰核苷酸与可能的离子/配体位点**：真实 RNA 常包含修饰，且金属离子（如 Mg²⁺）对折叠常关键；DSSR 被设计为能注释“modified nucleotides”等核酸结构特征。citeturn49search8turn49search0  

这些要素解释了为什么“把 RNA template 简化成蛋白那套 pseudo-beta+主链框架”往往不够：你可能能给网络一些几何先验，但会丢掉 RNA 的“符号结构”（配对/堆叠类型）与“化学多样性”（修饰/离子/配体）。

## RNA template tensor 设计方案

下面给出至少 3 种可行的 RNA template 张量设计，并显式对齐“可适配多种输入 shape”的要求：每种方案都以“pairwise 2D maps + per-residue 1D features”为核心，但在表达能力、与 Protenix 现模块对接难度、计算成本上不同。

### 方案一：Protenix TemplateEmbedder 兼容的 RNA 模板最小接口（推荐优先落地）

**张量维度与含义（canonical 版）**

设 RNA token 数为 N，模板数为 T（Protenix 默认 max_templates=4），则输出：

- `template_aatype`: `(T, N)` int32，取值空间为 `len(STD_RESIDUES_WITH_GAP)=32`。在 TemplateEmbedder 中会 one-hot 再扩展为 i/j 两份 32 维通道。citeturn47view0turn28view0  
- `template_distogram`: `(T, N, N, 39)` float32，base-anchor 距离的 39 桶 one-hot。citeturn47view0turn25view2  
- `template_pseudo_beta_mask`: `(T, N, N)` float32/bool，表示该对 (i,j) 在该模板下是否有有效“代表点距离”。TemplateEmbedder 会把它作为 1 通道特征使用。citeturn47view0turn25view2  
- `template_unit_vector`: `(T, N, N, 3)` float32，表示从 i 到 j 的相对位置向量在 i 的局部框架下的单位向量（或等价的归一化方向编码）。citeturn47view0turn32view1  
- `template_backbone_frame_mask`: `(T, N, N)` float32/bool，表示 i 的局部框架是否可用、以及该对是否应提供方向信息（通常会与 pair_mask、同链掩码相乘）。citeturn47view0turn32view1  

这五个键名与通道数完全对齐 Protenix 的 TemplateEmbedder：其 `single_template_forward()` 会拼接 `39 + 1 + 32 + 32 + 3 + 1 = 108` 维特征并嵌入。citeturn47view0

**数据来源与计算方法**

- 坐标主来源：entity["organization","wwPDB","worldwide protein data bank"] 的 PDBx/mmCIF（其作为结构数据归档与交换的主格式）。citeturn50search0turn50search3turn50search4  
- 结构注释与核苷酸几何：优先用 entity["organization","DSSR","nucleic acid structure annotator"]/entity["organization","3DNA","nucleic acid analysis suite"] 从模板结构中提取碱基参考系、配对与堆叠信息，并为“代表点/局部框架”提供稳健的原子集合与缺失处理策略。citeturn49search8turn49search24turn49search12  
- 同源/模板搜寻与家族比对（可选但强建议）：使用 entity["organization","Rfam","rna families database"] 的家族 MSA 与 covariance model 来获得更可靠的 RNA 比对（尤其当序列弱保守时）。citeturn49search7turn49search11  

**与 Protenix/AF3 现有模块对接方式**

- 最直接的对接是**不改模型结构**：只要你能在进入模型前把上面 5 个键填入 `input_feature_dict`（或等价输入结构），TemplateEmbedder 就会在前向中使用它们；如果缺失 `template_aatype`（或 n_blocks<1），它会直接返回 0，保持兼容。citeturn47view0  
- 现实挑战是 Protenix 自带模板检索/组装目前只对蛋白链启用：模板特征生成逻辑里存在 `chain_type != PROTEIN_CHAIN` 就跳过模板的判断。要让“标准 pipeline”支持 RNA template，你需要扩展这一判断与下游特征构建。citeturn30view1turn25view3  

**优缺点与计算成本**

优点是：接口已被 Protenix 的 TemplateEmbedder 固化，落地最快；pairwise 张量是标准 `(T,N,N,C)`，也最容易适配到其它 AF3-like 架构（无论其叫 template stack、template attention 还是 constraint head）。citeturn47view0  
缺点是：RNA 特有符号信息（配对类型、堆叠类型、糖-磷骨架构象、修饰/离子）没有显式通道，更多依赖几何“隐式编码”，在非规范配对与复杂三级折叠上可能不足。citeturn49search8turn49search9  
计算成本主要是 `(T*N^2)` 的 distogram 与 unit-vector 生成，T=4 时通常可控；但 N 很大（核糖体片段等）会使 `N^2` 成为主瓶颈。

### 方案二：显式编码碱基配对/堆叠/修饰的 RNA 图模板（需要并行或改动 TemplateEmbedder）

**张量维度与含义**

在方案一的基础上，增加一个“RNA-specific pair feature tensor”：

- `rna_template_pair`: `(T, N, N, C_rna)`  
  其中可包含：  
  - `bp_type_onehot`（canonical + 非规范类别；或更细粒度的 Leontis–Westhof 风格类别），  
  - `stacking_type_onehot`（堆叠/不堆叠/方向类），  
  - `is_pseudoknot_edge`（可选），  
  - `is_modified_i/j`、`mod_class_i/j`（可选）等。  

同时增加 per-residue 的 1D：

- `rna_template_single`: `(T, N, C_single)`，如 sugar pucker 类别、关键扭转角（sin/cos）、“是否缺失关键原子”等。

**数据来源与计算方法**

- 碱基配对、非规范配对、修饰核苷酸与堆叠：DSSR 对“分析与注释 RNA 三级结构”是核心目标，并被描述为可自动化处理经典结构中被忽略的特征。citeturn49search8turn49search12  
- 堆叠网络：MC-Annotate 被用于识别并计算 base-stacking network；你可以与 DSSR 输出做一致性检查，构造“置信度/一致性掩码”。citeturn49search13turn49search9  

**对接策略**

- 并行策略：保留 Protenix 原 TemplateEmbedder（吃方案一 5 键），另起一个 `RNATemplateEmbedder`（吃 `rna_template_pair/single`），两者输出到同一个 pair embedding `z` 上做加和或门控融合。  
- 替换/扩展策略：把 TemplateEmbedder 的输入通道从 108 扩展到 `108 + C_rna + ...`，需要重新训练/微调以适配新输入分布。citeturn47view0  

**优缺点与计算成本**

优点是能显式表达 RNA 的“离散拓扑”（配对类型、堆叠网络、修饰），更接近 RNA 结构决定因素；缺点是工程复杂度显著提高，且几乎必然要求训练或至少微调，否则新通道难以被有效利用。citeturn49search8turn49search9

### 方案三：原子级（dense atoms）RNA 模板 + 派生多视角几何图（适配更广但最重）

**张量维度与含义**

该方案把 RNA template 表达为更通用的“稠密原子模板”，再派生多种 pairwise map：

- `template_atom_positions`: `(T, N, A_dense, 3)`  
- `template_atom_mask`: `(T, N, A_dense)`  

其中 `A_dense` 可对齐 Protenix 当前 dense=24 的约定（其空模板默认 `num_dense=24`），对 RNA 选定 24 个代表性原子（骨架 + 糖环 + 若干碱基原子），缺失则 mask=0。citeturn32view2turn26view1  

再从这些原子派生多视角 pair map，如：

- base-center distogram（用于配对/堆叠），  
- phosphate–phosphate distogram（用于骨架拓扑），  
- base-frame orientation features（用于局部相对姿态），  
并最终至少生成方案一所需的 5 键以兼容 TemplateEmbedder。

**数据来源与计算方法**

- PDBx/mmCIF 作为结构主格式，提供足够完整的原子与化学组分信息，并由 wwPDB 的 mmCIF 资源体系维护其字典与类别。citeturn50search0turn50search3turn50search10  
- 对修饰核苷酸：可在 mmCIF atom_site 中读到实际 residue/atom 名称，再按“映射到 UNK_NUCLEIC 或近邻类别”的策略压缩到 `template_aatype`；同时保留 `is_modified` 通道供上层利用（若你实施方案二/三的扩展）。citeturn49search8turn28view0  

**对接策略**

- 若你的系统不是 Protenix 而是自研 AF3-like，可以直接把 `template_atom_positions/mask` 喂入一个 atom-level template encoder（类似“结构先验编码器”），与扩散/结构模块更强耦合。  
- 若你仍想复用 Protenix 现成模块，则把该方案视为“更通用的数据层”，最终仍要落到 Protenix 固定的 5 键上。citeturn47view0turn32view2  

**优缺点与计算成本**

优点是表达最丰富，适配最广（尤其对修饰/配体/离子附近的几何）；缺点是数据清洗与缺失原子处理最复杂，`T*N*A_dense` 的内存与 IO 压力也最大，推理时可能显著增大预处理开销。citeturn50search0turn49search8

## 最终推荐方案与实现细节

综合“可落地性、对 Protenix 的兼容性、以及对 RNA 特异结构信号的保留”，本报告推荐的最终方案是：

- 主路径采用**方案一（TemplateEmbedder 兼容的最小接口）**，确保你能在不改模型的情况下先验证“RNA template 是否带来增益”；  
- 预留扩展接口：在数据预处理阶段同时抽取 DSSR/MC-Annotate 的配对/堆叠信息，先以“诊断/评估信号”形式落盘（不进入模型），待主路径增益明确后，再升级到方案二做并行融合。citeturn47view0turn49search8turn49search9  

### 数据预处理流程

下面流程用“模板结构来自 PDBx/mmCIF；RNA 家族/同源来自 Rfam/RNAcentral；结构注释来自 DSSR/3DNA”的组合实现。

```mermaid
flowchart TD
  A[Query RNA sequence (+ chain info)] --> B[Template search / selection]
  B -->|Rfam CM / msa| C[Alignment: query<->template positions]
  B -->|PDBx/mmCIF| D[Parse template 3D coordinates]
  D --> E[DSSR/3DNA: base frames, pairing, stacking, modifications]
  C --> F[Map template residues to query indices]
  E --> G[Define RNA anchor & local frames]
  F --> H[Compute template_distogram / masks]
  G --> I[Compute template_unit_vector & backbone_frame_mask]
  H --> J[Assemble tensors: template_aatype + 4 pair maps]
  I --> J
  J --> K[Inject into Protenix input_feature_dict]
  K --> L[TemplateEmbedder -> pair embedding z update]
  L --> M[Pairformer + Diffusion -> 3D structure prediction]
```

### 必要工具/库建议

- PDBx/mmCIF 解析：使用任意可靠的 mmCIF parser（Biopython/biotite/自研均可），以 wwPDB 的 PDBx/mmCIF 用户指南与字典资源作为字段语义依据。citeturn50search0turn50search3turn50search7  
- RNA 结构注释：优先 DSSR/3DNA；其论文与站点材料都强调其对 RNA 三级结构要素的自动化注释能力。citeturn49search8turn49search12turn49search24  
- 家族/同源资源：Rfam（家族 MSA 与 covariance model）与 RNAcentral（大规模 ncRNA 序列枢纽）。citeturn49search7turn49search10turn49search2  

### 关键张量 shapes 示例与字段对齐

假设 `N=200`（200 nt RNA），`T=4`（最多 4 个模板，符合 Protenix 默认 max_templates=4 的常见设定），你最终交给模型的模板相关字段应至少包含：

| key | shape | dtype | 说明 |
|---|---|---|---|
| `template_aatype` | (4, 200) | int32 | 32 类 residue id；RNA 的 A/G/C/U 应映射到 Protenix 的 RNA ID 区间（其常量映射已给出）。citeturn28view0 |
| `template_distogram` | (4, 200, 200, 39) | float32 | 与 TemplateEmbedder 固定 39 通道一致。citeturn47view0turn25view2 |
| `template_pseudo_beta_mask` | (4, 200, 200) | float32 | 作为 1 通道特征使用，且会与同链/对偶 mask 相乘。citeturn47view0 |
| `template_unit_vector` | (4, 200, 200, 3) | float32 | 与 TemplateEmbedder 的 3 通道一致。citeturn47view0turn32view1 |
| `template_backbone_frame_mask` | (4, 200, 200) | float32 | 作为 1 通道特征使用。citeturn47view0 |

其中 Protenix 的 TemplateEmbedder 会在每个模板上构造拼接张量 `at` 并计算 `v = linear(z) + linear(at)`，再过一个 PairformerStack，最后对模板求平均并回投影到 `c_z` 维。citeturn47view0  

### RNA 代表点与局部框架的推荐定义

为了最大程度贴合 RNA 化学直觉且又保持工程稳定性，建议如下分层策略（避免某些模板缺失特定原子时直接崩）：

- **代表点（anchor，替代 pseudo-beta）**：  
  1）首选：base center（用 DSSR/3DNA 给出的碱基参考系/关键碱基原子集合计算几何中心）；  
  2）退化：C1' 或 C4'（糖环原子通常更稳定可得）；  
  3）再退化：若该残基缺失上述原子，则该位点 mask=0。citeturn49search8turn49search24  

- **局部框架（替代蛋白 N-CA-C 框架）**：  
  以 C4' 为原点，定义  
  - `v1 = P − C4'`（近似沿骨架方向），  
  - `v2 = C1' − C4'`（指向糖-碱基连接），  
  再用 Gram–Schmidt 得到 `e1,e2,e3`。若 P 或 C1'/C4' 缺失则框架 mask=0。该思路与 Protenix 蛋白框架的做法在数学形式上同构（都是三点定局部正交基），只是在原子选择上不同。citeturn32view1turn47view0  

### 归一化、置信度表示与缺失数据处理

- **距离归一化**：沿用 Protenix 模板 distogram 的 39 桶（min_bin=3.25, max_bin=50.75, num_bins=39）以确保与 TemplateEmbedder 输入通道严格对齐。citeturn25view2turn47view0  
- **方向归一化**：`template_unit_vector[i,j]` 始终做单位化；若距离过小或框架不可用，则将 mask=0 并置零向量。该策略与 Protenix `compute_template_unit_vector()` 里对单位向量与 mask 的处理精神一致（先检查关键原子 mask，再归一化）。citeturn32view1  
- **模板置信度（推荐做法）**：TemplateEmbedder 本身没有显式 `template_confidence` 通道；你可以用“缩放模板特征幅值”的方式注入权重（例如对某个模板 t 乘以 `w_t`）：  
  - `template_distogram[t] *= w_t`  
  - `template_pseudo_beta_mask[t] *= w_t`  
  - `template_unit_vector[t] *= w_t`  
  - `template_backbone_frame_mask[t] *= w_t`  
  这样会线性影响 `linear(at)` 的幅度，从而影响该模板对最终 `u` 的贡献（TemplateEmbedder 对模板做简单平均）。citeturn47view0  

- **缺失/插入**：当 query–template 比对导致某些 query 位点在模板中无对应残基时，把该位点的 anchor/frame mask 置 0；pairwise mask 由外积得到，使 distogram 与 unit-vector 在这些位置自动为 0。Protenix 空模板特征本就用全零 positions 与全零 mask 表示，你的实现应保持同样的“mask 优先”语义。citeturn32view2turn47view0  

### 伪代码示例

下面给出一个“只做方案一”的最小伪代码骨架（强调 shape 与 mask 逻辑；具体 mmCIF/DSSR 接口按你选用的库实现）。

```python
import numpy as np

def one_hot_bins(dist, bin_edges):
    # dist: (...,)
    # return: (..., n_bins) one-hot
    dist2 = dist * dist
    lower = bin_edges[:-1]**2
    upper = bin_edges[1:]**2
    # last bin as inf
    upper = np.concatenate([upper, np.array([1e8])])
    lower = np.concatenate([lower, np.array([lower[-1]])])
    # build
    return ((dist2[..., None] > lower) & (dist2[..., None] <= upper)).astype(np.float32)

def build_rna_frame(P, C4p, C1p, eps=1e-6):
    # returns (e1,e2,e3, valid_mask)
    v1 = P - C4p
    v2 = C1p - C4p
    n1 = np.linalg.norm(v1) + eps
    e1 = v1 / n1
    v2_ortho = v2 - np.dot(v2, e1) * e1
    n2 = np.linalg.norm(v2_ortho) + eps
    e2 = v2_ortho / n2
    e3 = np.cross(e1, e2)
    return e1, e2, e3, 1.0  # set 0.0 if atoms missing

def compute_template_tensors(query_len, templates, max_T=4):
    # templates: list of dict with mapped coords per query index
    # Each template provides:
    #  - aatype[N] int32
    #  - anchor_pos[N,3], anchor_mask[N]
    #  - frame_origin[N,3], frame_axes[N,3,3], frame_mask[N]
    T = min(len(templates), max_T)
    N = query_len

    template_aatype = np.zeros((max_T, N), dtype=np.int32)
    template_distogram = np.zeros((max_T, N, N, 39), dtype=np.float32)
    template_pb_mask = np.zeros((max_T, N, N), dtype=np.float32)
    template_uv = np.zeros((max_T, N, N, 3), dtype=np.float32)
    template_bb_mask = np.zeros((max_T, N, N), dtype=np.float32)

    bin_edges = np.linspace(3.25, 50.75, 39)  # follow Protenix bins

    for t in range(T):
        aatype = templates[t]["aatype"]          # (N,)
        anchor = templates[t]["anchor_pos"]      # (N,3)
        amask  = templates[t]["anchor_mask"]     # (N,)
        origin = templates[t]["frame_origin"]    # (N,3)
        axes   = templates[t]["frame_axes"]      # (N,3,3) columns=[e1,e2,e3]
        fmask  = templates[t]["frame_mask"]      # (N,)

        template_aatype[t] = aatype

        # pairwise masks
        pb2d = amask[:, None] * amask[None, :]
        bb2d = fmask[:, None] * amask[None, :]   # require i-frame & j-anchor (customize)

        # distances
        diff = anchor[:, None, :] - anchor[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dgram = one_hot_bins(dist, bin_edges)    # (N,N,39)

        # unit vectors: express r_ij in i frame
        r = (anchor[None, :, :] - origin[:, None, :])  # (N,N,3)
        # project: axes[i] @ r_ij
        proj = np.einsum("iab,ijb->ija", axes, r)       # (N,N,3)
        proj_norm = np.linalg.norm(proj, axis=-1, keepdims=True) + 1e-6
        uv = proj / proj_norm

        template_distogram[t] = dgram * pb2d[..., None]
        template_pb_mask[t] = pb2d
        template_uv[t] = uv * bb2d[..., None]
        template_bb_mask[t] = bb2d

    return {
        "template_aatype": template_aatype,
        "template_distogram": template_distogram,
        "template_pseudo_beta_mask": template_pb_mask,
        "template_unit_vector": template_uv,
        "template_backbone_frame_mask": template_bb_mask,
    }
```

### 与 Protenix 数据/模型接口的落点

从 Protenix 代码可以精确定位你需要对齐的“接口落点”：

- CLI/推理文档明确：启用模板需 `--use_template` 且 input JSON 里需要 `templatesPath`。citeturn39search0turn19view0  
- 但模板数据管线同时明确：`chain_type != PROTEIN_CHAIN` 时模板直接置空，且模板组装注释写明“Templates are currently only supported for protein chains”。citeturn30view1turn25view3  
- 真正进入模型的模板张量键名与通道数由 TemplateEmbedder 固化，如前所述。citeturn47view0  

因此你有两条实现路径：

1）**绕过 Protenix 的模板检索**：你自己完成 RNA 模板检索/对齐/张量化，然后在进入模型前把上述键注入 `input_feature_dict`（或你控制的等价输入结构）。这对“未指定输入 API/shape”的假设最友好：只要你的系统最后能构造出与 TemplateEmbedder 相同键名/shape 的张量，就可对接。citeturn47view0  

2）**扩展 Protenix 的模板检索与 featurizer**：将 `chain_type != PROTEIN_CHAIN` 的逻辑扩展为对 RNA_CHAIN 也支持模板，进一步实现 RNA 的原子映射、anchor 与 frame 计算。Protenix 常量已包含 RNA_CHAIN 类型及 RNA/DNA 的 residue-id 映射，说明扩展方向是顺着现有设计的。citeturn28view0turn30view1turn25view3  

## 实验验证计划

验证必须覆盖三类问题：模板注入是否提升 3D 准确度、提升来自哪些通道、以及在真实 RNA 困难场景（非规范配对、修饰、低分辨率/缺失原子、多构象）下是否稳健。

### 数据集建议

- **PDB RNA 结构集（PDBx/mmCIF）**：以 wwPDB 的 PDBx/mmCIF 作为原始结构数据来源，便于稳定解析核酸链类型、原子坐标与化学组分。citeturn50search0turn50search3turn50search10  
- **Rfam 家族级划分**：用于严格做同源拆分/训练测试隔离，降低 template 泄漏与家族记忆导致的高估。Rfam 的家族 MSA 与 covariance model 正是为“以结构-序列协变建模家族”服务。citeturn49search7turn49search11  
- **RNAcentral 序列扩展集（可选）**：用于构造更大的同源序列背景（尤其当你希望 template search 走“先家族定位再到结构库取模板”的路线）。citeturn49search10turn49search2  

（你提到的 RNA-Puzzles 等社区基准非常适合外部验证，但本报告在“必须引用来源”的约束下不对其细节作事实陈述；建议你将其作为独立测试集引入。）

### 训练/微调策略

- **零改模型的推理验证（首要）**：在不改 Protenix 权重的前提下，仅注入方案一模板特征，比较 `use_template=False` vs `use_template=True(注入RNA模板)` 的差异。Protenix 文档与代码均表明模板是可开关的特征通道。citeturn39search0turn47view0turn25view3  
- **小规模微调（可选）**：若零改验证显示有信号但不稳定，再考虑只微调 TemplateEmbedder 与其相邻层，使网络适应“RNA 的 distogram/unit vector 分布”，同时冻结其余主干以降低灾难性遗忘风险。citeturn47view0  
- **扩展通道（方案二）需要配套训练**：一旦你把 bp/stacking/modification 显式通道加入网络，必须配套训练或微调，否则额外输入通道大概率被随机初始化的线性层“当噪声”。citeturn49search8turn49search9  

### 消融实验设计

建议把消融围绕 TemplateEmbedder 实际使用的通道做，因为它们是确定的输入集合：citeturn47view0  

- 仅 `template_aatype`（其在 embedder 中被 one-hot 两次扩展到 i/j）  
- `template_distogram` only  
- `template_unit_vector` only  
- 去掉 `template_pseudo_beta_mask` 或 `template_backbone_frame_mask`（验证掩码是否仅起屏蔽作用还是提供显式信号）  
- 不同 anchor 定义（base center vs C4' vs C1'）  
- 不同 frame 定义（骨架 frame vs base frame）  

### 评估指标

- **全局几何误差**：RMSD（对齐后整体误差）、以及长度归一化的相似度（“TM-score-like”）来减少长度效应（可用任意结构比对工具实现；这里作为建议指标）。  
- **碱基配对质量**：从预测结构中用 DSSR（或与之等价的注释工具）提取碱基对集合，与真值比较计算 base-pair precision/recall/F1。DSSR 被明确用于 RNA 三级结构的分析与注释，因此适合做“结构→配对集合”的评估抽取器。citeturn49search8turn49search17  
- **接触图质量**：把 distogram/contact（例如以阈值转成 contacts）评估 precision/recall，与 template 注入前后对比。trRosettaRNA 一类方法本身就把距离/接触图作为核心输出与约束对象，说明这一视角对 RNA 建模有意义。citeturn0search3  
- **局部构象指标（可选）**：糖-磷骨架关键扭转角误差、堆叠/配对网络一致性（可依赖 DSSR/MC-Annotate 的网络抽取）。citeturn49search8turn49search13turn49search9  

## 风险与未决问题清单

第一，RNA 的多体堆叠与三级相互作用高度复杂，且不同注释工具在“是否算堆叠/是否算配对”的判据上可能不一致；这会让你在构造“bp/stacking 模板特征”时产生系统性噪声。缓解策略是：用 DSSR 与 MC-Annotate 的输出做交集/一致性评分，把“一致性”编码为置信度（先用于评估与过滤，再考虑进模型）。citeturn49search8turn49search9turn49search13

第二，RNA 存在显著的动力学异构体与构象集合；单一模板可能只覆盖其中一种状态，注入过强可能把模型拉向错误构象。缓解策略包括：多模板集成（T 个模板平均本就是 TemplateEmbedder 的默认聚合方式）、以及对模板特征施加权重衰减（用 `w_t` 缩放模板输入幅值），降低错误模板的影响。citeturn47view0

第三，修饰碱基与离子/配体位点在真实 RNA 中普遍存在，而你的最小接口（方案一）只能用“UNK_NUCLEIC”之类的类型压缩表达，细粒度化学差异会丢失。缓解策略是走分阶段路线：先用方案一验证“几何模板先验”收益；若收益明确，再用方案三的原子级模板或方案二的显式修饰通道增强表达。DSSR 相关资料明确提到其可识别 modified nucleotides 等核酸结构特征，说明数据侧是可抽取的。citeturn49search8turn49search0turn28view0

第四，低分辨率结构与缺失原子会导致 frame/anchor 不可用，从而在 pairwise 特征中形成大片 mask，严重时模板几乎“失声”。缓解策略是：多级退化的 anchor/frame 定义（base center→C1'/C4'→缺失），并在模板筛选阶段过滤过度缺失的模板；同时记录缺失比例作为模板置信度的一部分。Protenix 的模板空特征与 mask 设计为“mask 优先”，说明其下游 embedder 能自然处理大量缺失，只要你严格遵循 mask 语义。citeturn32view2turn47view0

第五（开放问题），RNA 模板检索本身比蛋白更难：序列保守性弱、插入缺失多、二级结构约束强。Rfam 以“家族 MSA + covariance model”形式编码序列与结构协变，为解决这一点提供了事实上的标准化途径；但把它与“结构库模板”高质量地关联、并在工程上高效实现，仍是你需要根据具体业务场景取舍的系统设计问题。citeturn49search7turn49search11turn49search3