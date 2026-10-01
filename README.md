# EmbeddingFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L1 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | EmbeddingFactory |
| 类型 | Factory |
| 真正有意思的核心 | Embedding → Index → Benchmark |

---

## 二、核心命题

这里研究的对象从 Token 变成：

```
x → f_θ(x) → z
```

即：**把「表示」与「检索」两个空间解耦**，而不是把它们混在一个 pipeline 里当成黑盒。

## 三、整体架构

```
                Dataset
                   │
                   ▼
             Embedding Model
                   │
                   ▼
                 Vector
                   │
                   ▼
             Index Builder
        ┌──────────┼───────────┐
        ▼          ▼           ▼
      FAISS       HNSW       Flat
        │          │           │
        └──────────┼───────────┘
                   ▼
               Retrieval
                   │
                   ▼
               Benchmark
          ┌────────┼────────┐
          ▼        ▼        ▼
        Recall    MRR     Latency
          │       nDCG      RAM
          └────────┼────────┘
                   ▼
                Compare
```

## 四、主流水线

```
Documents
 ↓
Chunk
 ↓
Embed
 ↓
Normalize
 ↓
Index
 ↓
Query
 ↓
Retrieve
 ↓
Evaluate
 ↓
Compare
```

## 五、关键：把两个变量分开

```
Representation Space
Embedding Model
      ×
Retrieval Space
Index + Distance + Search Params
```

于是：

```
Performance = f(Embedding, Chunking, Index, Distance, Search)
```

**这才是 Factory。** 因为这五个因素构成一个可搜索空间，而不是一串写死的调用。任意一环的参数变化都应能被独立消融（ablation），并归因到指标变化上。

## 六、评测维度

| 空间 | 变量 | 指标 |
| --- | --- | --- |
| Representation | Embedding Model | Recall@k、nDCG、MRR |
| Representation | Chunking 策略 | 召回完整度、边界切碎率 |
| Retrieval | Index 类型（FAISS / HNSW / Flat） | 构建时间、内存占用 |
| Retrieval | Distance（cos / L2 / IP） | 排序质量 |
| Retrieval | Search Params（efSearch / nprobe） | Latency vs Recall 曲线 |

## 七、阶段性落地计划

- **M0｜检索骨架**：定义 Chunk / Vector / Index Spec 的 schema，跑通 Chunk → Embed → Index → Query 一次（非自动）。
- **M1｜双空间变量化**：实现 Embedding / Chunking / Index / Distance / Search 五维可插拔，接入 Recall / nDCG / MRR / Latency / RAM 度量。
- **M2｜消融与搜索**：做单变量消融矩阵，接评分器做参数搜索，输出 Latency–Recall 前沿。
- **M3｜评测与发布**：可视化对比面板、示例与文档，沉淀为可复用检索组件。

## 八、验收标准

能端到端完成「EmbeddingFactory」，并在其「核心」（Embedding → Index → Benchmark）上给出可复现的量化结果：

- 给出 Embedding × Index × Distance × Search 的消融矩阵
- 输出 Latency–Recall Pareto 前沿，且每个结论可归因到具体变量

---

## 附录 A、工程骨架

```
L1-embeddingfactory/
├── README.md
├── main.py                    零安装入口：把 src/ 塞进 sys.path 后转交 cli:main
├── pyproject.toml             打包 + pytest 配置；dependencies = []，faiss extra 是「接工业 ANN 库」的扩展点
├── requirements.txt           注释说明「核心链路零依赖」
├── .gitignore
├── configs/
│   └── default.json           语料路径 / 表示空间（分块 × 嵌入）/ 检索空间（索引 × 距离 × 搜索档位）
│                              / 消融基准 / Pareto 目标对 / 选型预算
├── data/corpus/
│   ├── documents.jsonl        36 篇文档，12 个主题各 3 篇
│   └── queries.jsonl          24 条查询，每条 1~3 个文档级 ground truth
├── src/embeddingfactory/
│   ├── __init__.py            导出全部核心对象
│   ├── schema.py              Document / DocChunk / Query / CorpusStats / RetrievalSpec（**字段按两个空间分组**）
│   │                          / RetrievalMetrics / RetrievalRun / Version / RetrievalResult
│   ├── rng.py                 确定性随机：derive_seed / make_rng / **stable_hash64**（特征哈希用）
│   ├── lineage.py             LineageNode + LineageDAG（from_dict 重建 + 确定性导出 + merge_points / fan_out_ratio）
│   ├── config.py              配置加载 + **加载期**校验（含「三个 embedding 的 dim 必须一致」「参考配置必须在设计空间里」）
│   ├── corpus.py              CJK **bigram** 分词 + sentence/fixed 两种分块 + cut 标记（边界切碎率）
│   ├── embed.py               三个**真的拟合**的嵌入模型：tfidf / bm25 / 字符 n-gram 特征哈希（带 IDF）
│   ├── index.py               三种距离（cos / l2 / ip）+ 三种索引（flat / ivf / 单层 NSW）
│   │                          + **「距离计算次数」成本模型**
│   ├── retrieve.py            真的检索：嵌入 → 搜索 → chunk 归并回 doc → 排序 → 与 ground truth 比对
│   ├── metrics.py             五维度量 + **TIMING_METRICS**（哪些指标不可逐位复现）
│   ├── compare.py             支配 / 非支配 / 超体积（min 形式一套坐标）/ 膝点 / 选型 /
│   │                          消融组（**index 与 search 两轴特判**）/ 空间归因 / 达额成本 / 预算敏感性
│   ├── pipeline.py            扇形展开再收拢的编排 + 落盘（含 artifact 的**全精度**导出）
│   ├── report.py              报告表格行 + 十章 Markdown 报告 + explain
│   └── cli.py                 run / compare / frontier / ablation / spaces / explain / list / dag / query
├── tests/                     pytest（5 个文件 + conftest，211 个用例）
└── out/                       运行产物（每次 run 重建）
```

关于「三种索引」的如实说明：README 原稿写的是 `FAISS / HNSW / Flat`。本项目落地成
`flat / ivf / hnsw`，其中

- `flat` 是**全量暴力扫描**，也是本项目的**精确 oracle**（它的召回率就是该表示空间的上限）；
- `ivf` 是**倒排 + 确定性 k-means 分簇 + nprobe**，扮演原稿里 FAISS 的角色；
- `hnsw` 是**简化版单层 NSW**（近邻图 + `ef_search` 束搜索），**刻意不叫完整 HNSW** ——
  完整 HNSW 是多层跳表 + 启发式邻居选择，本语料只有一百多个向量，分层收益体现不出来。

手写而不是接现成库，理由有两条，都不是「为了炫技」：一是本项目要回答的是
「同一批向量下，索引 / 距离 / 搜索参数各自贡献了多少」，黑盒库只给一个 latency 数字，
给不出「为什么」；二是**手写才能精确统计「距离计算次数」**，而那正是本项目用来替代
不可复现的实测延迟的成本模型。生产环境当然该用 FAISS / hnswlib，所以留了 `faiss` extra。

## 附录 B、快速开始

```bash
# 核心链路零第三方依赖，直接用标准库跑
python main.py run --config configs/default.json --out out

# 消融矩阵（与报告第三节同一份行构造函数，不会两处分家）
python main.py compare --out out

# Pareto 前沿（--pair 可只看某一条目标对）
python main.py frontier --out out
python main.py frontier --out out --pair query_latency_ms+recall_at_k

# 单变量消融 + 空间归因 + 达额成本
python main.py ablation --out out

# 两个空间并排看（表示空间给上限，检索空间给成本）
python main.py spaces --out out

# 某个 spec 的完整解释（为什么它在/不在前沿上 + 逐查询样本）
python main.py explain tfidf-sentence-ivf-cos-np1 --out out

# 用**导出的 artifact** 真的跑一次查询 —— 「产物真的能用」的可执行检查
python main.py query tfidf-sentence-ivf-cos-np1 向量 检索 表示 --out out

# 列出全部 60 个配置（★ 入选 / ▲ 在主前沿上）
python main.py list --out out

# 血缘图（ASCII / Graphviz DOT；--selected-only 只画入选祖先链）
python main.py dag --out out
python main.py dag --out out --dot --selected-only > lineage_selected.dot

# 测试
pytest
```

设计约定（与同工作区其它项目共用一套）：

- **`--out` 只属于 `run`**。其余子命令是**只读观察者**，跑完产物目录一个字节都不变
  （实测：跑完全部 11 个只读命令后，`out/` 里每个文件的 md5 都没变；有测试守着）。
- **只读子命令不做检索**。两处例外都是*载入*而非重算：`query` 载入 artifact 并真的跑一次查询，
  `dag` 从 `lineage.json` 重建血缘图（否则 `--dot` 只能打印一句提示）。
- **未知 `spec_id` 返回 1 并给相近建议**，不抛栈。
- 刻意**没有** `rounds` 子命令（本项目不是一轮一轮迭代的），也**没有** `searchers`
  （配置里的交叉积就是全部候选）。

产物清单：

| 文件 | 内容 |
| --- | --- |
| `matrix.csv` | 消融矩阵（60 行 × 15 列；列序 = 五个设计轴 → 质量 → 成本 → 前沿） |
| `frontier.csv` | 全部目标对的前沿（`objective_pair` 列区分，一个点一行） |
| `runs.jsonl` | 每个 spec 一行，含**逐查询样本**（explain / query 的数据来源） |
| `result.json` | 矩阵（**超集**，含 `meta_*` 诊断字段）/ 前沿 / 消融 / counters / 版本 / 配置 |
| `ablations.json` | 单变量消融 / 空间归因 / 表示空间上限 / 达额成本 / 预算敏感性 |
| `artifacts/<spec>.json` | 入选配置的**可载入**检索组件（词表 / idf / 全精度向量 / chunk→doc 映射 / **种子**） |
| `lineage.json` / `lineage.dot` / `lineage_selected.dot` | 血缘图（全部 / 全部 DOT / 只画入选祖先链） |
| `version.json` | 内容寻址版本 |
| `report.md` | 十章报告（结构与两条验收标准对齐） |

## 附录 C、已验证状态

本骨架已实际跑通，不是纸面设计：

- `pytest` → **211 passed**（`test_primitives` 36 / `test_embed` 27 / `test_index` 40 /
  `test_metrics_compare` 49 / `test_end_to_end` 59），覆盖 schema / rng（含**跨进程
  `PYTHONHASHSEED` 反证**）/ 分词 / 分块 / 三个嵌入模型 / 三种距离与索引 / 度量口径 /
  支配与超体积 / 消融与归因 / 编排 / 落盘 / 报告 / CLI 契约与只读性 /
  **artifact 载入后行为一致**
- `python main.py run` → **60 个 spec**（3 embedding × 2 chunking × 3 index × 2 distance ×
  各索引的搜索档位），整条流水线 **4.2s**；DAG **131 节点 / 134 边 / 1 个收拢点**；
  版本 `v1-a8ebf7b8`

### 语料画像（所有结论的前提）

`documents.jsonl`：**36 篇文档 / 12 个主题各 3 篇**，总字符 3254，平均块长 35.4 字符。
`queries.jsonl`：**24 条查询**，平均 1.58 个相关文档。

分块画像（**边界切碎率**是分块策略的直接质量信号）：

| 分块策略 | 块数 | 被硬切开的块 | 切碎率 |
| --- | --- | --- | --- |
| `sentence`（`max_chars=50`） | 92 | 5 | 5.4% |
| `fixed`（`size=40, overlap=0`） | 108 | 70 | **64.8%** |

两者相差 12 倍 —— 这就是下面「`fixed` 把召回率上限砍掉 57 个百分点」的直接原因。

### 验收标准一 ✅：Embedding × Index × Distance × Search 的消融矩阵

60 行 × 15 列（`matrix.csv` 与报告第三节同一份数据），五个设计轴全在。
报告第四节另给**单变量消融**（7 行，每个轴只动一个变量），Δ 相对参考配置
`tfidf-sentence-flat-cos`：

| 轴 | 从 → 改为 | ΔRecall | Δ距离计算次数 |
| --- | --- | --- | --- |
| embedding | tfidf → bm25 | −0.0208 | +0.0 |
| embedding | tfidf → hashing | −0.2639 | +0.0 |
| chunking | sentence → fixed | **−0.5694** | +16.0 |
| index | flat → ivf（np1） | −0.1528 | **−66.8** |
| index | flat → hnsw（ef16） | −0.0625 | −33.1 |
| distance | cos → l2 | **−0.5764** | +0.0 |
| search | ef16 → ef64 | **+0.0625** | +32.1 |

三条读法：

1. **参考配置刻意选 `flat` + `cos`**：flat 是精确检索，所以基准行的召回率 0.8333
   就是该表示空间的上限 —— 这张表读起来是「从上限往下掉多少」，而不是「从一个任意起点
   往下掉多少」。
2. **换距离（−0.5764）和换分块（−0.5694）比换 embedding（−0.2639）影响更大**。
   只盯着 embedding 调参是本项目最想纠正的直觉。
3. **只有 `search` 轴的 ΔRecall 是正的**（ef16 → ef64 花 32.1 次额外距离计算换来
   +0.0625），但它的成本随之涨到 91.0 —— 已经接近精确检索的 92.0。所以这个轴不是
   「免费的午餐」，真正的取舍轴是 `index`（省 66.8 次距离，掉 0.1528 召回）。

### 验收标准二 ✅：Latency–Recall Pareto 前沿 + 可归因

主前沿的成本轴是**距离计算次数**（逐位可复现），**5 个点**，超体积 **58.4904**，
参考点 (116.55, −0.153125)：

| 排名 | spec | 表示空间 | 索引 | 距离计算次数 | Recall@5 | 支配数 | 超体积增量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `bm25-sentence-ivf-cos-np1` | bm25/sentence | ivf/cos/np1 | 22.7 | 0.6667 | 48 | 44.3338 |
| 2 | `bm25-sentence-ivf-cos-np3` | bm25/sentence | ivf/cos/np3 | 56.7 | 0.7917 | 33 | 6.6537 |
| 3 | `tfidf-sentence-hnsw-l2-ef16` | tfidf/sentence | hnsw/l2/ef16 | 21.0 | 0.1944 | 13 | 3.9481 |
| 4 | ★ `tfidf-sentence-ivf-cos-np1` | tfidf/sentence | ivf/cos/np1 | 25.2 | 0.6806 | 44 | 1.2681 |
| 5 | `tfidf-sentence-ivf-cos-np3` | tfidf/sentence | ivf/cos/np3 | 61.7 | 0.8333 | 34 | 2.2868 |

**归因**（报告第五节的极差，口径：其余轴固定在参考配置上）：

| 轴 | 极差 | 最优 → 最差 |
| --- | --- | --- |
| distance | **0.5764** | cos → l2 |
| chunking | **0.5694** | sentence → fixed |
| embedding | 0.2639 | tfidf → hashing |
| index | 0.1528 | flat → ivf |
| search | 0.0625 | ef64 → ef16 |

**达额成本**（前沿的可决策版本）：达到上限的 80% 只要精确检索的 **0.25×** 成本；
90% / 95% 要 0.62×；**100% 也只要 0.67×** —— 因为 `tfidf-sentence-ivf-cos-np3`
用 61.7 次距离就拿到了与 `flat` 相同的 0.8333。

**基线对照**：不比较就直接上 flat + cos，成本 92.0 次距离；同一召回率水平下最省的
是 `tfidf-sentence-ivf-cos-np3`，成本 61.7（**0.67×，省 33.0%**）。这就是「把两个
空间分开」这件事的直接收益。

**成本模型验证**（一个能被证伪的断言）：按索引类型看「距离计算次数 ↔ 实测延迟」的
同向率 —— flat 81%（29/36 对）、hnsw 90%（246/274）、ivf 88%（240/272）。同向率很高，
所以用距离计算次数代理延迟站得住。

**为什么主前沿不用实测延迟**：含实测轴的两条副前沿在两次运行之间换了成员 ——
`build_seconds` 前沿从 3 个点变 4 个点，`query_latency_ms` 前沿从 4 个点变 7 个点。
所以它们只作验证，报告里一律标注「⚠️ 含实测轴，不可逐位复现」，`matrix.csv` 的
「前沿」列也**只列可复现的前沿**。

### 表示空间决定上限 —— 而且「哪个 embedding 最好」会随分块翻转

精确检索（flat + cos）下每个表示空间的召回率上限：

| embedding | chunking | Recall@5 | 与上限差距 |
| --- | --- | --- | --- |
| tfidf | sentence | **0.8333** | ★ 上限 |
| bm25 | sentence | 0.8125 | −0.0208 |
| hashing | sentence | 0.5694 | −0.2639 |
| hashing | fixed | 0.5208 | −0.3125 |
| tfidf | fixed | 0.2639 | −0.5694 |
| bm25 | fixed | 0.2222 | −0.6111 |

**在 `sentence` 下 tfidf 最优、hashing 最差；换到 `fixed` 下正好反过来** ——
hashing 0.5208 反而成了最好的，tfidf 掉到 0.2639。原因是字符 n-gram 不依赖词边界，
块被切断时损失小；词袋则整词丢失。

所以报告第五节那条「**极差是局部量**」的警告不是客套话，它在这份语料上就是真的：
只报一个轴的最优值会误导，`embedding` 的结论必须连着 `chunking` 一起说。

### 跑起来才发现的问题

按「会不会让结论错」排序：

1. **cos 的距离符号写反了**（`1 − d²/2` 是相似度，不是距离）。召回率从 0.81 掉到
   0.0833（22 条查询零召回），而且**不报错** —— 只会让「cos 不如 l2」这种假结论出现。
   修好后 `distance` 轴的极差从「没意义」变成 **0.5764（影响最大的轴）**。现在有一条
   测试直接断言「cos 的排序与朴素点积实现逐位一致」。
2. **hashing 嵌入第一版没有 IDF 加权**，在中文查询上全军覆没（高频 n-gram 盖过关键词）。
   更糟的是「换 embedding」这一行里同时混进了「有没有 IDF」这个变量，结论无法归因。
   补上 `bucket_idf`（与词法模型同一套平滑公式）后，三个模型的唯一差别才回到
   **特征空间**本身。
3. **分块粒度太大，检索退化成文档级**：`max_chars=120` 大于多数文档长度 → 只切出
   36 块（= 文档数）。改成 50 / 40 后是 92 / 108 块，「分块策略」这个轴才开始有区分度。
4. **`index` 与 `search` 两个轴在消融表里静默少算**：参考配置是 `flat`，而 flat 没有
   搜索参数 —— 「只把 index 换成 ivf」造出的 spec_id 缺 `nprobe`，**根本不存在**，
   那一行被静默跳过。5 个轴里有 2 个看起来正常、实际什么都没比。修法：`index` 轴给每个
   索引配**它的第一档**参数，`search` 轴挑**档位最多**的索引当实验台，两处都写进 `note`
   并在报告与 CLI 里显示。
5. **artifact 载入后行为变了**：导出时把向量截断到 6 位小数（当时理由是「文件会小很多」），
   而 ivf 的 k-means++ 按 D² 采样选簇心 —— 末位差一点就换一组簇心，实测
   `tfidf-sentence-ivf-cos-np1` 的召回率从 0.6806 变成 0.7708。而且那个理由本身是错的：
   实测全精度只比 6 位小数大 **4%**（0.0 占绝大多数，Python 的 `repr` 本来就短）。
   现在向量与 idf 一律全精度落盘。
6. **artifact 没记录 `seed`**：`query` 子命令重建 ivf 时只能自己猜一个（原来传 0）→
   分簇不同 → 同一个 spec 的召回率变了。种子现在随产物落盘。
7. **HNSW 的 `ef_search` 档位几乎没有区分度**：初版 `[8, 32]` 在 l2 上都只扫 21 个向量、
   指标逐位相同，前沿上出现了两个**目标向量完全一样**的点。改成 `[16, 64]` 后有了区分度，
   并新增 `pareto_representatives` 按目标向量去重 —— 被去重的配置进
   `counters["frontier_duplicates"]`，因为「某个搜索档位完全无效」本身就是一个结论。
8. **前沿表头重复列**：无条件补一列「距离计算次数」，而主目标对的第一项恰好就是它 →
   表头出现两个同名列（TokenizerFactory 踩过同一个坑）。改成「只在标签里没有它时才补」，
   并由「任意两个不同指标组合都不得出现重复列名」的测试守着。
9. **`reach_cost` 的「上限配置」报错了对象**：并列时按 spec_id 取最大，结果挑中了更便宜的
   ivf 配置（0.8333 与 flat 持平），与报告第六节「上限由 flat 取得」自相矛盾。现在并列时
   优先挑 `flat`。
10. **配置里的参考配置写了缩写**（`tfidf-sent-flat-cos`），而 chunking 的全名是 `sentence`。
    validate 报的是「不在设计空间里」，但真正的原因只是少写了三个字母 —— 这类报错信息
    与真实原因不匹配的问题最费时间。现在 `DEFAULTS` 与 `configs/default.json` 都有一致性
    测试守着。
11. **性能**：手写距离循环让全空间扫描要几分钟。实测 `math.dist` 比手写循环快约 **7 倍**、
    `math.sumprod` 比 `sum(map(mul, ...))` 快约 **1.8 倍**。更重要的是把 cosine 也表达成
    一次 `math.dist`（经 `‖a−b‖²/2` 恒等式）—— 这样**所有距离的单次成本相等**，成本轴才
    能在不同距离之间可比。降到 4.2s。

### 可执行的不变量

这些不是文档里的承诺，是测试在守着的：

- **`flat` 是精确 oracle**：结果与暴力排序逐位一致。它一旦不成立，所有「近似索引掉了
  多少召回」的结论全部失去参照。
- **`cos` 的排序与朴素点积实现逐位一致**。
- **所有距离的单次成本相等**：同一批向量、同一个 k，cos / l2 / ip 报出的
  `n_distance_computations` 完全相同 —— 成本轴才在距离之间可比。
- **成本模型逐位可复现**：flat / ivf / hnsw 各建两次、搜两次，
  `n_distance_computations` 与结果 id 完全相同。
- **`ivf` 的 `nprobe = nlist` 与 `flat` 逐位一致**（扫遍所有簇 = 精确检索）。
- **`stable_hash64` 跨进程稳定**：换 `PYTHONHASHSEED` 重开进程，派生结果逐位相同；
  测试同时断言内置 `hash()` 在那两个进程里**确实不同**，以证明这个断言不是空转。
- **三个嵌入模型的 dim 相同**（配置校验挡着）；向量**不做 L2 归一化**（空文本给零向量
  就是反证 —— 归一化过的向量范数恒为 1，不可能是零向量）。
- **收拢点的输入恰好是主前沿候选**：`dag.parents("sel:main") == [bench:{sid} for sid in 主前沿]`
  —— 一个不多、一个不少。这是本形状存在的理由。
- **只读子命令不改产物**：跑完全部 11 个只读命令（含 `--dot` / `--selected-only` / `--pair`），
  `out/` 的 md5 一个都没变。
- **连跑两次，可复现字段逐位相同**（含矩阵、主前沿、超体积、参考点、版本 digest、
  血缘结构）；含实测轴的副前沿被显式排除，并有一条测试记录「为什么不用延迟」。
- **artifact 载入后行为一致**：重建模型与索引、重跑全部查询，召回率 / nDCG / MRR 与基准
  一致，成本逐位相同；并有一条反证断言「换个种子，ivf 的分簇确实会变」。
- **报告的任意两个不同指标组合都不会出现重复列名**，且每张表的行宽与表头严格相等。

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L1 深化设计稿；_
_附录 A/B/C 为落地后补写的工程记录。_
