

## 附录

### 什么是注意力？

当模型处理句子 “an apple and an orange” 时，注意力机制帮助模型理解 “apple” 的具体含义。

- “apple” 是一个多义词，既可以指水果，也可以指品牌（Apple 公司）。
- 句子中紧跟着的 “orange” 明确是一种水果，这个上下文信息通过注意力机制被捕捉到。
- 由于 “orange” 与 “apple” 在水果的语义空间中关系更密切，注意力机制会赋予 “orange” 较高的权重，从而帮助模型倾向于将 “apple” 理解为水果，而非品牌。

### 什么是 Q、K、V？

在 Transformer 模型中，**Q（Query）、K（Key）、V（Value）** 是自注意力机制（Self-Attention）的核心组成部分，用于计算输入序列中不同 token 之间的依赖关系，并融合上下文信息。

Q、K、V 都是由输入的词向量（embedding）通过不同的线性变换矩阵生成的：

$$
Q = X W^{Q},\quad K = X W^{K},\quad V = X W^{V}
$$

* 每个权重矩阵 $W^{Q}, W^{K}, W^{V}$ 都是可训练的参数，能够增强模型的表达能力；
* 这三个向量随后被送入注意力机制中进行信息交互与聚合。

**Query（Q）**  
   - **定义**：当前 token 提出的“问题”或“关注点”。
   - **作用**：用于与所有 token 的 Key 做点积，计算相似度（相关性分数）；

**Key（K）**  
   - **定义**：所有 token 的“身份标识”或“关键词”；
   - **作用**：与 Query 做点积，用于计算注意力分数，表示被关注的程度。

**Value（V）**  
   - **定义**：包含 token 的实际语义信息；
   - **作用**：在 softmax 得到的注意力权重下进行加权聚合，形成上下文感知表示。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/202505181241927.png)

Attention 的输出会传入一个 **Linear 层**映射成词表大小的 logits，然后经过 **softmax** 得到每个词的概率。最终生成阶段使用 greedy、sampling 或 beam search 等方式，从中选择下一个 token。

### KV Cache

- 在自回归生成任务（如大语言模型推理）中，每一步生成新 token 时，输入序列会不断变长。每次经过自注意力层时，都需要重新计算所有 token 的 Key（K）和 Value（V）向量。
- 对于已经生成过的 token（如 “Machine”, “learning”, “is”, “fun”），它们的 Key 和 Value 向量在每次新 token 推理时其实是完全相同的，并不会发生变化。
- 但如果不使用 kv cache，每次生成新 token 时，模型都要重复计算这些已经存在的 Key 和 Value，造成大量冗余计算，极大降低推理效率。

**kv cache 的作用**：

- kv cache 会把已经生成过的 token 的 Key 和 Value 向量缓存下来，后续每次推理时只需计算新 token 的 Key 和 Value，然后与缓存中的历史 Key 和 Value 拼接即可。
- 这样，模型每步只需计算一次新的 Key/Value，大幅减少重复计算，显著加快推理速度，尤其是在长序列推理时效果更明显。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/202505181349255.png)

### Parallel Sampling

**Parallel Sampling（并行采样）** 指的是对同一个输入 prompt，同时并行生成多个不同的输出结果，以提升生成内容的多样性和可选性。例如，ChatGPT 可以为同一个问题生成多个答案供你挑选。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/202505232255681.png)

### Beam Search

Beam Search（束搜索） 是一种在文本生成任务（如机器翻译、文本摘要、语音识别等）中常用的解码策略。它通过在每一步保留多个得分最高的候选序列，逐步扩展这些序列，以找到整体概率最高的输出。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/202505232256066.png)


每个时间步都有 A、B、C、D、E 五种可能的输出，设定 `num_beams = 2`，即每一步保留得分最高的两个候选序列。

1. **第一个时间步**：模型预测出 5 个候选词 A\~E。假设其中 A 和 C 得分最高，因此保留 `[A]` 和 `[C]` 两个分支，舍弃其余。
2. **第二个时间步**：从 `[A]` 和 `[C]` 分别扩展出 5 个新词，形成共 10 个候选序列（如 `[AB]`, `[AD]`, `[CE]` 等）。对这 10 个序列统一打分，保留得分最高的两个（假设为 `[AB]` 和 `[CE]`）。
3. **第三个时间步**：再从 `[AB]` 和 `[CE]` 各自扩展出 5 个词，得到 10 个新候选。重复相同的筛选过程，最终保留得分最高的两个序列（如 `[ABD]` 和 `[CED]`）。

最终输出 `[ABD]` 和 `[CED]`，它们是通过 Beam Search 筛选出的两个最优结果。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/202505232306767.png)

## 虚拟内存和分页

在传统的计算机系统中，CPU 直接操作物理内存地址。这意味着，应用程序认为自己是在操作物理内存。如果两个程序占用的内存有重叠，它们可能会互相干扰，导致程序崩溃。为了避免这种情况，操作系统引入了虚拟内存机制。每个进程被分配独立的一套虚拟地址空间，互不干涉。

**虚拟内存（Virtual Memory）** 是计算机系统内存管理的一种技术。它使得应用程序认为它拥有连续的可用的内存（一个连续完整的地址空间），而实际上，它通常是被分隔成多个物理内存碎片，还有部分暂时存储在外部磁盘存储器上，在需要时进行数据交换。

**分页（Paging）** 是一种将虚拟地址空间和物理内存划分为固定大小块的技术。在分页系统中：

* **虚拟内存**被划分为固定大小的块，称为 **页（Pages）**。
* **物理内存**被划分为与页大小相同的块，称为 **页框（Page Frames）**。
* **页表（Page Table）** 是一个数据结构，用于记录虚拟页与物理页框之间的映射关系。

当程序访问某个虚拟地址时，系统会通过页表将其转换为对应的物理地址。如果所需的页不在物理内存中，就会发生 **缺页中断（Page Fault）**，操作系统会将所需页从磁盘加载到内存中。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/202505241800931.png)

### 自回归分解（autoregressive decomposition）公式

语言建模的目标就是：计算一段文本（由一系列 token 构成）的概率是多少。由于语言本身具有自然的**顺序性**，所以我们通常会将这个**联合概率**分解为一系列**条件概率的乘积**，这也被称为 **自回归分解（autoregressive decomposition）**。

$$
P(x) = P(x_1) \cdot P(x_2 \mid x_1) \cdot \cdots \cdot P(x_n \mid x_1, ..., x_{n-1})
$$

整个句子的概率是：第一个词的概率 × 第二个词在第一个词出现的条件下的概率 × 第三个词在前两个词出现的条件下的概率 …… 依此类推。

### Causal Self-Attention（因果自注意力）公式

**Causal Self-Attention（因果自注意力）** 是一种特殊的自注意力机制，采用了因果掩码（Causal Masking），主要用于序列生成任务中，确保模型在生成当前位置的输出时，只依赖于该位置之前的信息，而不会利用未来位置的信息，从而保持自回归（autoregressive）属性。

$$
a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^i \exp(q_i^\top k_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^i a_{ij} v_j
$$

* **$q_i^\top k_j$**：第 $i$ 个 token 的 Query 向量与第 $j$ 个 token 的 Key 向量之间的点积（衡量相似度）。
* **除以 $\sqrt{d}$**：是为了防止点积值过大导致 softmax 输出太极端（归一化）。
* **exp / sum exp**：是 softmax 函数，用于将所有相似度变成一个概率分布。
* **上限是 $i$**：说明这里使用了**因果掩码（causal mask）**，也就是说 token $x_i$ 只能“看到”它自己和它之前的 tokens，不能访问未来的信息（用于生成任务）。


### Transformer 架构

Transformer 架构主要有三种变体：**Encoder-Only（仅编码器）**、**Decoder-Only（仅解码器）** 和 **Encoder-Decoder（编码器-解码器）**。它们在结构和应用场景上各有特点。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/202505241905140.png)

1. Encoder-Only（仅编码器）

* **结构特点**：仅包含编码器部分，使用双向自注意力机制（Bidirectional Self-Attention）。该机制允许模型在处理每个词时，同时关注该词前后的所有词，从而捕捉整个序列的上下文信息。
* **典型模型**：BERT（Bidirectional Encoder Representations from Transformers）
* **应用场景**：文本分类、命名实体识别、情感分析、问答系统等需要对输入文本进行理解的任务。
* **工作方式**：模型接收完整的输入序列，通过编码器生成上下文相关的表示，用于下游任务。

2. Decoder-Only（仅解码器）

* **结构特点**：仅包含解码器部分，采用自回归机制，使用因果掩码（Causal Masking）确保每个位置只能关注其之前的位置。
* **典型模型**：GPT 系列（Generative Pre-trained Transformer）
* **应用场景**：文本生成、自动补全、对话系统等需要生成文本的任务。
* **工作方式**：模型逐步生成文本，每一步根据之前生成的内容预测下一个词。

3. Encoder-Decoder（编码器-解码器）

* **结构特点**：包含编码器和解码器两个部分。编码器处理输入序列，解码器生成输出序列，解码器在生成每个词时，既关注之前生成的词，也关注编码器的输出。
* **典型模型**：原始 Transformer、T5（Text-to-Text Transfer Transformer）
* **应用场景**：机器翻译、文本摘要、问答系统等需要将输入序列转换为输出序列的任务。
* **工作方式**：编码器将输入序列编码为上下文表示，解码器根据这些表示和已生成的词逐步生成输出序列。


### PagedAttention Block 定义

$$
K_j = (k_{(j-1)B+1}, \dots, k_{jB})
$$

这表示第 $j$ 个 Key block 包含：

* 第 $(j-1)B+1$ 到第 $jB$ 个 token 的 Key 向量
* 一共刚好 $B$ 个 key，按顺序组合成一个 block


举个例子：

* Block size $B = 128$
* 那么第 1 个 block ($j = 1$) 包含：

  * Key 向量： $k_1, k_2, ..., k_{128}$
  * Value 向量： $v_1, v_2, ..., v_{128}$

* 第 2 个 block ($j = 2$) 包含：

  * Key 向量： $k_{129}, ..., k_{256}$
  * Value 向量： $v_{129}, ..., v_{256}$


### PagedAttention 中基于 block 的注意力计算公式


$$
A_{ij} = \frac{\exp(q_i^\top K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^\top K_t 1 / \sqrt{d})}
$$

* $A_{ij}$ 表示第 $i$ 个token（查询向量 $q_i$）与第 $j$ 个KV block（键向量 $K_j$）之间的注意力权重。
* $q_i$ 为第 $i$ 个token的查询向量（Query vector）。
* $K_j$ 为第 $j$ 个KV block的键向量（Key vector）。
* $d$ 表示模型维度大小，用于缩放注意力分数，防止过大的数值导致梯度消失。
* 分母是归一化因子，确保所有的注意力权重之和为1。

$$
o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top
$$

* $o_i$ 为第 $i$ 个token的注意力输出向量（Output vector）。
* $V_j$ 为第 $j$ 个KV block的值向量（Value vector）。
* $A_{ij}^\top$ 表示注意力权重向量的转置，与值向量 $V_j$ 相乘以加权求和，产生最终的输出。

* 每个 $A_{ij}$ 实际上是一个行向量，包含了第 $i$ 个token对于第 $j$ 个KV block内所有token的注意力得分：

$$
A_{ij} = (a_{i,(j-1)B+1}, \dots, a_{i,jB})
$$

* 每个 KV block 的大小固定为 $B$，这样通过分块计算注意力，可以显著降低内存使用，并提高计算效率。这就是 PagedAttention 机制的核心思想。

## CUDA kernel

CUDA kernel 是一种运行在 GPU 上的并行函数，它通过启动大量线程并行执行相同的代码，从而实现高效的并行计算，是 CUDA 编程模型的核心概念。

## 参考资料

- How To Reduce LLM Decoding Time With KV-Caching: https://newsletter.theaiedge.io/p/how-to-reduce-llm-decoding-time-with
- Introduction to vLLM and PagedAttention：https://blog.runpod.io/introduction-to-vllm-and-how-to-run-vllm-on-runpod-serverless/