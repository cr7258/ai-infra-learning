# PD 分离

## 吞吐量（Throughput）vs 有效吞吐量（Goodput）

目前，大多数 LLM 服务系统（如 vLLM、TensorRT-LLM）都以吞吐量（Throughput） 作为主要性能指标——即单位时间内处理的请求数（RPS）或生成的 token 数。这种度量方式直观，并且与成本（$/req）有直接关联，因此被广泛采用。

实际上，下游应用的类型多种多样，它们在用户体验上的延迟需求各异，因此需要满足的服务等级目标（SLO）也存在显著差异。大模型服务中最常用的 SLO 包括：

- TTFT (Time To First Token)：首 token 响应延迟，直接影响用户的等待体验。
- TPOT (Time Per Output Token)：衡量两个连续生成的 token 之间的平均延迟，决定交互的流畅程度。

例如，实时聊天机器人更关注 低 TTFT 以保证响应及时，而 TPOT 只需快于人类阅读速度（约 250 词/分钟）即可；相反，文档摘要则更强调 低 TPOT，以便更快地产生完整摘要。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250918230251121.png)

单纯依赖 Throughput 作为指标，并不能反映延迟表现，系统看似处理了大量请求，但其中不少未能满足 SLO，最终呈现给用户的仍是不理想的服务体验。

- Throughput（吞吐量）：通常指系统单位时间内处理的 token 数或请求数。很多工作把“提高吞吐量”作为主要优化目标，但在实际场景下，这并不直接代表用户体验。
- Goodput（有效吞吐量）：指系统在满足延迟约束（如 TTFT/TPOT SLO）的前提下，真正完成的请求数量。如果一个请求因为延迟过长而被用户放弃，或者超过服务约束而无效，那么即便它产生了 token，也不能算作有效产出。

在 [DistServe](https://arxiv.org/abs/2401.09670) 论文中，引入了 Goodput 概念，即在满足 SLO（TTFT 和 TPOT 要求）的前提下，每秒完成的有效请求数。与单纯的吞吐量相比，Goodput 是更优的衡量指标，因为它能够体现请求在满足 SLO 情况下的吞吐水平，从而同时反映成本效益与服务质量。

为了简要说明 Goodput，假设某个应用要求至少 90% 的请求满足 TTFT < 200ms 且 TPOT < 50ms，则可以得到如下定义：

> Goodput (P90 TTFT < 200ms 且 P90 TPOT < 50ms) 表示在至少 90% 的请求同时满足 TTFT < 200ms 和 TPOT < 50ms 的条件下，系统所能维持的最大每秒请求数。

下图展示了一个简单的例子：某应用的吞吐量为 10 RPS（每秒请求数），但由于延迟约束的限制，只有 3 RPS 的请求满足 SLO，因此该系统的 Goodput 仅为 3 RPS。可以想象，用户在这样一个 高吞吐但低 Goodput 的系统中，依然会感受到较差的服务体验。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250918232043135.png)


## Prefill 与 Decode 共置导致干扰

在 LLM 服务中请求的生命周期通常包含两个阶段：prefill（生成首个 token）和 decode（逐步生成后续 token）。大多数现有系统（如 vLLM、TensorRT-LLM）采用 continuous batching 技术，将 prefill 和 decode 混合在一起统一批处理。这种方式确实能够提升整体吞吐量，但由于两者计算特性和 SLO 目标差异显著，将它们共置在同一 GPU 上往往并不理想。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250919080109748.gif)

如下图所示，continuous batching 会带来明显的干扰。当 prefill 和 decode 被放在同一批次时，decode 请求的延迟（TPOT）会被显著拉长，而 prefill 请求的首 token 延迟（TTFT）也会有所增加。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250919074140327.png)

图中展示了三种不同的执行方式：

- 1P+nD（棕色柱子）：1 个 prefill 与 n 个 decode 混合批处理。
- nD（蓝色柱子）：仅包含 decode 请求的批处理。
- Prefill-only（红色虚线）：仅运行 prefill 请求的延迟。

在 prompt 长度为 128 时，相比仅包含 decode 的请求，延迟增加约 1.8 倍；而当 prompt 长度为 1024 时，干扰效应显著放大，decode 延迟提升至 12.6 倍。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250919074202351.png)

由于这种干扰，如下图所示，当服务必须同时满足 TTFT 和 TPOT 的 SLO 时，系统往往需要进行资源的过度配置才能达到延迟目标，尤其是在任一 SLO 要求较严格的情况下。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250919075355980.png)

## PD 分离的整体思路

直观的思路很简单：将 prefill 和 decode 分离到不同的 GPU 上，并为每个阶段定制并行策略。这自然解决了前面提到的两个问题：

- **没有干扰**：prefill 和 decode 各自独立运行，更快地完成计算，也更容易满足各自的 SLO。
- **资源分配与并行策略解耦**：可以针对 prefill 和 decode 分别进行优化。

下图展示了在这样一个分离式系统中，请求是如何被处理的。当一个请求到达系统时，它会先被分配到 prefill worker 完成 prefill 阶段；随后系统将其中间状态（主要是 KV Cache）迁移到 decode worker，并执行多步 decode 以生成后续 token；当生成完成后，请求才会离开系统。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250919080133122.gif)

让我们通过一个简单的实验来看看为什么 PD 分离是有益的。我们在一张 A100-80GB GPU 上运行一个 130 亿参数的 LLM，请求到达服从泊松分布，输入长度为 512，输出长度为 64。我们逐步增加请求速率（x 轴），并在下图测量两类延迟（P90 TTFT 和 P90 TPOT，y 轴）的变化。

假设我们设定 SLO：P90 TTFT = 0.4 秒，P90 TPOT = 0.04 秒（下图中的横线）。实验结果表明：在单卡情况下，现有系统大约可以在 3 rps 下满足 TTFT 的延迟约束，而 TPOT 只能维持在 1.6 rps（下图左边）。由于必须同时满足两个约束条件，现有共置系统的 Goodput = min(3, 1.6) = 1.6 rps/GPU。

在分离之后，性能得到了显著提升。如果单独处理一个阶段，prefill worker 和 decode worker 的 rps 都优于之前的结果 —— 如下图右边所示，一个 prefill worker 大约可达到 5.6 rps，一个 decode worker 大约可达到 10 rps。更重要的是，我们现在可以灵活地分配资源，例如配置 2 个 prefill worker + 1 个 decode worker（记作 2P1D），共 3 张 GPU。此时：

```bash
Goodput (2P1D) = min(5.6 × 2, 10) = 10 reqs/s ÷ 3 GPUs ≈ 3.3 rps/GPU。
```

这个实验表明，即便没有引入任何并行优化，仅仅通过简单的分离，Goodput 就提升了约 2 倍。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250919080653076.png)

## 分离式推理架构的优化方向

### 算力与存储

- prefill 阶段：拥有计算受限的性质（compute-bound），特别是在请求流量较大，用户的prompt也比较长的情况下。prefill 阶段算完 KV cache 并发给 decode 阶段后，理论上 prefill 就不再需要这个 KV cache 了（当然你也可以采用 LRU 等策略对 KV cache 的保存做管理，而不是一股脑地清除）。
- decode 阶段：拥有存储受限的性质（memory-bound），因为 token by token 的生成方式，decode 阶段要频繁从存储中读取 KV Cache，同时也意味着它需要尽可能保存 KV cache。

因此在分离式框架下，计算和存储可以朝着两个独立的方向做优化。

### Batching 策略

- prefill 阶段：随着 batch size 的增加，吞吐量的提升很快趋于平缓。这是因为 prefill 属于 compute-bound，当 batch 中的总 tokens 数超过一定规模后，GPU 的计算能力已经被完全吃满，再增加请求只会延长整体处理时间，而不会带来明显的吞吐提升。
- decode 阶段：随着 batch size 的增加，吞吐量的增长趋势越来越显著。这是因为 decode 阶段是 memory-bound，即相比于计算，读写数据的时间要更多。所以在 decode 阶段中，如果我们能提升 batch size，就能把计算强度提起来，吞吐量就上去了。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250920102734489.png)

在分离架构下，我们可以针对 prefill 和 decode 的特性对 batching 策略分别进行优化：
- 具体来说，对于 prefill 实例，需要事先结合特定的 LLM 和 GPU 做性能分析，找出输入长度的临界点——一旦超过这个点，prefill 就会进入 compute-bound，此时增加 batch size 只会拖慢整体处理速度。在实际应用中，用户的 prompt 往往已有数百个 tokens，因此 prefill 的 batch size 通常保持较小。
- 相对地，decode 阶段更适合采用较大的 batch size，以充分提升 GPU 利用率和整体吞吐。

### 并行策略

由于 prefill 和 decode 具有不同的计算模式和延迟目标，这两个阶段的最佳并行策略通常并不相同。例如，当 TTFT 要求严格而 TPOT 要求相对宽松时，prefill 更适合采用**张量并行**来满足低延迟，而 decode 则通常采用**数据并行**或**流水线**并行来提升吞吐。

## KV Cache 传输

PD 分离带来的代价是需要在 prefill 和 decode 的 GPU 之间传输中间状态（即 KV cache）。接下来，我们来看看 KV cache 传输的开销分析、传输方式以及相关的优化策略。

### KV Cache 传输开销

初看之下，KV cache 是 LLM 推理中巨大的内存开销，而 GPU 之间 KV cache 的传输似乎会成为瓶颈。然而，DistServe 的论文中展示了相反的结果：通过合理的放置，KV Cache 的传输开销可以被有效地最小化，甚至低于一次 decode 步骤的时间，这得益于当今高速互联网络（如 NVLink 和 PCI-e 5.0）。

假设我们在 GPU 之间使用 8 通道 PCIe 5.0 x16（每条链路 64GB/s）作为节点内互联。对于一个包含 2048 tokens 的请求，在服务 OPT-175B 时传输 KV cache 的延迟可以估算如下：

```bash
Latency = 2048 tokens * (4.5 MB/token) / (64GB/s * 8) = 17.6 ms
```

对于 OPT-175B，延迟小于单次 decode 步骤（在 A100 上约为 30-50 毫秒）。对于更大的模型、更长的序列或更先进的网络（例如带宽为 600GB/s 的 A100-NVLink），如下图所示，与单次 decoe 步骤相比，KV cache传输相关的相对开销变得不那么显著。总之，通过精心安排 prefill 和 decode 工作节点以利用高带宽网络，可以有效隐藏 KV cache 传输的开销。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250920113307725.png)

### KV Cache 传输方式

目前 KV cache 的传输主要有两种方式：**中心存储**和**点对点（P2P）**，当然在实际系统中也可能采用二者结合的混合方案。

- **中心存储**：建立一个跨设备的 KV store，由它统一管理 KV cache 的增、删、查和传递等操作。prefill 和 decode 实例只需与这个 KV store 交互，负责写入或读取数据。
- **P2P 传输**：每个实例独立管理自己的存储。例如，一个 prefill 实例完成计算后，会直接与目标 decode 实例建立通信，将 KV cache 传过去，不依赖统一的中介。

两种方式各有优劣：

- **中心存储**：更适合构建大规模集群，能充分利用多种存储介质和传输通道，并提升计算结果的复用效率，但在某些场景下性能可能受限，同时系统维护成本较高。
- **P2P 传输**：架构更简单，性能表现通常更好，但在扩展性和链路稳定性方面会面临挑战。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250920224320677.png)

### KV Cache 传输的网络堆栈

现有的物理数据链路可以分为 3 类：

- **Direct**，即 GPU 之间通过高速直连链路（如 NVLink 或 HCCS）相互连接。在这种情况下，可以利用底层的内存拷贝原语或集体通信库来完成数据传输。
- **Direct-NIC**，即 GPU 通过其配套的网卡（NIC）进行通信。在这里，可以使用定制化的库，通过 PCIe 和以太网（或 InfiniBand）进行数据传输。
- **Indirect**，即当 GPU 之间没有直接链路时，必须通过其 CPU 的 DRAM 中转数据，从而带来额外的内存拷贝开销。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250920225127533.png)

[图片来源：Inference without Interference:  Disaggregate LLM Inference for Mixed Downstream Workloads](https://arxiv.org/abs/2401.11181)

### KV Cache 传输粒度

KV cache 传输粒度可以分为 3 类：

- **请求级**：等到 prefill 阶段完成后，将 KV cache 一次性传输。这种方式的好处是能够减少网络传输次数，因为每次传输的数据量更大，从而降低了通信开销。然而当 KV cache 大小较大时，会影响 TTFT 的延迟。
- **层级**：Splitwise 通过在 prefill 阶段的计算与 KV cache 传输之间实现重叠来优化性能。每一层计算完成后，都会异步传输该层的 KV cache，同时继续执行下一层的计算，从而降低传输开销。层级传输还能带来额外优势，例如更早启动 decode 阶段，以及更早释放 prefill 端的内存。层级 KV cache 传输与下一层的 prefill 计算并行进行，这需要逐层的细粒度同步以确保正确性，因此可能会带来性能干扰并增加 TTFT，尤其是在小 prompt 的场景下。不过对于小 prompt 来说，KV cache 的总体规模很小，不需要层级传输来隐藏延迟。由于在计算开始时批次中的 token 数已经是已知的，**Splitwise 会选择最合适的 KV cache 传输方式：小 prompt 使用序列化传输，而大 prompt 使用层级传输。**

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250920232717989.png)

[图片来源：Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)

- **块级**：TetriInfer 在 PD 分离的基础上，还会将输入的 prompt 划分为固定大小的 chunk，以便让 GPU 始终运行在接近计算饱和的状态。因此，TetriInfer 论文中也提出了基于块级的 KV cache 传输方案。

### vLLM 的 PD 分离

vLLM 提供了 **KV Connector** 作为管理实例间 KV cache 交换的抽象层，它提供统一接口来实现 KV cache 的保存、加载与传输，使不同的 vLLM 实例（如 prefill 与 decode 实例）能够高效共享计算结果。通过实现这一接口，各类 Connector（例如通过文件系统的 SharedStorageConnector、通过网络的 NixlConnector 等）提供了灵活的 KV cache 传输方案，从而支持 PD 分离等高级功能。

`KVConnectorBase_V1` 是所有 connector 的基类。它是一个抽象基类，定义了以下 API：

- scheduler 侧方法：
  - build_connector_meta：构建元数据，scheduler 告诉 worker 需要保持/加载哪些 KV cache。
  - get_num_new_matched_tokens：获取远端已计算的 KV cache 的 token 数量。
  - update_state_after_alloc：block 开辟后，更新 connector 的状态。

- worker 侧方法：
  - **start_load_kv**：从 connector buffer 加载 KV cache，消费端调用。
  - wait_for_layer_load：阻塞直到指定层加载结束，消费端调用；
  - **save_kv_layer**：将 vLLM 的 KV buffer 中某一层的 KV cache 保存到 connector buffer 中，生产端调用。
  - wait_for_save：阻塞直到所有保存操作完成，生产端调用。

vLLM v1 中 connector 有两个执行角色（Role）：scheduler_connector 和 worker_connector，分别在 scheduler 线程和 worker 线程中执行。scheduler 负责指挥 worker 进行 KV cache 的传递，两者之间的信息桥梁是元数据（KVConnectorMetadata），worker 通过 metadata 知道哪些 KV 值需要从远端加载。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250921115717450.png)

当前 vLLM 支持 5 种类型的 connector，分别是：

- **SharedStorageConnector**：SharedStorageConnector 是 vLLM 中最简单的 KV Connector 实现，它通过共享文件系统（如本地磁盘或 NFS）在 prefill 和 decode 实例之间传递 KV cache，使用 MD5 哈希生成唯一文件名来存储和检索每个请求的 KV cache。prefill 实例将每层的 KV cache 序列化为 SafeTensors 格式保存到指定路径，decode 实例根据相同的 token_ids 计算哈希值找到对应文件并加载，整个过程没有显式的网络传输，完全依赖文件系统的读写操作。
- **P2pNcclConnector**：P2pNcclConnector 是基于 NCCL（NVIDIA Collective Communications Library）实现的高性能 KV Connector，它通过 NCCL 的 send/recv 原语实现 KV cache 在不同 GPU 之间的点对点传输，避免了文件系统的开销。
- **NixlConnector**：NixlConnector 使用 NIXL（NVIDIA Inference Xfer Library）库来加速 GPU 之间以及异构内存与存储之间的 KV cache 传输。
- **LMCacheConnectorV1**：通过与 LMCache 集成实现 KV cache 的外部存储和检索，支持多种[存储后端](https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html#supported-offloading-destinations)（如 CPU 内存、本地文件系统、Redis、 InfiniStore 等）。LMCache 通过重用缓存的 KV cache 来减少推理时间，消除冗余计算，适用于跨请求或跨会话的 KV cache 共享场景。
- **MultiConnector**：允许同时使用多个 KV connector 来实现 KV cache 的传输，它的核心逻辑是从第一个能提供可用 token 的 connector 加载 KV cache，但会向所有 connector 保存数据。MultiConnector 适用于需要同时向多个存储后端保存 KV cache 的场景，比如同时保存到本地存储和远程存储，提供数据冗余和可靠性保障。

```bash
--kv-transfer-config '{
   "kv_connector": "MultiConnector",
   "kv_connector_extra_config": {
      "connectors": [
         {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_both"
         },
         {
            "kv_connector": "SharedStorageConnector",
            "kv_connector_extra_config": {
               "shared_storage_path": "local_storage"
            },
            "kv_role": "kv_both"
         }
      ]
   },
   "kv_role": "kv_both"
}'
```

以上几个 connector 的运行实例代码可以在这里找到：https://docs.vllm.ai/en/latest/features/disagg_prefill.html#usage-example

#### 运行 vLLM 官方 PD 分离实例

##### 环境准备

执行以下命令安装 vLLM 以及 NVIDIA GPU Driver。

```bash
# 安装 UV Python 包管理工具
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 创建 Python 虚拟环境
uv venv --python 3.12 --seed
source .venv/bin/activate

# 安装 vLLM
uv pip install vllm

# 安装 NVIDIA GPU Driver
sudo apt install nvidia-driver-535

# 克隆 vLLM 代码：
git clone https://github.com/vllm-project/vllm.git -b v0.10.2
cd vllm
```

##### P2pNcclConnector

运行这个示例将会同时运行 1 个 prefill 和 decode 实例，总共需要 2 块 GPU。

```bash
# 把模型名改成 Qwen2.5-1.5B-Instruct，因为原示例中使用的 LLaMA 模型需要先在 Huggingface 上申请许可，并且配置 HUGGINGFACE_TOKEN 环境变量
sed -i 's|^MODEL_NAME=.*|MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct|' ./examples/online_serving/disaggregated_prefill.sh
bash ./examples/online_serving/disaggregated_prefill.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8100 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_port":21001,"kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"8000","http_port":"8100"}}' 
```

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --port 8200 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_port":21002,"kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"8000","http_port":"8200"}}'
```


```bash
cd benchmarks/disagg_benchmarks
python disagg_prefill_proxy_server.py
```


```bash
curl -X POST -s http://127.0.0.1:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "San Francisco is a"
}'
```

### PD 分离工业界项目


https://zhuanlan.zhihu.com/p/1906741007606878764

#### Mooncake

#### Dynamo

NVIDIA Dynamo 是一个开源的模块化推理框架，用于在分布式环境上实现生成式 AI 模型的服务化部署。Dynamo 通过动态资源调度、智能路由、内存优化与高速数据传输，无缝扩展大型 GPU 集群之间的推理工作负载。

Dynamo 采用推理引擎无关的设计（支持 TensorRT-LLM、vLLM、SGLang 等），包括以下 4 个核心组件：

- **NVIDIA Dynamo Planner**：一个智能规划和调度引擎，用于监控分布式推理中的容量与延迟，并在 prefill 与 decode 阶段之间灵活分配 GPU 资源，以最大化吞吐量和效率。Planner 会持续跟踪关键的 GPU 容量指标，并结合应用的 SLO（如 TTFT 和 ITL），智能决策是否采用分离式推理，或是否需要为 prefill/decode 阶段动态增加更多 GPU。
- **NVIDIA Dynamo Smart Router**：KV cache 感知的路由引擎，可在分布式推理环境中将请求转发到最佳的节点，从而最大限度减少 KV cache 的重复计算开销。
- **NVIDIA Dynamo Distributed KV Cache Manager**：通过将较旧或低频访问的 KV cache 卸载到更低成本的存储（如 CPU 内存、本地存储或对象存储等），大幅降低 GPU 内存占用。借助这种分层管理，开发者既能保留大规模 KV cache 重用的优势，又能释放宝贵的 GPU 资源，从而有效降低推理计算成本。
- **NVIDIA Inference Transfer Library (NIXL)**：高效的推理数据传输库，可加速 GPU 之间以及异构内存与存储之间的 KV cache 传输。通过减少同步开销和智能批处理，NIXL 显著降低了分布式推理中的通信延迟，使得在 prefill/decode 分离部署时，prefill 节点也能在毫秒级将大批量的 KV cache 传输至 decode 节点，从而避免跨节点数据交换成为性能瓶颈。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250905101610730.png)

在 Dynamo 的 PD 分离架构中，有 4 个核心组件：

- **(decode) worker**：执行 prefill 和 decode 请求。
- **prefill worker**：只执行 prefill 请求。
- **disaggregated router**：决定 prefill 阶段是在本地还是远程执行。
- **prefill queue**：缓存并负载均衡远程 prefill 请求。

当 worker 收到请求时，首先会通过 disaggregated router 判断 prefill 应该在本地还是远程完成，并分配相应的 KV block。
如果选择远程 prefill，请求会被推送到 prefill queue。随后，prefill worker 从队列中取出请求，读取 worker 中 prefix cache 命中的 KV block，执行 prefill 计算，并将生成的 KV block 回写给 worker。最后，worker 会继续完成剩余的 decode 阶段。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250905210843098.png)

Dynamo 提供了 Operator 方便我们在 Kubernetes 环境中以声明式的方式定义 PD 分离服务。只需在 `DynamoGraphDeployment` 配置中声明 Frontend、VllmDecodeWorker 和 VllmPrefillWorker 三个组件即可。`dynamoNamespace` 是 Dynamo 分布式运行时的逻辑隔离单元，而非 Kubernetes 的 namespace；同一 `dynamoNamespace` 内的组件可以相互发现并进行通信。

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
spec:
  services:
    Frontend:
      dynamoNamespace: vllm-disagg
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1
    VllmDecodeWorker:
      dynamoNamespace: vllm-disagg
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1
          workingDir: /workspace/components/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - "python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B"
    VllmPrefillWorker:
      dynamoNamespace: vllm-disagg
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1
          workingDir: /workspace/components/backends/vllm
          command:
            - /bin/sh
            - -c
          args:
            - "python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --is-prefill-worker"
```

Dynamo 详细的部署教程可以参考博客：https://cr7258.github.io/blogs/original/2025/20-dynamo#_3-%E8%BF%90%E8%A1%8C-dynamo

#### llm-d

#### AIBrix


## PD 分离相关论文

- DistServe：将预填充和解码计算分配至不同的 GPU，从而消除了二者之间的干扰。针对应用程序的总时延和每个 Token 运行时间的需求，DistServe 为每个阶段量身定制了资源分配和并行策略的优化方案。此外，DistServe 还根据服务集群的带宽来确定如何部署这两个阶段，以最小化由任务分解引起的通信。
- SplitWise：特点是分布式调度策略、PD分离、分层KV Cache传输，并且增设了第三个机器池，专门用于处理 Prefill 和 Decode 阶段的混合批处理，并且能够根据实时计算需求灵活调整其规模。
    - 分层 KV cache 传输
    - 和Mooncake最相似的是今年5月份发布的微软和华盛顿大学的工作Splitwise，它里面列出了Prefill和Decode不同的七个Insights值得大家仔细读一下。
    洞察一：不同的推理服务可能具有广泛不同的提示和标记分布。
    洞察二：混合连续批处理中的大部分时间都花在令牌阶段，很少有活动令牌进行批处理。
    洞察三：对于大多数请求，大部分E2E时间都花在令牌生成阶段。
    洞察四：提示阶段的批处理大小应限制在保证良好性能的范围内。相反，令牌生成阶段的批处理可以获得高吞吐量而没有任何不利影响。
    洞察五：在提示阶段进行批处理受计算限制，而令牌阶段受内存容量限制。
    洞察六：尽管提示阶段有效地利用了GPU的功耗预算，但令牌生成阶段没有。
    洞察七：生成阶段可以在更便宜、性能较低的硬件上运行，以获得更好的性能/功耗和性能/价格效益。
- TetriInfer：结合了分块预填充和两阶段分离，以及预测性的两阶段调度算法来优化资源利用率，且仅针对 prefill 进行分块处理。SplitWise、DistServe 和 TetriInfer的静态并行性和分区策略并不灵活，无法处理动态工作负载。
    - limiting the number of tokens processed in a single prefill iteration so that hardware is fully utilized without incurring extra penalties.
    - disaggregating prefill from decode phases. TetriInfer has dedicated prefill and decode instances
    - Third, to avoid interference running decode requests, we propose using a smart two-level scheduling algorithm augmented with predicted resource usage to avoid scheduling hotspots
    - TetriInfer incorporates an LLM-based length prediction model to speculate the number of generated tokens of decode requests, and then schedule them accordingly
    我们的想法有三方面。首先, 为了避免预填充时的干扰,我们建议限制单个预填充步 骤中处理的令牌数量,以便硬件得到充分利用而不产生 额外惩罚。其次,为了避免预填充和解码同时运行的干 扰,我们建议将预填充与解码分离,使它们独立运行。 第三,为了避免解码请求运行的干扰,我们建议使用一 种智能的两级调度算法,并辅以预测的资源使用情况, 以避免调度热点。
    - we have designed and implemented three scheduler policies: first-come-first-serve (FCFS), shortest-job-first (SJF), and longest-job-first (LJF).
    - 一项并行工 作 [32] 提出了按层传输KV缓存的方法,这与我们的块 级别方法一致。在本工作 中,我们仅为了简化而实现了请求级别传输,将块级别 传输留待未来研究。
    - 们将现有的物理数据链路分为三种类型
    - 实例翻转：Additionally, TetriInfer can also dynamically adjust the number of prefill and decode instances within fixed hardware resources.
- MemServe：用统一的分布式视角对KV Cache进行管理。
   - 这些利用依赖关 系的技术可以分为两类:请求间和请求内。
   - MemPool管理推理集群中的所有内存,包括CPU DRAM和GPU HBM。MemPool在每个推理实例中运 行,共同提供一组分布式内存池API
   - 为实现此索引, MemPool 利用 SGLang 提出的 radixtree [40], 并进行 了两项关键扩展。
    - 首先,由于 MemPool 管理 GPU HBM 和 CPU DRAM,我们使基数树能够引用系统中任 何位置的数据。
    - 其次,由于我们还将基数树用于在全局调 度器(§6)中构建全局提示树,我们添加了一个字段来 指示哪个推理实例持有数据
    - 按层传输 KV cache：。然而,随着负载的增加,这两种方法都会由于过多的网络传 输而产生非微不足道的开销。我们发现根本原因在于(1) 离散的内存布局和(2)不足的网络原始功能。
    - 成本模型：该模型 有两个主要目的:1)实现位置感知和负载均衡的全局调度,以及2)决定是否转移KV缓存或重新计算它们。
- Mooncake：构建了以 KV cache 存储为中心、基于RDMA的 P-D 分离调度集群和推理架构，形成了 prefill/decode pool 以及分布式异构介质 KV cache pool；融合了缓存感知、负载均衡和以服务水平目标（SLO）为导向的决策机制。
  - 我们坚信一个系统方案只有实际在场景中验证过的，才有分享出来的必要性。本论文与很多 Prefill/Decoding 分离的论文不同的是，这套方案已经在大规模集群上进行几个月的验证并证明了方案的有效性。https://zhuanlan.zhihu.com/p/705910725
  - 在prefill阶段,主要目标 是尽可能重用KVCache以避免冗余计算。然而,等待存储在低层存储上的KVCache可能会 违反TTFT SLO。
  - 相比之下,decoding阶段有不同的优化目标和约束。目标是在一个decoding批次中聚合 尽可能多的token以提高MFU。然而,这一目标不仅受TBT SLO的限制,还受聚合 KVCache总大小能否容纳在VRAM中的限制。
  - Conductor还负责预测KVCache块的未来使用情况,并相应执行调度操作,如交换 和复制。最热门的块应复制到多个节点以避免获取拥堵,而最冷的块应被交换出去以降低 保留成本。
  - 相比之 下,目前GPU/加速器的供应有限,许多MaaS提供商在高峰期面临严重的超载问题。在这 种情况下的调度带来了现有工作未曾探讨的独特挑战。例如,我们需要预测未来负载,并 在prefill阶段后如果没有可用的解码槽时提前拒绝某些请求,以节省浪费的计算资源。
  - 我们的主要目标是在遵守SLO的前提下最大化整体吞吐量,这一 概念在其他研究中被称为goodput [8, 14]。我们的方法不同之处在于,只有完全完成执行的请 求才计入goodput的衡量。否则,所有先前消耗/生成的token都不计入,且相应的资源被浪费。换句话说,如果请求无法在SLO内完成全部执行,则应尽早拒绝该请求。实现这一目标不仅需要 优化prefill和解码阶段的架构,还需要开发预测短期未来负载的能力。
  - 我们决定保持Mooncake的分离架构。只有当请求的prefill可以在不进 行分块且不影响TBT SLO的情况下转发时,才将其内联到解码批处理中。做出这一决定的主要 原因有两个:1)Prefill节点需要不同的跨节点并行设置以处理长上下文(§5.1);2)这为节 省VRAM提供了独特的机会
  - 为了解决这个问题,Mooncake 利用仅解码器变换器的自回归特性,并实现了用于长上下文预 填充的分块流水线并行(CPP)。
  - Layer-wise Prefill 通过在逐层计算时异步加载/存储 KVCache，把 I/O 和计算重叠起来，从而显著降低长上下文请求的延迟，并释放调度上的灵活性。
  - 当最优前缀实例负载过高时，系统会在 本地重算 vs 远程迁移 之间做权衡：如果计算更快就重算，否则迁移。这样既降低请求延迟，又能在不同机器之间自动复制热点 KVCache，提升整体负载均衡。：如前所述，由于实例负载过高，请求并不总是会被分配到拥有最长前缀缓存的 prefill 实例。在这种情况下，如果预估的额外预填充时间短于缓存迁移时间，调度器会将缓存位置和请求转发给另一个实例。该实例会主动从缓存持有者处获取 KVCache，并将其存储到本地。更重要的是，当远程最佳前缀匹配长度不大于当前本地可复用前缀长度乘以某一阈值时，我们更倾向于直接重新计算输入 tokens。这两种策略不仅减少了请求的预填充时间，还能促使热点缓存自动复制，并在多台机器之间更广泛地分布。
  - 为了解决此问题,自然的做法是将 decoding 实例的负载评估提前到 prefill 阶段开始之前。我 们将此策略称为 Early Rejection。
  - 此外,近期研究与我们对预填充和解码阶段分离的见解一致,提出了一种提升系统吞吐量 的解耦架构。Splitwise [7]的arXiv发表正处于Mooncake开发的早期阶段,进一步激励了 我们的进展。许多并行工作也证实了我们的发现,包括DistServe [8], ,它优化了每个阶段 的资源分配和并行策略以最大化GPU有效吞吐量,以及TetriInfer [9], ,它结合了分块预 填充和两阶段解耦,并采用预测性两阶段调度算法以优化资源利用。


下面介绍几篇关于分离式推理结构的论文：

1. DistServe: Disaggregating Prefill and Decode for Goodput-optimized Large Language Model Serving

这篇论文提出了 DistServe 系统，通过将大型语言模型（LLM）的预填充（prefill）和解码（decode）阶段分离，以优化服务性能。传统的 LLM 服务通常将这两个阶段合并处理，可能导致资源竞争和性能下降。DistServe 通过将预填充和解码分配到不同的 GPU 上，消除了相互干扰，并针对每个阶段的特定需求进行资源分配和并行策略优化，从而提高了每个 GPU 的有效吞吐量。

2. Splitwise: Efficient Generative LLM Inference Using Phase Splitting

该论文介绍了 Splitwise 技术，通过将 LLM 推理过程中的提示计算（prompt computation）和令牌生成（token generation）阶段分离到不同的机器上，以提高硬件利用率。提示计算阶段计算密集，而令牌生成阶段则受限于内存带宽。通过分离这两个阶段，Splitwise 能够针对每个阶段的特定需求进行资源管理，从而在相同的成本和功耗预算下，实现更高的吞吐量。

3. Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads

这篇论文提出了 TetriInfer 系统，通过将 LLM 推理过程中的预填充和解码阶段分离，以减少不同下游任务之间的干扰。TetriInfer 通过将输入提示分割成固定大小的块、独立运行预填充和解码实例，以及使用智能的两级调度算法，显著降低了首次令牌生成时间（TTFT）和作业完成时间（JCT），并提高了性能与成本的效率。

4. MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool

MemServe 系统针对分离式 LLM 服务中的上下文缓存管理问题，提出了弹性内存池的解决方案。通过高效的上下文缓存机制，MemServe 能够在不同的计算节点之间共享和管理内存资源，从而提高 LLM 服务的可扩展性和资源利用率。

5. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

Mooncake 提出了一种以键值缓存（KVCache）为中心的分离式 LLM 服务架构。通过优化 KVCache 的管理和传输，Mooncake 在满足服务水平目标（SLO）的前提下，实现了高达 525%的吞吐量提升。在实际工作负载下，Mooncake 使得 Kimi 系统的请求处理能力提高了 75%。



## Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

一是，将Prefill与Decode计算资源分开，这与前人工作无异，如splitwise和distserve等；Prefill阶段优化目标是利用request间存在共同前缀的机会，尽可能复用KVCache，同时满足TTFT（Time To First Token） SLO，最大化MFU（论文中似乎笔误成minimum）和KVCache小于CPU内存限制。Decode优化目标为最大化吞吐，满足TBT（Time between tokens ，Decode阶段两个Token之间的时间）SLO和KVCache小于GPU显存限制。

二是，将KVCache和计算分离开，它将GPU集群的CPU、DRAM、SSD和RDMA资源分组组成Distributed KVCache Pool，KVCache也是分块以Paged方式管理，KVCache Blocks如何在Pool中调度请求和复用KVCache乃本文精髓

Mooncake 采用了一种以 KV cache 为核心的解耦架构，将 prefill 集群与 decoding 集群分离开来。同时，它还利用 GPU 集群中未被充分使用的 CPU、DRAM 和 SSD 资源，来实现一个解耦式的 KV cache 缓存。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250913104732062.png)

Mooncake 的核心是其以 KV cache 为中心的调度器，它在最大化整体有效吞吐量的同时，满足与延迟相关的服务等级目标（SLO）要求。不同于传统研究中假设所有请求都会被处理的前提，Mooncake 在高负载场景下面临更多挑战。为此，Mooncake 提出了一种基于预测的提前拒绝策略来缓解问题。实验结果表明，Mooncake 在长上下文场景中表现尤为出色。与基线方法相比，Mooncake 在某些模拟场景下的吞吐量最高可提升 525%，同时仍能满足 SLO 要求。在真实工作负载下，Mooncake 的创新架构使 Kimi 能够多处理 75% 的请求。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250913105013510.png)

通过将 token 块(块大小为 512)哈希为包含当前块 及所有前置块的前缀哈希值生成的(详见图 3)。生 成的哈希值随后映射为全局唯一 ID。相同的 hash ID 表明一块 token 及其前置 token 是相同的,从而 允许在对应的 KVCache 中重用。例如,在提供的样 本中,前 12 个 hash ID 是相同的,表明它们可以共 享前 12*512=6,144 个 token 的前缀缓存。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250913105404446.png)

这些KVCache块在CPU和GPU之间的传输由一个独立的基于(GPUDirect)RDMA的组件 Messenger处理。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250913111146148.png)

预填充实例的选择考虑了更多因素——不仅是负载,还有前缀缓存 命中长度和可重用 KVCache 块的分布。虽然倾向于将请求路由到具有更长前缀缓存长度 的预填充实例以减少计算成本,但将它们调度到其他节点以确保整体系统平衡并满足 TTFT SLO 可能更有利。
为了解决这些复杂性,我们提出了一种缓存感知的全局调度算法, 该算法同时考虑了由于前缀缓存引起的预填充时间和与实例负载相关的排队时间。



## 配合模型并行传递需要注意的问题？

https://zhuanlan.zhihu.com/p/1906741007606878764

## KV cache传递一定是只有P端到D端吗？

## KV Cache 传输开销

针对KV Cache传输问题，为了避免迁移开销上升，需要合理放置 Worker。DistServe作者通过精心放置预填充和解码工作器来利用高带宽网络来有效地隐藏 KV Cache 传输开销。具体策略是：因为KVCache 按层存储，所以可以把模型按层划分多段，每段放到不同机器，不同段的模型采用 PP，然后再对同一段的模型进行单机的模型并行策略的搜索。这样可以保证：

跨机传输仅出现在 PP 层间。
Prefill 和 Decode Worker 相同层的 KVCache 在同一个机器内，Prefill 和 Decode Worker之间可以使用节点内NVLINK带宽进行传输。
从而显著减少传输延迟。并且当模型越大、序列越长、多卡通信设备带宽越高，KVCache 迁移开销占比越低。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250817215257920.png)


![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250912111152042.png)
[图片来源：Inference without Interference:  Disaggregate LLM Inference for Mixed Downstream Workloads](https://arxiv.org/abs/2403.02310)

## Chunked-Prefills VS PD 分离

分块预填充会导致预填充的计算开销，因此选择明显低于 GPU 饱和点的块大小会延长预填充任务的执行时间。
仍然存在prefill阶段无法最大化MFU的可能。因为在chunk-prefill中，我们只是用profiling估算出在特定设备上一个batch的最大tokens配额，这些tokens包括prefill和decode。这个size是对整体的，而不是单独对prefill或decode的。而且如果序列长度除不尽tiles尺寸，则又会产生额外的计算开销。
即使块大小被优化到几乎最大化 GPU 使用率，分块预填充也会显著增加预填充任务的内存访问量，因为需要将 KV Cache 从 GPU 的 HBM 加载到 SRAM 以用于每个后续块。而且长序列可能会持久地占据着KV cache的存储空间以及gpu的计算资源。
至于 TPOT，将预填充和解码在批次中合并实际上会降低所有这些解码任务的速度。

总之，分块预填充可能有助于最大化整体吞吐量，但由于动态分割无法完全解耦预填充和解码操作，会导致资源争用以及 TTFT 与 TPOT 之间的妥协。当应用程序无法在 TTFT 和 TPOT 之间进行权衡，而是要同时遵守这两者时，解耦就成为更好的选择。


例如,尽管许多研究人员 [7, 8, 9] 与我们有相同的直觉,倾向于使用分离架构,但随 着chunked prefill [15]的引入,是否仍有必要进行这种分离值得讨论。Chunked prefill 将输入token划分为多个小块,加入连续的批处理流程。这种方法有两个明显的好处:1) 没有分离时,所有节点被平等对待,使调度更简单;2)将chunked prefill内联到解码批 处理中可以提高解码批次的计算强度,从而带来更好的MFU。


我觉得最主要的一点，是chunked-prefills可能还没有完全实现在达到TPOT/TBT SLO的情况下，最大化prefill阶段对GPU FLOPS的利用率（MFU）。
https://zhuanlan.zhihu.com/p/710165390

## vLLM 的 PD 分离

https://docs.vllm.ai/en/stable/features/disagg_prefill.html

prefill阶段，会将计算的kvcache通过save_kv_layer函数保存到本地文件夹中。
decoder阶段，会通过start_load_kv函数从本地文件夹中读取kvcache。

![](https://chengzw258.oss-cn-beijing.aliyuncs.com/Article/20250913115458124.png)


Now supports 5 types of connectors:

SharedStorageConnector: refer to  examples/offline_inference/disaggregated-prefill-v1/run.sh for the example usage of SharedStorageConnector disaggregated prefilling.
LMCacheConnectorV1: refer to  examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh for the example usage of LMCacheConnectorV1 disaggregated prefilling which uses NIXL as the underlying KV transmission.
NixlConnector: refer to  tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh for the example usage of NixlConnector disaggregated prefilling which support fully async send/recv.
P2pNcclConnector: refer to  examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh for the example usage of P2pNcclConnector disaggregated prefilling.
MultiConnector: take advantage of the kv_connector_extra_config: dict[str, Any] already present in KVTransferConfig to stash all the connectors we want in an ordered list of kwargs.such as:

- vLLM PD分离方案浅析：https://zhuanlan.zhihu.com/p/1889243870430201414

V1版本中PD的角色关系已变得比较模糊，P和D既可以是生产者也可以是消费者，而且默认开启prefix KV cache特性后P可以从D实例中获取已计算好的Block（虽然这个功能暂未实现）。


## 参考资料

- Lecture 58: Disaggregated LLM Inference：https://www.youtube.com/watch?v=tIPDwUepXcA
- Throughput is Not All You Need: Maximizing Goodput in LLM Serving using Prefill-Decode Disaggregation：https://hao-ai-lab.github.io/blogs/distserve/
- Mooncake阅读笔记：深入学习以Cache为中心的调度思想，谱写LLM服务降本增效新篇章：https://zhuanlan.zhihu.com/p/706097807
- 探秘Transformer系列之（26）--- KV Cache优化 之 PD分离or合并：https://www.cnblogs.com/rossiXYZ/p/18815541
- 大模型推理分离架构五虎上将：https://zhuanlan.zhihu.com/p/706218732
- LLM关于PD分离的最新实测：https://zhuanlan.zhihu.com/p/1919794916504114120
- State of the Model Serving Communities - August 2025：https://inferenceops.substack.com/p/state-of-the-model-serving-communities
- 图解大模型训练系列：序列并行1，Megatron SP：https://zhuanlan.zhihu.com/p/4083427292
- 序列并行做大模型训练，你需要知道的六件事：https://zhuanlan.zhihu.com/p/698031151
- vLLM PD分离KV cache传递机制详解与演进分析：https://zhuanlan.zhihu.com/p/1906741007606878764
- vLLM PD分离方案浅析：https://zhuanlan.zhihu.com/p/1889243870430201414
- P/D Disaggregation of vLLM and Integration with Mooncake：https://docs.google.com/document/d/1Ab6TMW1E2CdHJJyCrpJnLhgmE2b_6leH5MVP9k72sjw/edit?tab=t.0#heading=h.611v2r4aqubz
- 0.5x提升:PD分离KV cache传输的实践经验：https://zhuanlan.zhihu.com/p/1946608360259577576
- 分布式推理优化思路：http://zhuanlan.zhihu.com/p/1937556222371946860
- 图解大模型计算加速系列：分离式推理架构2，模糊分离与合并边界的chunked-prefills：https://zhuanlan.zhihu.com/p/710165390
- 图解大模型计算加速系列：分离式推理架构1，从DistServe谈起：https://zhuanlan.zhihu.com/p/706761664
- LLM推理优化 - Prefill-Decode分离式推理架构：https://zhuanlan.zhihu.com/p/9433793184
- Shaping NIXL-based PD Disaggregation in vLLM V1：https://blog.lmcache.ai/2025-04-11-lmcache-vllmv1-nixl/
- vLLM P2P NCCL Connector：https://docs.vllm.ai/en/latest/design/p2p_nccl_connector.html
- vLLM Disaggregated Prefilling (experimental)：https://docs.vllm.ai/en/latest/features/disagg_prefill.html
- LMCache Example: Disaggregated prefill：https://docs.lmcache.ai/getting_started/quickstart/disaggregated_prefill.html
- Bringing State-Of-The-Art PD Speed to vLLM v1 with LMCache：https://blog.lmcache.ai/2025-04-29-pdbench/
- Demystify vLLM V1 KVconnector SharedStorageConnector：https://blog.diabloneo.com/demystify-vllm-v1-kvconnector-sharedstorageconnector-05a487627036
- vLLM源码之分离式架构：https://zhuanlan.zhihu.com/p/1933647687
- vLLM v1 PD分离设计：https://zhuanlan.zhihu.com/p/1894425784107632241
- Inside vLLM: Anatomy of a High-Throughput LLM Inference System：https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html
- [P/D][V1] KV Connector API V1：https://github.com/vllm-project/vllm/pull/15960
- vLLM PD Disaggregation discussion：https://docs.google.com/document/d/1uPGdbEXksKXeN4Q9nUm9hzotqEjQhYmnpAhidLuAsjk/edit?tab=t.0#heading=h.qhtgj3vmvwn
