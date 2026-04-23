# 🚀 TensorRT-LLM 模型部署指南（Triton Inference Server）

---

## 📖 简介

在参考b站程序员鲁哥的视频后，进行了相应的实验，与视频中有一些不同。本指南详细介绍如何使用 TensorRT-LLM 将 Llama3-8B 模型转换为 TensorRT Engine，并部署到 Triton Inference Server。

本教程面向希望学习大模型部署的开发者，通过本指南您将掌握：

- 如何将 HuggingFace 格式的模型转换为 TensorRT Engine
- 如何使用 Triton Inference Server 部署模型服务
- 完整的模型推理服务部署流程

**整体流程**：

```
🤗 HuggingFace Model → 🔄 FasterTransformer → ⚡ TensorRT Engine → 🖥️ Triton Server → 🌐 API 服务
```

***

## 📋 目录

- [简介](#简介)
- [版本信息](#版本信息)
- [Step 1：模型转换到 Engine](#step-1模型转换到-engine)
- [Step 2：Engine 部署到 Triton Server](#step-2engine-部署到-triton-server)
- [性能测试结果](#性能测试结果)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

---

## ℹ️ 版本信息

| 组件 | 版本 |
|------|------|
| 🧠 TensorRT-LLM（源码） | 1.0.0 |
| 🐳 TensorRT-LLM（镜像） | 1.0.0 |
| 🖥️ Triton Server | 25.09-trtllm-python-py3 |
| 📦 Backends | TensorRT-LLM源码中有 |

> **⚠️ 注意**：需要注意 tensorrt-llm 的版本一致性。

**版本匹配要求**：确保以下组件版本一致

- 🟢 tensorrt_llm: 1.0.0

---

## ⚙️ Step 1：模型转换到 Engine

### 1.1 准备模型

从 ModelScope 或 HuggingFace 下载 Llama3-8B 模型。

模型路径示例：`/root/Meta-Llama-3-8B-Instruct`

### 1.2 克隆 TensorRT-LLM 源代码

GitHub: <https://github.com/NVIDIA/TensorRT-LLM>

```bash
git clone --branch v1.0.0 https://github.com/NVIDIA/TensorRT-LLM.git TensorRT-LLM-1.0.0
```

# 1.3 拉取镜像并启动容器

```bash
# 拉取镜像
docker image pull nvcr.io/nvidia/tensorrt-llm/release:1.0.0

# 查看镜像，确认已经下载完成
docker image ls
```

启动容器：

```bash
docker run --rm \
-it \
--name=tensorrt-llm \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--gpus=all \
-v /mnt/sda/tensorrt-llm/TensorRT-LLM-1.0.0:/root/TensorRT-LLM \
-v /mnt/sda/tensorrt-llm/Meta-Llama-3-8B-Instruct:/root/llms/Meta-Llama-3-8B-Instruct \
-p 8000:8000 \
nvcr.io/nvidia/tensorrt-llm/release:1.0.0
```

### 1.4 配置 GPU 支持（确保容器中可以使用GPU)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

### 1.5 Transformer → FasterTransformer 格式转换

```bash
cd /root/TensorRT-LLM/examples

python ./models/core/llama/convert_checkpoint.py \
--model_dir /root/llms/Meta-Llama-3-8B-Instruct \
--output_dir ./models/core/llama/tllm_checkpoint \
--dtype float16 \
--use_weight_only \
--weight_only_precision int8
```

### 1.6 FasterTransformer → TensorRT Engine

```bash
cd /root/TensorRT-LLM/examples

trtllm-build \
--checkpoint_dir ./models/core/llama/tllm_checkpoint \
--output_dir ./models/core/llama/trt_engines \
--gemm_plugin float16 \
--max_num_tokens 8012 \
--max_batch_size 64 \
--max_input_len 512 \
--remove_input_padding enable
```

### 1.7 测试 Engine

```bash
cd /root/TensorRT-LLM/examples

python run.py \
--input_text "write a story" \
--max_output_len 500 \
--tokenizer_dir /root/llms/Meta-Llama-3-8B-Instruct \
--engine_dir ./models/core/llama/trt_engines
```

```bash
python run.py \
  --input_text "write a story" \
  --max_output_len 500 \
  --tokenizer_dir /root/llms/Meta-Llama-3-8B-Instruct \
  --engine_dir ./models/core/llama/trt_engines \
  --run_profiling
```
---

## 🖥️ Step 2：Engine 部署到 Triton Server

### 2.1 部署方式

| 方式            | 说明                  | 适用场景      |
| ------------- | ------------------- | --------- |
| trtllm-serve  | TensorRT-LLM 官方部署平台 | 快速测试、简单部署 |
| Triton Server | 生产环境部署              | 工业界标准部署方式 |

#### 方式一：在 tensorrt-llm 容器中快速部署(注意\*\*⚠️\*\*：当前1.0.0版本启动服务有点问题)

启动服务：

```bash
trtllm-serve /trt_engines/ --tokenizer /Owen2.5-0.5B-nstruct/
```

使用 HTTP 方式访问（遵循 OpenAI API 规范）：

```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "/Owen2.5-0.5B-Instruct",
  "prompt": "大模型为什么这么厉害?",
  "max_tokens": 16,
  "temperature": 0
}'
```

> **总结**：部署简单，工业界部署不常见

#### 方式二：Triton Server 部署

##### 拉取镜像

```bash
docker pull nvcr.io/nvidia/tritonserver:25.09-trtllm-python-py3
```

##### 启动容器

```bash
docker run \
-it \
--name=triton_server \
--gpus=all \
--net=host \
--shm-size=2g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v /mnt/sda/tensorrt-llm/Meta-Llama-3-8B-Instruct:/root/Meta-Llama-3-88-Instruct \
-v /mnt/sda/tensorrt-llm/TensorRT-LLM-1.0.0:/root/TensorRT-LLM \
-v /mnt/sda/tensorrt-llm/TensorRT-LLM-1.0.0/triton_backend:/root/tensorrtllm_backend \
nvcr.io/nvidia/tritonserver:25.09-trtllm-python-py3
```

<br />

### 2.2 修改配置文件

> **说明**：需要替换配置文件中的占位符，配置文件位于 `/root/tensorrtllm_backend/all_models/inflight_batcher_llm` 目录下。

需要修改的文件：

- ensemble：模型编织
- preprocessing：前处理：
- postprocessing：后处理：
- tensorrt\_llm：模型：
- tensorrt\_llm\_bls

#### 步骤 1：创建配置脚本

在容器内进入 `tensorrt_backend` 目录，创建 `fill_configpbtxt.sh` 脚本：

```bash
cd /root/tensorrtllm_backend

# 需要对应修改：ENGINE_DIR，TOKENIZER_DIR，MODEL_FOLDER，FILL_TEMPLATE_SCRIPT（/root/tensorrtllm_backend/tools/fill_template.py）
```

```bash
ENGINE_DIR=/root/TensorRT-LLM/examples/models/core/llama/trt_engines
TOKENIZER_DIR=/root/Meta-Llama-3-88-Instruct
MODEL_FOLDER=/root/tensorrtllm_backend/all_models/inflight_batcher_llm
TRITON_MAX_BATCH_SIZE=4
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=0
MAX_QUEUE_SIZE=0
FILL_TEMPLATE_SCRIPT=/root/tensorrtllm_backend/tools/fill_template.py
DECOUPLED_MODE=false
LOGITS_DATATYPE=TYPE_FP32

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}
```

#### 步骤 2：运行配置脚本

```bash
cd /root/tensorrtllm_backend
sh fill_configpbtxt.sh
```

### 2.3 启动 Triton Server

```bash
cd /root/tensorrtllm_backend
python3 scripts/launch_triton_server.py \
--world_size=1 \
--model_repo=all_models/inflight_batcher_llm
```

### 2.4 测试服务

```bash
curl -X POST localhost:8000/v2/models/ensemble/generate \
-d '{
  "text_input": "什么是人工智能?",
  "max_tokens": 20,
  "bad_words": "",
  "stop_words": ""
}'
```

---

## 📊 性能测试结果

> 💡 **测试环境**
> - 模型：Meta-Llama-3-8B-Instruct
> - GPU：NVIDIA RTX 3090 Ti (24GB)
> - TensorRT-LLM 版本：1.0.0

---

### 📦 模型大小对比

| 模型类型 | 大小 | 备注 |
|----------|------|------|
| 🤗 HuggingFace 原始 (FP16) | 15.1 GB | 4个safetensors文件 |
| 💾 INT8 Checkpoint | 8.5 GB | 量化后 |
| ⚡ TensorRT Engine | 8.5 GB | 优化后 |
| ✨ **节省空间** | **44%** | - |

---

### ⚡ Token 生成速度对比

| 部署方式 | Token 速度 | 500 tokens 耗时 | 提升 |
|----------|-----------|-----------------|------|
| 🤗 HuggingFace 原生 | 44.25 tokens/s | ~11.3 秒 | - |
| 🚀 TensorRT-LLM 直接运行 | ~101 tokens/s | ~4.9 秒 | **2.3x** |
| 🖥️ Triton Server | ~100.8 tokens/s | ~5.0 秒 | **2.3x** |

---

### 🎯 关键发现

- ✅ **TensorRT-LLM 提速约 2.3 倍** 相比 HuggingFace 原生模型
- ✅ **模型体积减少 44%**（INT8 量化）
- ✅ Triton Server 与直接运行性能基本一致
- ✅ 启用 Paged KV Cache 和 Context FMHA 显著提升性能

---

## 📚 参考资料

### 官方文档

- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)

### 镜像地址

- TensorRT-LLM 镜像：`nvcr.io/nvidia/tensorrt-llm/release:1.0.0`
- Triton Server 镜像：`nvcr.io/nvidia/tritonserver:25.09-trtllm-python-py3`

### 学习视频

- [TensorRT-LLM 实用指南 - Llama3 模型推理加速](https://www.bilibili.com/video/BV19ec2zzEeG)（来源：哔哩哔哩，up主：程序员-鲁哥）

***

