# Awesome Low-GPU ⚡

![Awesome](https://awesome.re/badge.svg)
![License: CC0](https://img.shields.io/badge/License-CC0-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

A curated list of open-source codebases and practical methodologies that reduce memory requirements or make machine-learning inference/training possible on consumer GPUs, integrated GPUs, CPUs, Apple Silicon, edge devices, or heterogeneous systems.

> **Goal**: Democratize AI by making it accessible on consumer hardware through quantization, offloading, optimization, and efficient inference techniques.

## What Counts as "Low-GPU"?

A project belongs here if it does at least one of the following:

- reduces model-weight memory through quantization, sparsity, or compression;
- reduces KV-cache, activation, optimizer-state, or attention memory;
- offloads weights/KV/compute to CPU RAM, disk/NVMe, unified memory, or another device;
- streams layers or MoE experts so the entire model does not need to reside on the GPU;
- enables sequential model swapping or memory-aware multi-model pipelines;
- provides an efficient CPU/native/edge runtime that removes the need for a discrete high-end GPU;
- pools multiple smaller devices to run a model that does not fit on one device;
- provides profiling/capacity-planning tools specifically useful for constrained-memory deployment.

General-purpose serving engines are included only when they expose relevant memory-saving features.

## Contents

- [Native and Low-Overhead Runtimes](#native-and-low-overhead-runtimes)
- [Local LLM Inference Runtimes](#local-llm-inference-runtimes)
- [Serving Engines with Memory-Saving Features](#serving-engines-with-memory-saving-features)
- [Quantization and Compression](#quantization-and-compression)
- [Offloading, Streaming, and Model Swapping](#offloading-streaming-and-model-swapping)
- [KV-Cache and Long-Context Memory](#kv-cache-and-long-context-memory)
- [Memory-Efficient Fine-Tuning and Training](#memory-efficient-fine-tuning-and-training)
- [Low-VRAM Generative Media](#low-vram-generative-media)
- [Speech and Audio](#speech-and-audio)
- [Distributed and Collaborative Inference](#distributed-and-collaborative-inference)
- [Hardware-Specific and Edge Deployment](#hardware-specific-and-edge-deployment)
- [Rust-Native Frameworks and Runtimes](#rust-native-frameworks-and-runtimes)
- [Compressed Model Formats](#compressed-model-formats)
- [Benchmarking and Capacity Planning](#benchmarking-and-capacity-planning)
- [Research, Experimental, and Legacy Projects](#research-experimental-and-legacy-projects)
- [Contributing](#contributing)

---

## Native and Low-Overhead Runtimes

Native runtimes can reduce framework/runtime overhead and often provide memory mapping, quantized kernels, and CPU/GPU hybrid execution.

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** - C/C++ LLM inference runtime and the primary GGUF ecosystem. Supports CPU inference, GPU acceleration/offload across multiple backends, memory mapping, and many low-bit quantization types.

- **[llamafile](https://github.com/mozilla-ai/llamafile)** - Packages llama.cpp-based models and runtime components into portable executables using Cosmopolitan Libc.

- **[BitNet](https://github.com/microsoft/BitNet)** - Microsoft's official inference framework for 1-bit/1.58-bit BitNet-family LLMs, designed around very low-bit weights.

- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** - Pure C/C++ inference for diffusion and related generative models, including Stable Diffusion, FLUX, Wan, Qwen Image, and others; supports GGUF/quantization and multiple compute backends.

- **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** - C/C++ implementation of Whisper with quantization and CPU/GPU acceleration for local speech recognition.

- **[llama2.c](https://github.com/karpathy/llama2.c)** - Minimal educational Llama 2 inference implementation in C.

- **[llm.c](https://github.com/karpathy/llm.c)** - Educational C/CUDA implementation focused on LLM training, especially GPT-2-style models.

---

## Local LLM Inference Runtimes

- **[Ollama](https://github.com/ollama/ollama)** - Local model runner and API server with model management and support for quantized local models.

- **[KoboldCpp](https://github.com/LostRuins/koboldcpp)** - Easy-to-run local text-generation application built around llama.cpp-derived backends, with GGUF support and CPU/GPU offloading.

- **[text-generation-webui](https://github.com/oobabooga/textgen)** - Extensible local LLM interface supporting multiple loaders/backends and quantization formats. Its low-VRAM behavior depends on the selected backend.

- **[LocalAI](https://github.com/mudler/LocalAI)** - Open-source local AI API layer supporting CPU-only and multiple GPU backends across text, vision, audio, image, and video workloads.

- **[GPT4All](https://github.com/nomic-ai/gpt4all)** - Open-source local LLM application and SDK focused on desktop/CPU-friendly model execution.

- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3)** - Optimized quantization and inference library for LLMs on modern consumer GPUs, including EXL3 low-bit quantization and cache quantization.

- **[mistral.rs](https://github.com/EricLBuehler/mistral.rs)** - Rust inference runtime with GGUF/GPTQ/AWQ/HQQ/other quantization support, hardware-aware tuning, PagedAttention, and multiple accelerator backends.

- **[CTranslate2](https://github.com/OpenNMT/CTranslate2)** - Efficient C++/Python Transformer inference runtime using weight quantization, layer fusion, reduced precision, and CPU/GPU optimizations.

- **[MLC LLM](https://github.com/mlc-ai/mlc-llm)** - Compilation-based deployment stack for LLMs across GPUs, CPUs, mobile devices, and WebGPU-capable browsers.

- **[ExecuTorch](https://github.com/pytorch/executorch)** - PyTorch's edge/on-device inference runtime for mobile, embedded, and other resource-constrained targets.

---

## Serving Engines with Memory-Saving Features

These projects are primarily optimized for throughput or production serving rather than for the smallest possible GPU. They are still relevant because they implement quantization, paged KV-cache management, offloading, or other memory controls.

- **[vLLM](https://github.com/vllm-project/vllm)** - High-throughput LLM serving engine. PagedAttention and quantization support can improve KV-cache utilization and model fit, but vLLM is not a low-VRAM-first runtime.

- **[SGLang](https://github.com/sgl-project/sglang)** - High-performance serving/runtime stack with quantization, cache management, structured generation, and distributed execution features.

- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** - NVIDIA inference stack with low-precision kernels, quantization, paged KV caching, and multi-GPU support. Best suited to supported NVIDIA hardware.

---

## Quantization and Compression

### General LLM quantization

- **[GPTQModel](https://github.com/ModelCloud/GPTQModel)** - Actively developed GPTQ-centered quantization and inference toolkit with support for multiple quantization schemes and hardware-accelerated backends. It is the recommended modern replacement for archived AutoGPTQ.

- **[AWQ / llm-awq](https://github.com/mit-han-lab/llm-awq)** - Activation-aware weight quantization for low-bit LLM inference, commonly used at 4-bit.

- **[HQQ](https://github.com/dropbox/hqq)** - Calibration-free half-quadratic quantization supporting low-bit weights and integrations with common inference stacks.

- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** - 8-bit and 4-bit quantization primitives widely used for low-memory inference and QLoRA training.

- **[torchao](https://github.com/pytorch/ao)** - PyTorch-native quantization and sparsity tooling for training and inference.

- **[AutoRound](https://github.com/intel/auto-round)** - Low-bit post-training quantization toolkit for LLMs with CPU/XPU/CUDA support and integrations with common inference engines.

- **[LLM Compressor](https://github.com/vllm-project/llm-compressor)** - Transformers-compatible library for applying quantization/compression recipes for deployment, especially in the vLLM ecosystem.

- **[compressed-tensors](https://github.com/vllm-project/compressed-tensors)** - Safetensors extension/schema for storing sparse and quantized tensors and their compression metadata.

- **[ExLlamaV3 / EXL3](https://github.com/turboderp-org/exllamav3)** - Consumer-GPU-focused inference plus variable-bit EXL3 quantization.

### Diffusion / generative-model quantization

- **[Nunchaku](https://github.com/nunchaku-ai/nunchaku)** - High-performance 4-bit neural-network inference engine implementing SVDQuant. The project reports major memory reductions for FLUX-class diffusion models and supports CPU offloading for even lower VRAM configurations.

- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** - ComfyUI nodes for running transformer/DiT components and text encoders from quantized GGUF files on lower-end GPUs.

---

## Offloading, Streaming, and Model Swapping

This category is central to the repository: these tools deliberately trade PCIe, CPU, unified-memory, disk/NVMe bandwidth, and latency for lower accelerator-memory requirements. Techniques include layer streaming, expert streaming, CPU/GPU partitioning, model swapping, and memory-aware execution.

- **[AirLLM](https://github.com/lyogavin/airllm)** - Streams model layers or MoE experts instead of keeping the full model on the GPU. The project reports extremely low VRAM footprints for very large models; substantial disk/system-memory capacity and much lower throughput than fully resident inference should be expected.

- **[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)** - Portable C99 inference engine for the 2.78-trillion-parameter Kimi K3. Streams the dense trunk from disk and keeps routed experts off-RAM, with a reported 8.24 GB peak RSS on an 8 GB machine and no GPU. Requires roughly 1.56 TB for the original checkpoint plus 109 GB for the packed streaming trunk.

- **[BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge)** - MoE inference engine built on llama.cpp that streams only the experts selected for each token directly from flash storage instead of keeping the full model resident. The project reports DeepSeek V4 Flash 284B running CPU-only on a 12 GB phone, with other models above the device's RAM also supported.

- **[WARP](https://github.com/sqliteai/warp)** - Dependency-free C inference engine that keeps a model's shared trunk in RAM while streaming selected MoE experts directly from disk and using remaining RAM as a bounded expert cache. Targets extremely large models such as Kimi K3; the project reports running the full 2.78T model with about 29 GB minimum RAM and roughly 0.45–0.62 tok/s on a 64 GB MacBook Pro.

- **[Pulsar](https://github.com/giannisanni/pulsar)** - Rust + CUDA SSD-streaming inference engine for large MoE models on consumer GPUs. Automatically measures PCIe bandwidth and places attention and frequently used experts according to available GPU memory, while streaming the remaining weights from NVMe. Benchmarks include 295B Hy3 at 6 tok/s and 744B GLM-5.2 at 2.7 tok/s on two 16 GB GPUs.

- **[TurboFieldfare](https://github.com/drumih/turbo-fieldfare)** - Experimental Swift + Metal inference runtime for Gemma 4 26B-A4B on Apple Silicon. Keeps the shared model core and KV cache in memory while streaming routed MoE experts from SSD, enabling the 26B model to run with roughly 2 GB of model/KV memory on an 8 GB Apple Silicon Mac. The project uses 4-bit weights and an SSD-backed expert cache; it is currently model-specific and requires macOS 26, Metal 4, and Swift 6.2+.

- **[KTransformers](https://github.com/kvcache-ai/ktransformers)** - CPU/GPU heterogeneous inference and fine-tuning framework. Particularly relevant to large MoE models because selected operators/experts can run on CPU while other work remains on GPU.

- **[mmgp](https://github.com/deepbeepmeep/mmgp)** - "Memory Management for the GPU Poor" module used by WanGP and related applications. Provides configurable RAM/VRAM profiles, on-the-fly quantization, model slicing, and asynchronous transfers.

- **[Low-GPU-Multi-Model-Pipeline-Builder](https://github.com/kemalariboga/Low-GPU-Multi-Model-Pipeline-Builder)** - Multi-model orchestration layer on top of AirLLM. It loads a requested model, runs inference, unloads it, and then moves to the next model so sequential pipelines can share one constrained GPU instead of keeping all models resident simultaneously.

- **[FlexLLMGen (FlexGen)](https://github.com/FMInference/FlexLLMGen)** - Research system for throughput-oriented LLM inference using GPU/CPU/NVMe offloading and compression. Its famous OPT-175B result used a 16GB GPU plus large CPU RAM and SSD capacity and should not be read as "175B fits in 16GB total memory."

- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** - Includes ZeRO-Offload/ZeRO-Inference techniques for partitioning or offloading model states/weights to CPU and NVMe. Most useful as a framework component rather than a lightweight standalone local runtime.

- **[Hugging Face Accelerate](https://github.com/huggingface/accelerate)** - Device-placement/distributed-execution toolkit used by the Hugging Face ecosystem; useful for mixed-device execution and big-model workflows when paired with Transformers/PEFT/DeepSpeed.

---

## KV-Cache and Long-Context Memory

At long context lengths, KV cache can become a major fraction of runtime memory even when model weights are quantized.

- **[KIVI](https://github.com/jy-yuan/KIVI)** - Research implementation of tuning-free asymmetric 2-bit KV-cache quantization for LLM inference.

- **[KVQuant](https://github.com/SqueezeAILab/KVQuant)** - Research code for low-bit KV-cache quantization aimed at reducing memory needed for long-context inference.

- **[H2O](https://github.com/FMInference/H2O)** - Heavy-Hitter Oracle research project for retaining important KV entries while evicting less useful ones.

- **[StreamingLLM](https://github.com/mit-han-lab/streaming-llm)** - Attention-sink method for streaming long sequences with a bounded cache window rather than retaining the full historical KV cache.

- **[PagedAttention](https://arxiv.org/abs/2309.06180)** - Used by systems such as vLLM to manage KV cache in paged blocks, reducing allocation waste/fragmentation and improving batching. It is a memory-utilization technique, not a guarantee that total KV-cache demand disappears.

- **[TurboQuant](https://github.com/0xsero/turboquant)** - Google's new near-optimal KV cache quantization for LLM inference (3-bit keys, 2-bit values) with Triton kernels + vLLM integration.

- **KV-cache quantization** - Supported in several modern runtimes (including ExLlama-family and other engines) to trade some numerical precision for longer context or more concurrent sequences.

**Prefix caching** stores reusable prompt-prefix state to avoid recomputing it. It can greatly reduce repeated prefill compute, but the cache itself consumes memory/storage; therefore it belongs under compute/throughput optimization rather than pure VRAM reduction.

---

## Memory-Efficient Fine-Tuning and Training

- **[PEFT](https://github.com/huggingface/peft)** - Parameter-efficient fine-tuning library implementing LoRA and many related adapter methods so only a small subset of parameters needs to be trained/stored.

- **[QLoRA](https://github.com/artidoro/qlora)** - Reference implementation/paper code for training LoRA adapters through a frozen 4-bit quantized base model.

- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** - Core 4-bit/8-bit building blocks used by many QLoRA and low-memory optimizer workflows.

- **[Unsloth](https://github.com/unslothai/unsloth)** - Fine-tuning stack focused on reducing training memory and increasing speed for LoRA/QLoRA/full-fine-tuning workflows.

- **[Liger Kernel](https://github.com/linkedin/Liger-Kernel)** - Triton kernels for LLM training/post-training that fuse operations to reduce memory traffic and peak memory while improving throughput.

- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** - IO-aware exact attention implementation with substantially better attention-memory scaling than materializing the full attention matrix.

- **[xFormers](https://github.com/facebookresearch/xformers)** - Optimized Transformer building blocks including memory-efficient attention, widely useful in both language and diffusion workloads.

- **[GaLore](https://github.com/jiaweizzhao/GaLore)** - Gradient low-rank projection for reducing optimizer/gradient memory while retaining full-parameter learning.

---

## Low-VRAM Generative Media

### General

- **[mmgp](https://github.com/deepbeepmeep/mmgp)** - Shared memory/offload module used by several low-VRAM generative-media projects.

### Image generation

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** - Node-based generative-media runtime with explicit execution graphs and model unloading/offloading behavior that can be effective on constrained VRAM.

- **[Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)** - Resource-optimized Stable Diffusion WebUI fork with memory-management and backend improvements.

- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** - Native C/C++ diffusion runtime supporting quantization/GGUF and multiple backends.

- **[Nunchaku](https://github.com/nunchaku-ai/nunchaku)** - 4-bit inference engine for diffusion/DiT models. The project reports FLUX configurations as low as roughly 4 GiB with per-layer CPU offloading and later Qwen-Image configurations around 3 GiB.

- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** - GGUF quantized model/text-encoder loaders for ComfyUI; especially relevant to transformer-based image models such as FLUX.

- **[qwen-image-fast](https://github.com/xlite-dev/qwen-image-fast)** - Acceleration work for Qwen Image-class workloads.

### Video generation

- **[WanGP / Wan2GP](https://github.com/deepbeepmeep/Wan2GP)** - Multi-model generative app specifically oriented toward "GPU poor" systems. The project states that select models can run with as little as 6GB VRAM; requirements vary substantially by model, resolution, duration, and profile.

- **[FramePack](https://github.com/lllyasviel/FramePack)** - Video-generation system that packs temporal context so workload does not grow like a naïve full-history approach. The project reports 13B-model, one-minute generation with a 6GB minimum GPU-memory requirement on supported NVIDIA GPUs.

- **[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)** - Official 8.3B video-generation model/repository positioned for consumer-grade GPU use. It is a smaller model architecture, not itself a general offloading framework.

### 3D generation

- **[Hunyuan3D-2GP](https://github.com/deepbeepmeep/Hunyuan3D-2GP)** - Low-VRAM adaptation of Hunyuan3D 2. The project's model table reports about 6GB for shape generation, while full shape + texture generation can require around 24.5GB; therefore it should not be described as universally "<6GB for mesh and texture."

---

## Speech and Audio

- **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** - Quantized native Whisper inference on CPU and multiple GPU backends.

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** - Whisper inference built on CTranslate2, benefiting from reduced-precision and optimized CPU/GPU execution.

- **[CTranslate2](https://github.com/OpenNMT/CTranslate2)** - Efficient Transformer runtime used by faster-whisper and other speech/NLP applications.

- **[Kokoro](https://github.com/hexgrad/kokoro)** - Small 82M-parameter open-weight TTS model suitable for relatively modest hardware. Actual CPU/GPU requirements depend on the inference implementation.

- **[Piper](https://github.com/OHF-Voice/piper1-gpl)** - Fast local neural TTS with CLI, web, Python, and C/C++ APIs.

---

## Distributed and Collaborative Inference

Pooling devices is another way to overcome per-device memory limits, but network bandwidth/latency can become the new bottleneck.

- **[Petals](https://github.com/bigscience-workshop/petals)** - Distributed/collaborative LLM inference and fine-tuning where model blocks are hosted across machines.

- **[exo](https://github.com/exo-explore/exo)** - Local AI cluster software with automatic device discovery and topology-aware model splitting.

- **[mycoSwarm](https://github.com/msb-msb/mycoSwarm)** - Coordinates AI workloads across heterogeneous consumer devices.

- **[LocalAI](https://github.com/mudler/LocalAI)** - Also includes distributed deployment/routing features in addition to its local backend abstraction.

---

## Hardware-Specific and Edge Deployment

### Apple Silicon

- **[MLX](https://github.com/ml-explore/mlx)** - Apple's array/ML framework for Apple Silicon, designed around the platform's unified-memory architecture.

- **[MLX LM](https://github.com/ml-explore/mlx-lm)** - LLM generation/fine-tuning package built on MLX, with built-in model quantization and distributed inference/fine-tuning support.

- **llama.cpp Metal backend** - Uses Metal acceleration and can split work between CPU and GPU/unified memory on Apple Silicon.

- **[TurboFieldfare](https://github.com/drumih/turbo-fieldfare)** - Model-specific Swift + Metal runtime for Gemma 4 26B-A4B on Apple Silicon. Uses 4-bit weights and SSD-backed MoE expert streaming to minimize resident memory; the project reports roughly 2 GB of resident model/KV memory on an 8 GB M2 MacBook Air. Requires macOS 26 and Metal 4.

### AMD / Vulkan / cross-vendor GPU paths

- **llama.cpp HIP/Vulkan backends** - Provide AMD and cross-vendor accelerator paths; actual model support/performance depends on GPU and driver stack.

- **LocalAI** - Exposes backend options for NVIDIA, AMD/ROCm, Intel, Apple Silicon, Vulkan, and CPU-only deployment.

### Intel CPU/GPU/NPU and edge

- **[OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)** - Generative-AI APIs on OpenVINO Runtime for supported CPU, GPU, and NPU devices.

- **[ExecuTorch](https://github.com/pytorch/executorch)** - On-device PyTorch deployment for mobile/embedded/edge targets.

- **[MLC LLM](https://github.com/mlc-ai/mlc-llm)** - Compiles models for mobile, desktop, and browser/WebGPU deployment.

---

## Rust-Native Frameworks and Runtimes

Rust itself does not guarantee lower VRAM, but native runtimes can avoid Python dependencies and expose efficient backends suitable for local/embedded deployment.

- **[Candle](https://github.com/huggingface/candle)** - Hugging Face's minimalist Rust ML framework with CPU/CUDA/Metal and WASM-oriented use cases.

- **[Burn](https://github.com/tracel-ai/burn)** - Rust deep-learning framework with multiple execution backends.

- **[mistral.rs](https://github.com/EricLBuehler/mistral.rs)** - Rust LLM inference runtime with consumer-GPU quantization, device mapping, and cache-management features.

---

## Compressed Model Formats

A container format is not automatically memory-efficient; the benefit comes when it stores low-bit weights, supports memory mapping, or preserves compression metadata needed by efficient runtimes.

- **GGUF** - llama.cpp ecosystem format supporting metadata, memory-mapped loading, and many quantized tensor types. Widely used for CPU/GPU hybrid local inference.

- **EXL3** - Variable-bit quantized format used by ExLlamaV3 for consumer-GPU LLM inference.

- **compressed-tensors** - Safetensors-compatible compression metadata/storage approach supporting quantized and sparse tensors in the vLLM ecosystem.

- **GPTQ/AWQ checkpoints** - Common low-bit weight layouts/metadata consumed by multiple GPU inference engines. Treat the algorithm and the serialization convention as related but distinct concepts.

---

## Benchmarking and Capacity Planning

- **llama.cpp tools** - `llama-bench` for performance measurements and `llama-perplexity` for model-quality/perplexity checks.

- **[llm-optimizer](https://github.com/Fai305/llm-optimizer)** - Memory-budget planner that can reason about quantization, KV-cache precision, context length, and device/offload placement, then produce runnable configurations/commands.

- **[GPUQuantizer](https://github.com/ayinedjimi/GPUQuantizer)** - Community toolkit for quantizing and benchmarking LLMs across formats on NVIDIA GPUs.

- **vLLM built-in benchmarks** - Prefer the benchmark tooling maintained in the main vLLM repository.

---

## Research, Experimental, and Legacy Projects

These projects are useful historically or experimentally, but should not be presented as the default modern recommendation.

### Research / hardware-specific

- **[PowerInfer](https://github.com/Tiiny-AI/PowerInfer)** - Research system exploiting activation sparsity by placing "hot" neurons on GPU and "cold" neurons on CPU. Headline speedups are model/hardware-specific and depend on supported sparse-activation architectures; do not generalize the maximum figure to arbitrary LLMs.

- **[AdaLLM / NVFP4-on-4090-vLLM](https://github.com/BenChaliah/NVFP4-on-4090-vLLM)** - Experimental NVFP4/FP8 inference runtime targeting Ada Lovelace hardware.

- **[ZipServ](https://github.com/HPMLL/ZipServ_ASPLOS26)** - Research prototype for lossless compression of LLM weights during serving. Its published speed/size figures are benchmark maxima from the prototype, not general guarantees.

### Archived / superseded

- **[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)** - Archived and unmaintained since 2025. Its own README recommends GPTQModel for new model support and fixes.

- **[rhasspy/piper](https://github.com/rhasspy/piper)** - Archived former Piper repository. Development moved to [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl).

### Legacy / limited current model coverage

- **[CTransformers](https://github.com/marella/ctransformers)** - Python bindings around older GGML-era C/C++ Transformer implementations. Still useful for legacy setups, but llama.cpp and newer runtimes have much broader current-model support.

---

## Contributing

Contributions are welcome.

### Criteria for Inclusion

A core-list entry should:

- be open source under an explicit OSI-approved license; if a public repository has no license, label it **License not specified** until that is resolved;
- materially reduce GPU/accelerator-memory requirements or enable useful ML workloads without a high-end discrete GPU;
- have public documentation or reproducible code;
- clearly state the technique used (quantization, offload, streaming, CPU execution, distributed execution, etc.);
- avoid unsupported universal speed/VRAM claims.

Research prototypes and archived projects are welcome when clearly labeled.

### Required information for new entries

When possible, include:

- supported model families/modalities;
- supported hardware/backends;
- minimum tested VRAM and required system RAM;
- quantization/precision used;
- context length, batch size, resolution, or video duration used for the claim;
- whether a benchmark is independently reproduced or only project-reported;
- project status: active, research, experimental, archived, or legacy.

### How to Contribute

1. Fork this repository.
2. Add the project or methodology in the most specific category.
3. Link to a primary source (official repository, documentation, or paper).
4. Use qualified wording for benchmark claims.
5. Submit a pull request.

---

Special thanks to the developers and researchers making modern AI usable on smaller, cheaper, and more diverse hardware.

**Star this repo** ⭐ if you find it useful, and open a PR when you discover another reproducible low-GPU technique or codebase.
