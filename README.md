# GLMRT

## What

GLMRT is a Rust-based Attention–FFN Disaggregation (AFD) inference engine for
GLM-5.2. It runs the model across one RTX PRO 6000 Blackwell 96 GB coordinator
and four NVIDIA DGX Sparks: the coordinator handles attention, scheduling,
cache, sampling, and the OpenAI-compatible API, while the Sparks run the routed
MoE experts.

It supports the `lukealonso/GLM-5.2-NVFP4` and `nvidia/GLM-5.2-NVFP4` exports,
continuous batching, long context, tool use, reasoning, vision, and plain,
native-MTP, or dSpark decoding. It is built for this particular hardware
layout, but is open source for anyone who wants to adapt it to their own.

## Why

I wanted a high-quality local GLM-5.2 using the hardware I already had, and a
custom AFD engine was the best way to make the large GPU and four Sparks work
together.

I built GLMRT for fun in an agentic process lasting roughly 30 days. One agent
did the implementation; a second was a discussion partner that helped sharpen
my thinking before I gave instructions to the coding agent. The project
started with GPT-5.5 + DeepSeek V4 Pro and later switched to GPT-5.6 Sol +
Grok 4.5, which was a big upgrade.

I am releasing it because it is cool, intelligence should be everywhere, and
it may be useful to someone building a customized inference engine for their
own hardware.

## How to use it

Clone the repository and edit the four Spark host names and network addresses
in `glmrt.config` for your setup. The other defaults can be left alone.

To build both Docker images and run GLMRT:

```bash
./build.sh
./run.sh
```

`build.sh` builds the coordinator image locally, builds the ARM expert image
on the first configured Spark, and copies it to the other Sparks.

To use the prebuilt images instead:

```bash
docker pull ghcr.io/tpurtell/glmrt-coordinator:latest

for host in spark-a spark-b spark-c spark-d; do
  ssh "$host" docker pull ghcr.io/tpurtell/glmrt-spark-expert:latest
done
```

Set the corresponding image names in `glmrt.config`:

```ini
COORDINATOR_DOCKER_INFERENCE=ghcr.io/tpurtell/glmrt-coordinator:latest
SPARK_EXPERT_DOCKER_INFERENCE=ghcr.io/tpurtell/glmrt-spark-expert:latest
```

Then run:

```bash
./run.sh
```

The server exposes an OpenAI-compatible API at
`http://localhost:8000/v1` by default. Model weights are not included; the
selected GLM-5.2 checkpoint must already be present in the Hugging Face cache
on each host.

## Benchmarks

Final release benchmarks will be added after the public images are built and
measured. They will cover scalar decode, dSpark decode, prefill, long context,
and concurrent throughput.

Agents customizing the engine should start with [`DEVELOPER.md`](DEVELOPER.md).
GLMRT is released under the [MIT License](LICENSE).
