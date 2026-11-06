# Built by prepare-image.py from two verified native release artifacts only.
# The pinned base supplies ORT 1.28.2 CUDA12, NVIDIA libraries and healthcheck.
FROM registry.100credit.cn/xdynamic/x86_64/nvidia/ollaya@sha256:af5e33cf6989959a2bfcc19b4dad61d03048f83b1e8baedfcc43ea0f70909544
USER 0:0
COPY --chmod=0755 ollaya /usr/bin/ollaya
COPY --chmod=0755 ollaya /usr/lib/ollaya/ollaya-cuda-runner
COPY --chmod=0755 parity_elephant /usr/libexec/ollaya/parity_elephant
COPY build.json /usr/share/ollaya/elephant-build.json
ENV ORT_DYLIB_PATH=/usr/lib/ollaya/cuda_v12/libonnxruntime.so.1 \
    LD_LIBRARY_PATH=/usr/lib/ollaya/cuda_v12
LABEL org.opencontainers.image.source="https://github.com/MonadKai/ollaya" \
      org.opencontainers.image.revision="@SOURCE@" \
      org.opencontainers.image.version="@VERSION@" \
      org.ollaya.layout="elephant-1-v1"
USER 1000:1000
