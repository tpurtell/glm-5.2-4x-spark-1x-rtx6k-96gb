variable "BASE_IMAGE" {
  default = "nvcr.io/nvidia/pytorch:26.05-py3"
}

group "default" {
  targets = ["oliver", "spark"]
}

target "oliver" {
  dockerfile = "docker/Dockerfile.dev"
  tags = ["glmrt-dev:oliver"]
  args = {
    BASE_IMAGE = BASE_IMAGE
    GLMRT_ROLE = "coordinator"
    CUDA_ARCH = "120"
    TARGET_PLATFORM = "linux/amd64"
  }
  platforms = ["linux/amd64"]
}

target "spark" {
  dockerfile = "docker/Dockerfile.dev"
  tags = ["glmrt-dev:spark"]
  args = {
    BASE_IMAGE = BASE_IMAGE
    GLMRT_ROLE = "expert"
    CUDA_ARCH = "121"
    TARGET_PLATFORM = "linux/arm64"
  }
  platforms = ["linux/arm64"]
}

