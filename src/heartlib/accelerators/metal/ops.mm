#include <torch/extension.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <string>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace fs = std::filesystem;

namespace {

// Must match `RMSNormParams` in `rmsnorm.metal`.
struct RMSNormParams {
  uint32_t d_model;
  float eps;
  uint32_t stride_row;
};

// Must match `RoPEParams` in `rope.metal`.
struct RoPEParams {
  uint32_t d_model;
  uint32_t rot_dim;
  uint32_t half_rot;
  uint32_t seq_len;
};

constexpr NSUInteger kThreadsPerThreadgroup = 256;

static id<MTLLibrary> g_lib = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_fwd_inv = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_fwd_inv_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight_fwd_inv = nil;
static id<MTLComputePipelineState> g_pipeline_rmsnorm_noweight_fwd_inv_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rope = nil;
static id<MTLComputePipelineState> g_pipeline_rope_fp32 = nil;
static id<MTLComputePipelineState> g_pipeline_rope_bwd = nil;
static id<MTLComputePipelineState> g_pipeline_rope_bwd_fp32 = nil;
static std::mutex g_pipeline_mutex;

static std::string metallib_path_for_this_module() {
  Dl_info info;
  if (dladdr((void*)&metallib_path_for_this_module, &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  fs::path so_path(info.dli_fname);
  fs::path lib_path = so_path.parent_path() / "heartlib_ops.metallib";
  return lib_path.string();
}

static void ensure_library_locked(id<MTLDevice> device) {
  if (g_lib != nil) {
    return;
  }

  const std::string lib_path = metallib_path_for_this_module();
  TORCH_CHECK(!lib_path.empty(), "heartlib_metal_ops: failed to locate extension path via dladdr()");

  NSString* ns_path = [NSString stringWithUTF8String:lib_path.c_str()];
  NSURL* url = [NSURL fileURLWithPath:ns_path];
  NSError* err = nil;
  g_lib = [device newLibraryWithURL:url error:&err];
  if (g_lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "heartlib_metal_ops: failed to load metallib at ", lib_path, ": ", msg);
  }
}

static id<MTLComputePipelineState> ensure_pipeline(
    id<MTLDevice> device,
    id<MTLComputePipelineState> __strong* pipeline,
    const char* fn_name) {
  std::lock_guard<std::mutex> lock(g_pipeline_mutex);
  ensure_library_locked(device);

  if (*pipeline != nil) {
    return *pipeline;
  }

  NSString* ns_fn = [NSString stringWithUTF8String:fn_name];
  id<MTLFunction> fn = [g_lib newFunctionWithName:ns_fn];
  TORCH_CHECK(fn != nil, "heartlib_metal_ops: function `", fn_name, "` not found in metallib");

  NSError* err = nil;
  *pipeline = [device newComputePipelineStateWithFunction:fn error:&err];
  if (*pipeline == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown error";
    TORCH_CHECK(false, "heartlib_metal_ops: failed to create compute pipeline: ", msg);
  }

  TORCH_CHECK(
      (*pipeline).maxTotalThreadsPerThreadgroup >= kThreadsPerThreadgroup,
      "heartlib_metal_ops: pipeline maxTotalThreadsPerThreadgroup (",
      (int)(*pipeline).maxTotalThreadsPerThreadgroup,
      ") < expected threads (",
      (int)kThreadsPerThreadgroup,
      ")");
  return *pipeline;
}

static inline id<MTLBuffer> storage_as_mtlbuffer(const at::Tensor& t) {
  const auto& dp = t.storage().data_ptr();
  void* ctx = dp.get_context();
  TORCH_CHECK(
      ctx != nullptr,
      "heartlib_metal_ops: expected MPS storage to provide an MTLBuffer context (got null).");
  return (__bridge id<MTLBuffer>)ctx;
}

static inline NSUInteger storage_offset_bytes(const at::Tensor& t) {
  return (NSUInteger)(t.storage_offset() * (int64_t)t.element_size());
}

torch::Tensor rmsnorm(
    at::Tensor x,
    at::Tensor weight,
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "rmsnorm: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "rmsnorm: weight must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rmsnorm: x must be fp16 or fp32");
  TORCH_CHECK(weight.dtype() == x.dtype(), "rmsnorm: weight dtype must match x");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "rmsnorm: weight must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "rmsnorm: weight must have numel == x.size(-1)");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm: x.numel must be divisible by D");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_fp32, "rmsnorm_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm, "rmsnorm_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm: failed to get MTLComputeCommandEncoder from MPS stream");

  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(out, 2);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:3];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  return out;
}

torch::Tensor rmsnorm_noweight(
    at::Tensor x,
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_noweight: x must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rmsnorm_noweight: x must be fp16 or fp32");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_noweight: x must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm_noweight: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_noweight: invalid last dim");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_noweight: x.numel must be divisible by D");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_noweight_fp32, "rmsnorm_noweight_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_noweight, "rmsnorm_noweight_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_noweight: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_noweight: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_noweight: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(out, 1);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:2];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  return out;
}

std::vector<torch::Tensor> rmsnorm_forward_with_inv(
    at::Tensor x,
    at::Tensor weight,
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_forward_with_inv: x must be on MPS");
  TORCH_CHECK(weight.device().is_mps(), "rmsnorm_forward_with_inv: weight must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rmsnorm_forward_with_inv: x must be fp16 or fp32");
  TORCH_CHECK(weight.dtype() == x.dtype(), "rmsnorm_forward_with_inv: weight dtype must match x");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_forward_with_inv: x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "rmsnorm_forward_with_inv: weight must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm_forward_with_inv: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_forward_with_inv: invalid last dim");
  TORCH_CHECK(weight.numel() == D, "rmsnorm_forward_with_inv: weight must have numel == x.size(-1)");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_forward_with_inv: x.numel must be divisible by D");
  auto inv = torch::empty({rows}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_fwd_inv_fp32, "rmsnorm_fwd_inv_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_fwd_inv, "rmsnorm_fwd_inv_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_forward_with_inv: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_forward_with_inv: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_forward_with_inv: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(weight, 1);
  set_tensor(out, 2);
  set_tensor(inv, 3);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  return {out, inv};
}

std::vector<torch::Tensor> rmsnorm_noweight_forward_with_inv(
    at::Tensor x,
    double eps) {
  TORCH_CHECK(x.device().is_mps(), "rmsnorm_noweight_forward_with_inv: x must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rmsnorm_noweight_forward_with_inv: x must be fp16 or fp32");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_noweight_forward_with_inv: x must be contiguous");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm_noweight_forward_with_inv: x must have dim >= 1");

  const int64_t D = x.size(-1);
  TORCH_CHECK(D > 0, "rmsnorm_noweight_forward_with_inv: invalid last dim");

  auto out = torch::empty_like(x);
  const int64_t rows = x.numel() / D;
  TORCH_CHECK(rows * D == x.numel(), "rmsnorm_noweight_forward_with_inv: x.numel must be divisible by D");
  auto inv = torch::empty({rows}, x.options());

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rmsnorm_noweight_fwd_inv_fp32, "rmsnorm_noweight_fwd_inv_fp32")
      : ensure_pipeline(device, &g_pipeline_rmsnorm_noweight_fwd_inv, "rmsnorm_noweight_fwd_inv_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rmsnorm_noweight_forward_with_inv: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rmsnorm_noweight_forward_with_inv: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rmsnorm_noweight_forward_with_inv: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(out, 1);
  set_tensor(inv, 2);

  RMSNormParams params;
  params.d_model = (uint32_t)D;
  params.eps = (float)eps;
  params.stride_row = (uint32_t)D;
  [encoder setBytes:&params length:sizeof(RMSNormParams) atIndex:3];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)rows, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  return {out, inv};
}

torch::Tensor rope(
    at::Tensor x,
    at::Tensor cos,
    at::Tensor sin,
    int64_t rot_dim) {
  TORCH_CHECK(x.device().is_mps(), "rope: x must be on MPS");
  TORCH_CHECK(cos.device().is_mps(), "rope: cos must be on MPS");
  TORCH_CHECK(sin.device().is_mps(), "rope: sin must be on MPS");
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "rope: x must be fp16 or fp32");
  TORCH_CHECK(cos.dtype() == x.dtype(), "rope: cos dtype must match x");
  TORCH_CHECK(sin.dtype() == x.dtype(), "rope: sin dtype must match x");
  TORCH_CHECK(x.is_contiguous(), "rope: x must be contiguous");
  TORCH_CHECK(cos.is_contiguous(), "rope: cos must be contiguous");
  TORCH_CHECK(sin.is_contiguous(), "rope: sin must be contiguous");
  TORCH_CHECK(x.dim() == 4, "rope: x must be (B,H,T,D)");
  TORCH_CHECK(cos.dim() == 2, "rope: cos must be (T, rot/2)");
  TORCH_CHECK(sin.dim() == 2, "rope: sin must be (T, rot/2)");
  TORCH_CHECK(rot_dim > 0, "rope: rot_dim must be > 0");
  TORCH_CHECK((rot_dim % 2) == 0, "rope: rot_dim must be even");

  const int64_t B = x.size(0);
  const int64_t H = x.size(1);
  const int64_t T = x.size(2);
  const int64_t D = x.size(3);
  TORCH_CHECK(rot_dim <= D, "rope: rot_dim must be <= head_dim");

  const int64_t half_rot = rot_dim / 2;
  TORCH_CHECK(cos.size(0) == T && cos.size(1) == half_rot, "rope: cos shape mismatch");
  TORCH_CHECK(sin.size(0) == T && sin.size(1) == half_rot, "rope: sin shape mismatch");

  auto out = torch::empty_like(x);
  const int64_t n_vec = B * H * T;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (x.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rope_fp32, "rope_fp32")
      : ensure_pipeline(device, &g_pipeline_rope, "rope_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rope: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rope: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rope: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(x, 0);
  set_tensor(cos, 1);
  set_tensor(sin, 2);
  set_tensor(out, 3);

  RoPEParams params;
  params.d_model = (uint32_t)D;
  params.rot_dim = (uint32_t)rot_dim;
  params.half_rot = (uint32_t)half_rot;
  params.seq_len = (uint32_t)T;
  [encoder setBytes:&params length:sizeof(RoPEParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)n_vec, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  return out;
}

torch::Tensor rope_backward(
    at::Tensor grad_y,
    at::Tensor cos,
    at::Tensor sin,
    int64_t rot_dim) {
  TORCH_CHECK(grad_y.device().is_mps(), "rope_backward: grad_y must be on MPS");
  TORCH_CHECK(cos.device().is_mps(), "rope_backward: cos must be on MPS");
  TORCH_CHECK(sin.device().is_mps(), "rope_backward: sin must be on MPS");
  TORCH_CHECK(grad_y.dtype() == at::kHalf || grad_y.dtype() == at::kFloat, "rope_backward: grad_y must be fp16 or fp32");
  TORCH_CHECK(cos.dtype() == grad_y.dtype(), "rope_backward: cos dtype must match grad_y");
  TORCH_CHECK(sin.dtype() == grad_y.dtype(), "rope_backward: sin dtype must match grad_y");
  TORCH_CHECK(grad_y.is_contiguous(), "rope_backward: grad_y must be contiguous");
  TORCH_CHECK(cos.is_contiguous(), "rope_backward: cos must be contiguous");
  TORCH_CHECK(sin.is_contiguous(), "rope_backward: sin must be contiguous");
  TORCH_CHECK(grad_y.dim() == 4, "rope_backward: grad_y must be (B,H,T,D)");
  TORCH_CHECK(rot_dim > 0, "rope_backward: rot_dim must be > 0");
  TORCH_CHECK((rot_dim % 2) == 0, "rope_backward: rot_dim must be even");

  const int64_t B = grad_y.size(0);
  const int64_t H = grad_y.size(1);
  const int64_t T = grad_y.size(2);
  const int64_t D = grad_y.size(3);
  TORCH_CHECK(rot_dim <= D, "rope_backward: rot_dim must be <= head_dim");

  const int64_t half_rot = rot_dim / 2;
  TORCH_CHECK(cos.size(0) == T && cos.size(1) == half_rot, "rope_backward: cos shape mismatch");
  TORCH_CHECK(sin.size(0) == T && sin.size(1) == half_rot, "rope_backward: sin shape mismatch");

  auto grad_x = torch::empty_like(grad_y);
  const int64_t n_vec = B * H * T;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = (grad_y.dtype() == at::kFloat)
      ? ensure_pipeline(device, &g_pipeline_rope_bwd_fp32, "rope_bwd_fp32")
      : ensure_pipeline(device, &g_pipeline_rope_bwd, "rope_bwd_fp16");

  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  TORCH_CHECK(stream != nullptr, "rope_backward: failed to get current MPS stream");
  id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
  TORCH_CHECK(encoder != nil, "rope_backward: failed to get MTLComputeCommandEncoder from MPS stream");
  [encoder setComputePipelineState:pipeline];

  auto set_tensor = [&](const at::Tensor& t, int idx) {
    id<MTLBuffer> buf = storage_as_mtlbuffer(t);
    TORCH_CHECK(buf != nil, "rope_backward: tensor has null MTLBuffer");
    const NSUInteger off = storage_offset_bytes(t);
    [encoder setBuffer:buf offset:off atIndex:(NSUInteger)idx];
  };

  set_tensor(grad_y, 0);
  set_tensor(cos, 1);
  set_tensor(sin, 2);
  set_tensor(grad_x, 3);

  RoPEParams params;
  params.d_model = (uint32_t)D;
  params.rot_dim = (uint32_t)rot_dim;
  params.half_rot = (uint32_t)half_rot;
  params.seq_len = (uint32_t)T;
  [encoder setBytes:&params length:sizeof(RoPEParams) atIndex:4];

  const MTLSize tg = MTLSizeMake(kThreadsPerThreadgroup, 1, 1);
  const MTLSize grid = MTLSizeMake((NSUInteger)n_vec, 1, 1);
  [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
  return grad_x;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm, "RMSNorm Forward (Metal/MPS)");
  m.def("rmsnorm_noweight", &rmsnorm_noweight, "RMSNorm Forward (no weight, Metal/MPS)");
  m.def("rmsnorm_forward_with_inv", &rmsnorm_forward_with_inv, "RMSNorm Forward with inv cache (Metal/MPS)");
  m.def("rmsnorm_noweight_forward_with_inv", &rmsnorm_noweight_forward_with_inv, "RMSNorm Forward (no weight) with inv cache (Metal/MPS)");
  m.def("rope", &rope, "RoPE Apply (Metal/MPS)");
  m.def("rope_backward", &rope_backward, "RoPE Backward (Metal/MPS)");
}

