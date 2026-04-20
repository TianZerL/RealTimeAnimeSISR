import cv2
import numpy as np
import pyopencl as cl
from typing import Tuple

from .common import Processor

class USMSharpener(Processor):
    def __init__(self, radius: int = 50, sigma: int = 0):
        super(USMSharpener, self).__init__()

        if radius % 2 == 0:
            radius += 1

        kernel = cv2.getGaussianKernel(radius, sigma)
        self.kernel = np.dot(kernel, kernel.transpose())

    def process_numpy(self, image: np.ndarray, weight: float = 0.5, threshold: int = 10) -> np.ndarray:
        image_type = image.dtype
        is_integer_image = np.issubdtype(image_type, np.integer)
        integer_scale_factor = np.iinfo(image_type).max if is_integer_image else None
    
        if is_integer_image:
            image = image / integer_scale_factor

        blur = cv2.filter2D(image, -1, self.kernel)
        residual = image - blur

        mask = (np.abs(residual) > (threshold / 255)).astype(image.dtype)
        soft_mask = cv2.filter2D(mask, -1, self.kernel)

        sharp = image + weight * residual
        sharp = np.clip(sharp, 0, 1)

        output =  soft_mask * sharp + (1 - soft_mask) * image

        if is_integer_image:
            output = (output * integer_scale_factor).round().astype(image_type)

        return output

class RCASSharpener(Processor):
    def __init__(self, opencl_platform_device_id: Tuple[int, int] = (0, 0)):
        super(RCASSharpener, self).__init__()
        # RCAS kernel ref FSR2: https://github.com/GPUOpen-Effects/FidelityFX-FSR2/blob/1680d1edd5c034f88ebbbb793d8b88f8842cf804/src/ffx-fsr2-api/shaders/ffx_fsr1.h#L592-L771
        self.kernel_string = """
constant sampler_t n_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline float fast_rcpf(float a)
{
    float b = as_float(0x7ef19fffu - as_uint(a));
    return b * (-b * a + 2.0f);
}

kernel void rcas_rgba(
    read_only image2d_t src,
    write_only image2d_t dst,
    const float sharpness
)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float4 b = read_imagef(src, n_sampler, (int2)(x  , y-1));
    float4 d = read_imagef(src, n_sampler, (int2)(x-1, y  ));
    float4 e = read_imagef(src, n_sampler, (int2)(x  , y  ));
    float4 f = read_imagef(src, n_sampler, (int2)(x+1, y  ));
    float4 h = read_imagef(src, n_sampler, (int2)(x  , y+1));

    float bl = dot(b.xyz, (float3)(0.5f, 1.0f, 0.5f));
    float dl = dot(d.xyz, (float3)(0.5f, 1.0f, 0.5f));
    float el = dot(e.xyz, (float3)(0.5f, 1.0f, 0.5f));
    float fl = dot(f.xyz, (float3)(0.5f, 1.0f, 0.5f));
    float hl = dot(h.xyz, (float3)(0.5f, 1.0f, 0.5f));

    float nz = 0.25f * (bl + dl + fl + hl) - el;
    nz = clamp(fabs(nz) * fast_rcpf(fmax(fmax(fmax(bl, dl), fmax(fl, hl)), el) - fmin(fmin(fmin(bl, dl), fmin(fl, hl)), el)), 0.0f, 1.0f);
    nz = -0.5f * nz + 1.0f;

    float4 ring_min = fmin(fmin(b, d), fmin(f, h));
    float4 ring_max = fmax(fmax(b, d), fmax(f, h));

    float4 hit_min = -ring_min / (4.0f * ring_max);
    float4 hit_max = (1.0f - ring_max) / (4.0f * (ring_min - 1.0f));
    float3 lobe = fmax(hit_min, hit_max).xyz;

    float weight = fmax(-(0.25f - (1.0f / 16.0f)), fmin(fmax(fmax(lobe.x, lobe.y), lobe.z), 0.0f)) * exp2(-sharpness) * nz;
    float4 output = (weight * (b + d + f + h) + e) * fast_rcpf(4.0f * weight + 1.0f);

    write_imagef(dst, (int2)(x, y), (float4)(clamp(output, 0.0f, 1.0f).xyz, e.w));
}

kernel void rcas_gray(
    read_only image2d_t src,
    write_only image2d_t dst,
    const float sharpness
)
{
    const int x = get_global_id(0), y = get_global_id(1);
    if(x >= get_image_width(src) || y >= get_image_height(src)) return;

    float b = read_imagef(src, n_sampler, (int2)(x  , y-1)).x;
    float d = read_imagef(src, n_sampler, (int2)(x-1, y  )).x;
    float e = read_imagef(src, n_sampler, (int2)(x  , y  )).x;
    float f = read_imagef(src, n_sampler, (int2)(x+1, y  )).x;
    float h = read_imagef(src, n_sampler, (int2)(x  , y+1)).x;

    float ring_min = fmin(fmin(b, d), fmin(f, h));
    float ring_max = fmax(fmax(b, d), fmax(f, h));

    float nz = 0.25f * (b + d + f + h) - e;
    nz = clamp(fabs(nz) * fast_rcpf(fmax(ring_max, e) - fmin(ring_min, e)), 0.0f, 1.0f);
    nz = -0.5f * nz + 1.0f;

    float hit_min = -ring_min / (4.0f * ring_max);
    float hit_max = (1.0f - ring_max) / (4.0f * (ring_min - 1.0f));

    float weight = fmax(-(0.25f - (1.0f / 16.0f)), fmin(fmax(hit_min, hit_max), 0.0f)) * exp2(-sharpness) * nz;
    float output = (weight * (b + d + f + h) + e) * fast_rcpf(4.0f * weight + 1.0f);

    write_imagef(dst, (int2)(x, y), (float4)(clamp(output, 0.0f, 1.0f), 0.0f, 0.0f, 1.0f));
}
    """

        self.context = cl.create_some_context(answers = list(map(str, opencl_platform_device_id)))
        self.queue = cl.CommandQueue(self.context)
        self.program = cl.Program(self.context, self.kernel_string).build()

        self.rcas_gray_kernel = cl.Kernel(self.program, 'rcas_gray')
        self.rcas_rgba_kernel = cl.Kernel(self.program, 'rcas_rgba')

        self.type_map = {
            np.dtype(np.uint8): cl.channel_type.UNORM_INT8,
            np.dtype(np.uint16): cl.channel_type.UNORM_INT16,
            np.dtype(np.float16): cl.channel_type.HALF_FLOAT,
            np.dtype(np.float32): cl.channel_type.FLOAT,
        }

    def align(self, v: int, n: int) -> int:
        return (v + n - 1) & -n

    def process_numpy(self, image: np.ndarray, sharpness: float = 1.0) -> np.ndarray:
        sharpness = max(1.0 / (sharpness if sharpness else 1e-6) - 1.0, 0.0) 

        h, w = image.shape[0:2]
        range_w = self.align(w, 16)
        range_h = self.align(h, 8)

        channel_num = image.shape[2] if len(image.shape) > 2 else 1
        assert channel_num == 1 or channel_num == 3 or channel_num == 4

        format = cl.ImageFormat(cl.channel_order.R if channel_num == 1 else cl.channel_order.RGBA, self.type_map[image.dtype])
        rcas_kernel = self.rcas_gray_kernel if channel_num == 1 else self.rcas_rgba_kernel

        if channel_num == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        output = np.empty_like(image)

        input_buf = cl.create_image(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.HOST_WRITE_ONLY, format, (w, h))
        output_buf = cl.create_image(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_READ_ONLY, format, (w, h))

        cl.enqueue_copy(self.queue, input_buf, image, origin=(0, 0, 0), region=(w, h, 1), pitches=(image.strides[0],), is_blocking = False)
        rcas_kernel(self.queue, (range_w, range_h), (16, 8), input_buf, output_buf, np.float32(sharpness))
        cl.enqueue_copy(self.queue, output, output_buf, origin=(0, 0, 0), region=(w, h, 1), pitches=(output.strides[0],), is_blocking = True)

        if channel_num == 3:
            output = cv2.cvtColor(output, cv2.COLOR_RGBA2RGB)

        return output
