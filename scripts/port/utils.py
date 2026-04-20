import torch
import numpy as np

from pathlib import Path

def format_array(name, data, align=32, per_line=8):
    output = [f"alignas({align}) constexpr float {name}[] = {{"]
    for i in range(0, len(data), per_line):
        line = ", ".join(f"{x:+.7f}f" for x in data[i:i+per_line])
        output.append(f"  {line},")
    output.append("};")
    return "\n".join(output)

def write_weights(output_dir, prefix, layout, kernels, biases, alphas):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = layout.upper()
    with open(output_dir / f'{prefix}.txt', 'w') as f:
        if kernels:
            f.write(format_array(f'{prefix}_{layout}_kernels', kernels))
        if biases:
            f.write("\n\n")
            f.write(format_array(f'{prefix}_{layout}_biases', biases))
        if alphas:
            f.write("\n\n")
            f.write(format_array(f'{prefix}_{layout}_alphas', alphas))

def get_torch_weights(model : torch.nn.Module, layout='nhwc'):
    kernels = []
    biases = []
    alphas = []

    if layout == 'nchw':
        axes = [0, 1, 2, 3]
    elif layout == 'nhwc':
        axes = [0, 2 ,3, 1]
    elif layout == 'hwnc':
        axes = [2 ,3, 0, 1]
    elif layout == 'hwcn':
        axes = [2 ,3, 1, 0]
    else:
        raise ValueError('Unsupported layout')

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            weight = layer.weight.detach().cpu().numpy()
            # PyTorch [N, C, H, W] -> C++ [N, H, W, C]
            weight_nhwc = weight.transpose(*axes)
            kernels.extend(weight_nhwc.flatten().tolist())

            if layer.bias is not None:
                biases.extend(layer.bias.detach().cpu().numpy().tolist())
        elif isinstance(layer, torch.nn.PReLU):
            alphas.extend(layer.weight.detach().cpu().numpy().tolist())
        elif isinstance(layer, torch.nn.ConvTranspose2d):
            weight = layer.weight.detach().cpu().numpy()
            weight = weight.transpose(1, 0, 2, 3)  # [out_channels, in_channels, H, W]
            weight_nhwc = weight.transpose(*axes)  # [N, H, W, C]
            kernels.extend(weight_nhwc.flatten().tolist())

            if layer.bias is not None:
                biases.extend(layer.bias.detach().cpu().numpy().tolist())

    return kernels, biases, alphas

def get_torch_weights_numpy(model : torch.nn.Module, layout='nhwc'):
    kernels = []
    biases = []
    alphas = []

    if layout == 'nchw':
        axes = [0, 1, 2, 3]
    elif layout == 'nhwc':
        axes = [0, 2 ,3, 1]
    elif layout == 'hwnc':
        axes = [2 ,3, 0, 1]
    elif layout == 'hwcn':
        axes = [2 ,3, 1, 0]
    else:
        raise ValueError('Unsupported layout')

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            weight = layer.weight.detach().cpu().numpy()
            # PyTorch [N, C, H, W] -> C++ [N, H, W, C]
            weight_nhwc = weight.transpose(*axes)
            kernels.append(weight_nhwc)

            if layer.bias is not None:
                biases.append(layer.bias.detach().cpu().numpy())
        elif isinstance(layer, torch.nn.PReLU):
            alphas.append(layer.weight.detach().cpu().numpy())
        elif isinstance(layer, torch.nn.ConvTranspose2d):
            weight = layer.weight.detach().cpu().numpy()
            weight = weight.transpose(1, 0, 2, 3)  # [out_channels, in_channels, H, W]
            weight_nhwc = weight.transpose(*axes)  # [N, H, W, C]
            kernels.append(weight_nhwc)

            if layer.bias is not None:
                biases.append(layer.bias.detach().cpu().numpy())

    return kernels, biases, alphas

class Identity:
    def __call__(self, v: str, out_group: int):
        return f'{v}'

class ReLU:
    def __call__(self, v: str, out_group: int):
        return f'max({v}, 0.0)'

class LReLU:
    def __init__(self, a: float):
        self.a = a
    def __call__(self, v: str, out_group: int):
        return f'max({v}, {v} * {self.a})'

class PReLU:
    def __init__(self, alphas: np.ndarray):
        self.alphas = alphas
    def __call__(self, v: str, out_group: int):
        alpha = self.alphas[out_group * 4 : out_group * 4 + 4]
        return f'max({v}, vec4(0.0)) + vec4({', '.join(map(str, alpha.flatten()))}) * min({v}, vec4(0.0))'

class ResidualArg:
    def __init__(self, texture: str, scale: float, channels: int) -> None:
        self.texture = texture
        self.scale = scale
        self.channels = channels

    def texture_name(self, out_group: int) -> str:
        return f'{self.texture}_{out_group}' if self.channels >= 4 else self.texture

    def __call__(self, v: str, out_group: int) -> str:
        if self.scale != 1.0:
            residual = f'{v} * {self.scale}'
        else:
            residual = v

        swizzle = '.x' if self.channels == 1 else ''

        return f'{residual} + {self.texture_name(out_group)}_texOff(vec2(0.0, 0.0)){swizzle}'

def conv_to_glsl(
    input_name: str,
    output_name: str,
    num_in_ch: int,
    num_out_ch: int,
    kernels: np.ndarray,
    biases: np.ndarray,
    activation,
    kernel_size: int,
    desc: str,
    residual_args=None,
    factor: float = 1.2,
) -> str:

    if residual_args is None:
        residual_args = []

    if biases is None:
        biases = np.zeros(num_out_ch)

    if num_in_ch < 4:
        input_textures = [input_name]
    else:
        input_textures = [f'{input_name}_{i}' for i in range(num_in_ch // 4)]

    input_bind_lines = [f'//!BIND {tex}' for tex in input_textures]
    input_bind_section = '\n'.join(input_bind_lines)

    indent = '    '
    shader_parts = []
    half_kernel = kernel_size // 2

    for out_group in range(num_out_ch // 4):
        residual_bind_lines = [f'//!BIND {arg.texture_name(out_group)}' for arg in residual_args]
        residual_bind_section = '\n'.join(residual_bind_lines)

        header_lines = [
            f'//!DESC {desc} part {out_group}',
            f'//!WHEN OUTPUT.w LUMA.w / {factor} > OUTPUT.h LUMA.h / {factor} > *',
            f'//!COMPONENTS 4',
            f'//!HOOK LUMA',
            f'//!SAVE {output_name}_{out_group}',
        ]

        if input_bind_section:
            header_lines.append(input_bind_section)

        if residual_bind_section:
            header_lines.append(residual_bind_section)

        header = '\n'.join(header_lines)

        # HWCN
        kernel_group = kernels[:, :, :, out_group * 4 : out_group * 4 + 4]
        bias_group = biases[out_group * 4 : out_group * 4 + 4]

        body_lines = [f'{indent}vec4 result = vec4({', '.join(map(str, bias_group.flatten()))});']

        for tex_idx, tex_name in enumerate(input_textures):
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    weight_data = kernel_group[ky, kx, tex_idx * 4 : tex_idx * 4 + 4, :].flatten()

                    if num_in_ch > 1 and len(weight_data) <= 16:
                        weight_data = np.pad(weight_data, (0, 16 - len(weight_data)), constant_values=0)

                    vec_type = 'vec4' if len(weight_data) == 4 else 'mat4'
                    offset_x = float(kx - half_kernel)
                    offset_y = float(ky - half_kernel)

                    channel_swizzle = '.x' if num_in_ch == 1 else ''

                    body_lines.append(
                        f'{indent}result += {vec_type}({', '.join(map(str, weight_data))}) * {tex_name}_texOff(vec2({offset_x}, {offset_y})){channel_swizzle};'
                    )

        if not isinstance(activation, Identity):
            body_lines.append(f'{indent}result = {activation('result', out_group)};')

        for arg in residual_args:
            body_lines.append(f'{indent}result = {arg('result', out_group)};')

        body_lines.append(f'{indent}return result;')

        shader_parts.append(f'{header}\nvec4 hook() {{\n{chr(10).join(body_lines)}\n}}\n')

    return '\n'.join(shader_parts) + '\n\n'

def pixelsuffle_4to1_to_glsl(input_name: str, desc: str, factor: float = 1.2):
    texture = input_name + '_0'

    header_lines = [
        f'//!DESC {desc}',
        f'//!WHEN OUTPUT.w LUMA.w / {factor} > OUTPUT.h LUMA.h / {factor} > *',
        f'//!COMPONENTS 1',
        f'//!WIDTH LUMA.w 2 *',
        f'//!HEIGHT LUMA.h 2 *',
        f'//!HOOK LUMA',
        f'//!BIND {texture}',
    ]

    header = '\n'.join(header_lines)

    shader =  f'''{header}
vec4 hook() {{
    vec2 f0 = fract({texture}_pos * {texture}_size);
    ivec2 i0 = ivec2(f0 * vec2(2.0));
    float c0 = {texture}_tex({texture}_pos + (vec2(0.5) - f0) * {texture}_pt)[i0.y * 2 + i0.x];
    return vec4(clamp(c0, 0.0, 1.0), 0.0, 0.0, 1.0);
}}
'''
    return shader

def deconv4x4_8to1_to_glsl(input_name: str, kernels: np.ndarray, desc: str, factor: float = 1.2):
    input_textures = [f'{input_name}_{i}' for i in range(8 // 4)]

    input_bind_lines = [f'//!BIND {tex}' for tex in input_textures]
    input_bind_section = '\n'.join(input_bind_lines)

    header_lines = [
        f'//!DESC {desc}',
        f'//!WHEN OUTPUT.w LUMA.w / {factor} > OUTPUT.h LUMA.h / {factor} > *',
        f'//!COMPONENTS 1',
        f'//!WIDTH LUMA.w 2 *',
        f'//!HEIGHT LUMA.h 2 *',
        f'//!HOOK LUMA',
    ]

    header_lines.append(input_bind_section)

    header = '\n'.join(header_lines)

    shader =  f'''{header}
vec4 hook() {{
    vec2 f0 = fract({input_textures[0]}_pos * {input_textures[0]}_size);
    vec2 p0 = {input_textures[0]}_pos + (vec2(0.5) - f0) * {input_textures[0]}_pt;
    ivec2 i0 = ivec2(f0 * vec2(2.0));

    vec4 r0 = {input_textures[0]}_tex(p0);
    vec4 r1 = {input_textures[1]}_tex(p0);

    float result = 0;

    switch(i0.y * 2 + i0.x)
    {{
    case 0:
        result += dot(vec4({', '.join(map(str, kernels[0, 0, 0:4, 0]))}), r0);
        result += dot(vec4({', '.join(map(str, kernels[0, 0, 4:8, 0]))}), r1);
        break;
    case 1:
        result += dot(vec4({', '.join(map(str, kernels[0, 1, 0:4, 0]))}), r0);
        result += dot(vec4({', '.join(map(str, kernels[0, 1, 4:8, 0]))}), r1);
        break;
    case 2:
        result += dot(vec4({', '.join(map(str, kernels[1, 0, 0:4, 0]))}), r0);
        result += dot(vec4({', '.join(map(str, kernels[1, 0, 4:8, 0]))}), r1);
        break;
    case 3:
        result += dot(vec4({', '.join(map(str, kernels[1, 1, 0:4, 0]))}), r0);
        result += dot(vec4({', '.join(map(str, kernels[1, 1, 4:8, 0]))}), r1);
        break;
    }}

    return vec4(clamp(result, 0.0, 1.0), 0.0, 0.0, 1.0);
}}
'''
    return shader
