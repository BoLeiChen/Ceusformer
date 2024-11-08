import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch import Tensor
from timesformer.cupy_layers.utils import Dtype, Stream, load_kernel

CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

_aggregation_zeropad_mix_merge_forward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_mix_merge_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int kernel_num = 2;
    const int n = index / kernel_num / ${weight_heads} / ${input_channels} / ${top_height} / ${top_width};
    const int kernel_idx = (index / ${top_width} / ${top_height} / ${input_channels} / ${weight_heads}) % kernel_num;
    const int head = (index / ${top_width} / ${top_height} / ${input_channels}) % ${weight_heads};
    const int c = (index / ${top_width} / ${top_height}) % ${input_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};

    ${Dtype} value = 0;
    if (kernel_idx == 0) {
      for (int kh = 0; kh < ${kernel1_h}; ++kh) {
        for (int kw = 0; kw < ${kernel1_w}; ++kw) {
          const int h_in = -${pad1_h} + h * ${stride_h} + kh * ${dilation_h};
          const int w_in = -${pad1_w} + w * ${stride_w} + kw * ${dilation_w};
          if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
            const int offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
            const int offset_weight = n * ${weight_heads} * ${weight_channels} * (${kernel1_h}*${kernel1_w}+${kernel2_h}*${kernel2_w}) * ${top_height} * ${top_width}
               + ((head * ${weight_channels} + c % ${weight_channels}) * ${kernel1_h} * ${kernel1_w} + (kh * ${kernel1_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
            
            value += weight_data[offset_weight] * bottom_data[offset_bottom];
          }
        }
      }
    }else {
      for (int kh = 0; kh < ${kernel2_h}; ++kh) {
        for (int kw = 0; kw < ${kernel2_w}; ++kw) {
          const int h_in = -${pad2_h} + h * ${stride_h} + kh * ${dilation_h};
          const int w_in = -${pad2_w} + w * ${stride_w} + kw * ${dilation_w};
          if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
            const int offset_bottom = ((n * ${input_channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
            const int offset_weight = n * ${weight_heads} * ${weight_channels} * (${kernel1_h}*${kernel1_w}+${kernel2_h}*${kernel2_w}) * ${top_height} * ${top_width}
               + ${weight_heads} * ${weight_channels} * ${kernel1_h}*${kernel1_w} * ${top_height} * ${top_width}
               + ((head * ${weight_channels} + c % ${weight_channels}) * ${kernel2_h} * ${kernel2_w} + (kh * ${kernel2_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;
            value += weight_data[offset_weight] * bottom_data[offset_bottom];
          }
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_aggregation_zeropad_mix_merge_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_mix_merge_input_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${input_channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${input_channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    ${Dtype} value = 0;

    for (int head = 0; head < ${weight_heads}; ++head) {
        for (int kh = 0; kh < ${kernel1_h}; ++kh) {
          for (int kw = 0; kw < ${kernel1_w}; ++kw) {
            const int h_out_s = h + ${pad1_h} - kh * ${dilation_h};
            const int w_out_s = w + ${pad1_w} - kw * ${dilation_w};
            if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
              const int h_out = h_out_s / ${stride_h};
              const int w_out = w_out_s / ${stride_w};
              if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
                const int offset_top = ((((n * 2 + 0) * ${weight_heads} + head) * ${input_channels} + c) * ${top_height} + h_out) * ${top_width} + w_out;
                const int offset_weight = n * ${weight_heads} * ${weight_channels} * (${kernel1_h}*${kernel1_w}+${kernel2_h}*${kernel2_w}) * ${top_height} * ${top_width}
                   + ((head * ${weight_channels} + c % ${weight_channels}) * ${kernel1_h} * ${kernel1_w} + (kh * ${kernel1_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
                value += weight_data[offset_weight] * top_diff[offset_top];
              }
            }
          }
        }
    }

    for (int head = 0; head < ${weight_heads}; ++head) {
        for (int kh = 0; kh < ${kernel2_h}; ++kh) {
          for (int kw = 0; kw < ${kernel2_w}; ++kw) {
            const int h_out_s = h + ${pad2_h} - kh * ${dilation_h};
            const int w_out_s = w + ${pad2_w} - kw * ${dilation_w};
            if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
              const int h_out = h_out_s / ${stride_h};
              const int w_out = w_out_s / ${stride_w};
              if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
                const int offset_top = ((((n * 2 + 1) * ${weight_heads} + head) * ${input_channels} + c) * ${top_height} + h_out) * ${top_width} + w_out;
                const int offset_weight = n * ${weight_heads} * ${weight_channels} * (${kernel1_h}*${kernel1_w}+${kernel2_h}*${kernel2_w}) * ${top_height} * ${top_width}
                   + ${weight_heads} * ${weight_channels} * ${kernel1_h}*${kernel1_w} * ${top_height} * ${top_width}
                   + ((head * ${weight_channels} + c % ${weight_channels}) * ${kernel2_h} * ${kernel2_w} + (kh * ${kernel2_w} + kw)) * ${top_height} * ${top_width} + h_out * ${top_width} + w_out;
                value += weight_data[offset_weight] * top_diff[offset_top];
              }
            }
          }
        }
    }

    bottom_diff[index] = value;
  }
}
'''

_aggregation_zeropad_mix_merge_weight_backward_kernel = kernel_loop + '''
extern "C"
__global__ void aggregation_zeropad_mix_merge_weight_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* weight_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int kernel_num = 2;
    const int n = index / kernel_num / ${weight_heads} / ${weight_channels} / ${top_height} / ${top_width};
    const int kernel_idx = (index / ${top_width} / ${top_height} / ${weight_channels} / ${weight_heads}) % kernel_num;
    const int head = (index / ${top_width} / ${top_height} / ${weight_channels}) % ${weight_heads};
    const int c = (index / ${top_width} / ${top_height}) % ${weight_channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};

    if (kernel_idx == 0) {
      for (int kh = 0; kh < ${kernel1_h}; ++kh) {
        for (int kw = 0; kw < ${kernel1_w}; ++kw) {
          const int h_in = -${pad1_h} + h * ${stride_h} + kh * ${dilation_h};
          const int w_in = -${pad1_w} + w * ${stride_w} + kw * ${dilation_w};
          const int offset_weight = n * ${weight_heads} * ${weight_channels} * (${kernel1_h}*${kernel1_w}+${kernel2_h}*${kernel2_w}) * ${top_height} * ${top_width}
             + ((head * ${weight_channels} + c) * ${kernel1_h} * ${kernel1_w} + (kh * ${kernel1_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;

          ${Dtype} value = 0;
          if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
            for (int cc = c; cc < ${input_channels}; cc += ${weight_channels}) {
              const int offset_bottom = ((n * ${input_channels} + cc) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
              const int offset_top = ((((n * 2 + 0) * ${weight_heads} + head) * ${input_channels} + cc) * ${top_height} + h) * ${top_width} + w;
              value += bottom_data[offset_bottom] * top_diff[offset_top];
            }
          }
          weight_diff[offset_weight] = value;
        }
      }
      
    } else {
      for (int kh = 0; kh < ${kernel2_h}; ++kh) {
        for (int kw = 0; kw < ${kernel2_w}; ++kw) {
          const int h_in = -${pad2_h} + h * ${stride_h} + kh * ${dilation_h};
          const int w_in = -${pad2_w} + w * ${stride_w} + kw * ${dilation_w};
          const int offset_weight = n * ${weight_heads} * ${weight_channels} * (${kernel1_h}*${kernel1_w}+${kernel2_h}*${kernel2_w}) * ${top_height} * ${top_width}
             + ${weight_heads} * ${weight_channels} * ${kernel1_h}*${kernel1_w} * ${top_height} * ${top_width}
             + ((head * ${weight_channels} + c) * ${kernel2_h} * ${kernel2_w} + (kh * ${kernel2_w} + kw)) * ${top_height} * ${top_width} + h * ${top_width} + w;

          ${Dtype} value = 0;
          if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
            for (int cc = c; cc < ${input_channels}; cc += ${weight_channels}) {
              const int offset_bottom = ((n * ${input_channels} + cc) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
              const int offset_top = ((((n * 2 + 1) * ${weight_heads} + head) * ${input_channels} + cc) * ${top_height} + h) * ${top_width} + w;
              value += bottom_data[offset_bottom] * top_diff[offset_top];
            }
          }
          weight_diff[offset_weight] = value;
        }
      }
    }
  }
}
'''

class AggregationZeropadMixMerge(Function):
    @staticmethod
    def forward(ctx, input, weight, head_num, w_channels, kernel_size1, kernel_size2, stride, padding1, padding2, dilation):
        kernel_size1, kernel_size2, stride, padding1, padding2, dilation = _pair(kernel_size1), _pair(kernel_size2), _pair(stride), _pair(padding1), _pair(padding2), _pair(dilation)
        ctx.head_num, ctx.w_channels, ctx.kernel_size1, ctx.kernel_size2, ctx.stride, ctx.padding1, ctx.padding2, ctx.dilation = head_num, w_channels, kernel_size1, kernel_size2, stride, padding1, padding2, dilation
        assert input.dim() == 4 and input.is_cuda and weight.is_cuda
        batch_size, input_channels, input_height, input_width = input.size()
        weight_heads = head_num
        weight_channels = w_channels
        weight_height = weight.size()[-2]
        weight_width = weight.size()[-1]

        output_height = int((input_height + 2 * padding1[0] - (dilation[0] * (kernel_size1[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding1[1] - (dilation[1] * (kernel_size1[1] - 1) + 1)) / stride[1] + 1)
        assert output_height * output_width == weight_height * weight_width
        output = input.new(batch_size, weight_heads * input_channels * 2, output_height, output_width)
        n = output.numel()
        if not input.is_contiguous():
            input = input.detach().clone()
        if not weight.is_contiguous():
            weight = weight.detach().clone()

        with torch.cuda.device_of(input):
            f = load_kernel('aggregation_zeropad_mix_merge_forward_kernel', _aggregation_zeropad_mix_merge_forward_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, input_channels=input_channels, 
                            weight_heads=weight_heads, weight_channels=weight_channels,
                            bottom_height=input_height, bottom_width=input_width,
                            top_height=output_height, top_width=output_width,
                            kernel1_h=kernel_size1[0], kernel1_w=kernel_size1[1],
                            kernel2_h=kernel_size2[0], kernel2_w=kernel_size2[1],
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad1_h=padding1[0], pad1_w=padding1[1],
                            pad2_h=padding2[0], pad2_w=padding2[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        head_num, w_channels, kernel_size1, kernel_size2, stride, padding1, padding2, dilation = ctx.head_num, ctx.w_channels, ctx.kernel_size1, ctx.kernel_size2, ctx.stride, ctx.padding1, ctx.padding2, ctx.dilation
        input, weight = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        batch_size, input_channels, input_height, input_width = input.size()
        weight_heads = head_num
        weight_channels = w_channels
        weight_height = weight.size()[-2]
        weight_width = weight.size()[-1]

        output_height, output_width = grad_output.size()[2:]
        grad_input, grad_weight = None, None
        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, input_channels=input_channels, 
                   weight_heads=weight_heads, weight_channels=weight_channels,
                   bottom_height=input_height, bottom_width=input_width,
                   top_height=output_height, top_width=output_width,
                   kernel1_h=kernel_size1[0], kernel1_w=kernel_size1[1],
                   kernel2_h=kernel_size2[0], kernel2_w=kernel_size2[1],
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad1_h=padding1[0], pad1_w=padding1[1],
                   pad2_h=padding2[0], pad2_w=padding2[1])
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())
                n = grad_input.numel()
                opt['nthreads'] = n
                f = load_kernel('aggregation_zeropad_mix_merge_input_backward_kernel', _aggregation_zeropad_mix_merge_input_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())
                n = grad_weight.numel() // weight.shape[3]
                opt['nthreads'] = n
                f = load_kernel('aggregation_zeropad_mix_merge_weight_backward_kernel', _aggregation_zeropad_mix_merge_weight_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input, grad_weight, None, None, None, None, None, None, None, None

def aggregation_zeropad_mix_merge(input, weight, head_num, w_channels, kernel_size1=3, kernel_size2=5, stride=1, padding1=0, padding2=0, dilation=1):
    assert input.shape[0] == weight.shape[0] and (input.shape[1] % w_channels == 0)
    assert weight.shape[1] == head_num * w_channels * (kernel_size1*kernel_size1 + kernel_size2*kernel_size2)
    if input.is_cuda:
        out = AggregationZeropadMixMerge.apply(input, weight, head_num, w_channels, kernel_size1, kernel_size2, stride, padding1, padding2, dilation)
    else:
        #raise NotImplementedError
        out = AggregationZeropadMixMerge.apply(input.cuda(), weight.cuda(), head_num, w_channels, kernel_size1, kernel_size2, stride, padding1, padding2, dilation)
        torch.cuda.synchronize()
        out = out.cpu()
    return out

class LocalConvolutionMixMerge(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        head_num: int,
        w_channels: int,
        kernel_size1: int,
        kernel_size2: int,
        stride: int = 1,
        padding1: int = 0,
        padding2: int = 0,
        dilation: int = 1,
        pad_mode: int = 0,
    ):
        super(LocalConvolutionMixMerge, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_num = head_num
        self.w_channels = w_channels
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.stride = stride
        self.padding1 = padding1
        self.padding2 = padding2
        self.dilation = dilation
        self.pad_mode = pad_mode
        

    def forward(self, input: Tensor, weight: Tensor):
        out = aggregation_zeropad_mix_merge(
            input, 
            weight, 
            head_num = self.head_num,
            w_channels = self.w_channels,
            kernel_size1=self.kernel_size1, 
            kernel_size2=self.kernel_size2, 
            stride=self.stride, 
            padding1=self.padding1, 
            padding2=self.padding2, 
            dilation=self.dilation)
        return out

def test_aggregation_zeropad_mix_merge():
  kernel_size1, kernel_size2, stride, dilation = 3, 5, 1, 1
  padding1 = (dilation * (kernel_size1 - 1) + 1) // 2
  padding2 = (dilation * (kernel_size2 - 1) + 1) // 2
  head_num = 2
  n, c_x, c_w, in_height, in_width = 2, 8, 4, 6, 6
  out_height = int((in_height + 2 * padding1 - (dilation * (kernel_size1 - 1) + 1)) / stride + 1)
  out_width = int((in_width + 2 * padding1 - (dilation * (kernel_size1 - 1) + 1)) / stride + 1)
  x = torch.randn(n, c_x, in_height, in_width, requires_grad=True).double().cuda()
  w1 = torch.randn(n, head_num, c_w, pow(kernel_size1, 2), out_height, out_width, requires_grad=True).double().cuda()
  w2 = torch.randn(n, head_num, c_w, pow(kernel_size2, 2), out_height, out_width, requires_grad=True).double().cuda()
  
  _w1 = w1.view(n, -1, out_height, out_width)
  _w2 = w2.view(n, -1, out_height, out_width)
  w = torch.cat([_w1, _w2], dim=1)

  y1 = aggregation_zeropad_mix_merge(x, w, head_num=head_num, w_channels = c_w,
      kernel_size1=kernel_size1, kernel_size2=kernel_size2, stride=stride, 
      padding1=padding1, padding2=padding2, dilation=dilation)

  unfold_j1 = torch.nn.Unfold(kernel_size=kernel_size1, dilation=dilation, padding=padding1, stride=stride)
  x21 = unfold_j1(x).view(n, c_x // c_w, c_w, pow(kernel_size1, 2), out_height, out_width)
  y21 = (w1.unsqueeze(2) * x21.unsqueeze(1)).sum(-3).view(n, head_num * c_x, out_height, out_width)
  unfold_j2 = torch.nn.Unfold(kernel_size=kernel_size2, dilation=dilation, padding=padding2, stride=stride)
  x22 = unfold_j2(x).view(n, c_x // c_w, c_w, pow(kernel_size2, 2), out_height, out_width)
  y22 = (w2.unsqueeze(2) * x22.unsqueeze(1)).sum(-3).view(n, head_num * c_x, out_height, out_width)
  y2 = torch.cat([y21, y22], dim=1)
  assert (y1 - y2).abs().max() < 1e-9

  gx1 = torch.autograd.grad(y1.mean(), x, retain_graph=True)[0]
  gx2 = torch.autograd.grad(y2.mean(), x, retain_graph=True)[0]
  assert (gx1 - gx2).abs().max() < 1e-9

  gw1 = torch.autograd.grad(y1.mean(), w1, retain_graph=True)[0]
  gw2 = torch.autograd.grad(y2.mean(), w1, retain_graph=True)[0]
  assert (gw1 - gw2).abs().max() < 1e-9

  gw1 = torch.autograd.grad(y1.mean(), w2, retain_graph=True)[0]
  gw2 = torch.autograd.grad(y2.mean(), w2, retain_graph=True)[0]
  assert (gw1 - gw2).abs().max() < 1e-9

  from functools import partial
  assert torch.autograd.gradcheck(partial(aggregation_zeropad_mix, kernel_size1=kernel_size1, kernel_size2=kernel_size2, stride=stride, padding1=padding1, padding2=padding2, dilation=dilation), (x, w1, w2))
  print('test case passed')

if __name__ == '__main__':
    test_aggregation_zeropad_mix_merge()