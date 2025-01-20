use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::prelude::{Backend, Config, Device, Module};
use burn::tensor::activation::silu;
use burn::tensor::Tensor;

pub fn expand(num_channels: usize, factor: f64) -> usize {
	(num_channels as f64 * factor).floor() as usize
}

#[derive(Module, Debug)]
pub enum Conv<B: Backend> {
	BaseConv(BaseConv<B>),
	DwsConv(DwsConv<B>),
}

impl <B: Backend> Conv<B> {
	pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
		match self {
			Self::BaseConv(conv) => conv.forward(x),
			Self::DwsConv(conv) => conv.forward(x),
		}
	}
}


#[derive(Config)]
pub struct ConvConfig {
	in_channels: usize,
	out_channels: usize,
	kernel_size: usize,
	stride: usize,
	depthwise: bool,
}

impl ConvConfig {
	pub fn init<B: Backend>(&self, device: &B::Device) -> Conv<B> {
		if self.depthwise {
			Conv::DwsConv(
				DwsConvConfig::new(
					self.in_channels,
					self.out_channels,
					self.kernel_size,
					self.stride
				).init(device)
			)
		} else {
			Conv::BaseConv(
				BaseConvConfig::new(
					self.in_channels,
					self.out_channels,
					self.kernel_size,
					self.stride,
					1
				).init(device)
			)
		}
	}
}

#[derive(Module, Debug)]
pub struct BaseConv<B: Backend> {
	conv: Conv2d<B>,
	bn: BatchNorm<B, 2>,
}

impl <B: Backend> BaseConv<B> {
	pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
		let x = self.conv.forward(x);
		let x = self.bn.forward(x);
		
		silu(x)
	}
}

pub struct BaseConvConfig {
	conv: Conv2dConfig,
	bn: BatchNormConfig,
}

impl BaseConvConfig {
	pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, groups: usize) -> Self {
		let pad = (kernel_size - 1) / 2;
		
		let conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
			.with_stride([stride, stride])
			.with_padding(PaddingConfig2d::Explicit(pad, pad))
			.with_groups(groups)
			.with_bias(false);
		
		let bn = BatchNormConfig::new(out_channels)
			.with_epsilon(1e-3)
			.with_momentum(0.03);
		
		Self {
			conv,
			bn
		}
	}
	
	pub fn init<B: Backend>(&self, device: &B::Device) -> BaseConv<B> {
		BaseConv {
			conv: self.conv.init(device),
			bn: self.bn.init(device),
		}
	}
}

#[derive(Module, Debug)]
pub struct DwsConv<B: Backend> {
	dconv: BaseConv<B>,
	pconv: BaseConv<B>,
}

impl <B: Backend> DwsConv<B> {
	pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
		let x = self.dconv.forward(x);
		self.pconv.forward(x)
	}
}

pub struct DwsConvConfig {
	dconv: BaseConvConfig,
	pconv: BaseConvConfig,
}

impl DwsConvConfig {
	pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
		let dconv = BaseConvConfig::new(in_channels, in_channels, kernel_size, stride, in_channels);
		let pconv = BaseConvConfig::new(in_channels, out_channels, 1, 1, 1);
		
		Self {
			dconv,
			pconv
		}
	}
	
	pub fn init<B: Backend>(&self, device: &B::Device) -> DwsConv<B> {
		DwsConv {
			dconv: self.dconv.init(device),
			pconv: self.pconv.init(device),
		}
	}
}

#[derive(Module, Debug)]
pub struct Focus<B: Backend> {
	conv: BaseConv<B>
}

impl <B: Backend> Focus<B> {
	pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
		let device = x.device();
		let [_, _, h, w] = x.dims();

		let top_idx = Tensor::arange_step(0..h as i64, 2, &device);
		let bottom_idx = Tensor::arange_step(1..h as i64, 2, &device);
		let left_idx = Tensor::arange_step(0..w as i64, 2, &device);
		let right_idx = Tensor::arange_step(1..w as i64, 2, &device);

		let patch_top_left = x
			.clone()
			.select(2, top_idx.clone())
			.select(3, left_idx.clone());
		let patch_top_right = x.clone().select(2, top_idx).select(3, right_idx.clone());
		let patch_bottom_left = x.clone().select(2, bottom_idx.clone()).select(3, left_idx);
		let patch_bottom_right = x.select(2, bottom_idx).select(3, right_idx);

		let x = Tensor::cat(
			vec![
				patch_top_left,
				patch_bottom_left,
				patch_top_right,
				patch_bottom_right,
			],
			1,
		);

		self.conv.forward(x)
	}
}

pub struct FocusConfig {
	conv: BaseConvConfig
}

impl FocusConfig {
	pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
		let conv = BaseConvConfig::new(in_channels * 4, out_channels, kernel_size, stride, 1);
		
		Self {
			conv
		}
	}

	pub fn init<B: Backend>(&self, device: &B::Device) -> Focus<B> {
		Focus {
			conv: self.conv.init(device),
		}
	}
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
	conv0: Conv<B>,
	conv1: Conv<B>,
}

impl <B: Backend> ConvBlock<B> {
	pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
		let x = self.conv0.forward(x);
		self.conv1.forward(x)
	}
}

pub struct ConvBlockConfig {
	conv0: ConvConfig,
	conv1: ConvConfig,
}

impl ConvBlockConfig {
	pub fn new(channels: usize, kernel_size: usize, stride: usize, depthwise: bool) -> Self {
		let conv0 = ConvConfig::new(channels, channels, kernel_size, stride, depthwise);
		let conv1 = ConvConfig::new(channels, channels, kernel_size, stride, depthwise);

		Self {
			conv0,
			conv1
		}
	}

	pub fn init<B: Backend>(&self, device: &Device<B>) -> ConvBlock<B> {
		ConvBlock {
			conv0: self.conv0.init(device),
			conv1: self.conv1.init(device),
		}
	}
}