use burn::nn::PaddingConfig2d;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::prelude::{Backend, Module, Tensor};
use crate::blocks::{expand, BaseConv, BaseConvConfig, Conv, ConvConfig};

pub(crate) const SPP_POOLING: [usize; 3] = [5, 9, 13];

#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
	conv1: BaseConv<B>,
	conv2: Conv<B>,
	shortcut: bool,
}

impl<B: Backend> Bottleneck<B> {
	pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
		let identity = x.clone();
		
		let x = self.conv1.forward(x);
		let mut x = self.conv2.forward(x);
		
		if self.shortcut {
			x = x + identity;
		}
		
		x
	}
}

struct BottleneckConfig {
	conv1: BaseConvConfig,
	conv2: ConvConfig,
	shortcut: bool,
}

impl BottleneckConfig {
	pub fn new(in_channels: usize, out_channels: usize, shortcut: bool, depthwise: bool) -> Self {
		let hidden_channels = out_channels;
		
		let conv1 = BaseConvConfig::new(in_channels, hidden_channels, 1, 1, 1);
		let conv2 = ConvConfig::new(hidden_channels, out_channels, 3, 1, depthwise);
		
		Self {
			conv1,
			conv2,
			shortcut
		}
	}
	
	pub fn init<B: Backend>(&self, device: &B::Device) -> Bottleneck<B> {
		Bottleneck {
			conv1: self.conv1.init(device),
			conv2: self.conv2.init(device),
			shortcut: self.shortcut,
		}
	}
}

#[derive(Module, Debug)]
pub struct SppBottleneck<B: Backend> {
	conv1: BaseConv<B>,
	conv2: BaseConv<B>,
	m: Vec<MaxPool2d>
}

impl <B: Backend> SppBottleneck<B> {
	pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
		if self.m.is_empty() {
			panic!("SPP bottleneck requires max pooling layers");
		}
		
		let x = self.conv1.forward(x);
		
		let x: Vec<_> = vec![x.clone()]
			.into_iter()
			.chain(self.m.iter().map(|m| m.forward(x.clone())))
			.collect();
		let x = Tensor::cat(x, 1);
		
		self.conv2.forward(x)
	}
}

pub struct SppBottleneckConfig {
	conv1: BaseConvConfig,
	conv2: BaseConvConfig,
	m: Vec<MaxPool2dConfig>
}

impl SppBottleneckConfig {
	pub fn new(in_channels: usize, out_channels: usize) -> Self {
		let hidden_channels = in_channels / 2;
		let conv2_channels = hidden_channels * 4;
		
		let conv1 = BaseConvConfig::new(in_channels, hidden_channels, 1, 1, 1);
		let conv2 = BaseConvConfig::new(conv2_channels, out_channels, 1, 1, 1);
		let m: Vec<_> = SPP_POOLING
			.into_iter()
			.map(|k| {
				let pad = k / 2;
				MaxPool2dConfig::new([k, k]).with_padding(PaddingConfig2d::Explicit(pad, pad))
			})
			.collect();
		
		Self {
			conv1,
			conv2,
			m
		}
	}
	
	pub fn init<B: Backend>(&self, device: &B::Device) -> SppBottleneck<B> {
		SppBottleneck {
			conv1: self.conv1.init(device),
			conv2: self.conv2.init(device),
			m: self.m.iter().map(|m| m.init()).collect()
		}
	}
}

#[derive(Module, Debug)]
pub struct CspBottleneck<B: Backend> {
	conv1: BaseConv<B>,
	conv2: BaseConv<B>,
	conv3: BaseConv<B>,
	m: Vec<Bottleneck<B>>
}

impl <B: Backend> CspBottleneck<B> {
	pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
		let x1 = self.conv1.forward(x.clone());
		let x2 = self.conv2.forward(x);
		
		let x1 = self.m.iter().fold(x1, |x_i, bottleneck| bottleneck.forward(x_i));
		
		let x = Tensor::cat(vec![x1, x2], 1);
		
		self.conv3.forward(x)
	}
}

pub struct CspBottleneckConfig {
	conv1: BaseConvConfig,
	conv2: BaseConvConfig,
	conv3: BaseConvConfig,
	m: Vec<BottleneckConfig>,
}

impl CspBottleneckConfig {
	pub fn new(
		in_channels: usize,
		out_channels: usize,
		num_blocks: usize,
		expansion: f64,
		shortcut: bool,
		depthwise: bool,
	) -> Self {
		assert!(
			expansion > 0.0 && expansion <= 1.0,
			"expansion should be in range (0, 1]"
		);

		let hidden_channels = expand(out_channels, expansion);

		let conv1 = BaseConvConfig::new(in_channels, hidden_channels, 1, 1, 1);
		let conv2 = BaseConvConfig::new(in_channels, hidden_channels, 1, 1, 1);
		let conv3 = BaseConvConfig::new(2 * hidden_channels, out_channels, 1, 1, 1);
		let m = (0..num_blocks)
			.map(|_| BottleneckConfig::new(hidden_channels, hidden_channels, shortcut, depthwise))
			.collect();

		Self {
			conv1,
			conv2,
			conv3,
			m,
		}
	}

	pub fn init<B: Backend>(&self, device: &B::Device) -> CspBottleneck<B> {
		CspBottleneck {
			conv1: self.conv1.init(device),
			conv2: self.conv2.init(device),
			conv3: self.conv3.init(device),
			m: self.m.iter().map(|b| b.init(device)).collect(),
		}
	}
}