use burn::nn::{BatchNorm, Relu};
use burn::nn::conv::Conv2d;
use burn::prelude::{Backend, Module, Tensor};

#[derive(Debug, Module)]
pub enum ResidualBlock<B: Backend> {
	Basic(BasicBlock<B>),
	Bottleneck(Bottleneck<B>)
}

impl <B: Backend> ResidualBlock<B> {
	pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
		match self {
			ResidualBlock::Basic(block) => block.forward(input),
			ResidualBlock::Bottleneck(block) => block.forward(input)
		}
	}
}

#[derive(Debug, Module)]
pub struct BasicBlock<B: Backend> {
	pub(crate) conv1: Conv2d<B>,
	pub(crate) conv2: Conv2d<B>,
	pub(crate) bn1: BatchNorm<B, 2>,
	pub(crate) bn2: BatchNorm<B, 2>,
	pub(crate) relu: Relu,
	pub(crate) downsample: Option<Downsample<B>>
}

impl<B: Backend> BasicBlock<B> {
	pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
		let identity = input.clone();
		
		let x = self.conv1.forward(input);
		let x = self.bn1.forward(x);
		let x = self.relu.forward(x);
		
		let x = self.conv2.forward(x);
		let x = self.bn2.forward(x);
		
		// skip
		let x = match &self.downsample {
			Some(downsample) => x + downsample.forward(&identity),
			None => x + identity
		};
		
		self.relu.forward(x)
	}
}

#[derive(Debug, Module)]
pub struct Bottleneck<B: Backend> {
	pub(crate) conv1: Conv2d<B>,
	pub(crate) conv2: Conv2d<B>,
	pub(crate) conv3: Conv2d<B>,
	pub(crate) bn1: BatchNorm<B, 2>,
	pub(crate) bn2: BatchNorm<B, 2>,
	pub(crate) bn3: BatchNorm<B, 2>,
	pub(crate) relu: Relu,
	pub(crate) downsample: Option<Downsample<B>>
}

impl <B: Backend> Bottleneck<B> {
	pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
		let identity = input.clone();
		
		let x = self.conv1.forward(input);
		let x = self.bn1.forward(x);
		let x = self.relu.forward(x);
		
		let x = self.conv2.forward(x);
		let x = self.bn2.forward(x);
		let x = self.relu.forward(x);
		
		let x = self.conv3.forward(x);
		let x = self.bn3.forward(x);
		
		let x = match &self.downsample {
			Some(downsample) => x + downsample.forward(&identity),
			None => x + identity
		};
		
		self.relu.forward(x)
	}
}

#[derive(Debug, Module)]
pub struct Downsample<B: Backend> {
	pub(crate) conv: Conv2d<B>,
	pub(crate) bn: BatchNorm<B, 2>
}

impl<B: Backend> Downsample<B> {
	pub fn forward(&self, input: &Tensor<B, 4>) -> Tensor<B, 4> {
		let x = self.conv.forward(input.clone());
		self.bn.forward(x)
	}
}

#[derive(Debug, Module)]
pub struct LayerBlock<B: Backend> {
	pub(crate) blocks: Vec<ResidualBlock<B>>
}

impl<B: Backend> LayerBlock<B> {
	pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
		let mut x = input;
		for block in &self.blocks {
			x = block.forward(x);
		}
		x
	}
}

