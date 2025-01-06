use burn::nn::conv::Conv2dConfig;
use burn::nn::{BatchNormConfig, Initializer, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::{Backend, Config, Device};
use std::f64::consts::SQRT_2;
use burn::nn::pool::{AdaptiveAvgPool2dConfig, MaxPool2dConfig};
use crate::block::{BasicBlock, Bottleneck, Downsample, LayerBlock, ResidualBlock};
use crate::resnet::ResNet;

#[derive(Config)]
struct ResidualBlockConfig {
	in_channels: usize,
	out_channels: usize,
	stride: usize,
	bottleneck: bool,
}

impl ResidualBlockConfig {
	fn init<B: Backend>(&self, device: &Device<B>) -> ResidualBlock<B> {
		if self.bottleneck {
			ResidualBlock::Bottleneck(
				BottleneckConfig::new(self.in_channels, self.out_channels, self.stride)
					.init(device),
			)
		} else {
			ResidualBlock::Basic(
				BasicBlockConfig::new(self.in_channels, self.out_channels, self.stride)
					.init(device),
			)
		}
	}
}

struct BasicBlockConfig {
	conv1: Conv2dConfig,
	bn1: BatchNormConfig,
	conv2: Conv2dConfig,
	bn2: BatchNormConfig,
	downsample: Option<DownsampleConfig>,
}

impl BasicBlockConfig {
	fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
		// conv3x3
		let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
			.with_stride([stride, stride])
			.with_padding(PaddingConfig2d::Explicit(1, 1))
			.with_bias(false);
		let bn1 = BatchNormConfig::new(out_channels);

		// conv3x3
		let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
			.with_stride([1, 1])
			.with_padding(PaddingConfig2d::Explicit(1, 1))
			.with_bias(false);
		let bn2 = BatchNormConfig::new(out_channels);

		let downsample = {
			if in_channels != out_channels {
				Some(DownsampleConfig::new(in_channels, out_channels, stride))
			} else {
				None
			}
		};

		Self {
			conv1,
			bn1,
			conv2,
			bn2,
			downsample,
		}
	}

	/// Initialize a new [basic residual block](BasicBlock) module.
	fn init<B: Backend>(&self, device: &Device<B>) -> BasicBlock<B> {
		// Conv initializer
		let initializer = Initializer::KaimingNormal {
			gain: SQRT_2, // recommended value for ReLU
			fan_out_only: true,
		};

		BasicBlock {
			conv1: self
				.conv1
				.clone()
				.with_initializer(initializer.clone())
				.init(device),
			bn1: self.bn1.init(device),
			relu: Relu::new(),
			conv2: self
				.conv2
				.clone()
				.with_initializer(initializer)
				.init(device),
			bn2: self.bn2.init(device),
			downsample: self.downsample.as_ref().map(|d| d.init(device)),
		}
	}
}

struct BottleneckConfig {
	conv1: Conv2dConfig,
	bn1: BatchNormConfig,
	conv2: Conv2dConfig,
	bn2: BatchNormConfig,
	conv3: Conv2dConfig,
	bn3: BatchNormConfig,
	downsample: Option<DownsampleConfig>,
}

impl BottleneckConfig {
	fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
		// Intermediate output channels w/ expansion = 4
		let int_out_channels = out_channels / 4;
		// conv1x1
		let conv1 = Conv2dConfig::new([in_channels, int_out_channels], [1, 1])
			.with_stride([1, 1])
			.with_padding(PaddingConfig2d::Explicit(0, 0))
			.with_bias(false);
		let bn1 = BatchNormConfig::new(int_out_channels);
		// conv3x3
		let conv2 = Conv2dConfig::new([int_out_channels, int_out_channels], [3, 3])
			.with_stride([stride, stride])
			.with_padding(PaddingConfig2d::Explicit(1, 1))
			.with_bias(false);
		let bn2 = BatchNormConfig::new(int_out_channels);
		// conv1x1
		let conv3 = Conv2dConfig::new([int_out_channels, out_channels], [1, 1])
			.with_stride([1, 1])
			.with_padding(PaddingConfig2d::Explicit(0, 0))
			.with_bias(false);
		let bn3 = BatchNormConfig::new(out_channels);

		let downsample = {
			if in_channels != out_channels {
				Some(DownsampleConfig::new(in_channels, out_channels, stride))
			} else {
				None
			}
		};

		Self {
			conv1,
			bn1,
			conv2,
			bn2,
			conv3,
			bn3,
			downsample,
		}
	}

	fn init<B: Backend>(&self, device: &Device<B>) -> Bottleneck<B> {
		let initializer = Initializer::KaimingNormal {
			gain: SQRT_2, // recommended value for ReLU
			fan_out_only: true,
		};

		Bottleneck {
			conv1: self
				.conv1
				.clone()
				.with_initializer(initializer.clone())
				.init(device),
			bn1: self.bn1.init(device),
			relu: Relu::new(),
			conv2: self
				.conv2
				.clone()
				.with_initializer(initializer.clone())
				.init(device),
			bn2: self.bn2.init(device),
			conv3: self
				.conv3
				.clone()
				.with_initializer(initializer)
				.init(device),
			bn3: self.bn3.init(device),
			downsample: self.downsample.as_ref().map(|d| d.init(device)),
		}
	}
}

struct DownsampleConfig {
	conv: Conv2dConfig,
	bn: BatchNormConfig,
}

impl DownsampleConfig {
	fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
		// conv1x1
		let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
			.with_stride([stride, stride])
			.with_padding(PaddingConfig2d::Explicit(0, 0))
			.with_bias(false);
		let bn = BatchNormConfig::new(out_channels);

		Self { conv, bn }
	}

	/// Initialize a new [downsample](Downsample) module.
	fn init<B: Backend>(&self, device: &B::Device) -> Downsample<B> {
		// Conv initializer
		let initializer = Initializer::KaimingNormal {
			gain: SQRT_2, // recommended value for ReLU
			fan_out_only: true,
		};

		Downsample {
			conv: self.conv.clone().with_initializer(initializer).init(device),
			bn: self.bn.init(device),
		}
	}
}

#[derive(Config)]
pub struct LayerBlockConfig {
	num_blocks: usize,
	in_channels: usize,
	out_channels: usize,
	stride: usize,
	bottleneck: bool,
}

impl LayerBlockConfig {
	pub fn init<B: Backend>(&self, device: &Device<B>) -> LayerBlock<B> {
		let blocks = (0..self.num_blocks)
			.map(|b| {
				if b == 0 {
					// First block uses the specified stride
					ResidualBlockConfig::new(
						self.in_channels,
						self.out_channels,
						self.stride,
						self.bottleneck,
					)
						.init(device)
				} else {
					// Other blocks use a stride of 1
					ResidualBlockConfig::new(
						self.out_channels,
						self.out_channels,
						1,
						self.bottleneck,
					)
						.init(device)
				}
			})
			.collect();

		LayerBlock { blocks }
	}
}

pub struct ResNetConfig {
	pub conv1: Conv2dConfig,
	pub bn1: BatchNormConfig,
	pub maxpool: MaxPool2dConfig,
	pub layer1: LayerBlockConfig,
	pub layer2: LayerBlockConfig,
	pub layer3: LayerBlockConfig,
	pub layer4: LayerBlockConfig,
	pub avgpool: AdaptiveAvgPool2dConfig,
	pub fc: LinearConfig,
}

impl ResNetConfig {
	pub fn new(blocks: [usize; 4], num_classes: usize, expansion: usize) -> Self {
		// `new()` is private but still check just in case...
		assert!(
			expansion == 1 || expansion == 4,
			"ResNet module only supports expansion values [1, 4] for residual blocks"
		);

		// 7x7 conv, 64, /2
		let conv1 = Conv2dConfig::new([3, 64], [7, 7])
			.with_stride([2, 2])
			.with_padding(PaddingConfig2d::Explicit(3, 3))
			.with_bias(false);
		let bn1 = BatchNormConfig::new(64);

		// 3x3 maxpool, /2
		let maxpool = MaxPool2dConfig::new([3, 3])
			.with_strides([2, 2])
			.with_padding(PaddingConfig2d::Explicit(1, 1));

		// Residual blocks
		let bottleneck = expansion > 1;
		let layer1 = LayerBlockConfig::new(blocks[0], 64, 64 * expansion, 1, bottleneck);
		let layer2 =
			LayerBlockConfig::new(blocks[1], 64 * expansion, 128 * expansion, 2, bottleneck);
		let layer3 =
			LayerBlockConfig::new(blocks[2], 128 * expansion, 256 * expansion, 2, bottleneck);
		let layer4 =
			LayerBlockConfig::new(blocks[3], 256 * expansion, 512 * expansion, 2, bottleneck);

		// Average pooling [B, 512 * expansion, H, W] -> [B, 512 * expansion, 1, 1]
		let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]);

		// Output layer
		let fc = LinearConfig::new(512 * expansion, num_classes);

		Self {
			conv1,
			bn1,
			maxpool,
			layer1,
			layer2,
			layer3,
			layer4,
			avgpool,
			fc,
		}
	}
	
	pub fn init<B: Backend>(self, device: &Device<B>) -> ResNet<B> {
		// Conv initializer
		let initializer = Initializer::KaimingNormal {
			gain: SQRT_2, // recommended value for ReLU
			fan_out_only: true,
		};

		ResNet {
			conv1: self.conv1.with_initializer(initializer).init(device),
			bn1: self.bn1.init(device),
			relu: Relu::new(),
			maxpool: self.maxpool.init(),
			layer1: self.layer1.init(device),
			layer2: self.layer2.init(device),
			layer3: self.layer3.init(device),
			layer4: self.layer4.init(device),
			avgpool: self.avgpool.init(),
			fc: self.fc.init(device),
		}
	}
}