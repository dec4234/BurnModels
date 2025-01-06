use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::pool::{AdaptiveAvgPool2d, MaxPool2d};
use burn::nn::{BatchNorm, Linear, Relu};
use burn::prelude::{Backend, Device};
use burn::record::{FullPrecisionSettings, Recorder, RecorderError};
use burn::tensor::Tensor;
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use crate::block::LayerBlock;
use crate::config::ResNetConfig;
use crate::weights;
use crate::weights::{Weights, WeightsMeta};

const RESNET18_BLOCKS: [usize; 4] = [2, 2, 2, 2];

#[derive(Debug, Module)]
pub struct ResNet<B: Backend> {
	pub(crate) conv1: Conv2d<B>,
	pub(crate) bn1: BatchNorm<B, 2>,
	pub(crate) relu: Relu,
	pub(crate) maxpool: MaxPool2d,
	pub(crate) layer1: LayerBlock<B>,
	pub(crate) layer2: LayerBlock<B>,
	pub(crate) layer3: LayerBlock<B>,
	pub(crate) layer4: LayerBlock<B>,
	pub(crate) avgpool: AdaptiveAvgPool2d,
	pub(crate) fc: Linear<B>,
}

impl<B: Backend> ResNet<B> {
	pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
		let x = self.conv1.forward(input);
		let x = self.bn1.forward(x);
		let x = self.relu.forward(x);
		let x = self.maxpool.forward(x);

		let x = self.layer1.forward(x);
		let x = self.layer2.forward(x);
		let x = self.layer3.forward(x);
		let x = self.layer4.forward(x);

		let x = self.avgpool.forward(x);
		let x = x.flatten(1, 3);

		self.fc.forward(x)
	}

	pub fn resnet18(num_classes: usize, device: &Device<B>) -> Self {
		ResNetConfig::new(RESNET18_BLOCKS, num_classes, 1).init(device)
	}

	pub fn resnet18_pretrained(weights: weights::ResNet18, device: &Device<B>) -> Result<Self, RecorderError> {
		let weights = weights.weights();
		let record = Self::load_weights_record(&weights, device)?;
		let model = ResNet::<B>::resnet18(weights.num_classes, device).load_record(record);

		Ok(model)
	}

	pub fn load_weights_record(weights: &Weights, device: &Device<B>) -> Result<ResNetRecord<B>, RecorderError> {
		let torch_weights = weights.download().map_err(|err| {
			RecorderError::Unknown(format!("Could not download weights.\nError: {err}"))
		})?;

		// Load weights from torch state_dict
		let load_args = LoadArgs::new(torch_weights)
			// Map *.downsample.0.* -> *.downsample.conv.*
			.with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
			// Map *.downsample.1.* -> *.downsample.bn.*
			.with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
			// Map layer[i].[j].* -> layer[i].blocks.[j].*
			.with_key_remap("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3");
		let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;

		Ok(record)
	}
}

