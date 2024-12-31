use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use crate::data::AnimalBatch;

#[derive(Debug, Module)]
pub struct AnimalClassModel<B: Backend> {
	conv1: Conv2d<B>,
	conv2: Conv2d<B>,
	pool: AdaptiveAvgPool2d,
	dropout: Dropout,
	linear1: Linear<B>,
	linear2: Linear<B>,
	activation: Relu,
}

impl <B: Backend> AnimalClassModel<B> {
	pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
		let images = images.squeeze(0); // [3, 230, 400] -> [230, 400]
		let [batch_size, height, width] = images.dims();

		let x = images.reshape([batch_size, 1, height, width]);

		let x= self.conv1.forward(x);
		let x = self.dropout.forward(x);
		let x = self.conv2.forward(x);
		let x = self.dropout.forward(x);
		let x = self.activation.forward(x);

		let x = self.pool.forward(x);
		let x = x.reshape([batch_size, 16 * 8 * 8]);
		let x = self.linear1.forward(x);
		let x= self.dropout.forward(x);
		let x = self.activation.forward(x);

		self.linear2.forward(x)
	}

	pub fn forward_classification(&self, images: Tensor<B, 4>, targets: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
		let output = self.forward(images);

		let loss = CrossEntropyLossConfig::new().init(&output.device())
			.forward(output.clone(), targets.clone());

		ClassificationOutput::new(loss, output, targets)
	}
}

impl <B: AutodiffBackend> TrainStep<AnimalBatch<B>, ClassificationOutput<B>> for AnimalClassModel<B> {
	fn step(&self, batch: AnimalBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
		let item = self.forward_classification(batch.images, batch.targets);

		TrainOutput::new(self, item.loss.backward(), item)
	}
}

impl<B: Backend> ValidStep<AnimalBatch<B>, ClassificationOutput<B>> for AnimalClassModel<B> {
	fn step(&self, batch: AnimalBatch<B>) -> ClassificationOutput<B> {
		self.forward_classification(batch.images, batch.targets)
	}
}

#[derive(Debug, Config)]
pub struct AnimalClassConfig {
	#[config(default = 4)]
	pub num_classes: usize,
	#[config(default = 0.5)]
	dropout: f64,
	#[config(default = 512)]
	hidden_size: usize,
}

impl AnimalClassConfig {
	pub fn init<B: Backend>(&self, device: &B::Device) -> AnimalClassModel<B> {
		AnimalClassModel {
			conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
			conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
			pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
			activation: Relu::new(),
			linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
			linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
			dropout: DropoutConfig::new(self.dropout).init(),
		}
	}
}