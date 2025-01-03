use crate::data::ClassificationBatch;
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};

pub const NUM_CLASSES: usize = 4;

#[derive(Debug, Module)]
pub struct AnimalClassModel<B: Backend> {
	activation: Relu,
	//dropout: Dropout,
	pool: MaxPool2d,
	conv1: Conv2d<B>,
	conv2: Conv2d<B>,
	conv3: Conv2d<B>,
	fc1: Linear<B>,
}

impl <B: Backend> AnimalClassModel<B> {
	pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
		let x = self.conv1.forward(images);
		let x = self.activation.forward(x);
		let x = self.pool.forward(x);
		
		let x = self.conv2.forward(x);
		let x = self.activation.forward(x);
		let x = self.pool.forward(x);
		//let x = self.dropout.forward(x);

		let x = self.conv3.forward(x);
		let x = self.activation.forward(x);
		let x = self.pool.forward(x);

		let x = x.flatten(1, 3);

		let x = self.fc1.forward(x);
		let x = self.activation.forward(x);
		//let x = self.dropout.forward(x);
		
		x
	}

	pub fn forward_classification(&self, images: Tensor<B, 4>, targets: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
		let output = self.forward(images);

		let loss = CrossEntropyLossConfig::new().init(&output.device())
			.forward(output.clone(), targets.clone());

		ClassificationOutput::new(loss, output, targets)
	}
}

impl <B: AutodiffBackend> TrainStep<ClassificationBatch<B>, ClassificationOutput<B>> for AnimalClassModel<B> {
	fn step(&self, batch: ClassificationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
		let item = self.forward_classification(batch.images, batch.targets);

		TrainOutput::new(self, item.loss.backward(), item)
	}
}

impl<B: Backend> ValidStep<ClassificationBatch<B>, ClassificationOutput<B>> for AnimalClassModel<B> {
	fn step(&self, batch: ClassificationBatch<B>) -> ClassificationOutput<B> {
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
		let conv1 = Conv2dConfig::new([3, 32], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);
		
		let conv2 = Conv2dConfig::new([32, 64], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);

		let conv3 = Conv2dConfig::new([64, 128], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);

		let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

		let fc1 = LinearConfig::new(115_200, 4).init(device);

		let dropout = DropoutConfig::new(0.3).init();

		AnimalClassModel {
			activation: Relu::new(),
			//dropout,
			pool,
			conv1,
			conv2,
			conv3,
			fc1,
		}
	}
}