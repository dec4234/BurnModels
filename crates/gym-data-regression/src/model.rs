use crate::data::GymBatch;
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::Reduction::Mean;
use burn::nn::loss::{CrossEntropyLossConfig, MseLoss};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use burn::train::{RegressionOutput, TrainOutput, TrainStep, ValidStep};
// https://github.com/tracel-ai/burn/blob/main/examples/simple-regression/src/model.rs

pub const NUM_FEATURES: usize = 2; // TODO: change back to 14

#[derive(Debug, Module)]
pub struct WeightModel<B: Backend> {
	input: Linear<B>,
	output: Linear<B>,
	activation: Relu,
}

impl <B: Backend> WeightModel<B> {
	pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
		let x = self.input.forward(input);
		let x = self.activation.forward(x);
		self.output.forward(x)
	}
	
	pub fn forward_step(&self, item: GymBatch<B>) -> RegressionOutput<B> {
		let targets: Tensor<B, 2> = item.targets.unsqueeze_dim(1);
		let output: Tensor<B, 2> = self.forward(item.inputs);
		
		let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

		RegressionOutput {
			loss,
			output,
			targets,
		}
	}
}

impl <B: AutodiffBackend> TrainStep<GymBatch<B>, RegressionOutput<B>> for WeightModel<B> {
	fn step(&self, item: GymBatch<B>) -> TrainOutput<RegressionOutput<B>> {
		let item = self.forward_step(item);

		TrainOutput::new(self, item.loss.backward(), item)
	}
}

impl<B: Backend> ValidStep<GymBatch<B>, RegressionOutput<B>> for WeightModel<B> {
	fn step(&self, batch: GymBatch<B>) -> RegressionOutput<B> {
		self.forward_step(batch)
	}
}

#[derive(Config)]
pub struct WeightModelConfig {
	#[config(default = 64)]
	pub hidden_size: usize,
}

impl WeightModelConfig {
	pub fn init<B: Backend>(&self, device: &B::Device) -> WeightModel<B> {
		let input = LinearConfig::new(NUM_FEATURES, self.hidden_size)
			.with_bias(true)
			.init(device);
		let output = LinearConfig::new(self.hidden_size, 1)
			.with_bias(true)
			.init(device);

		WeightModel {
			input,
			output,
			activation: Relu::new(),
		}
	}
}