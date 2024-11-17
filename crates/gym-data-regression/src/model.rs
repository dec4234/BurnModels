use burn::module::Module;
use burn::nn::Linear;
use burn::nn::loss::{BinaryCrossEntropyLossConfig, CrossEntropyLossConfig};
use burn::prelude::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::RegressionOutput;

// https://github.com/tracel-ai/burn/blob/main/examples/simple-regression/src/model.rs

#[derive(Debug, Module)]
pub struct WeightModel<B: Backend> {
	linear: Linear<B>,
}

impl <B: Backend> WeightModel<B> {
	pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
		self.linear.forward(input)
	}
	
	pub fn forward_regression(&self, input: Tensor<B, 1>, target: Tensor<B, 1, Int>) -> RegressionOutput<B> {
		let output = self.forward(input);
		
		let loss = BinaryCrossEntropyLossConfig::new().init(&output.device())
			.forward(output.clone(), target.clone());
		
		RegressionOutput::new(loss, output.reshape([1, 1]), target.reshape([1, 1]).float())
	}
}