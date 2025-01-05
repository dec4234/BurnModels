use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu};
use burn::prelude::{Backend, Tensor};
use burn::tensor::backend::AutodiffBackend;

pub const NUM_CLASSES: usize = 4;

#[derive(Debug, Module)]
pub struct AnimalClassModel<B: Backend> {
	activation: Relu,
	con_dropout: Dropout,
	fcn_dropout: Dropout,
	pool: MaxPool2d,
	conv1: Conv2d<B>,
	conv2: Conv2d<B>,
	conv3: Conv2d<B>,
	conv4: Conv2d<B>,
	conv5: Conv2d<B>,
	conv6: Conv2d<B>,
	fc1: Linear<B>,
	fc2: Linear<B>,
}

impl <B: Backend> AnimalClassModel<B> {
	pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
		let x = self.conv1.forward(images);
		let x = self.activation.forward(x);
		let x = self.conv2.forward(x);
		let x = self.activation.forward(x);
		let x = self.pool.forward(x);
		let x = self.con_dropout.forward(x);

		let x = self.conv3.forward(x);
		let x = self.activation.forward(x);
		let x = self.conv4.forward(x);
		let x = self.activation.forward(x);
		let x = self.pool.forward(x);
		let x = self.con_dropout.forward(x);

		let x = self.conv5.forward(x);
		let x = self.activation.forward(x);
		let x = self.conv6.forward(x);
		let x = self.activation.forward(x);
		let x = self.pool.forward(x);
		let x = self.con_dropout.forward(x);

		let x = x.flatten(1, 3);

		let x = self.fc1.forward(x);
		let x = self.activation.forward(x);
		let x = self.fcn_dropout.forward(x);

		self.fc2.forward(x)
	}
}

#[derive(Debug, Config)]
pub struct AnimalClassConfig {
	#[config(default = 4)]
	pub num_classes: usize,
	#[config(default = 0.5)]
	dropout: f64,
}

impl AnimalClassConfig {
	pub fn init<B: Backend>(&self, device: &B::Device) -> AnimalClassModel<B> {
		let conv1 = Conv2dConfig::new([3, 32], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);
		let conv2 = Conv2dConfig::new([32, 32], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);

		let conv3 = Conv2dConfig::new([32, 64], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);
		let conv4 = Conv2dConfig::new([64, 64], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);

		let conv5 = Conv2dConfig::new([64, 128], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);
		let conv6 = Conv2dConfig::new([128, 128], [3, 3])
			.with_padding(PaddingConfig2d::Same)
			.init(device);

		let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

		let fc1 = LinearConfig::new(2048, 128).init(device);
		let fc2 = LinearConfig::new(128, self.num_classes).init(device);

		let con_dropout = DropoutConfig::new(0.3).init();
		let fcn_dropout = DropoutConfig::new(0.3).init();

		AnimalClassModel {
			activation: Relu::new(),
			con_dropout,
			fcn_dropout,
			pool,
			conv1,
			conv2,
			conv3,
			conv4,
			conv5,
			conv6,
			fc1,
			fc2,
		}
	}
}