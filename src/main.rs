use std::time::SystemTime;
use burn::backend::{Autodiff, LibTorch, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::{MnistDataset, MnistItem};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, Relu};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use burn::train::metric::{AccuracyMetric, CpuTemperature, CpuUse, CudaMetric, LossMetric};
use burn_tch::LibTorchDevice;

fn main() {
	assert!(
		tch::utils::has_cuda(),
		"Could not detect valid CUDA configuration"
	);
	//type Backend = Wgpu<f32, i32>;

	let time = SystemTime::now();
	
	let device = LibTorchDevice::Cuda(0);

	type LibTorchBackend = LibTorch;
	type MyAutoDiffBackend = Autodiff<LibTorchBackend>;

	let artifact_dir = "models/mnist_autodiff_libtorch";

	train::<MyAutoDiffBackend>(artifact_dir, TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()), device.clone());
	
	println!("Time: {}", time.elapsed().unwrap().as_secs());
}

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
	conv1: Conv2d<B>,
	conv2: Conv2d<B>,
	pool: AdaptiveAvgPool2d,
	dropout: Dropout,
	linear1: Linear<B>,
	linear2: Linear<B>,
	activation: Relu,
}

impl <B: Backend> Model<B> {
	pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
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
}

impl<B: Backend> Model<B> {
	pub fn forward_classification(&self, images: Tensor<B, 3>, targets: Tensor<B, 1, Int>) -> ClassificationOutput<B> {
		let output = self.forward(images);

		let loss = CrossEntropyLossConfig::new().init(&output.device())
			.forward(output.clone(), targets.clone());

		ClassificationOutput::new(loss, output, targets)
	}
}

impl <B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
	fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
		let item = self.forward_classification(batch.images, batch.targets);

		TrainOutput::new(self, item.loss.backward(), item)
	}
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
	fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
		self.forward_classification(batch.images, batch.targets)
	}
}

#[derive(Debug, Config)]
pub struct ModelConfig {
	num_classes: usize,
	hidden_size: usize,
	#[config(default = 0.5)]
	dropout: f64,
}

impl ModelConfig {
	pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
		Model {
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

#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
	device: B::Device
}

impl <B: Backend> MnistBatcher<B> {
	pub fn new(device: B::Device) -> Self {
		Self { device }
	}
}

#[derive(Debug, Clone)]
pub struct MnistBatch<B: Backend> {
	pub images: Tensor<B, 3>,
	pub targets: Tensor<B, 1, Int>,
}

impl <B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
	fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
		let images = items
			.iter()
			.map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
			.map(|data| Tensor::<B, 2>::from_data(data, &self.device))
			.map(|tensor| tensor.reshape([1, 28, 28]))
			.map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
			.collect();

		let targets = items
			.iter()
			.map(|item| {
				Tensor::<B, 1, Int>::from_data(
					[(item.label as i64).elem::<B::IntElem>()],
					&self.device,
				)
			})
			.collect();

		let images = Tensor::cat(images, 0).to_device(&self.device);
		let targets = Tensor::cat(targets, 0).to_device(&self.device);

		MnistBatch { images, targets }
	}
}

#[derive(Config)]
pub struct TrainingConfig {
	pub model: ModelConfig,
	pub optimizer: AdamConfig,
	#[config(default = 10)]
	pub num_epochs: usize,
	#[config(default = 64)]
	pub batch_size: usize,
	#[config(default = 4)]
	pub num_workers: usize,
	#[config(default = 42)]
	pub seed: u64,
	#[config(default = 1.0e-4)]
	pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
	// Remove existing artifacts before to get an accurate learner summary
	std::fs::remove_dir_all(artifact_dir).ok();
	std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
	create_artifact_dir(artifact_dir);
	config
		.save(format!("{artifact_dir}/config.json"))
		.expect("Config should be saved successfully");

	B::seed(config.seed);

	let batcher_train = MnistBatcher::<B>::new(device.clone());
	let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

	let dataloader_train = DataLoaderBuilder::new(batcher_train)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(MnistDataset::train());

	let dataloader_test = DataLoaderBuilder::new(batcher_valid)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(MnistDataset::test());

	let learner = LearnerBuilder::new(artifact_dir)
		.metric_train_numeric(AccuracyMetric::new())
		.metric_valid_numeric(AccuracyMetric::new())
		.metric_train_numeric(LossMetric::new())
		.metric_valid_numeric(LossMetric::new())

		.metric_train(CudaMetric::new())
		.metric_train(CpuUse::new())

		.with_file_checkpointer(CompactRecorder::new())
		.devices(vec![device.clone()])
		.num_epochs(config.num_epochs)
		.summary()
		.build(
			config.model.init::<B>(&device),
			config.optimizer.init(),
			config.learning_rate,
		);

	let model_trained = learner.fit(dataloader_train, dataloader_test);

	model_trained
		.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
		.expect("Trained model should be saved successfully");
}