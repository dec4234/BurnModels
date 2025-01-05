use crate::data::{AnimalClassDataset, ClassificationBatch, ClassificationBatcher, ARTIFACT_DIR};
use crate::model::{AnimalClassConfig, AnimalClassModel, NUM_CLASSES};
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::ImageFolderDataset;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::SgdConfig;
use burn::prelude::{Backend, Int, Tensor};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, CpuUse, CudaMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep};
use std::time::Instant;
use burn::data::dataset::transform::ShuffledDataset;
use rand::rngs::StdRng;
use rand::SeedableRng;

impl<B: Backend> AnimalClassModel<B> {
	pub fn forward_classification(
		&self,
		images: Tensor<B, 4>,
		targets: Tensor<B, 1, Int>,
	) -> ClassificationOutput<B> {
		let output = self.forward(images);
		let loss = CrossEntropyLossConfig::new()
			.init(&output.device())
			.forward(output.clone(), targets.clone());

		ClassificationOutput::new(loss, output, targets)
	}
}

impl<B: AutodiffBackend> TrainStep<ClassificationBatch<B>, ClassificationOutput<B>> for AnimalClassModel<B> {
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

#[derive(Config)]
pub struct TrainingConfig {
	pub optimizer: SgdConfig,
	#[config(default = 300)]
	pub num_epochs: usize,
	#[config(default = 800)]
	pub batch_size: usize,
	#[config(default = 4)]
	pub num_workers: usize,
	#[config(default = 42)]
	pub seed: u64,
	#[config(default = 0.02)]
	pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
	// Remove existing artifacts before to get an accurate learner summary
	std::fs::remove_dir_all(artifact_dir).ok();
	std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) {
	create_artifact_dir(ARTIFACT_DIR);

	config
		.save(format!("{ARTIFACT_DIR}/config.json"))
		.expect("Config should be saved successfully");

	B::seed(config.seed);

	// Dataloaders
	let batcher_train = ClassificationBatcher::<B>::new(device.clone());
	let batcher_valid = ClassificationBatcher::<B::InnerBackend>::new(device.clone());

	let dataloader_train = DataLoaderBuilder::new(batcher_train)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(ShuffledDataset::new(ImageFolderDataset::animal_class_train(), &mut StdRng::seed_from_u64(config.seed)));

	// NOTE: we use the CIFAR-10 test set as validation for demonstration purposes
	let dataloader_test = DataLoaderBuilder::new(batcher_valid)
		.batch_size(config.batch_size)
		.num_workers(config.num_workers)
		.build(ShuffledDataset::new(ImageFolderDataset::animal_class_test(), &mut StdRng::seed_from_u64(config.seed)));

	// Learner config
	let learner = LearnerBuilder::new(ARTIFACT_DIR)
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
			AnimalClassConfig::new().init(&device),
			config.optimizer.init(),
			config.learning_rate,
		);

	// Training
	let now = Instant::now();
	let model_trained = learner.fit(dataloader_train, dataloader_test);
	let elapsed = now.elapsed().as_secs();
	println!("Training completed in {}m{}s", (elapsed / 60), elapsed % 60);

	model_trained
		.save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
		.expect("Trained model should be saved successfully");
}