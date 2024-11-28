use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use burn::prelude::Module;
use burn::record::{CompactRecorder, NoStdTrainingRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn::train::LearnerBuilder;
use burn::train::metric::{AccuracyMetric, CpuUse, CudaMetric, LossMetric};
use crate::data::{GymBatcher, GymDataset};
use crate::model::WeightModelConfig;

#[derive(Config)]
pub struct TrainingConfig {
	#[config(default = 100)]
	pub num_epochs: usize,

	#[config(default = 2)]
	pub num_workers: usize,

	#[config(default = 1337)]
	pub seed: u64,

	pub optimizer: AdamConfig,

	#[config(default = 256)]
	pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
	// Remove existing artifacts before to get an accurate learner summary
	std::fs::remove_dir_all(artifact_dir).ok();
	std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
	create_artifact_dir(artifact_dir);

	// Config
	let optimizer = AdamConfig::new();
	let config = TrainingConfig::new(optimizer);
	let model = WeightModelConfig::new().init(&device);
	B::seed(config.seed);
	
	let dataset = GymDataset::from("crates/gym-data-regression/gym_members_exercise_tracking.csv");

	// Define train/valid datasets and dataloaders
	let train_dataset = dataset.train_data();
	let valid_dataset = dataset.test_data();

	println!("Train Dataset Size: {}", train_dataset.len());
	println!("Valid Dataset Size: {}", valid_dataset.len());

	let batcher_train = GymBatcher::<B>::new(device.clone(), &dataset);

	let batcher_test = GymBatcher::<B::InnerBackend>::new(device.clone(), &dataset);

	let dataloader_train = DataLoaderBuilder::new(batcher_train)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(train_dataset);

	let dataloader_test = DataLoaderBuilder::new(batcher_test)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(valid_dataset);

	// Model
	let learner = LearnerBuilder::new(artifact_dir)
		.metric_train_numeric(LossMetric::new())
		.metric_valid_numeric(LossMetric::new())
		.with_file_checkpointer(CompactRecorder::new())

		.metric_train(CudaMetric::new())
		.metric_train(CpuUse::new())
		
		.devices(vec![device.clone()])
		.num_epochs(config.num_epochs)
		.summary()
		.build(model, config.optimizer.init(), 1e-3);

	let model_trained = learner.fit(dataloader_train, dataloader_test);

	config
		.save(format!("{artifact_dir}/config.json").as_str())
		.unwrap();

	model_trained
		.save_file(
			format!("{artifact_dir}/model"),
			&NoStdTrainingRecorder::new(),
		)
		.expect("Failed to save trained model");
}