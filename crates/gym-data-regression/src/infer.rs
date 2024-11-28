use crate::data::{GymBatcher, GymDataset, GymGoer};
use crate::training::TrainingConfig;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::Backend;
use burn::record::{CompactRecorder, Recorder};
use std::path::Path;
use burn::module::Module;

pub fn infer<B: Backend, P: AsRef<Path>>(artifact_dir: P, device: B::Device, item: GymGoer) {
	let artifact_dir = artifact_dir.as_ref().to_str().unwrap().to_string();
	let model_dir = format!("{}/model", artifact_dir);
	let artifact_dir = format!("{}/config.json", artifact_dir);

	let config = TrainingConfig::load(artifact_dir).unwrap();
	let record = CompactRecorder::new().load(model_dir.parse().unwrap(), &device).unwrap();

	let model = config.model.init::<B>(&device).load_record(record);

	let label = item.weight;
	
	let min = vec![18.0, 3.4028235e38, 1.5, 160.0, 120.0, 50.0, 0.5, 303.0, 3.4028235e38, 10.0, 1.5, 2.0, 1.0, 12.32];
	let max = vec![59.0, 1.0, 2.0, 199.0, 169.0, 74.0, 2.0, 1783.0, 3.0, 35.0, 3.7, 5.0, 3.0, 49.84];
	
	let min = vec![18.0, 1.5]; // age, height
	let max = vec![59.0, 2.0];
	
	let batcher = GymBatcher::<B>::new_from_precomputed(device.clone(), &min, &max);
	let batch = batcher.batch(vec![item]);
	let output = model.forward(batch.inputs);
	let predicted = output.into_scalar();

	println!("Label: {}, Predicted: {}", label, predicted);
}