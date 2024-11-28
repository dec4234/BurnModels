use crate::data::{GymBatcher, GymDataset, GymGoer};
use crate::training::TrainingConfig;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::Backend;
use burn::record::{CompactRecorder, Recorder};
use std::path::Path;

pub fn infer<B: Backend, P: AsRef<Path>>(artifact_dir: P, device: B::Device, item: GymGoer) {
	let artifact_dir = artifact_dir.as_ref().to_str().unwrap().to_string();
	let model_dir = format!("{}/model", artifact_dir);
	let artifact_dir = format!("{}/config.json", artifact_dir);

	let config = TrainingConfig::load(artifact_dir).unwrap();
	let record = CompactRecorder::new().load(model_dir.parse().unwrap(), &device).unwrap();

	let model = config.model.init::<B>(&device).load_record(record);

	let dataset = GymDataset {
		data: vec![item],
	};
	
	let label = item.label;
	let batcher = GymBatcher::<B>::new(device.clone(), &dataset);
	let batch = batcher.batch(vec![item]);
	let output = model.forward(batch.images);
	let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

	println!("Label: {}, Predicted: {}", label, predicted);
}