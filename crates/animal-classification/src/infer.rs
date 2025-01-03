use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::{Annotation, ImageDatasetItem};
use burn::module::Module;
use burn::prelude::Backend;
use burn::record::{CompactRecorder, Recorder};
use crate::data::ClassificationBatcher;
use crate::model::{AnimalClassConfig, AnimalClassModel, NUM_CLASSES};

pub fn infer_run<B: Backend>(artifact_dir: &str, device: B::Device, item: ImageDatasetItem) {
	let record = CompactRecorder::new()
		.load(format!("{artifact_dir}/model").into(), &device)
		.expect("Trained model should exist");

	let model: AnimalClassModel<B> = AnimalClassConfig::new().init(&device).load_record(record);

	let mut label = 0;
	if let Annotation::Label(category) = item.annotation {
		label = category;
	};
	let batcher = ClassificationBatcher::new(device);
	let batch = batcher.batch(vec![item]);
	let output = model.forward(batch.images);
	let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
	println!("Predicted {} Expected {:?}", predicted, label);
}