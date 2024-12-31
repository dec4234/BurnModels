use std::time::SystemTime;
use burn::backend::{Autodiff, LibTorch};
use burn::backend::libtorch::LibTorchDevice;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::vision::{Annotation, ImageFolderDataset};
use burn::module::Module;
use burn::optim::momentum::MomentumConfig;
use burn::optim::SgdConfig;
use burn::record::{CompactRecorder, Recorder};
use rand::seq::IteratorRandom;
use rand::thread_rng;
use crate::data::ClassificationBatcher;
use crate::dataset::CIFAR10Loader;
use crate::model::Cnn;
use crate::training::TrainingConfig;

mod data;
mod dataset;
mod inference;
mod model;
mod training;

const NUM_CLASSES: u8 = 10;
const ARTIFACT_DIR: &str = "/tmp/custom-image-dataset";

fn main() {
    //train();
    infer();
}

// 5 mins
pub fn train() {
    println!("Checking CUDA configuration for training...");
    assert!(tch::utils::has_cuda(), "Could not detect valid CUDA configuration");

    let time = SystemTime::now();
    
    let device = LibTorchDevice::Cuda(0);

    training::train::<Autodiff<LibTorch>>(
        TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
            momentum: 0.9,
            dampening: 0.,
            nesterov: false,
        }))),
        device,
    );

    println!("Time to train: {}", time.elapsed().unwrap().as_millis() as f64 / 1000.0);
}

pub fn infer() {
    println!("Checking CUDA configuration for inference...");
    assert!(tch::utils::has_cuda(), "Could not detect valid CUDA configuration");

    type Backend = Autodiff<LibTorch>;

    let device = LibTorchDevice::Cuda(0);

    let record = CompactRecorder::new()
        .load(format!("{ARTIFACT_DIR}/model").into(), &device)
        .expect("Trained model should exist");

    let model: Cnn<Backend> = Cnn::new(NUM_CLASSES.into(), &device).load_record(record);

    let dataset = ImageFolderDataset::cifar10_test();
    let item = dataset.iter().choose(&mut thread_rng()).unwrap();

    println!("Label: {:?}", item.annotation);

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
