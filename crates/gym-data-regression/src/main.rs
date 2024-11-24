//! Dataset from: https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset
mod data;
mod model;

use std::time::SystemTime;
use burn::backend::Autodiff;
use burn_tch::{LibTorch, LibTorchDevice};

type LibTorchBackend = LibTorch;
type MyAutoDiffBackend = Autodiff<LibTorchBackend>;

fn main() {
    assert!(tch::utils::has_cuda(), "Could not detect valid CUDA configuration");

    let time = SystemTime::now();

    let device = LibTorchDevice::Cuda(0);

    println!("Time to train: {}", time.elapsed().unwrap().as_millis() as f64 / 1000.0);
}


