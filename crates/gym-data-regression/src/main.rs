//! Dataset from: https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset
mod data;
mod model;
mod training;
mod infer;

use std::time::SystemTime;
use burn::backend::Autodiff;
use burn_tch::{LibTorch, LibTorchDevice};
use crate::data::{Gender, GymGoer, WorkoutType};

type LibTorchBackend = LibTorch;
type MyAutoDiffBackend = Autodiff<LibTorchBackend>;

fn main() {
    println!("Checking CUDA configuration...");
    assert!(tch::utils::has_cuda(), "Could not detect valid CUDA configuration");

    let time = SystemTime::now();

    let device = LibTorchDevice::Cuda(0);
    
    training::run::<MyAutoDiffBackend>("artifacts", device);

    println!("Time to train: {}", time.elapsed().unwrap().as_millis() as f64 / 1000.0);
}

#[test]
fn infer() {
    assert!(tch::utils::has_cuda(), "Could not detect valid CUDA configuration");

    let time = SystemTime::now();

    let device = LibTorchDevice::Cuda(0);
    
    infer::infer::<MyAutoDiffBackend, &str>("../../artifacts/", device, GymGoer::default());

    println!("Time to infer: {}", time.elapsed().unwrap().as_millis() as f64 / 1000.0);
}


