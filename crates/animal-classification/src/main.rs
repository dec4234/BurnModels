use crate::data::{AnimalClassDataset, ARTIFACT_DIR};
use burn::data::dataset::vision::ImageFolderDataset;
use burn::data::dataset::Dataset;
use image::{DynamicImage, GenericImageView};
use std::path::PathBuf;
use std::time::SystemTime;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, SgdConfig};
use burn_tch::{LibTorch, LibTorchDevice};
use crate::train::TrainingConfig;

mod data;
mod error;
mod model;
mod train;
mod infer;

fn main() {
    train();
}

fn train() {
    println!("Checking CUDA configuration for training...");
    assert!(tch::utils::has_cuda(), "Could not detect valid CUDA configuration");

    let time = SystemTime::now();

    let device = LibTorchDevice::Cuda(0);

    train::train::<Autodiff<LibTorch>>(TrainingConfig::new(SgdConfig::new()), device);

    println!("Time to train: {}", time.elapsed().unwrap().as_millis() as f64 / 1000.0);
}

fn infer() {
    println!("Checking CUDA configuration for training...");
    assert!(tch::utils::has_cuda(), "Could not detect valid CUDA configuration");

    let time = SystemTime::now();

    let device = LibTorchDevice::Cuda(0);
    
    let item = ImageFolderDataset::animal_class_test().get(325).unwrap();
    infer::infer_run::<Autodiff<LibTorch>>(ARTIFACT_DIR, device, item);
    
    println!("Time to train: {}", time.elapsed().unwrap().as_millis() as f64 / 1000.0);
}

fn read_first_pixel() {
    let data = ImageFolderDataset::new_classification("E:/Dev/Datasets/AnimalClassificationX4/Data/train/").unwrap();
    
    let mut i = 0;
    
    while i < data.len() {
        let item = data.get(i).unwrap();
        
        println!("{i} = {:?}", item.annotation);
        i += 1;
    }
}

// Proves that ImageDatasetItem is held in R, G, B, R, G, B, ...
fn read_pixel() {
    let image = image::open("E:/Dev/Datasets/AnimalClassificationX4/Data/train/sample/Zebra_101.jpg").unwrap();
    let pixel = image.get_pixel(0, 0);
    println!("{:?}", pixel);
}

// Resize all images in the dataset to the same size
fn resize_dataset() {
    println!("Resizing dataset...");
    let paths = vec![
        PathBuf::from("E:/Dev/Datasets/AnimalClassificationX4/Data/train/Buffalo/"),
        PathBuf::from("E:/Dev/Datasets/AnimalClassificationX4/Data/train/Rhino/"),
        PathBuf::from("E:/Dev/Datasets/AnimalClassificationX4/Data/train/Elephant/"),
        PathBuf::from("E:/Dev/Datasets/AnimalClassificationX4/Data/train/Zebra/"),
        
        PathBuf::from("E:/Dev/Datasets/AnimalClassificationX4/Data/test/Buffalo/"),
        PathBuf::from("E:/Dev/Datasets/AnimalClassificationX4/Data/test/Rhino/"),
        PathBuf::from("E:/Dev/Datasets/AnimalClassificationX4/Data/test/Elephant/"),
        PathBuf::from("E:/Dev/Datasets/AnimalClassificationX4/Data/test/Zebra/"),
    ];
    for path in paths {
        path.read_dir().unwrap().for_each(|entry| {
            let path = entry.unwrap().path();
            println!("Resizing {:?}", path);
            let image = image::open(&path).unwrap();
            let resized = image::imageops::resize(&image, data::SIDE_LENGTH, data::SIDE_LENGTH, image::imageops::FilterType::Lanczos3);
            let resized = DynamicImage::from(resized);
            let rgb = resized.into_rgb8();
            rgb.save(path).unwrap();
        });
    }
}