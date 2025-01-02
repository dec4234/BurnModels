use crate::data::AnimalClassDataset;
use burn::data::dataset::vision::ImageFolderDataset;
use burn::data::dataset::Dataset;
use image::{DynamicImage, GenericImageView};
use rand::seq::IteratorRandom;
use std::path::PathBuf;

mod data;
mod error;
mod model;

fn main() {
    read_first_pixel();
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
            let resized = image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Lanczos3);
            let resized = DynamicImage::from(resized);
            let rgb = resized.into_rgb8();
            rgb.save(path).unwrap();
        });
    }
}