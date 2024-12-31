// Data folder E:\Dev\Datasets\AnimalClassificationX4\Data

use std::fs::read_dir;
use crate::error::AnimalClassError;
use burn::prelude::{Backend, ElementConversion, Int};
use burn::tensor::{Tensor, TensorData};
use image::{imageops, RgbImage};
use std::path::{Path, PathBuf};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::vision::MnistItem;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct AnimalDataset<B: Backend> {
	pub data: Vec<AnimalImage<B>>,
}

impl <B: Backend> AnimalDataset<B> {
	pub fn from<A: AsRef<Path>>(data_folder: A, device: &B::Device) -> Result<Self, AnimalClassError> {
		let mut data_folder = data_folder.as_ref().read_dir()?;
		
		let buff = data_folder.find(|entry| entry.as_ref().unwrap().path().file_name().unwrap() == "buffalo").ok_or(AnimalClassError::FolderNotFound)??.path();
		let elep = data_folder.find(|entry| entry.as_ref().unwrap().path().file_name().unwrap() == "elephant").ok_or(AnimalClassError::FolderNotFound)??.path();
		let rhin = data_folder.find(|entry| entry.as_ref().unwrap().path().file_name().unwrap() == "rhino").ok_or(AnimalClassError::FolderNotFound)??.path();
		let zeb = data_folder.find(|entry| entry.as_ref().unwrap().path().file_name().unwrap() == "zebra").ok_or(AnimalClassError::FolderNotFound)??.path();

		let size = buff.read_dir().iter().count() + elep.read_dir().iter().count() + rhin.read_dir().iter().count() + zeb.read_dir().iter().count();

		let mut vec = Vec::with_capacity(size);
		
		buff.read_dir()?.for_each(|entry| {
			let entry = entry.unwrap();
			let image = image::open(entry.path()).unwrap().to_rgb8();
			let image: AnimalImage<B> = AnimalImage::new(image, AnimalLabel::Buffalo, device).unwrap();
			vec.push(image);
		});
		
		elep.read_dir()?.for_each(|entry| {
			let entry = entry.unwrap();
			let image = image::open(entry.path()).unwrap().to_rgb8();
			let image: AnimalImage<B> = AnimalImage::new(image, AnimalLabel::Elephant, device).unwrap();
			vec.push(image);
		});
		
		rhin.read_dir()?.for_each(|entry| {
			let entry = entry.unwrap();
			let image = image::open(entry.path()).unwrap().to_rgb8();
			let image: AnimalImage<B> = AnimalImage::new(image, AnimalLabel::Rhino, device).unwrap();
			vec.push(image);
		});
		
		zeb.read_dir()?.for_each(|entry| {
			let entry = entry.unwrap();
			let image = image::open(entry.path()).unwrap().to_rgb8();
			let image: AnimalImage<B> = AnimalImage::new(image, AnimalLabel::Zebra, device).unwrap();
			vec.push(image);
		});

		vec.shuffle(&mut thread_rng());
		
		Ok(Self { data: vec })
	}
	
	pub fn train_data(&self) -> Self {
		Self {
			data: self.data[0..(self.data.len() * 4 / 5)].to_vec()
		}
	}
	
	pub fn test_data(&self) -> Self {
		Self {
			data: self.data[(self.data.len() * 4 / 5)..].to_vec()
		}
	}
}

#[derive(Debug, Clone)]
pub struct AnimalImage<B: Backend> {
	pub image: Tensor<B, 3>, // TODO
	pub label: AnimalLabel,
}

impl <B: Backend> AnimalImage<B> {
	pub fn new(mut image: RgbImage, animal_label: AnimalLabel, device: &B::Device) -> Result<Self, AnimalClassError> {
		if image.width() != 400 {
			return Err(AnimalClassError::InvalidImageWidth);
		}
		
		if image.height() != 230 {
			image = imageops::resize(&image, 400, 230, imageops::FilterType::Lanczos3);
		}
		
		let data = TensorData::new(image.into_raw().into_iter().map(|x| (x as f32) / 255.0).collect(), [230, 400, 3]);
		let image: Tensor<B, 3> = Tensor::from_data(data, device).permute([2, 0, 1]); // [3, 230, 400]
		
		Ok(
			Self {
				image,
				label: animal_label,
			}
		)
	}
}

#[derive(Debug, Copy, Clone)]
pub enum AnimalLabel {
	Buffalo,
	Elephant,
	Rhino,
	Zebra,
}

impl From<i32> for AnimalLabel {
	fn from(label: i32) -> Self {
		match label {
			0 => AnimalLabel::Buffalo,
			1 => AnimalLabel::Elephant,
			2 => AnimalLabel::Rhino,
			3 => AnimalLabel::Zebra,
			_ => panic!("Invalid label"),
		}
	}
}

#[derive(Clone)]
pub struct AnimalBatcher<B: Backend> {
	device: B::Device
}

impl <B: Backend> AnimalBatcher<B> {
	pub fn new(device: B::Device) -> Self {
		Self { device }
	}
}

#[derive(Debug, Clone)]
pub struct AnimalBatch<B: Backend> {
	pub images: Tensor<B, 3>,
	pub targets: Tensor<B, 1, Int>,
}

impl <B: Backend> Batcher<AnimalImage<B>, AnimalBatch<B>> for AnimalBatcher<B> {
	fn batch(&self, items: Vec<AnimalImage<B>>) -> AnimalBatch<B> {
		let images = items
			.iter()
			.map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
			.map(|data| Tensor::<B, 2>::from_data(data, &self.device))
			.map(|tensor| tensor.reshape([1, 28, 28]))
			.map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
			.collect();

		let targets = items
			.iter()
			.map(|item| {
				Tensor::<B, 1, Int>::from_data(
					[(item.label as i64).elem::<B::IntElem>()],
					&self.device,
				)
			})
			.collect();

		let images = Tensor::cat(images, 0).to_device(&self.device);
		let targets = Tensor::cat(targets, 0).to_device(&self.device);

		AnimalBatch { 
			images, 
			targets 
		}
	}
}