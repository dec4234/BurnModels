use rayon::iter::ParallelIterator;
use std::ops::AddAssign;
use std::sync::{Arc, Mutex};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::data::dataset::vision::{Annotation, ImageDatasetItem, ImageFolderDataset, PixelDepth};
use burn::prelude::Backend;
use burn::tensor::{ElementConversion, Int, Shape, Tensor, TensorData};
use rayon::iter::IntoParallelIterator;
use crate::dataset::CIFAR10Loader;

const MEAN: [f32; 3] = [0.4914, 0.48216, 0.44653];
const STD: [f32; 3] = [0.24703, 0.24349, 0.26159];

#[derive(Clone)]
pub struct Normalizer<B: Backend> {
	pub mean: Tensor<B, 4>,
	pub std: Tensor<B, 4>,
}

impl <B: Backend> Normalizer<B> {
	pub fn new(device: &B::Device) -> Self {
		let mean = Tensor::<B, 1>::from_floats(MEAN, device).reshape([1, 3, 1, 1]);
		let std = Tensor::<B, 1>::from_floats(STD, device).reshape([1, 3, 1, 1]);
		Self { mean, std }
	}
	
	pub fn normalize(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
		(input - self.mean.clone()) / self.std.clone()
	}
}

#[derive(Clone, Debug)]
pub struct ClassificationBatch<B: Backend> {
	pub images: Tensor<B, 4>,
	pub targets: Tensor<B, 1, Int>,
	pub images_path: Vec<String>,
}

#[derive(Clone)]
pub struct ClassificationBatcher<B: Backend> {
	normalizer: Normalizer<B>,
	device: B::Device,
}

impl <B: Backend> ClassificationBatcher<B> {
	pub fn new(device: B::Device) -> Self {
		Self {
			normalizer: Normalizer::<B>::new(&device),
			device,
		}
	}
}

impl<B: Backend> Batcher<ImageDatasetItem, ClassificationBatch<B>> for ClassificationBatcher<B> {
	fn batch(&self, items: Vec<ImageDatasetItem>) -> ClassificationBatch<B> {
		fn image_as_vec_u8(item: ImageDatasetItem) -> Vec<u8> {
			// Convert Vec<PixelDepth> to Vec<u8> (we know that CIFAR images are u8)
			item.image
				.into_iter()
				.map(|p: PixelDepth| -> u8 { p.try_into().unwrap() })
				.collect::<Vec<u8>>()
		}

		let targets = items
			.iter()
			.map(|item| {
				// Expect class label (int) as target
				if let Annotation::Label(y) = item.annotation {
					Tensor::<B, 1, Int>::from_data(
						TensorData::from([(y as i64).elem::<B::IntElem>()]),
						&self.device,
					)
				} else {
					panic!("Invalid target type")
				}
			})
			.collect();

		// Original sample path
		let images_path: Vec<String> = items.iter().map(|item| "item_image_path".to_string()).collect();

		let images = items
			.into_iter()
			.map(|item| TensorData::new(image_as_vec_u8(item), Shape::new([32, 32, 3])))
			.map(|data| {
				Tensor::<B, 3>::from_data(data.convert::<B::FloatElem>(), &self.device)
					// permute(2, 0, 1)
					.swap_dims(2, 1) // [H, C, W]
					.swap_dims(1, 0) // [C, H, W]
			})
			.map(|tensor| tensor / 255) // normalize between [0, 1]
			.collect();

		let images = Tensor::stack(images, 0);
		let targets = Tensor::cat(targets, 0);

		let images = self.normalizer.normalize(images);

		ClassificationBatch {
			images,
			targets,
			images_path,
		}
	}
}

pub fn compress(pixel: u8) -> f32 {
	pixel as f32 / 255.0
}

// Outputs from this match the provided mean and STD values from the example
#[ignore]
#[test]
pub fn calculate_mean() {
	println!("Calculating mean and std...");

	let data = ImageFolderDataset::cifar10_train();
	let r_sum = Arc::new(Mutex::new(0.0f64));
	let g_sum = Arc::new(Mutex::new(0.0f64));
	let b_sum = Arc::new(Mutex::new(0.0f64));
	let count = Arc::new(Mutex::new(0));

	(0..data.len()).into_par_iter().for_each(|i| {
		let item = data.get(i).unwrap();
		let image = item.image;
		let mut r = 0.0f64;
		let mut g = 0.0f64;
		let mut b = 0.0f64;
		let mut c = 0;

		for i in 0..image.len() {
			if let PixelDepth::U8(u) = image[i] {
				if i % 3 == 0 {
					r += compress(u) as f64;
					c += 1;
				} else if i % 3 == 1 {
					g += compress(u) as f64;
				} else {
					b += compress(u) as f64;
				}
			}
		}

		count.lock().unwrap().add_assign(c);

		r_sum.lock().unwrap().add_assign(r);
		g_sum.lock().unwrap().add_assign(g);
		b_sum.lock().unwrap().add_assign(b);
	});

	let count = count.lock().unwrap().clone();

	let r_mean = r_sum.lock().unwrap().clone() / count as f64;
	let g_mean = g_sum.lock().unwrap().clone() / count as f64;
	let b_mean = b_sum.lock().unwrap().clone() / count as f64;

	let r_sum_std = Arc::new(Mutex::new(0.0f64));
	let g_sum_std = Arc::new(Mutex::new(0.0f64));
	let b_sum_std = Arc::new(Mutex::new(0.0f64));

	(0..data.len()).into_par_iter().for_each(|i| {
		let item = data.get(i).unwrap();
		let image = item.image;
		let mut r_s = 0.0f64;
		let mut g_s = 0.0f64;
		let mut b_s = 0.0f64;

		for i in 0..image.len() {
			if let PixelDepth::U8(u) = image[i] {
				if i % 3 == 0 {
					r_s += (compress(u) as f64 - r_mean).powi(2);
				} else if i % 3 == 1 {
					g_s += (compress(u) as f64 - g_mean).powi(2);
				} else {
					b_s += (compress(u) as f64 - b_mean).powi(2);
				}
			}
		}

		r_sum_std.lock().unwrap().add_assign(r_s);
		g_sum_std.lock().unwrap().add_assign(g_s);
		b_sum_std.lock().unwrap().add_assign(b_s);
	});

	let std_r = f32::sqrt((r_sum_std.lock().unwrap().clone() / count as f64) as f32);
	let std_g = f32::sqrt((g_sum_std.lock().unwrap().clone() / count as f64) as f32);
	let std_b = f32::sqrt((b_sum_std.lock().unwrap().clone() / count as f64) as f32);

	println!("Mean: [{}, {}, {}]", r_mean, g_mean, b_mean);
	println!("Std: [{}, {}, {}]", std_r, std_g, std_b);
	println!("Count: {}", count);
	println!(
		"Sums: [{}, {}, {}]",
		r_sum.lock().unwrap(),
		g_sum.lock().unwrap(),
		b_sum.lock().unwrap()
	);
}