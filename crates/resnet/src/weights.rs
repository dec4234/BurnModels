use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;
use burn::data::network::downloader;

pub struct Weights {
	pub url: &'static str,
	pub num_classes: usize,
}

impl Weights {
	pub fn download(&self) -> Result<PathBuf, std::io::Error> {
		// Model cache directory
		let model_dir = dirs::home_dir()
			.expect("Should be able to get home directory")
			.join(".cache")
			.join("resnet-burn");

		if !model_dir.exists() {
			create_dir_all(&model_dir)?;
		}

		let file_base_name = self.url.rsplit_once('/').unwrap().1;
		let file_name = model_dir.join(file_base_name);
		if !file_name.exists() {
			// Download file content
			let bytes = downloader::download_file_as_bytes(self.url, file_base_name);

			// Write content to file
			let mut output_file = File::create(&file_name)?;
			let bytes_written = output_file.write(&bytes)?;

			if bytes_written != bytes.len() {
				return Err(std::io::Error::new(
					std::io::ErrorKind::InvalidData,
					"Failed to write the whole model weights file.",
				));
			}
		}

		Ok(file_name)
	}
}

pub trait WeightsMeta {
	fn weights(&self) -> Weights;
}

pub enum ResNet18 {
	/// These weights reproduce closely the results of the original paper.
	/// Top-1 accuracy: 69.758%.
	/// Top-5 accuracy: 89.078%.
	ImageNet1kV1,
	
}
impl WeightsMeta for ResNet18 {
	fn weights(&self) -> Weights {
		Weights {
			url: "https://download.pytorch.org/models/resnet18-f37072fd.pth",
			num_classes: 1000,
		}
	}
}