use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;
use burn::data::network::downloader;

pub struct Weights {
	pub(super) url: &'static str,
	pub(super) num_classes: usize,
}

impl Weights {
	/// Download the pre-trained weights to the local cache directory.
	pub fn download(&self) -> Result<PathBuf, std::io::Error> {
		// Model cache directory
		let model_dir = dirs::home_dir()
			.expect("Should be able to get home directory")
			.join(".cache")
			.join("yolox-burn");

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

pub enum YoloxX {
	/// These weights were released after the original paper implementation with slightly better results.
	/// mAP (test2017): 51.5
	Coco,
}
impl WeightsMeta for YoloxX {
	fn weights(&self) -> Weights {
		Weights {
			url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
			num_classes: 80,
		}
	}
}

pub enum YoloxNano {
	/// These weights were released after the original paper implementation with slightly better results.
	/// mAP (val2017): 25.8
	Coco,
}
impl WeightsMeta for YoloxNano {
	fn weights(&self) -> Weights {
		Weights {
			url: "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
			num_classes: 80,
		}
	}
}