use std::path::{Path, PathBuf};
use burn::data::dataset::vision::ImageFolderDataset;
use burn::data::network::downloader;
use flate2::read::GzDecoder;

const URL: &str = "https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz";

pub trait CIFAR10Loader {
	fn cifar10_train() -> Self;
	fn cifar10_test() -> Self;
}

impl CIFAR10Loader for ImageFolderDataset {
	fn cifar10_train() -> Self {
		let root = download();
		
		Self::new_classification(root.join("train")).unwrap()
	}

	fn cifar10_test() -> Self {
		let root = download();
		
		Self::new_classification(root.join("test")).unwrap()
	}
}

pub fn download() -> PathBuf {
	let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
	let cifar_dir = example_dir.join("cifar10");
	
	let labels_file = cifar_dir.join("labels.txt");
	if !labels_file.exists() {
		let bytes = downloader::download_file_as_bytes(URL, "cifar10.tgz");
		
		let gz_buffer = GzDecoder::new(&bytes[..]);
		let mut archive = tar::Archive::new(gz_buffer);
		archive.unpack(example_dir).unwrap();
	}
	
	cifar_dir
}