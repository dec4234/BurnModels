use thiserror::Error;

#[derive(Debug, Error)]
pub enum AnimalClassError {
	#[error("Invalid image width")]
	InvalidImageWidth,
	#[error("Std IO error")]
	StdIoError(#[from] std::io::Error),
	#[error("Folder not found")]
	FolderNotFound,
}