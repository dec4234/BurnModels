use std::path::Path;
use std::time::SystemTime;
use burn::backend::Autodiff;
use burn::prelude::{Backend, Device, Module, TensorData};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{Element, Tensor};
use burn_tch::{LibTorch, LibTorchDevice};
use crate::imagenet::Normalizer;
use crate::resnet::ResNet;
use crate::weights::ResNet18;

mod block;
mod config;
mod resnet;
mod weights;
mod imagenet;

const SIDE: u32 = 224;
const MODEL_PATH: &str = "resnet18-ImageNet1k";

fn main() {
    println!("Checking CUDA configuration for training...");
    assert!(tch::utils::has_cuda(), "Could not detect valid CUDA configuration");

    let time = SystemTime::now();

    let device = LibTorchDevice::Cuda(0);
    
    let model: ResNet<Autodiff<LibTorch>> = ResNet::resnet18_pretrained(ResNet18::ImageNet1kV1, &device).unwrap();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(MODEL_PATH, &recorder).unwrap();
    let model = model.load_file(MODEL_PATH, &recorder, &device).unwrap();
    
    infer("C:/Users/decar/RustroverProjects/uascv25/crates/resnet/sock.jpg", &model, &device);

    println!("Time to infer: {}", time.elapsed().unwrap().as_millis() as f64 / 1000.0);
}

pub fn infer<A: AsRef<Path>>(path: A, resnet: &ResNet<Autodiff<LibTorch>>, device: &Device<LibTorch>) {
    let img = resized_image(path);
    let tensor = to_tensor(img.into_rgb8().into_raw(), [SIDE as usize, SIDE as usize, 3], device).unsqueeze::<4>();
    
    let x = Normalizer::new(device).normalize(tensor);
    
    let out = resnet.forward(x);
    let (score, idx) = out.max_dim_with_indices(1);
    let idx = idx.into_scalar() as usize;
    
    println!("Predicted: {:?} with score: {}", imagenet::CLASSES[idx], score.into_scalar());
}

pub fn resized_image<A: AsRef<Path>>(path: A) -> image::DynamicImage {
    println!("Path is: {:?}", path.as_ref());
    let img = image::open(path).unwrap();
    img.resize_exact(SIDE, SIDE, image::imageops::FilterType::Lanczos3)
}

pub fn to_tensor<B: Backend, T: Element>(data: Vec<T>, shape: [usize; 3], device: &Device<B>) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(TensorData::new(data, shape).convert::<B::FloatElem>(), device).permute([2, 0 ,1]) / 255
}
