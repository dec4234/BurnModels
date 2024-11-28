use std::path::Path;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::Dataset;
use burn::prelude::Backend;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct GymNormalizer<B: Backend> {
	pub min: Tensor<B, 2>,
	pub max: Tensor<B, 2>,
}

impl <B: Backend> GymNormalizer<B> {
	pub fn new(min: Tensor<B, 2>, max: Tensor<B, 2>) -> Self {
		Self { min, max }
	}

	pub fn new_from_slice(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
		Self {
			min: Tensor::<B, 1>::from_floats(min, device).unsqueeze(),
			max: Tensor::<B, 1>::from_floats(max, device).unsqueeze(),
		}
	}

	pub fn normalize(&self, data: Tensor<B, 2>) -> Tensor<B, 2> {
		(data - self.min.clone()) / (self.max.clone() - self.min.clone())
	}

	pub fn denormalize(&self, data: Tensor<B, 2>) -> Tensor<B, 2> {
		(data * (self.max.clone() - self.min.clone())) + self.min.clone()
	}
}

#[derive(Debug, Clone)]
pub struct GymBatcher<B: Backend> {
	device: B::Device,
	normalizer: GymNormalizer<B>
}

impl <B: Backend> GymBatcher<B> {
	pub fn new(device: B::Device, dataset: &GymDataset) -> Self {
		let min = dataset.min_per_category();
		let max = dataset.max_per_category();

		Self {
			device: device.clone(),
			normalizer: GymNormalizer::new_from_slice(&device, &min, &max)
		}
	}
}

impl <B: Backend> Batcher<GymGoer, GymBatch<B>> for GymBatcher<B> {
	fn batch(&self, items: Vec<GymGoer>) -> GymBatch<B> {
		let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

		for item in items.iter() {
			let input = Tensor::<B, 1>::from_floats(
				[
				item.age as f32,
				item.gender.into(),
				//item.weight,
				item.height,
				item.max_bpm as f32,
				item.average_bpm as f32,
				item.resting_bpm as f32,
				item.session_duration,
				item.calories_burned,
				item.workout_type.into(),
				item.fat_percentage,
				item.water_intake,
				item.workout_frequency as f32,
				item.experience_level as f32,
				item.bmi
			],
			&self.device);

			inputs.push(input.unsqueeze());
		}

		let inputs = Tensor::cat(inputs, 0);
		let inputs = self.normalizer.normalize(inputs);

		let targets = items
			.iter()
			.map(|item| Tensor::<B, 1>::from_floats([item.weight], &self.device))
			.collect();

		let targets = Tensor::cat(targets, 0);
		
		GymBatch { 
			inputs, 
			targets 
		}
	}
}

#[derive(Debug, Clone)]
pub struct GymBatch<B: Backend> {
	pub inputs: Tensor<B, 2>,
	pub targets: Tensor<B, 1>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GymDataset {
	pub data: Vec<GymGoer>,
}

impl GymDataset {
	pub fn from<A: AsRef<Path>>(path: A) -> Self {
		let mut vec = Vec::new();
		
		csv::Reader::from_path(path).unwrap().deserialize().for_each(|record| {
			let record: GymGoer = record.unwrap();
			vec.push(record);
		});
		
		Self { data: vec }
	}
	
	// 963 entries?
	pub fn train_data(&self) -> GymDataset {
		Self {
			data: self.data[0..900].to_vec()
		}
	}
	
	pub fn test_data(&self) -> GymDataset {
		Self {
			data: self.data[900..].to_vec()
		}
	}

	pub fn min_per_category(&self) -> Vec<f32> {
		let mut min = vec![f32::MAX; 14];

		for goer in &self.data {
			min[0] = min[0].min(goer.age as f32);
			// skip gender since its a binary value
			//min[2] = min[2].min(goer.weight);
			min[2] = min[2].min(goer.height);
			min[3] = min[3].min(goer.max_bpm as f32);
			min[4] = min[4].min(goer.average_bpm as f32);
			min[5] = min[5].min(goer.resting_bpm as f32);
			min[6] = min[6].min(goer.session_duration);
			min[7] = min[7].min(goer.calories_burned);
			// skip workout type since its a categorical value
			min[9] = min[9].min(goer.fat_percentage);
			min[10] = min[10].min(goer.water_intake);
			min[11] = min[11].min(goer.workout_frequency as f32);
			min[12] = min[12].min(goer.experience_level as f32);
			min[13] = min[13].min(goer.bmi);
		}

		min
	}

	pub fn max_per_category(&self) -> Vec<f32> {
		let mut max = vec![f32::MIN; 14];

		for goer in &self.data {
			max[0] = max[0].max(goer.age as f32);
			max[1] = 1.0;
			// max[2] = max[2].max(goer.weight);
			max[2] = max[2].max(goer.height);
			max[3] = max[3].max(goer.max_bpm as f32);
			max[4] = max[4].max(goer.average_bpm as f32);
			max[5] = max[5].max(goer.resting_bpm as f32);
			max[6] = max[6].max(goer.session_duration);
			max[7] = max[7].max(goer.calories_burned);
			max[8] = 3.0;
			max[9] = max[9].max(goer.fat_percentage);
			max[10] = max[10].max(goer.water_intake);
			max[11] = max[11].max(goer.workout_frequency as f32);
			max[12] = max[12].max(goer.experience_level as f32);
			max[13] = max[13].max(goer.bmi);
		}

		max
	}
}

impl Dataset<GymGoer> for GymDataset {
	fn get(&self, index: usize) -> Option<GymGoer> {
		self.data.get(index).copied()
	}

	fn len(&self) -> usize {
		self.data.len()
	}
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GymGoer {
	#[serde(rename = "Age")]
	pub age: u8,
	#[serde(rename = "Gender")]
	pub gender: Gender,
	#[serde(rename = "Weight (kg)")]
	pub weight: f32,
	#[serde(rename = "Height (m)")]
	pub height: f32,
	#[serde(rename = "Max_BPM")]
	pub max_bpm: u8,
	#[serde(rename = "Avg_BPM")]
	pub average_bpm: u8,
	#[serde(rename = "Resting_BPM")]
	pub resting_bpm: u8,
	#[serde(rename = "Session_Duration (hours)")]
	pub session_duration: f32,
	#[serde(rename = "Calories_Burned")]
	pub calories_burned: f32,
	#[serde(rename = "Workout_Type")]
	pub workout_type: WorkoutType,
	#[serde(rename = "Fat_Percentage")]
	pub fat_percentage: f32,
	#[serde(rename = "Water_Intake (liters)")]
	pub water_intake: f32,
	#[serde(rename = "Workout_Frequency (days/week)")]
	pub workout_frequency: u8,
	#[serde(rename = "Experience_Level")]
	pub experience_level: u8,
	#[serde(rename = "BMI")]
	pub bmi: f32,
}

impl From<[f32; 15]> for GymGoer {
	fn from(value: [f32; 15]) -> Self {
		GymGoer {
			age: value[0] as u8,
			gender: value[1].into(),
			weight: value[2],
			height: value[3],
			max_bpm: value[4] as u8,
			average_bpm: value[5] as u8,
			resting_bpm: value[6] as u8,
			session_duration: value[7],
			calories_burned: value[8],
			workout_type: value[9].into(),
			fat_percentage: value[10],
			water_intake: value[11],
			workout_frequency: value[12] as u8,
			experience_level: value[13] as u8,
			bmi: value[14],
		}
	}
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Gender {
	Male,
	Female
}

impl From<f32> for Gender {
	fn from(value: f32) -> Self {
		match value {
			0.0 => Gender::Male,
			1.0 => Gender::Female,
			_ => panic!("Unknown float value for Gender")
		}
	}
}

impl From<Gender> for f32 {
	fn from(value: Gender) -> Self {
		match value {
			Gender::Male => 0.0,
			Gender::Female => 1.0
		}
	}
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WorkoutType {
	Cardio,
	HIIT,
	Strength,
	Yoga
}

impl From<f32> for WorkoutType {
	fn from(value: f32) -> Self {
		match value {
			0.0 => WorkoutType::Cardio,
			1.0 => WorkoutType::HIIT,
			2.0 => WorkoutType::Strength,
			3.0 => WorkoutType::Yoga,
			_ => panic!("Unknown float value for WorkoutType")
		}
	}
}

impl From<WorkoutType> for f32 {
	fn from(value: WorkoutType) -> Self {
		match value {
			WorkoutType::Cardio => 0.0,
			WorkoutType::HIIT => 1.0,
			WorkoutType::Strength => 2.0,
			WorkoutType::Yoga => 3.0,
		}
	}
}