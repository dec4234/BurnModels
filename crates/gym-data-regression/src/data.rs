use std::path::Path;
use burn::data::dataloader::Dataset;
use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

pub struct GymBatch<B: Backend> {
	device: B::Device,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GymDataset {
	pub data: Vec<GymGoer>,
}

impl GymDataset {
	pub fn from<A: AsRef<Path>>(path: A) -> Self {
		let mut vec = Vec::new();
		
		csv::Reader::from_path(path).unwrap().deserialize().for_each(|record| {
			let record: [f32; 15] = record.unwrap();
			vec.push(record.into());
		});
		
		Self { data: vec }
	}
	
	// 963 entries?
	pub fn train_data(&self) -> &[GymGoer] {
		&self.data[0..800]
	}
	
	pub fn test_data(&self) -> &[GymGoer] {
		&self.data[900..]
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