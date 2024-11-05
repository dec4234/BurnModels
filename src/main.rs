// https://github.com/twistedfall/opencv-rust/issues/118#issuecomment-619608278

use log::debug;
use opencv::{
    core::{self, Point, Scalar, Size, Vector},
    imgcodecs,
    imgproc::{self, bounding_rect, contour_area, find_contours, threshold, THRESH_BINARY},
    prelude::*,
    videoio::{self, VideoCapture, CAP_ANY},
    Result
};
use opencv::types::{VectorOfVectorOfPoint, VectorOfi32};
use simple_logger::SimpleLogger;

fn main() {
    
}

fn box_contours(mut out: Mat, mat: &Mat) -> Result<()> {
    let mut contours = VectorOfVectorOfPoint::new();
    find_contours(
        &mat,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    debug!("Found {} contours", contours.len());

    for contour in contours.iter() {
        let epsilon = 0.02 * imgproc::arc_length(&contour, true)?;
        let mut approx = Mat::default();
        imgproc::approx_poly_dp(&contour, &mut approx, epsilon, true)?;

        let area = contour_area(&approx, false)?;

        if area < 100.0 {
            continue;
        }

        let rect = bounding_rect(&approx)?;

        match approx.total() {
            3 => {
                debug!("Found triangle at {}, {}", rect.x, rect.y);
            }
            4 => {
                debug!("Found square at {}, {}", rect.x, rect.y);
            }
            5 => {
                debug!("Found pentagon at {}, {}", rect.x, rect.y);
            }
            12 => {
                debug!("Found cross at {}, {}", rect.x, rect.y);
            }
            _ => {
                debug!("Found contour with {} epsilons", approx.total());
            }
        }

        imgproc::rectangle(
            &mut out,
            rect,
            Scalar::new(66.0, 87.0, 245.0, 1.0), // Red bounding box
            5,
            imgproc::LINE_8,
            0,
        )?;
    }

    imgcodecs::imwrite("output.png", &out, &Vector::new())?;
    
    Ok(())
}

#[test]
fn test_filter_background() -> Result<()> {
    SimpleLogger::new().init().unwrap();
    
    let frame = imgcodecs::imread("images/fake/simple.png", imgcodecs::IMREAD_COLOR)?;
    let out = frame.clone();

    let mut hsv = Mat::default();
    imgproc::cvt_color(&frame, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

    let mut green = Mat::default();
    let lower_green = Scalar::new(35.0, 50.0, 50.0, 0.0); // Lower bound for green
    let upper_green = Scalar::new(85.0, 255.0, 255.0, 0.0); // Upper bound for green
    core::in_range(&hsv, &lower_green, &upper_green, &mut green)?;
    
    let mut yellow = Mat::default(); // TODO: what if the shape overlaps multiple background colors?
    let lower_yellow = Scalar::new(20.0, 100.0, 100.0, 0.0); // Lower bound for yellow
    let upper_yellow = Scalar::new(30.0, 255.0, 255.0, 0.0); // Upper bound for yellow
    core::in_range(&hsv, &lower_yellow, &upper_yellow, &mut yellow)?;
    
    let mut mask = Mat::default();
    core::bitwise_or(&green, &yellow, &mut mask, &core::no_array())?; // combine all color masks

    let mut inv_mask = Mat::default();
    core::bitwise_not(&mask, &mut inv_mask, &core::no_array())?; // invert to find objects of interest

    box_contours(out, &inv_mask)?;
    Ok(())
}

#[test]
fn test_filter_targets() -> Result<()> {
    let img = imgcodecs::imread("images/real/1.png", imgcodecs::IMREAD_COLOR)?;
    let out = img.clone();

    // Convert the image to HSV
    let mut hsv = Mat::default();
    imgproc::cvt_color(&img, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

    // Define color ranges for red, orange, and blue
    let lower_red1 = Scalar::new(0.0, 100.0, 100.0, 0.0);
    let upper_red1 = Scalar::new(10.0, 255.0, 255.0, 0.0);

    let lower_red2 = Scalar::new(160.0, 100.0, 100.0, 0.0);
    let upper_red2 = Scalar::new(180.0, 255.0, 255.0, 0.0);

    let lower_orange = Scalar::new(11.0, 100.0, 100.0, 0.0);
    let upper_orange = Scalar::new(25.0, 255.0, 255.0, 0.0);

    let lower_blue = Scalar::new(100.0, 150.0, 0.0, 0.0);
    let upper_blue = Scalar::new(140.0, 255.0, 255.0, 0.0);
    
    let lower_white = Scalar::new(167.0, 174.0, 181.0, 0.0);
    let upper_white = Scalar::new(256.0, 256.0, 256.0, 0.0);
    
    let lower_green = Scalar::new(170.0, 50.0, 0.0, 0.0); 
    let upper_green = Scalar::new(255.0, 255.0, 0.0, 0.0);

    // Create masks for each color
    let mut mask_red1 = Mat::default();
    let mut mask_red2 = Mat::default();
    let mut mask_orange = Mat::default();
    let mut mask_blue = Mat::default();
    let mut mask_white = Mat::default();
    let mut mask_green = Mat::default();

    core::in_range(&hsv, &lower_red1, &upper_red1, &mut mask_red1)?;
    core::in_range(&hsv, &lower_red2, &upper_red2, &mut mask_red2)?;
    core::in_range(&hsv, &lower_orange, &upper_orange, &mut mask_orange)?;
    core::in_range(&hsv, &lower_blue, &upper_blue, &mut mask_blue)?;
    core::in_range(&hsv, &lower_white, &upper_white, &mut mask_white)?;
    core::in_range(&hsv, &lower_green, &upper_green, &mut mask_green)?;

    // Combine the red masks
    let mut mask_red = Mat::default();
    core::bitwise_or(&mask_red1, &mask_red2, &mut mask_red, &core::no_array())?;

    // Combine all the masks
    let mut mask_combined = Mat::default();
    core::bitwise_or(&mask_red, &mask_orange, &mut mask_combined, &core::no_array())?;
    core::bitwise_or(&mask_combined.clone(), &mask_white, &mut mask_combined, &core::no_array())?;
    core::bitwise_or(&mask_combined.clone(), &mask_blue, &mut mask_combined, &core::no_array())?;
    core::bitwise_or(&mask_combined.clone(), &mask_green, &mut mask_combined, &core::no_array())?;

    // Apply the mask to the original image to extract the shapes
    let mut result = Mat::default();
    core::bitwise_and(&img, &img, &mut result, &mask_combined)?;
    
    let mut gray = Mat::default();
    imgproc::cvt_color(&result, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    
    box_contours(out, &gray)?;
    
    Ok(())
}

