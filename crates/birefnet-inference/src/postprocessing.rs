//! Post-processing utilities for BiRefNet examples.
//!
//! This module provides common post-processing operations for
//! segmentation masks and images.

use std::collections::VecDeque;

use anyhow::{Context, Result};
use birefnet_util::{
    StructuringElement, closing, custom_filter, dynamic_image_to_tensor, gaussian_kernel, opening,
    tensor_to_dynamic_image,
};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use image::{self, imageops::FilterType};

/// Apply threshold to create binary mask.
///
/// # Arguments
/// * `mask` - Input mask tensor with shape [N, 1, H, W]
/// * `threshold` - Threshold value (0.0 to 1.0)
///
/// # Returns
/// Binary mask tensor
pub fn apply_threshold<B: Backend>(mask: Tensor<B, 4>, threshold: f64) -> Tensor<B, 4> {
    mask.greater_elem(threshold).float()
}

/// Apply Gaussian blur to soften mask edges.
///
/// # Arguments
/// * `mask` - Input mask tensor with shape [N, 1, H, W]
/// * `kernel_size` - Size of the Gaussian kernel (odd number)
/// * `sigma` - Standard deviation of the Gaussian kernel
///
/// # Returns
/// Blurred mask tensor
pub fn gaussian_blur<B: Backend>(
    mask: Tensor<B, 4>,
    kernel_size: usize,
    sigma: f64,
) -> Tensor<B, 4> {
    let normalized_kernel_size = normalize_kernel_size(kernel_size);
    let effective_sigma = if sigma.is_finite() && sigma > 0.0 {
        sigma
    } else {
        // Approximate sigma used by common CV toolkits when not explicitly provided.
        normalized_kernel_size as f64 / 6.0
    };

    let device = mask.device();
    let kernel = gaussian_kernel(normalized_kernel_size, effective_sigma, &device);
    custom_filter(mask, kernel)
}

/// Morphological opening operation (erosion followed by dilation).
///
/// # Arguments
/// * `mask` - Input binary mask tensor with shape [N, 1, H, W]
/// * `kernel_size` - Size of the structuring element
///
/// # Returns
/// Processed mask tensor
pub fn morphological_opening<B: Backend>(mask: Tensor<B, 4>, kernel_size: usize) -> Tensor<B, 4> {
    let normalized_kernel_size = normalize_kernel_size(kernel_size);
    let device = mask.device();
    let structuring_element =
        StructuringElement::rectangle(normalized_kernel_size, normalized_kernel_size, &device);
    opening(mask, &structuring_element)
}

/// Morphological closing operation (dilation followed by erosion).
///
/// # Arguments
/// * `mask` - Input binary mask tensor with shape [N, 1, H, W]
/// * `kernel_size` - Size of the structuring element
///
/// # Returns
/// Processed mask tensor
pub fn morphological_closing<B: Backend>(mask: Tensor<B, 4>, kernel_size: usize) -> Tensor<B, 4> {
    let normalized_kernel_size = normalize_kernel_size(kernel_size);
    let device = mask.device();
    let structuring_element =
        StructuringElement::rectangle(normalized_kernel_size, normalized_kernel_size, &device);
    closing(mask, &structuring_element)
}

/// Remove small connected components from binary mask.
///
/// # Arguments
/// * `mask` - Input binary mask tensor with shape [N, 1, H, W]
/// * `min_size` - Minimum size of components to keep
///
/// # Returns
/// Cleaned mask tensor
pub fn remove_small_components<B: Backend>(mask: Tensor<B, 4>, min_size: usize) -> Tensor<B, 4> {
    if min_size <= 1 {
        return mask;
    }

    let dims = mask.dims();
    let device = mask.device();
    let [batch, channels, height, width] = dims;

    let mut binary = tensor_to_binary_vec(mask);
    let pixels_per_channel = height * width;
    let pixels_per_batch = channels * pixels_per_channel;

    for b in 0..batch {
        for c in 0..channels {
            let start = b * pixels_per_batch + c * pixels_per_channel;
            let end = start + pixels_per_channel;
            remove_small_components_in_place(&mut binary[start..end], height, width, min_size);
        }
    }

    binary_vec_to_tensor(binary, dims, &device)
}

/// Fill holes in binary mask.
///
/// # Arguments
/// * `mask` - Input binary mask tensor with shape [N, 1, H, W]
///
/// # Returns
/// Mask with holes filled
pub fn fill_holes<B: Backend>(mask: Tensor<B, 4>) -> Tensor<B, 4> {
    let dims = mask.dims();
    let device = mask.device();
    let [batch, channels, height, width] = dims;

    let mut binary = tensor_to_binary_vec(mask);
    let pixels_per_channel = height * width;
    let pixels_per_batch = channels * pixels_per_channel;

    for b in 0..batch {
        for c in 0..channels {
            let start = b * pixels_per_batch + c * pixels_per_channel;
            let end = start + pixels_per_channel;
            fill_holes_in_place(&mut binary[start..end], height, width);
        }
    }

    binary_vec_to_tensor(binary, dims, &device)
}

/// Comprehensive postprocessing pipeline.
///
/// # Arguments
/// * `mask` - Input mask tensor with shape [N, 1, H, W]
/// * `threshold` - Threshold for binarization
/// * `blur_kernel_size` - Size of blur kernel (0 to skip)
/// * `blur_sigma` - Sigma for Gaussian blur
/// * `morphology_kernel_size` - Size of morphology kernel (0 to skip)
/// * `min_component_size` - Minimum component size (0 to skip)
/// * `fill_holes_flag` - Whether to fill holes
///
/// # Returns
/// Processed mask tensor
pub fn postprocess_mask<B: Backend>(
    mask: Tensor<B, 4>,
    threshold: f64,
    blur_kernel_size: usize,
    blur_sigma: f64,
    morphology_kernel_size: usize,
    min_component_size: usize,
    fill_holes_flag: bool,
) -> Tensor<B, 4> {
    let mut processed = mask;

    // Apply threshold
    processed = apply_threshold(processed, threshold);

    // Apply Gaussian blur if requested
    if blur_kernel_size > 0 {
        processed = gaussian_blur(processed, blur_kernel_size, blur_sigma);
    }

    // Apply morphological operations if requested
    if morphology_kernel_size > 0 {
        processed = morphological_opening(processed, morphology_kernel_size);
        processed = morphological_closing(processed, morphology_kernel_size);
    }

    // Remove small components if requested
    if min_component_size > 0 {
        processed = remove_small_components(processed, min_component_size);
    }

    // Fill holes if requested
    if fill_holes_flag {
        processed = fill_holes(processed);
    }

    processed
}

/// Convert tensor to image data for saving.
///
/// # Arguments
/// * `tensor` - Input tensor with shape [1, 1, H, W]
///
/// # Returns
/// Vector of u8 pixel values
pub fn tensor_to_image_data<B: Backend>(tensor: Tensor<B, 4>) -> Vec<u8> {
    let [_n, _c, h, w] = tensor.dims();
    let data = tensor.to_data();

    let mut image_data = Vec::with_capacity(h * w);
    for value in data.iter::<f64>() {
        image_data.push((value.clamp(0.0, 1.0) * 255.0) as u8);
    }

    image_data
}

/// Resize tensor to target size.
///
/// # Arguments
/// * `tensor` - Input tensor with shape [N, C, H, W]
/// * `target_height` - Target height
/// * `target_width` - Target width
///
/// # Returns
/// Resized tensor
pub fn resize_tensor<B: Backend>(
    tensor: Tensor<B, 4>,
    target_height: usize,
    target_width: usize,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let [_batch_size, _channels, current_height, current_width] = tensor.dims();

    if current_height == target_height && current_width == target_width {
        return Ok(tensor);
    }

    // This approach converts tensor to image, resizes, then converts back to tensor.
    // It's inefficient but works without a direct tensor interpolation implementation.
    let dynamic_image = tensor_to_dynamic_image(tensor, false)
        .context("Failed to convert tensor to image for resizing")?;

    let resized_image = dynamic_image.resize_exact(
        target_width as u32,
        target_height as u32,
        FilterType::Lanczos3,
    );

    // Convert back to tensor
    dynamic_image_to_tensor(resized_image, device)
        .context("Failed to convert resized image back to tensor")
}

fn normalize_kernel_size(kernel_size: usize) -> usize {
    match kernel_size {
        0 => 1,
        even if even % 2 == 0 => even + 1,
        odd => odd,
    }
}

fn tensor_to_binary_vec<B: Backend>(mask: Tensor<B, 4>) -> Vec<u8> {
    let data = mask.to_data();
    let mut binary = Vec::with_capacity(data.num_elements());
    for value in data.iter::<f64>() {
        binary.push(u8::from(value > 0.5));
    }
    binary
}

fn binary_vec_to_tensor<B: Backend>(
    binary: Vec<u8>,
    dims: [usize; 4],
    device: &B::Device,
) -> Tensor<B, 4> {
    let float_mask: Vec<f32> = binary.into_iter().map(f32::from).collect();
    let data = TensorData::new(float_mask, dims).convert::<B::FloatElem>();
    Tensor::from_data(data, device)
}

fn remove_small_components_in_place(
    binary: &mut [u8],
    height: usize,
    width: usize,
    min_size: usize,
) {
    let mut visited = vec![false; binary.len()];
    let mut component = Vec::new();
    let mut queue = VecDeque::new();

    for idx in 0..binary.len() {
        if visited[idx] || binary[idx] == 0 {
            continue;
        }

        visited[idx] = true;
        queue.push_back(idx);
        component.clear();

        while let Some(current) = queue.pop_front() {
            component.push(current);
            for neighbor in neighbors(current, height, width) {
                if !visited[neighbor] && binary[neighbor] == 1 {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        if component.len() < min_size {
            for &pixel in &component {
                binary[pixel] = 0;
            }
        }
    }
}

fn fill_holes_in_place(binary: &mut [u8], height: usize, width: usize) {
    let mut visited = vec![false; binary.len()];
    let mut component = Vec::new();
    let mut queue = VecDeque::new();

    for idx in 0..binary.len() {
        if visited[idx] || binary[idx] == 1 {
            continue;
        }

        visited[idx] = true;
        queue.push_back(idx);
        component.clear();

        let mut touches_border = false;

        while let Some(current) = queue.pop_front() {
            component.push(current);
            let row = current / width;
            let col = current % width;
            if row == 0 || row + 1 == height || col == 0 || col + 1 == width {
                touches_border = true;
            }

            for neighbor in neighbors(current, height, width) {
                if !visited[neighbor] && binary[neighbor] == 0 {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        if !touches_border {
            for &pixel in &component {
                binary[pixel] = 1;
            }
        }
    }
}

fn neighbors(index: usize, height: usize, width: usize) -> [usize; 4] {
    let row = index / width;
    let col = index % width;

    let up = if row > 0 { index - width } else { index };
    let down = if row + 1 < height {
        index + width
    } else {
        index
    };
    let left = if col > 0 { index - 1 } else { index };
    let right = if col + 1 < width { index + 1 } else { index };

    [up, down, left, right]
}

#[cfg(test)]
mod tests {
    use burn::backend::Cpu;

    use super::*;

    type TestBackend = Cpu;

    #[test]
    fn gaussian_blur_changes_local_values() {
        let device = Default::default();
        let mask = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, //
                    0.0, 0.0, 0.0, //
                ],
                [1, 1, 3, 3],
            ),
            &device,
        );

        let blurred = gaussian_blur(mask, 3, 1.0);
        let values = blurred
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("blurred tensor should convert to vec");

        assert!(values[4] < 1.0, "center pixel should be smoothed");
        assert!(values[1] > 0.0, "neighboring pixel should receive energy");
    }

    #[test]
    fn remove_small_components_drops_noise() {
        let device = Default::default();
        let mask = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 1.0, 1.0, 0.0, 0.0, //
                    0.0, 1.0, 0.0, 0.0, 1.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 0.0, 0.0, //
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );

        let cleaned = remove_small_components(mask, 2);
        let values = cleaned
            .to_data()
            .convert::<u8>()
            .to_vec::<u8>()
            .expect("cleaned tensor should convert to vec");

        let isolated_pixel = values[2 * 5 + 4];
        let main_component = values[5 + 1];

        assert_eq!(isolated_pixel, 0);
        assert_eq!(main_component, 1);
    }

    #[test]
    fn fill_holes_fills_enclosed_background() {
        let device = Default::default();
        let mask = Tensor::<TestBackend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 1.0, 1.0, 1.0, 1.0, //
                    1.0, 0.0, 0.0, 0.0, 1.0, //
                    1.0, 0.0, 1.0, 0.0, 1.0, //
                    1.0, 0.0, 0.0, 0.0, 1.0, //
                    1.0, 1.0, 1.0, 1.0, 1.0, //
                ],
                [1, 1, 5, 5],
            ),
            &device,
        );

        let filled = fill_holes(mask);
        let values = filled
            .to_data()
            .convert::<u8>()
            .to_vec::<u8>()
            .expect("filled tensor should convert to vec");

        assert_eq!(values[5 + 1], 1);
        assert_eq!(values[2 * 5 + 3], 1);
    }
}
