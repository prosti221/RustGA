pub use crate::config::Config;
use ndarray::{Array, ShapeBuilder};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Clone, Debug)]
pub struct Layer {
    pub layer_type: LayerType,
    pub config: Config,
    pub input_shape: (usize, usize),
    pub weights: Array<f64, ndarray::Dim<[usize; 1]>>,
}

impl Layer {
    pub fn new(layer_type: LayerType, config: Config, input_shape: (usize, usize)) -> Layer {
        // Initialize random weights
        let mut weights: Array<f64, ndarray::Dim<[usize; 1]>> = Array::random(input_shape.0 * input_shape.1, Uniform::new(-1.0, 1.0));
        Layer {
            layer_type: layer_type,
            config: config,
            input_shape: input_shape,
            weights: weights,
        }
    }

    pub fn mutate(&self) {
        //TODO: Implement
        panic!("Not implemented");
    }

    pub fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        //TODO: Implement
        panic!("Not implemented");
    }

    pub fn crossover(&self, other: Layer) -> Layer {
        //TODO: Implement
        panic!("Not implemented");
    }
    
}

#[derive(Copy, Clone, Debug)]
pub enum LayerType {
    Conv2d,
    Linear,
    Softmax,
    Relu,
    MaxPool2d,
}