
use ndarray::{Array, ShapeBuilder};

pub fn compute_l2(a: &Array<f64, ndarray::Dim<[usize; 1]>>, b: &Array<f64, ndarray::Dim<[usize; 1]>>) -> f64 {
    let mut distance: f64 = 0.0;
    distance = (a - b).mapv(|x| x.powi(2)).sum();

    return distance;
}

