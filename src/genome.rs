
pub use crate::config::Config;
pub use crate::config::GaParams;
pub use crate::layers::Layer;
pub use crate::layers::LayerType;
pub use crate::utility::compute_l2;

#[derive(Clone, Debug)]
pub struct Genome {
    pub config: Config,
    pub input_shape: (usize, usize),
    pub layer_types: Vec<LayerType>,
    pub layers: Vec<Layer>,
    pub fitness: f64,
}

impl Genome {
    pub fn new(config: Config, input_shape: (usize, usize), layer_types: Vec<LayerType>) -> Genome {
        let mut layers: Vec<Layer> = Vec::new();

        for i in 0..layer_types.len() {
            let layer = Layer::new(layer_types[i], config, input_shape);
            layers.push(layer);
        }
        Genome {
            config: config,
            input_shape: input_shape,
            layer_types: layer_types,
            layers: layers,
            fitness: 0.0,
        }
    }

    pub fn compute_distance(&self, other: Genome) -> f64 {
        let mut distance: f64 = 0.0;

        for i in 0..self.layers.len() {
            let layer_dist : f64 = compute_l2(&self.layers[i].weights, &other.layers[i].weights);

            distance += layer_dist;

            }
        return distance / self.layers.len() as f64

    }
    fn mutate(&self) {
        // Mutate the weights and biases of the layer/genome
        for layer in self.layers.iter() {
            layer.mutate();
        }
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        // Inference here will be done in pytorch somehow.
        let mut output = input;
        for layer in self.layers.iter() {
            output = layer.predict(output);
        }
        output
    }

    fn crossover(&self, secondary_genome: Genome) -> Genome { 
        // Crossover the weights and biases of the layer/genome
        if self.layers.len() != secondary_genome.layers.len() {
            panic!("Genomes must have the same number of layers to crossover");
        }
        let mut new_layers: Vec<Layer> = Vec::new();

        for i in 0..self.layers.len() {
            let new_layer = self.layers[i].crossover(secondary_genome.layers[i].clone());
            new_layers.push(new_layer);
        }
        Genome {
            config: self.config,
            input_shape: self.input_shape,
            layer_types: self.layer_types.clone(),
            layers: new_layers,
            fitness: 0.0,
        }
    }
}