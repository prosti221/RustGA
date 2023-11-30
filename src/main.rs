// This needs to be refactored eventually

//############### Configuration #################
struct Config{
    parameters: Param,
}
struct Param {
    mutation_probability: f64,
    weight_mutation_probability: f64,
    weight_perturbation_magnitude: f64,
    weight_reset_probability: f64,
    mutation_stdev: f64,
    mating_probability: f64,
}
//############### Configuration #################


//############### Population #################
struct Population {
    config: Config,
    genomes: Vec<Genome>,

}
//############### Population #################

//############### Genome #################
struct Genome {
    config: Config,
    input_shape: (usize, usize, usize),
    layer_types: Vec<LayerType>,
    layers: Vec<Layer>,
    fitness: f64,
}

impl Genome {
    fn new(config: Config, input_shape: (usize, usize, usize), layer_types: Vec<LayerType>) -> Genome {
        let mut layers = Vec<Layer>::new();
        for layer_type in layer_types.iter() {
            // Build all of the layers here
            // ...
        }
        Genome {
            config: config,
            input_shape: input_shape,
            layer_types: layer_types,
            layers: layers,
            fitness: 0.0,
        }
    }
}

impl Mutate for Genome {
    fn mutate(&self) {
        // Mutate the weights and biases of the layer/genome
        for layer in self.layers.iter() {
            layer.mutate();
        }
    }
}

impl Predict for Genome {
    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        // Inference here will be done in pytorch somehow.
        let mut output = input;
        for layer in self.layers.iter() {
            output = layer.predict(output);
        }
        output
    }
}

impl Crossover<Genome> for Genome {
    fn crossover(&self, secondary_genome: Genome) -> Genome { 
        // Crossover the weights and biases of the layer/genome
        if self.layers.len() != secondary_genome.layers.len() {
            panic!("Genomes must have the same number of layers to crossover");
        }
        let mut new_layers = Vec<Layer>::new();

        for i in 0..self.layers.len() {
            let new_layer = self.layers[i].crossover(secondary_genome.layers[i]);
            new_layers.push(new_layer);
        }
        Genome {
            config: self.config,
            input_shape: self.input_shape,
            layer_types: self.layer_types,
            layers: new_layers,
            fitness: 0.0,
        }
    }
}
//############### Genome #################

//############### Layer struct and traits #################
struct Layer {
    layer_type: LayerType,
    config: Config,
    input_shape: (usize, usize, usize),
}
enum LayerType {
    Conv2d,
    Linear,
    Softmax,
    Relu,
    MaxPool2d,
}
trait BuildLayer {
    fn build_layer(&self, input_shape: (usize, usize, usize));
    // Should reshape the flattened weights and biases into the correct shape
}

trait Predict {
    fn predict(&self, input: Vec<f64>) -> Vec<f64>;
    // Inference here will be done in pytorch somehow.
}

trait Mutate {
    fn mutate(&self);
    // Mutate the weights and biases of the layer/genome
}

trait Crossover<T>{
    fn crossover(&self, other: T) -> T;
    // Crossover the weights and biases of the layer/genome
}

trait GetWeights {
    fn get_weights(&self) -> Vec<f64>;
    // Get the weights and biases of the layer/genome
}

trait GetInputShape {
    fn get_input_shape(&self) -> (usize, usize, usize);
    // Get the input shape of the layer/genome
}

//############### Layer struct and traits #################

fn main() {
    println!("Hello, world!");
}



