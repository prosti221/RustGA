/*
This needs to be refactored into seperate files eventually
    TODO:
        - Add python bindings with code for layer building and pytorch inference
        - Add mutation and crossover functionality on the layer level
        - Figure out how to do multi-threading in rust
 */
use core::panic;
use ndarray::{Array, ShapeBuilder};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;


//############### Utility functions #################
pub fn compute_l2(a: &Array<f64, ndarray::Dim<[usize; 1]>>, b: &Array<f64, ndarray::Dim<[usize; 1]>>) -> f64 {
    let mut distance: f64 = 0.0;
    distance = (a - b).mapv(|x| x.powi(2)).sum();

    return distance.sqrt();
}

//############### Utility functions #################

//############### Configuration #################
#[derive(Copy, Clone, Debug)]
struct Config{
    parameters: GaParams,

}
#[derive(Copy, Clone, Debug)]
struct GaParams {
    population_size: usize,
    mutation_probability: f64,
    weight_mutation_probability: f64,
    weight_perturbation_magnitude: f64,
    weight_reset_probability: f64,
    mutation_stdev: f64,
    mating_probability: f64,
}
//############### Configuration #################


//############### Population #################

#[derive(Clone, Debug)]
struct Population {
    config: Config,
    genomes: Vec<Genome>,
}

impl Population {
    fn new(config: Config, input_shape: (usize, usize), layer_types: Vec<LayerType>) -> Population {
        let mut genomes: Vec<Genome> = Vec::new();
        for _ in 0..config.parameters.population_size {
            let genome = Genome::new(config, input_shape, layer_types.clone());
            genomes.push(genome);
            println!("Genomes: {:?}", genomes.len());
        }
        Population {
            config: config,
            genomes: genomes,
        }
    }
    fn compute_distances(&self) -> Vec<f64> {
        let mut distances: Vec<f64> = Vec::new();
        for i in 0..self.genomes.len() {
            for j in i..self.genomes.len() {
                let distance = self.genomes[i].compute_distance(self.genomes[j].clone());
                distances.push(distance);
            }
        }
        distances
    }

    fn compute_fitness(&self) -> Vec<f64> {
        let mut fitnesses: Vec<f64> = Vec::new();
        for genome in self.genomes.iter() {
            let fitness = genome.fitness;
            fitnesses.push(fitness);
        }
        fitnesses
    }
}

//############### Population #################

//############### Genome #################
#[derive(Clone, Debug)]
struct Genome {
    config: Config,
    input_shape: (usize, usize),
    layer_types: Vec<LayerType>,
    layers: Vec<Layer>,
    fitness: f64,
}

impl Genome {
    fn new(config: Config, input_shape: (usize, usize), layer_types: Vec<LayerType>) -> Genome {
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

    fn compute_distance(&self, other: Genome) -> f64 {
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
//############### Genome #################

//############### Layer struct #################
#[derive(Clone, Debug)]
struct Layer {
    layer_type: LayerType,
    config: Config,
    input_shape: (usize, usize),
    weights: Array<f64, ndarray::Dim<[usize; 1]>>,
}

impl Layer {
    fn new(layer_type: LayerType, config: Config, input_shape: (usize, usize)) -> Layer {
        // Initialize random weights
        let mut weights: Array<f64, ndarray::Dim<[usize; 1]>> = Array::random(input_shape.0 * input_shape.1, Uniform::new(-1.0, 1.0));
        Layer {
            layer_type: layer_type,
            config: config,
            input_shape: input_shape,
            weights: weights,
        }
    }

    fn mutate(&self) {
        //TODO: Implement
        panic!("Not implemented");
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        //TODO: Implement
        panic!("Not implemented");
    }

    fn crossover(&self, other: Layer) -> Layer {
        //TODO: Implement
        panic!("Not implemented");
    }
    
}

#[derive(Copy, Clone, Debug)]
enum LayerType {
    Conv2d,
    Linear,
    Softmax,
    Relu,
    MaxPool2d,
}
//############### Layer struct #################

fn main() {
    let config : Config = Config {
        parameters: GaParams {
            population_size: 10,
            mutation_probability: 0.1,
            weight_mutation_probability: 0.1,
            weight_perturbation_magnitude: 0.1,
            weight_reset_probability: 0.1,
            mutation_stdev: 0.1,
            mating_probability: 0.1,
        }
    };

    let input_shape: (usize, usize) = (248, 512);
    let layer_types: Vec<LayerType> = vec![LayerType::Linear, LayerType::Linear, LayerType::Linear];
    let mut population: Population = Population::new(config, input_shape, layer_types);
    let mut genome1 = population.genomes[0].clone();
    let mut genome2 = population.genomes[1].clone();

    println!("Distance between genomes: {:?}", genome1.compute_distance(genome2));
    println!("Population size: {:?}, Each genome has {:?} layers. Eeach layer has {:?} parameters",
        population.genomes.len(), population.genomes[0].layers.len(), population.genomes[0].layers[0].weights.len()
    );

    println!("Weights: {:?}", population.genomes[0].layers[0].weights);
}
