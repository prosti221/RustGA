/*
    TODO:
        - Add python bindings with code for layer building and pytorch inference
        - Add mutation and crossover functionality on the layer level
        - Figure out how to do multi-threading in rust
 */
mod layers;
mod genome;
mod population;
mod config;
mod utility;
pub use config::Config;
pub use config::GaParams;
pub use layers::LayerType;
pub use population::Population;

fn main() {
    let config : Config = Config {
        parameters: GaParams {
            population_size: 256,
            generations: 2000,
            tournament_size: 4,
            mutation_stdev: 0.2,
            mutation_probability: 0.4,
        }
    };

    let input_shape: (usize, usize) = (248, 512);
    let layer_types: Vec<LayerType> = vec![LayerType::Linear, LayerType::Linear, LayerType::Linear];
    let population: Population = Population::new(config, input_shape, layer_types);
    let genome1 = population.genomes[0].clone();
    let genome2 = population.genomes[1].clone();

    println!("Distance between genomes: {:?}", genome1.compute_distance(genome2));
    println!("Population size: {:?}, Each genome has {:?} layers. Eeach layer has {:?} parameters",
        population.genomes.len(), population.genomes[0].layers.len(), population.genomes[0].layers[0].weights.len()
    );

    println!("Weights: {:?}", population.genomes[0].layers[0].weights);
}
