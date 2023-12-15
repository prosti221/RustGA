pub use crate::config::Config;
pub use crate::genome::Genome;
pub use crate::layers::LayerType;

#[derive(Clone, Debug)]
pub struct Population {
    pub config: Config,
    pub genomes: Vec<Genome>,
}

impl Population {
    pub fn new(config: Config, input_shape: (usize, usize), layer_types: Vec<LayerType>) -> Population {
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
