#[derive(Copy, Clone, Debug)]
pub struct Config{
    pub parameters: GaParams,

}
#[derive(Copy, Clone, Debug)]
pub struct GaParams {
    pub population_size: usize,
    pub generations: usize,
    pub tournament_size: usize,
    pub mutation_stdev: f64,
    pub mutation_probability: f64,
}