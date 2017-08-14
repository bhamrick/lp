#![cfg(test)]

use problem::StandardForm;
use rand::{Rng, SeedableRng, XorShiftRng};
use rand::distributions::normal::StandardNormal;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

// Generate a random dense LP problem with the specified size and seed
// For use in benchmarks.
pub fn dense_seeded(rows: usize, cols: usize, seed: [u32; 4]) -> StandardForm {
    assert!(rows <= cols);
    let mut rng: XorShiftRng = SeedableRng::from_seed(seed);
    let a = loop {
        let mut a_data : Vec<f32> = Vec::new();
        for _ in 0..rows {
            for _ in 0..cols {
                let StandardNormal(v) = rng.gen();
                a_data.push(v as f32);
            }
        }
        let a = Matrix::new(rows, cols, a_data);
        // TODO: Verify that A has full rank
        // But random matrices will be full rank with probability 1,
        // so we should be okay.
        break a;
    };
    let mut x_data : Vec<f32> = Vec::new();
    for _ in 0..cols {
        let StandardNormal(v) = rng.gen();
        x_data.push((v.abs() + 0.01) as f32);
    }
    let x = Vector::new(x_data);
    let b = a.clone()*x;
    let mut c_data : Vec<f32> = Vec::new();
    for _ in 0..cols {
        let StandardNormal(v) = rng.gen();
        c_data.push(v as f32);
    }
    let c = Vector::new(c_data);
    StandardForm {
        a: a,
        b: b,
        c: c,
    }
}