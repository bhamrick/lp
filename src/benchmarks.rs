#![cfg(test)]

use bench_util::dense_seeded;
use simplex;
use interior;

use test::Bencher;

#[bench]
fn simplex_dense_20x40(b: &mut Bencher) {
    b.iter(|| {
        let problem = dense_seeded(20, 40, [13, 37, 58, 23]);
        simplex::solve(problem);
    });
}

#[bench]
fn interior_dense_20x40(b: &mut Bencher) {
    b.iter(|| {
        let problem = dense_seeded(20, 40, [13, 37, 58, 23]);
        interior::solve(problem);
    });
}
