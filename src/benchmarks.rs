#![cfg(test)]

use bench_util::dense_seeded;
use simplex;
use interior;

use test::Bencher;

// These seeds have all been verified to have optimal solutions.
#[bench]
fn simplex_dense_10x20(b: &mut Bencher) {
    b.iter(|| {
        let problem = dense_seeded(10, 20, [1, 3, 3, 7]);
        simplex::solve(problem)
            .expect("Simplex solve should not fail");
    });
}

#[bench]
fn interior_dense_10x20(b: &mut Bencher) {
    b.iter(|| {
        let problem = dense_seeded(10, 20, [1, 3, 3, 7]);
        interior::solve(problem)
            .expect("Interior solve should not fail");
    });
}

#[bench]
fn simplex_dense_20x40(b: &mut Bencher) {
    b.iter(|| {
        let problem = dense_seeded(20, 40, [13, 37, 58, 23]);
        simplex::solve(problem)
            .expect("Simplex solve should not fail");
    });
}

#[bench]
fn interior_dense_20x40(b: &mut Bencher) {
    b.iter(|| {
        let problem = dense_seeded(20, 40, [13, 37, 58, 23]);
        interior::solve(problem)
            .expect("Interior solve should not fail");
    });
}

#[bench]
fn simplex_dense_40x80(b: &mut Bencher) {
    b.iter(|| {
        let problem = dense_seeded(40, 80, [32, 102, 87, 6]);
        simplex::solve(problem)
            .expect("Simplex solve should not fail");
    });
}

#[bench]
fn interior_dense_40x80(b: &mut Bencher) {
    b.iter(|| {
        let problem = dense_seeded(40, 80, [32, 102, 87, 6]);
        interior::solve(problem)
            .expect("Interior solve should not fail");
    });
}
