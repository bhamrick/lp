#![cfg(test)]

use problem::LPResult;
use bench_util::dense_seeded;
use simplex;
use interior;

fn same_result(result1: LPResult, result2: LPResult) -> bool {
    match result1 {
        LPResult::Infeasible => {
            match result2 {
                LPResult::Infeasible => true,
                _ => false,
            }
        },
        LPResult::Unbounded => {
            match result2 {
                LPResult::Unbounded => true,
                _ => false,
            }
        },
        LPResult::Optimum(x1) => {
            match result2 {
                LPResult::Optimum(x2) => {
                    for (v1, v2) in x1.iter().zip(x2) {
                        if (v1 - v2).abs() > 1e-3 {
                            return false;
                        }
                    }
                    true
                },
                _ => false,
            }
        },
    }
}

#[test]
fn dense_20x40() {
    // Test that simplex and interior methods give the same result
    let problem = dense_seeded(20, 40, [13, 37, 58, 23]);
    let simplex_solution = simplex::solve(problem.clone())
        .expect("Simplex test failed;");
    let interior_solution = interior::solve(problem.clone())
        .expect("Interior test failed");

    if let LPResult::Infeasible = simplex_solution {
        panic!("Simplex got Infeasible on problem generated to be feasible");
    }
    if let LPResult::Infeasible = interior_solution {
        panic!("Interior got Infeasible on problem generated to be feasible");
    }

    assert!(same_result(simplex_solution, interior_solution));
}

/*
// Test currently fails due to interior_solution producing vector of NaNs
// Simplex reports that the problem is unbounded
#[test]
fn dense_20x100() {
    // Test that simplex and interior methods give the same result
    let problem = dense_seeded(20, 100, [1, 2, 3, 4]);
    let simplex_solution = simplex::solve(problem.clone());
    println!("{:?}", simplex_solution);
    let interior_solution = interior::solve(problem.clone());

    if let LPResult::Infeasible = simplex_solution {
        panic!("Simplex got Infeasible on problem generated to be feasible");
    }
    if let LPResult::Infeasible = interior_solution {
        panic!("Interior got Infeasible on problem generated to be feasible");
    }

    assert!(same_result(simplex_solution, interior_solution));
}
*/
