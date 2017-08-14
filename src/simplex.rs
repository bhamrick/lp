use std::collections::HashSet;
use std::vec::Vec;

use rulinalg::matrix::BaseMatrix;
use rulinalg::vector::Vector;
use rulinalg::matrix::Matrix;

use problem::*;
use error::Error;

#[derive(Debug, Clone)]
struct SimplexState {
    basis: Vec<usize>,
    x_b: Vector<f32>,
    problem: StandardForm,
}

#[derive(Debug, Clone)]
enum PivotResult {
    Done(LPResult),
    Pivoted(SimplexState),
}

impl SimplexState {
    // Implementation following https://en.wikipedia.org/wiki/Revised_simplex_method
    fn from_basis(problem: StandardForm, basis: &Vec<usize>)
        -> Result<SimplexState, Error> {
        let mat_b = problem.a.select_cols(basis.iter());

        let x_b = mat_b.solve(problem.b.clone())?;

        Ok(
            SimplexState{
                basis: basis.clone(),
                x_b: x_b,
                problem: problem,
            }
        )
    }

    fn pivot(self) -> Result<PivotResult, Error> {
        let num_vars = self.problem.a.cols();
        let basis_indices : HashSet<usize> = self.basis.iter().cloned().collect();
        let mut nonbasis_indices = Vec::new();
        for i in 0..num_vars {
            if !basis_indices.contains(&i) {
                nonbasis_indices.push(i);
            }
        }

        let c_b = self.problem.c.select(&self.basis);
        let c_n = self.problem.c.select(&nonbasis_indices);
        let mat_b = self.problem.a.select_cols(self.basis.iter());
        let mat_n = self.problem.a.select_cols(nonbasis_indices.iter());
        let lambda = mat_b.transpose().solve(c_b)?;
        let s_n = c_n - mat_n.transpose() * lambda;

        let mut entering_index = None;
        for (i, &idx) in nonbasis_indices.iter().enumerate() {
            if s_n[i] < 0.0 {
                entering_index = Some(idx);
                break;
            }
        }

        let q = match entering_index {
            Some(q) => q,
            None => {
                let mut x = Vec::new();
                x.resize(num_vars, 0.0);
                for (i, &idx) in self.basis.iter().enumerate() {
                    x[idx] = self.x_b[i];
                }
                return Ok(PivotResult::Done(LPResult::Optimum(Vector::new(x))));
            },
        };

        let a_q = self.problem.a.col(q).into();
        let d = mat_b.solve(a_q)?;

        let mut leaving_index = None;
        let mut limiting_ratio = None;
        for (i, _) in self.basis.iter().enumerate() {
            if d[i] > 0.0 {
                let ratio = self.x_b[i] / d[i];
                match limiting_ratio {
                    None => {
                        leaving_index = Some(i);
                        limiting_ratio = Some(ratio);
                    },
                    Some(r) => {
                        if ratio < r {
                            leaving_index = Some(i);
                            limiting_ratio = Some(ratio);
                        }
                    },
                }
            }
        }

        let i_p = match leaving_index {
            None => return Ok(PivotResult::Done(LPResult::Unbounded)),
            Some(i) => i,
        };
        let x_q = limiting_ratio
            .expect("Limiting ratio should always exist when leaving index does");

        let mut new_x_b = self.x_b - d * x_q;
        let mut new_basis = self.basis;

        new_basis[i_p] = q;
        new_x_b[i_p] = x_q;

        Ok(PivotResult::Pivoted(SimplexState {
            basis: new_basis,
            x_b: new_x_b,
            problem: self.problem,
        }))
    }

    fn optimize(self) -> Result<LPResult, Error> {
        let mut state = self;
        loop {
            let pivot_result = state.pivot()?;
            match pivot_result {
                PivotResult::Done(result) => {
                    return Ok(result);
                },
                PivotResult::Pivoted(new_state) => {
                    state = new_state;
                },
            }
        }
    }

    fn optimized_state(self) -> Result<SimplexState, Error> {
        let mut state = self;
        loop {
            let pivot_result = state.clone().pivot()?;
            match pivot_result {
                PivotResult::Done(_) => {
                    return Ok(state);
                },
                PivotResult::Pivoted(new_state) => {
                    state = new_state;
                },
            }
        }
    }
}

fn phase1_start(problem: &StandardForm) -> SimplexState {
    let num_cols = problem.a.cols();
    let mut num_zs = 0;

    for (i, _) in problem.a.row_iter().enumerate() {
        if problem.b[i] != 0.0 {
            num_zs += 1;
        }
    }

    let mut z_count = 0;
    let mut phase1_a_data = Vec::new();
    for (i, row) in problem.a.row_iter().enumerate() {
        phase1_a_data.extend_from_slice(row.raw_slice());
        let mut z_coeffs = Vec::new();
        z_coeffs.resize(num_zs, 0.0);
        if problem.b[i] > 0.0 {
            z_coeffs[z_count] = 1.0;
            z_count += 1;
        } else if problem.b[i] < 0.0 {
            z_coeffs[z_count] = -1.0;
            z_count += 1;
        }
        phase1_a_data.extend(z_coeffs);
    }

    let mut phase1_c_data = Vec::new();
    for _ in 0..num_cols {
        phase1_c_data.push(0.0);
    }
    for _ in 0..num_zs {
        phase1_c_data.push(1.0);
    }

    let phase1_problem = StandardForm {
        a: Matrix::new(problem.a.rows(), num_cols + num_zs, phase1_a_data),
        b: problem.b.clone(),
        c: Vector::new(phase1_c_data),
    };

    let mut phase1_basis = Vec::new();
    for i in num_cols .. num_cols + num_zs{
        phase1_basis.push(i);
    }

    let phase1_start_state =
        SimplexState::from_basis(phase1_problem, &phase1_basis)
        .expect("Phase 1 basis should always be valid");

    phase1_start_state
}

pub fn solve(problem: StandardForm) -> Result<LPResult, Error> {
    let phase1_start = phase1_start(&problem);
    let phase1_optimized = phase1_start.optimized_state()?;
    for &i in phase1_optimized.basis.iter() {
        if i >= problem.a.cols() {
            return Ok(LPResult::Infeasible);
        }
    }
    let phase2_start =
        SimplexState::from_basis(problem, &phase1_optimized.basis)?;
    phase2_start.optimize()
}

#[test]
fn test_simplex() {
    let problem = StandardForm {
        a: Matrix::new(2, 5,vec![
            3.0, 2.0, 1.0, 1.0, 0.0,
            2.0, 5.0, 3.0, 0.0, 1.0,
        ]),
        b: Vector::new(vec![10.0, 15.0]),
        c: Vector::new(vec![-2.0, -3.0, -4.0, 0.0, 0.0]),
    };
    let state =
        SimplexState::from_basis(problem,&vec![3, 4])
            .expect("This basis is valid");
    assert_eq!(state.x_b.data(), &vec![10.0, 15.0]);

    let simplex_result = state.optimize().expect("Optimize should not fail");
    let expected_result = vec![0.0, 0.0, 5.0, 5.0, 0.0];

    match simplex_result {
        LPResult::Unbounded => panic!("Expected optimum, got unbounded"),
        LPResult::Infeasible => panic!("Expected optimum, got infeasible"),
        LPResult::Optimum(x) => {
            for (i, v) in x.iter().enumerate() {
                assert!((v - expected_result[i]).abs() < 1.0e-4);
            }
        }
    }
}

#[test]
fn test_feasible_phase1() {
    let problem = StandardForm {
        a: Matrix::new(2, 5,vec![
            3.0, 2.0, 1.0, 1.0, 0.0,
            2.0, 5.0, 3.0, 0.0, 1.0,
        ]),
        b: Vector::new(vec![10.0, 15.0]),
        c: Vector::new(vec![-2.0, -3.0, -4.0, 0.0, 0.0]),
    };
    let phase1_start = phase1_start(&problem);
    let phase1_optimized = phase1_start.optimized_state()
        .expect("Optimizing phase 1 should not fail");
    // Test example is feasible
    for i in phase1_optimized.basis {
        assert!(i < problem.a.cols());
    }
}

#[test]
fn test_solve() {
    let problem = StandardForm {
        a: Matrix::new(2, 5,vec![
            3.0, 2.0, 1.0, 1.0, 0.0,
            2.0, 5.0, 3.0, 0.0, 1.0,
        ]),
        b: Vector::new(vec![10.0, 15.0]),
        c: Vector::new(vec![-2.0, -3.0, -4.0, 0.0, 0.0]),
    };
    let result = solve(problem)
        .expect("Solve should not fail");
    let expected_result = [0.0, 0.0, 5.0, 5.0, 0.0];
    match result {
        LPResult::Unbounded => panic!("Expected optimum, got unbounded"),
        LPResult::Infeasible => panic!("Expected optimum, got infeasible"),
        LPResult::Optimum(x) => {
            for (i, v) in x.iter().enumerate() {
                assert!((v - expected_result[i]).abs() < 1.0e-4);
            }
        }
    }
}

#[test]
fn test_solve_infeasible() {
    let problem = StandardForm {
        a: Matrix::new(3, 3, vec![
            1.0, 1.0, 0.0,
            0.0, -1.0, 1.0,
            1.0, 0.0, 1.0,
        ]),
        b: Vector::new(vec![5.0, 10.0, 12.0]),
        c: Vector::new(vec![-1.0, -1.0, -1.0]),
    };
    let result = solve(problem)
        .expect("Solve should not fail");
    assert_eq!(result, LPResult::Infeasible);
}

#[test]
fn test_solve_unbounded() {
    let problem = StandardForm {
        a: Matrix::new(1, 2, vec![
            1.0, -2.0,
        ]),
        b: Vector::new(vec![5.0]),
        c: Vector::new(vec![-1.0, -1.0]),
    };
    let result = solve(problem)
        .expect("Solve should not fail");
    assert_eq!(result, LPResult::Unbounded);
}
