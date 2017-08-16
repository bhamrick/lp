// Primal-dual barrier method following section 4.4.1 of
// http://www.ams.org/journals/bull/2005-42-01/S0273-0979-04-01040-7/S0273-0979-04-01040-7.pdf

use std::collections::HashSet;

use rulinalg::vector::Vector;
use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};

use problem::{StandardForm, LPResult};
use error::Error;

#[derive(Debug, Clone)]
struct InteriorState {
    x: Vector<f32>,
    y: Vector<f32>,
    z: Vector<f32>,
    mu: f32,
    problem: StandardForm,
}

impl InteriorState {
    // Returns a measure of how big the step attempted was,
    // and the actual step size as a ratio of that.
    // For each component, considers the smaller of the
    // absolute and relative changes, returns the largest
    // of those.

    // If the second returned value is 1.0, the entire Newton
    // step was feasible, and as a result the primal and
    // dual are both strictly feasible.
    fn newton_step(&mut self) -> Result<(f32, f32), Error> {
        // azx is the matrix AZ^-1X
        let mut azx = self.problem.a.clone();
        for (i, mut c) in azx.col_iter_mut().enumerate() {
            *c *= self.x[i] / self.z[i];
        }

        // Compute AZ^-1XA^T
        let m = &azx * self.problem.a.transpose();

        // Compute RHS: AZ^-1X(c - muX^-1e - A^Ty) + b - Ax
        // Compute v1 = c - A^Ty - muX^-1e
        let mut v1 = &self.problem.c - self.problem.a.transpose() * &self.y;
        for (i, entry) in v1.iter_mut().enumerate() {
            *entry -= self.mu / self.x[i];
        }

        let rhs = azx * v1 + &self.problem.b - &self.problem.a * &self.x;

        // Compute newton step for y
        let p_y = m.solve(rhs)?;

        // Compute newton step for z
        // A^Tp_y + p_z = c - A^Ty - z
        // => p_z = c - A^Ty - z - A^Tp_y
        // = c - A^T(y+p_y) - z
        let p_z = &self.problem.c
            - self.problem.a.transpose()*(&self.y + &p_y)
            - &self.z;

        // Compute newton step for x
        // Zp_x + Xp_z = mu*e - XZe
        // p_x = Z^-1(mu*e - XZe - Xp_z)
        // (p_x)_i = mu/z_i - x_i - x_i/z_i * (p_z)_i
        let mut p_x_data = Vec::new();
        for i in 0..self.x.size() {
            p_x_data.push(self.mu/self.z[i] - self.x[i] - self.x[i] * p_z[i] / self.z[i]);
        }
        let p_x = Vector::new(p_x_data);

        // Find largest step in (p_x, p_y, p_z) direction that keeps x and z positive
        let mut alpha = 1.0;
        for (i, x_i) in self.x.iter().enumerate() {
            // Constant of 0.9 is to ensure that we stay strictly positive.
            let max_step = -0.9 * x_i / p_x[i];
            if max_step > 0.0 && max_step < alpha {
                alpha = max_step;
            }
        }
        for (i, z_i) in self.z.iter().enumerate() {
            // Constant of 0.9 is to ensure that we stay strictly positive.
            let max_step = -0.9 * z_i / p_z[i];
            if max_step > 0.0 && max_step < alpha {
                alpha = max_step;
            }
        }
        let mut step_size = 0.0;

        for (i, val) in p_x.iter().enumerate() {
            let change_size = if self.x[i].abs() < 1.0 {
                val.abs()
            } else {
                (val/self.x[i]).abs()
            };
            if change_size > step_size {
                step_size = change_size;
            }
        }
        for (i, val) in p_y.iter().enumerate() {
            let change_size = if self.y[i].abs() < 1.0 {
                val.abs()
            } else {
                (val/self.y[i]).abs()
            };
            if change_size > step_size {
                step_size = change_size;
            }
        }
        for (i, val) in p_z.iter().enumerate() {
            let change_size = if self.z[i].abs() < 1.0 {
                val.abs()
            } else {
                (val/self.z[i]).abs()
            };
            if change_size > step_size {
                step_size = change_size;
            }
        }

        self.x += &p_x * alpha;
        self.y += &p_y * alpha;
        self.z += &p_z * alpha;

        Ok((step_size, alpha))
    }

    // Round the current interior point to a vertex and check if that
    // vertex is the optimum of the linear program. Returns the vertex
    // if so, None otherwise.
    fn check_rounded(&self) -> Result<Option<LPResult>, Error> {
        let mut dimensions = Vec::new();
        for (i, _) in self.x.iter().enumerate() {
            dimensions.push(i)
        }
        dimensions.sort_by(|&i, &j| self.x[j].partial_cmp(&self.x[i]).unwrap());

        let mut basis = Vec::new();

        // Pick the largest components to be in our basis (allowed nonzero components)
        for i in 0..self.problem.a.rows() {
            basis.push(dimensions[i]);
        }

        let basis_index_set : HashSet<usize> = basis.iter().cloned().collect();
        let mut nonbasis_indices = Vec::new();
        for i in 0..self.problem.a.cols() {
            if !basis_index_set.contains(&i) {
                nonbasis_indices.push(i);
            }
        }

        let c_b = self.problem.c.select(&basis);
        let c_n = self.problem.c.select(&nonbasis_indices);

        let mat_b = self.problem.a.select_cols(basis.iter());

        let lambda = mat_b.transpose().solve(c_b)?;
        let x_b = mat_b.solve(self.problem.b.clone())?;

        // If any coordinates in x_b are negative, then this is not a feasible
        // basis, and therefore not optimal.
        for &v in x_b.iter() {
            if v < 0.0 {
                return Ok(None);
            }
        }

        let mat_n = self.problem.a.select_cols(nonbasis_indices.iter());
        let s_n = c_n - mat_n.transpose() * lambda;

        let mut entering_index = None;
        for (i, &idx) in nonbasis_indices.iter().enumerate() {
            if s_n[i] < 0.0 {
                entering_index = Some(idx);
                break;
            }
        }

        if let Some(_) = entering_index {
            Ok(None)
        } else {
            // No entering index exists, so this vertex is optimal. Convert
            // the compressed vector form to the full vector and return.
            let mut opt_data = Vec::new();
            opt_data.resize(self.problem.a.cols(), 0.0);
            for (i, &idx) in basis.iter().enumerate() {
                opt_data[idx] = x_b[i];
            }
            Ok(Some(LPResult::Optimum(Vector::new(opt_data))))
        }
    }
}

pub fn solve(problem: StandardForm) -> Result<LPResult, Error> {
    internal_solve(problem, false, false)
}

fn internal_solve(problem: StandardForm, mut known_feasible: bool, mut known_bounded: bool) -> Result<LPResult, Error> {
    // TODO: Improve selection of initial mu and how it decreases.
    let mut initial_mu = 1.0;
    for c_i in problem.c.iter() {
        if c_i.abs() > initial_mu {
            initial_mu = c_i.abs();
        }
    }

    let num_vars = problem.a.cols();
    let num_duals = problem.a.rows();

    let mut state = InteriorState {
        x: Vector::ones(num_vars),
        y: Vector::zeros(num_duals),
        z: Vector::ones(num_vars),
        mu: initial_mu,
        problem: problem,
    };

    loop {
        // Run Newton's method to almost convergence
        let (step_size, alpha) = state.newton_step()?;
        println!("{:?} {:?} {:?}", step_size, alpha, state);
        println!("{:?}", state.problem.c.dot(&state.x));
        println!("{:?}", state.problem.a.transpose() * &state.y);
        println!("{:?}", state.problem.b.dot(&state.y));

        if alpha == 1.0 {
            // If we ever take a full newton step, we have
            // strictly feasible primal and dual points.
            known_feasible = true;
            known_bounded = true;
        }
        if !known_feasible {
            // Check if y is a certificate of infeasibility.
            // If A^Ty <= 0, but b^T y > 0, then the original problem
            // is infeasible. Proof:
            // Suppose x is such that Ax = b, x >= 0.
            // Then
            // (A^Ty)^Tx <= 0 (because A^Ty <= 0, x >= 0),
            // but (A^Ty)^Tx = y^TAx = y^Tb = b^Ty > 0.
            // Thus we have a contradiction.
            if state.problem.b.dot(&state.y) > 0.0 {
                let vec = state.problem.a.transpose() * &state.y;
                let mut is_certificate = true;
                for &v in vec.iter() {
                    if v > 0.0 {
                        is_certificate = false;
                        break;
                    }
                }
                if is_certificate {
                    return Ok(LPResult::Infeasible);
                }
            }
        }

        if step_size < 1e-2 {
            state.mu *= 0.5;
        }

        // Check for optimality
        match state.check_rounded().expect("check_rounded should not error") {
            Some(res) => {
                println!("{:?}", res);
                return Ok(res);
            },
            None => {},
        }
    }
}

#[test]
fn test_newton() {
    let problem = StandardForm {
        a: Matrix::new(2, 5,vec![
            3.0, 2.0, 1.0, 1.0, 0.0,
            2.0, 5.0, 3.0, 0.0, 1.0,
        ]),
        b: Vector::new(vec![10.0, 15.0]),
        c: Vector::new(vec![-2.0, -3.0, -4.0, 0.0, 0.0]),
    };

    let mut state = InteriorState {
        x: Vector::ones(5),
        y: Vector::zeros(2),
        z: Vector::ones(5),
        mu: 5.0,
        problem: problem,
    };

    for _ in 0..20 {
        match state.newton_step() {
            Ok(_) => {},
            Err(e) => panic!("Newton step returned an error: {:?}", e),
        }
    }

    // Verify that the desired equations approximately hold:
    // Ax = b
    let ax = &state.problem.a * &state.x;
    for (i, entry) in ax.iter().enumerate() {
        assert!((entry - state.problem.b[i]).abs() < 1e-4);
    }

    // A^Ty + z = c
    let aty = state.problem.a.transpose() * &state.y;
    for (i, entry) in state.problem.c.iter().enumerate() {
        assert!((entry - (aty[i] + state.z[i])).abs() < 1e-4);
    }

    // Xz = mu*1
    for (i, entry) in state.x.iter().enumerate() {
        assert!((entry * state.z[i] - state.mu).abs() < 1e-4);
    }
}

#[test]
fn test_mu_decreasing() {
    let problem = StandardForm {
        a: Matrix::new(2, 5,vec![
            3.0, 2.0, 1.0, 1.0, 0.0,
            2.0, 5.0, 3.0, 0.0, 1.0,
        ]),
        b: Vector::new(vec![10.0, 15.0]),
        c: Vector::new(vec![-2.0, -3.0, -4.0, 0.0, 0.0]),
    };

    let mut state = InteriorState {
        x: Vector::ones(5),
        y: Vector::zeros(2),
        z: Vector::ones(5),
        mu: 5.0,
        problem: problem,
    };

    for _ in 0..10 {
        for _ in 0..5 {
            match state.newton_step() {
                Ok(_) => {},
                Err(e) => panic!("Newton step returned an error: {:?}", e),
            }
        }
        state.mu *= 0.5;
    }
}

#[test]
fn test_rounding() {
    let problem = StandardForm {
        a: Matrix::new(2, 5,vec![
            3.0, 2.0, 1.0, 1.0, 0.0,
            2.0, 5.0, 3.0, 0.0, 1.0,
        ]),
        b: Vector::new(vec![10.0, 15.0]),
        c: Vector::new(vec![-2.0, -3.0, -4.0, 0.0, 0.0]),
    };

    let mut state = InteriorState {
        x: Vector::ones(5),
        y: Vector::zeros(2),
        z: Vector::ones(5),
        mu: 0.5,
        problem: problem,
    };

    for _ in 0..5 {
        match state.newton_step() {
            Ok(_) => {},
            Err(e) => panic!("Newton step returned an error: {:?}", e),
        }
    }

    let rounded_result = state.check_rounded();

    match rounded_result {
        Ok(Some(LPResult::Optimum(x))) => {
            let expected_result = [0.0, 0.0, 5.0, 5.0, 0.0];
            for (i, v) in x.iter().enumerate() {
                assert!((v - expected_result[i]).abs() < 1e-4);
            }
        },
        Ok(Some(LPResult::Infeasible)) => {
            panic!("Expected optimum vertex, got infeasible");
        },
        Ok(Some(LPResult::Unbounded)) => {
            panic!("Expected optimum vertex, got unbounded");
        },
        Ok(None) => {
            panic!("Expected optimum vertex, got nothing");
        },
        Err(_) => {
            panic!("Expected optimum vertex, got error");
        },
    }
}

#[test]
fn test_solve_simple() {
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

    match result {
        LPResult::Infeasible => panic!("Expected optimum, got infeasible"),
        LPResult::Unbounded => panic!("Expected optimum, got unbounded"),
        LPResult::Optimum(x) => {
            let expected_result = [0.0, 0.0, 5.0, 5.0, 0.0];
            for (i, v) in x.iter().enumerate() {
                assert!((v - expected_result[i]).abs() < 1e-4);
            }
        },
    }
}

#[test]
fn test_solve_unbounded() {
    let problem = StandardForm {
        a: Matrix::new(1, 2,vec![
            1.0, -2.0,
        ]),
        b: Vector::new(vec![10.0]),
        c: Vector::new(vec![-1.0, 1.9]),
    };
    let result = solve(problem)
        .expect("Solve should not fail");

    match result {
        LPResult::Infeasible => panic!("Expected unbounded, got infeasible"),
        LPResult::Unbounded => {},
        LPResult::Optimum(_) => panic!("Expected unbounded, got optimum"),
    }
}

#[test]
fn test_solve_infeasible() {
    let problem = StandardForm {
        a: Matrix::new(3, 4, vec![
            1.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, -1.0,
            1.0, 1.0, 0.0, 0.0,
        ]),
        b: Vector::new(vec![6.0, 7.0, 11.0]),
        c: Vector::new(vec![-1.0, -1.0, -1.0, -1.0]),
    };
    let result = solve(problem)
        .expect("Solve should not fail");

    match result {
        LPResult::Infeasible => {},
        LPResult::Unbounded => panic!("Expected infeasible, got unbounded"),
        LPResult::Optimum(_) => panic!("Expected infeasible, got optimum"),
    }
}

/*
#[test]
fn test_feasibility_check() {
    let feasible_problem = StandardForm {
        a: Matrix::new(2, 5, vec![
            3.0, 2.0, 1.0, 1.0, 0.0,
            2.0, 5.0, 3.0, 0.0, 1.0,
        ]),
        b: Vector::new(vec![10.0, 15.0]),
        c: Vector::new(vec![-2.0, -3.0, -4.0, 0.0, 0.0]),
    };
    assert!(is_feasible(&feasible_problem)
        .expect("Feasibility test should not fail"));

    let infeasible_problem = StandardForm {
        a: Matrix::new(3, 4, vec![
            1.0, 0.0, -1.0, 0.0,
            0.0, 1.0, 0.0, -1.0,
            1.0, 1.0, 0.0, 0.0,
        ]),
        b: Vector::new(vec![6.0, 7.0, 11.0]),
        c: Vector::new(vec![-1.0, -1.0, -1.0, -1.0]),
    };
    assert!(!is_feasible(&infeasible_problem)
        .expect("Feasibility test should not fail"));
}
*/
