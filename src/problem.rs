use rulinalg::matrix::{BaseMatrix, Matrix};
use rulinalg::vector::Vector;

#[derive(Debug, Clone)]
pub struct StandardForm {
    // Standard form linear program:
    // Minimize c^Tx
    // subject to Ax = b
    // and x >= 0
    pub a: Matrix<f32>,
    pub b: Vector<f32>,
    pub c: Vector<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LPResult {
    Unbounded,
    Infeasible,
    Optimum(Vector<f32>),
}

impl StandardForm {
    pub fn dual(&self) -> StandardForm {
        // The dual of the standard form problem (when put into standard form) is
        // Minimize -b^T (y_+ - y_-)
        // Subject to
        // A^T(y_+ - y_-) + z = c
        // y_+, y_-, z >= 0
        // Note that the resulting objective value will be negated from
        // the primal.

        let a_t = self.a.transpose();
        let new_a = a_t
            .hcat(&(-(&a_t)))
            .hcat(&Matrix::identity(a_t.rows()));
        let new_b = self.c.clone();
        let mut new_c_data = Vec::new();
        for &v in self.b.iter() {
            new_c_data.push(-v);
        }
        for &v in self.b.iter() {
            new_c_data.push(v);
        }
        let new_c = Vector::new(new_c_data);
        StandardForm {
            a: new_a,
            b: new_b,
            c: new_c,
        }
    }
}
