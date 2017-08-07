use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

#[derive(Debug, Clone)]
pub struct StandardForm {
    // Standard form linear program:
    // Maximize c^Tx
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
