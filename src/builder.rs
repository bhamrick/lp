use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::vec::Vec;

use rulinalg::matrix::Matrix;
#[cfg(test)]
use rulinalg::matrix::BaseMatrix;
use rulinalg::vector::Vector;

use problem::*;

#[derive(Debug, Clone)]
pub struct Constraint {
    coefficients: HashMap<usize, f32>,
    direction: Ordering,
    value: f32,
}

impl Constraint {
    pub fn new() -> Constraint {
        Constraint {
            coefficients: HashMap::new(),
            direction: Ordering::Equal,
            value: 0.0,
        }
    }

    pub fn add(mut self, var_index: usize, coeff: f32) -> Constraint {
        match self.coefficients.entry(var_index) {
            Entry::Occupied(ent) => {
                *ent.into_mut() += coeff;
            },
            Entry::Vacant(ent) => {
                ent.insert(coeff);
            },
        }
        self
    }

    pub fn value(mut self, dir: Ordering, val: f32) -> Constraint {
        self.direction = dir;
        self.value = val;
        self
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone)]
pub struct Objective {
    coefficients: HashMap<usize, f32>,
    direction: ObjectiveDirection,
}

impl Objective {
    pub fn new() -> Objective {
        Objective {
            coefficients: HashMap::new(),
            direction: ObjectiveDirection::Maximize,
        }
    }

    pub fn add(mut self, var_index: usize, coeff: f32) -> Objective {
        match self.coefficients.entry(var_index) {
            Entry::Occupied(ent) => {
                *ent.into_mut() += coeff;
            },
            Entry::Vacant(ent) => {
                ent.insert(coeff);
            },
        }
        self
    }

    pub fn direction(mut self, dir: ObjectiveDirection) -> Objective {
        self.direction = dir;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Unbounded,
    Positive,
}

#[derive(Debug, Clone)]
pub struct Problem {
    variables: Vec<VariableType>,
    constraints: Vec<Constraint>,
    objective: Objective,
}

// Mapping from specified problem to standard form problem
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableMapping {
    Direct(usize),
    Difference(usize, usize),
}

impl Problem {
    pub fn new() -> Problem {
        Problem {
            variables: Vec::new(),
            constraints: Vec::new(),
            objective: Objective::new(),
        }
    }

    pub fn new_variable(&mut self, var_type: VariableType) -> usize {
        let result = self.variables.len();
        self.variables.push(var_type);
        result
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    pub fn set_objective(&mut self, objective: Objective) {
        self.objective = objective;
    }

    pub fn standard_form(&self) -> (StandardForm, Vec<VariableMapping>) {
        let mut var_count: usize = 0;
        let mut var_mapping: Vec<VariableMapping> = Vec::new();
        let mut slack_variables: Vec<Option<usize>> = Vec::new();

        for &var_type in self.variables.iter() {
            match var_type {
                VariableType::Positive => {
                    let var_index = var_count;
                    var_count += 1;
                    var_mapping.push(VariableMapping::Direct(var_index));
                },
                VariableType::Unbounded => {
                    let pos_var = var_count;
                    var_count += 1;
                    let neg_var = var_count;
                    var_count += 1;
                    var_mapping.push(VariableMapping::Difference(pos_var, neg_var));
                },
            }
        }

        for constraint in self.constraints.iter() {
            match constraint.direction {
                Ordering::Less => {
                    let slack_var = var_count;
                    var_count += 1;
                    slack_variables.push(Some(slack_var));
                },
                Ordering::Greater => {
                    let slack_var = var_count;
                    var_count += 1;
                    slack_variables.push(Some(slack_var));
                },
                Ordering::Equal => {
                    slack_variables.push(None);
                },
            }
        }

        let mut a_data = Vec::new();
        let mut b_data = Vec::new();
        let mut c_data = Vec::new();

        for (i, constraint) in self.constraints.iter().enumerate() {
            let mut constraint_row = Vec::new();
            constraint_row.resize(var_count, 0.0);
            for (&var, &coeff) in &constraint.coefficients {
                match var_mapping[var] {
                    VariableMapping::Direct(std_var) => {
                        constraint_row[std_var] = coeff;
                    },
                    VariableMapping::Difference(pos_var, neg_var) => {
                        constraint_row[pos_var] = coeff;
                        constraint_row[neg_var] = -coeff;
                    },
                }
            }
            match constraint.direction {
                Ordering::Less => {
                    let slack_var = slack_variables[i]
                        .expect("Non-Equal constraint must have slack variable");
                    constraint_row[slack_var] = 1.0;
                },
                Ordering::Greater => {
                    let slack_var = slack_variables[i]
                        .expect("Non-Equal constraint must have slack variable");
                    constraint_row[slack_var] = -1.0;
                },
                Ordering::Equal => {},
            }

            a_data.extend(constraint_row.iter());
            b_data.push(constraint.value);
        }

        c_data.resize(var_count, 0.0);
        for (&var, &coeff) in &self.objective.coefficients {
            let sign = match self.objective.direction {
                ObjectiveDirection::Maximize => 1.0,
                ObjectiveDirection::Minimize => -1.0,
            };
            match var_mapping[var] {
                VariableMapping::Direct(std_var) => {
                    c_data[std_var] = sign * coeff;
                },
                VariableMapping::Difference(pos_var, neg_var) => {
                    c_data[pos_var] = sign * coeff;
                    c_data[neg_var] = -sign * coeff;
                },
            }
        }

        let standard_form = StandardForm {
            a: Matrix::new(self.constraints.len(), var_count, a_data),
            b: Vector::new(b_data),
            c: Vector::new(c_data),
        };

        (standard_form, var_mapping)
    }
}

#[test]
fn build_constraint() {
    let constraint = Constraint::new()
        .add(0, 1.0)
        .add(1, 2.0)
        .add(0, -3.0)
        .value(Ordering::Less, 5.0);
    assert_eq!(constraint.coefficients.len(), 2);
    assert_eq!(constraint.coefficients[&0], -2.0);
    assert_eq!(constraint.coefficients[&1], 2.0);
    assert_eq!(constraint.direction, Ordering::Less);
    assert_eq!(constraint.value, 5.0);
}

#[test]
fn build_problem() {
    let mut problem = Problem::new();
    let var1 = problem.new_variable(VariableType::Positive);
    let var2 = problem.new_variable(VariableType::Unbounded);
    let constraint1 = Constraint::new()
        .add(var1, 1.0)
        .add(var2, 2.0)
        .value(Ordering::Less, 5.0);
    problem.add_constraint(constraint1);
    let constraint2 = Constraint::new()
        .add(var1, 5.0)
        .add(var2, 3.0)
        .value(Ordering::Less, 10.0);
    problem.add_constraint(constraint2);
    let constraint3 = Constraint::new()
        .add(var2, 1.0)
        .value(Ordering::Greater, 2.0);
    problem.add_constraint(constraint3);
    let objective = Objective::new()
        .add(var1, 1.0)
        .add(var2, 1.0)
        .direction(ObjectiveDirection::Maximize);
    problem.set_objective(objective);
    assert_eq!(problem.variables.len(), 2);
    assert_eq!(problem.constraints.len(), 3);
}

#[test]
fn standard_form() {
    let mut problem = Problem::new();
    let var1 = problem.new_variable(VariableType::Positive);
    let var2 = problem.new_variable(VariableType::Unbounded);
    let constraint1 = Constraint::new()
        .add(var1, 1.0)
        .add(var2, 2.0)
        .value(Ordering::Less, 5.0);
    problem.add_constraint(constraint1);
    let constraint2 = Constraint::new()
        .add(var1, 5.0)
        .add(var2, 3.0)
        .value(Ordering::Less, 10.0);
    problem.add_constraint(constraint2);
    let constraint3 = Constraint::new()
        .add(var2, 1.0)
        .value(Ordering::Greater, 2.0);
    problem.add_constraint(constraint3);
    let objective = Objective::new()
        .add(var1, 1.0)
        .add(var2, 1.0)
        .direction(ObjectiveDirection::Maximize);
    problem.set_objective(objective);
    let (standard_form, mapping) = problem.standard_form();
    assert_eq!(standard_form.a.data(),
               &vec![1.0, 2.0, -2.0, 1.0, 0.0, 0.0,
                     5.0, 3.0, -3.0, 0.0, 1.0, 0.0,
                     0.0, 1.0, -1.0, 0.0, 0.0, -1.0]);
    assert_eq!(standard_form.a.rows(), 3);
    assert_eq!(standard_form.a.cols(), 6);
    assert_eq!(standard_form.b.data(),
               &vec![5.0, 10.0, 2.0]);
    assert_eq!(standard_form.c.data(),
               &vec![1.0, 1.0, -1.0, 0.0, 0.0, 0.0]);
    assert_eq!(mapping[0], VariableMapping::Direct(0));
    assert_eq!(mapping[1], VariableMapping::Difference(1, 2));
}
