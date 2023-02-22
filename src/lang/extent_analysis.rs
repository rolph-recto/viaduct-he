use interval::ops::Hull;
use std::hash::Hash;

use super::*;

#[derive(Copy,Clone,Debug,Eq,Hash,PartialEq)]
pub struct ConstraintVar(usize);

#[derive(Copy,Clone,Debug,Eq,Hash,PartialEq)]
pub struct ShapeId(usize);

enum ExtentConstraint {
    Equals(ConstraintVar, ConstraintVar),
    AtLeast(ConstraintVar, Interval<i64>),
}

type ShapeConstraint = Vec<ConstraintVar>;

/// dataflow analysis in a lattice of intervals
/// computes the minimum padding required to make array extents "fit"
pub struct ExtentAnalysis {
    cur_constraint_id: usize,
    cur_shape_id: usize,
    constraints: Vec<ExtentConstraint>,
    constraint_vars: Vec<ConstraintVar>,
    shape_map: HashMap<ShapeId, Vec<ConstraintVar>>,
}

impl ExtentAnalysis {
    pub fn new() -> Self {
        ExtentAnalysis {
            cur_constraint_id: 0,
            cur_shape_id: 0,
            constraints: Vec::new(),
            constraint_vars: Vec::new(),
            shape_map: HashMap::new(),
        }
    }

    fn fresh_constraint_var(&mut self) -> ConstraintVar {
        let id = self.cur_constraint_id;
        self.cur_constraint_id += 1;
        let var = ConstraintVar(id);
        self.constraint_vars.push(var);
        var
    }

    fn fresh_shape_id(&mut self) -> ShapeId {
        let id = self.cur_shape_id;
        self.cur_shape_id += 1;
        ShapeId(id)
    }

    pub fn register_shape(&mut self, dims: usize) -> ShapeId {
        let constraints: Vec<ConstraintVar> =
            (0..dims).map(|_| self.fresh_constraint_var()).collect();
        let id = self.fresh_shape_id();
        self.shape_map.insert(id, constraints);
        id
    }

    pub fn add_equals_constraint(&mut self, id1: ShapeId, head1: usize, id2: ShapeId, head2: usize) {
        if let (Some(vars1), Some(vars2)) = (self.shape_map.get(&id1), self.shape_map.get(&id2)) {
            let suf1 = &vars1[head1..];
            let suf2= &vars2[head2..];
            if suf1.len() == suf1.len() {
                for (v1, v2) in suf1.iter().zip(suf2.iter()) {
                    self.constraints.push(ExtentConstraint::Equals(*v1, *v2));
                }
            } else {
                panic!("trying to add equality constraints to shapes with different dimensions")
            }

        } else {
            panic!("trying to add equality constraint to unregistered shapes")
        }
    }

    pub fn add_atleast_constraint(&mut self, id: ShapeId, head: usize, shape: im::Vector<Interval<i64>>) {
        if let Some(vars) = self.shape_map.get(&id) {
            let suf = &vars[head..];
            if suf.len() == shape.len() {
                for (v, extent) in vars.iter().zip(shape.iter()) {
                    self.constraints.push(ExtentConstraint::AtLeast(*v, *extent));
                }

            } else {
                panic!("trying to add equality constraints to shapes with different dimensions")
            }

        } else {
            panic!("trying to add atleast constraint to unregistered shape")
        }
    }

    pub fn solve(&mut self) -> HashMap<ShapeId, im::Vector<Interval<i64>>> {
        let mut solution: HashMap<ConstraintVar, Interval<i64>> = HashMap::new();
        let mut quiesce = false;

        // find fixpoint solution to constraints;
        // this just implements a simple linear pass instead of doing
        // the usual dataflow analysis optimizations like
        // keeping track of which constraints to wake when a solution is updated,
        // toposorting the connected components of the graph, etc.
        while !quiesce {
            quiesce = true;
            for constraint in self.constraints.iter() {
                match constraint {
                    ExtentConstraint::Equals(var1, var2) => {
                        let new_sol_opt =
                            match (solution.get(var1), solution.get(var2)) {
                                (None, None) => None,
                                (None, Some(extent2)) => Some(*extent2),
                                (Some(extent1), None) => Some(*extent1),
                                (Some(extent1), Some(extent2)) => {
                                    Some(extent1.hull(extent2))
                                }
                            };

                        if let Some(new_sol) = new_sol_opt {
                            solution.insert(*var1, new_sol);
                            solution.insert(*var2, new_sol);
                        }
                    },

                    ExtentConstraint::AtLeast(var, extent) => {
                        match solution.get(var) {
                            Some(cur_sol) => {
                                solution.insert(*var, extent.hull(cur_sol));
                            },

                            None => {
                                solution.insert(*var, *extent);
                            }
                        }
                    }
                }
            }
        }

        // collect solutions into 
        let mut shape_solution: HashMap<ShapeId, im::Vector<Interval<i64>>> = HashMap::new();
        for (&node, extent_vars) in self.shape_map.iter() {
            let shape: im::Vector<Interval<i64>> =
                extent_vars.iter()
                .map(|var| {
                    if let Some(extent) = solution.get(var) {
                        *extent
                    } else {
                        panic!("constraint var has no solution because it is unconstrained")
                    }
                })
                .collect();

            shape_solution.insert(node, shape);
        }

        shape_solution
    }
}
