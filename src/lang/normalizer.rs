/// normalizer.rs
/// normalizes source expressions into an index-free representation.

use std::{collections::HashMap, cmp::max};

use interval::{Interval, ops::{Range, Hull}};
use gcollections::ops::{bounded::Bounded, Subset};

use super::{SourceExpr, NormalizedExpr};

use crate::lang::*;

#[derive(Clone)]
enum PathInfo {
    Index { index: IndexName, extent: Extent },
    Reduce { op: ExprOperator }
}

#[derive(Copy,Clone,Debug,Eq,Hash,PartialEq)]
struct ConstraintVar(usize);

struct ExtentConstraint { var1: ConstraintVar, var2: ConstraintVar }

pub struct ExprNormalizer {
    cur_expr_id: usize,
    cur_constraint_id: usize,
    constraints: Vec<ExtentConstraint>,
    constraint_vars: Vec<ConstraintVar>,
    solution: HashMap<ConstraintVar, Extent>,
    node_vars: HashMap<ExprId, Vec<ConstraintVar>>,
}

impl ExprNormalizer {
    pub fn new() -> Self {
        ExprNormalizer {
            cur_expr_id: 0,
            cur_constraint_id: 0,
            constraints: Vec::new(),
            constraint_vars: Vec::new(),
            solution: HashMap::new(),
            node_vars: HashMap::new(),
        }
    }

    fn fresh_expr_id(&mut self) -> usize {
        let id = self.cur_expr_id;
        self.cur_expr_id += 1;
        id
    }

    fn fresh_constraint_var(&mut self) -> ConstraintVar {
        let id = self.cur_constraint_id;
        self.cur_constraint_id += 1;
        let var = ConstraintVar(id);
        self.constraint_vars.push(var);
        var
    }

    fn index_expr_to_interval(&self, index_expr: &IndexExpr, index_store: &HashMap<String,Extent>) -> Extent {
        match index_expr {
            IndexExpr::IndexVar(var) => {
                index_store[var]
            },

            IndexExpr::IndexLiteral(val) => {
                Interval::new(*val, *val)
            }

            IndexExpr::IndexOp(op, expr1, expr2) => {
                let interval1 = self.index_expr_to_interval(expr1, index_store);
                let interval2 = self.index_expr_to_interval(expr2, index_store);
                match op {
                    ExprOperator::OpAdd => interval1 + interval2,
                    ExprOperator::OpSub => interval1 - interval2,
                    ExprOperator::OpMul => interval1 * interval2,
                }
            }
        }
    }

    /// transformation that removes indices from source expressions.
    fn lower(&mut self, expr: &SourceExpr, store: &ExtentStore, path: &im::Vector<PathInfo> ) -> NormalizedExpr {
        match expr {
            SourceExpr::ForNode(index, extent, body) => {
                let new_path = 
                    &im::Vector::unit(PathInfo::Index {
                        index: index.clone(), extent: *extent
                    }) + path;

                self.lower(body, store, &new_path)
            },

            SourceExpr::ReduceNode(op, body) => {
                let new_path = 
                    &im::Vector::unit(PathInfo::Reduce { op: *op }) + path;
                let new_body = self.lower(body, store, &new_path);

                NormalizedExpr::ReduceNode(*op, Box::new(new_body))
            },

            SourceExpr::OpNode(op, expr1, expr2) => {
                let new_expr1 = self.lower(expr1, store, path);
                let new_expr2 = self.lower(expr2, store, path);

                NormalizedExpr::OpNode(*op, Box::new(new_expr1), Box::new(new_expr2))
            },

            // TODO for now, assume indexing nodes are scalar (0-dim)
            SourceExpr::IndexingNode(arr, index_list) => {
                // first, compute the required shape of the array
                let mut required_shape: Vec<(IndexName, Extent)> = Vec::new();
                let mut reduce_ind: usize = 0;

                // in-scope indices and their extents
                let mut index_store: HashMap<IndexName, Extent> = HashMap::new();

                for info in path.iter() {
                    match info {
                        PathInfo::Index { index, extent } => {
                            required_shape.insert(reduce_ind, (index.clone(), *extent));
                            index_store.insert(index.clone(), *extent);
                        },

                        PathInfo::Reduce { op: _ } => {
                            reduce_ind += 1;
                        }
                    }
                }

                // next, compute the transformations from the array's
                // original shape to the required shape

                // first, compute the original shape
                let mut orig_shape: Vec<IndexName> = Vec::new();
                for index_expr in index_list.iter() {
                    match index_expr.get_single_var() {
                        Some(var) => {
                            orig_shape.push(var)
                        },

                        None => panic!("only one index var allowed per dimension")
                    }
                }

                // compute fills
                // fills are added to the FRONT of the dimension list!
                // so new filled dimensions will be in the front of extent_list,
                // unless they are permuted by the transpose below
                let mut extent_list: Vec<Extent> = Vec::new();
                let missing_indices: Vec<(IndexName, Extent)> = 
                    required_shape.clone().into_iter().filter(|(index, _)|
                        !orig_shape.contains(index)
                    ).collect();
                let mut new_shape: Vec<String> = orig_shape.clone();
                let mut fill_sizes: Vec<usize> = Vec::new();
                for (index, extent) in missing_indices.iter() {
                    extent_list.push(*extent);
                    fill_sizes.push((extent.upper() - extent.lower() + 1) as usize);
                    new_shape.insert(0, index.clone());
                }

                // compute padding
                // initialize with padding for filled dimensions,
                // which should always be (0,0)
                let mut pad_sizes: Vec<PadSize> =
                    (0..fill_sizes.len()).into_iter().map(|_| (0, 0)).collect();

                for (i, index_expr) in index_list.iter().enumerate() {
                    let index_interval = self.index_expr_to_interval(index_expr, &index_store);
                    let dim_interval = store[arr][i];
                    let pad_min = max(0, dim_interval.lower() - index_interval.lower());
                    let pad_max = max(0, index_interval.upper() - dim_interval.upper());
                    let extent =
                        Interval::new(dim_interval.lower() - pad_min, dim_interval.upper() + pad_max);
                    pad_sizes.push((pad_min as usize, pad_max as usize));
                    extent_list.push(extent);
                }

                // compute transposition
                let mut transpose: Vec<usize> = (0..required_shape.len()).collect();
                for i in 0..new_shape.len() {
                    let cur_index = &required_shape[i].0;
                    transpose[i] =
                        new_shape.iter()
                        .position(|index| index == cur_index)
                        .unwrap();
                }

                // apply transposition
                let transposed_pad_sizes: Vec<PadSize> = 
                    transpose.iter().map(|&i| pad_sizes[i]).collect();

                let transposed_extent_list: Vec<Extent> = 
                    transpose.iter().map(|&i| extent_list[i]).collect();

                // finally, assemble the array transform
                NormalizedExpr::TransformNode(
                    self.fresh_expr_id(), 
                    arr.clone(),
                    ArrayTransform {
                        fill_sizes,
                        transpose,
                        pad_sizes: transposed_pad_sizes,
                        extent_list: transposed_extent_list,
                    }
                )
            }
        }
    }

    fn collect_extent_constraints(&mut self, expr: &NormalizedExpr) -> (usize, Vec<ConstraintVar>) {
        match expr {
            NormalizedExpr::ReduceNode(_, body) => {
                let (i, extent_list) = self.collect_extent_constraints(body);
                (i+1, extent_list)
            },

            NormalizedExpr::OpNode(_, expr1, expr2) => {
                let (i1, extent_list1) = self.collect_extent_constraints(expr1);
                let (i2, extent_list2) = self.collect_extent_constraints(expr2);

                assert!(extent_list1[i1..].len() == extent_list2[i2..].len());

                let zipped_extents = extent_list1[i1..].iter().zip(extent_list2[i2..].iter());
                for (&extent1, &extent2) in zipped_extents {
                    self.constraints.push(
                        ExtentConstraint { var1: extent1, var2: extent2 }
                    )
                }

                // arbitrarily pick one extent list from operands to return
                (i1, extent_list1)
            },

            NormalizedExpr::TransformNode(id, _, transform) => {
                let mut extent_vars: Vec<ConstraintVar> = Vec::new();
                for extent in transform.extent_list.iter() {
                    let extent_var = self.fresh_constraint_var();
                    extent_vars.push(extent_var);
                    self.solution.insert(extent_var, *extent);
                }

                self.node_vars.insert(*id, extent_vars.clone());
                (0, extent_vars)
            }
        }
    }

    fn solve_extent_constraints(&mut self) -> HashMap<ExprId, Vec<Extent>> {
        let mut quiesce = false;

        // find fixpoint solution to constraints;
        // this just implements a simple linear pass instead of doing
        // the usual dataflow analysis optimizations like
        // keeping track of which constraints to wake when a solution is updated,
        // toposorting the connected components of the graph, etc.
        while !quiesce {
            quiesce = true;
            for c in self.constraints.iter() {
                let sol1 = self.solution[&c.var1];
                let sol2 = self.solution[&c.var2];
                if sol1 != sol2 {
                    let new_sol = sol1.hull(&sol2);
                    self.solution.insert(c.var1, new_sol);
                    self.solution.insert(c.var2, new_sol);
                    quiesce = false;
                }
            }
        }

        let mut node_solutions: HashMap<ExprId, Vec<Extent>> = HashMap::new();
        for (node, extent_vars) in self.node_vars.iter() {
            let extent_sol: Vec<Extent> =
                extent_vars.iter()
                .map(|var| self.solution[var])
                .collect();
            node_solutions.insert(*node, extent_sol);
        }

        node_solutions
    }

    fn apply_extent_solution(&self, expr: &NormalizedExpr, node_solution: &HashMap<ExprId, Vec<Extent>>) -> NormalizedExpr {
        match expr {
            NormalizedExpr::ReduceNode(op, body) => {
                let new_body = self.apply_extent_solution(body, node_solution);
                NormalizedExpr::ReduceNode(*op, Box::new(new_body))
            },

            NormalizedExpr::OpNode(op, expr1, expr2) => {
                let new_expr1 = self.apply_extent_solution(expr1, node_solution);
                let new_expr2 = self.apply_extent_solution(expr2, node_solution);
                NormalizedExpr::OpNode(*op, Box::new(new_expr1), Box::new(new_expr2))
            },

            NormalizedExpr::TransformNode(id, arr, transform) => {
                if node_solution.contains_key(id) {
                    let mut new_pad_sizes: Vec<PadSize> = Vec::new();
                    let zipped_iter =
                        transform.pad_sizes.iter().zip(
                            transform.extent_list.iter().zip(
                                node_solution[id].iter()
                            )
                        );

                    for (pad, (cur_extent, sol_extent)) in zipped_iter {
                        assert!(cur_extent.is_subset(sol_extent), "extent solution should only add padding, not remove it");
                        if cur_extent != sol_extent {
                            let new_pad_min = ((cur_extent.lower() - sol_extent.lower()) as usize) + pad.0;
                            let new_pad_max = ((sol_extent.upper() - cur_extent.upper()) as usize) + pad.1;
                            new_pad_sizes.push((new_pad_min, new_pad_max));
                        }
                    }

                    NormalizedExpr::TransformNode(
                        *id,
                        arr.clone(),
                        ArrayTransform {
                            fill_sizes: transform.fill_sizes.clone(),
                            transpose: transform.transpose.clone(),
                            pad_sizes: new_pad_sizes,
                            extent_list: node_solution[id].clone()
                        }
                    )

                } else {
                    expr.clone()
                }
            }
        }
    }

    pub fn run(&mut self, expr: &SourceExpr, store: &ExtentStore) -> NormalizedExpr {
        let norm_expr = self.lower(expr, store, &im::Vector::new());
        self.collect_extent_constraints(&norm_expr);
        let node_solution = self.solve_extent_constraints();
        self.apply_extent_solution(&norm_expr, &node_solution)
        // self.lower(expr, store, &im::Vector::new())
    }
}