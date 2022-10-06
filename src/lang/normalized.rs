use std::cmp::max;

use interval::ops::{Hull, Range};
use gcollections::ops::{bounded::Bounded, Subset};

use crate::lang::{*, source::{*, IndexExpr::*}};

#[derive(Clone, Debug)]
pub struct NormalizedProgram {
    pub store: ArrayEnvironment,
    pub expr: NormalizedExpr
}

#[derive(Clone,Debug)]
pub enum NormalizedExpr {
    ReduceNode(ExprOperator, Box<NormalizedExpr>),
    OpNode(ExprOperator, Box<NormalizedExpr>, Box<NormalizedExpr>),
    TransformNode(ExprId, ArrayName, ArrayTransform),
    LiteralNode(i64)
}

type PadSize = (u64, u64);

#[derive(Clone,Debug)]
pub struct ArrayTransform {
    fill_sizes: Vec<usize>,
    transpose: Vec<usize>,
    pad_sizes: Vec<PadSize>,
    extent_list: Vec<Extent>
}

impl Display for NormalizedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NormalizedExpr::ReduceNode(op, body) => {
                let reduce_op_str = 
                    match op {
                        ExprOperator::OpAdd => "sum",
                        ExprOperator::OpSub => "sum_sub",
                        ExprOperator::OpMul => "product"
                    };

                write!(f, "{}({})", reduce_op_str, body)
            },

            NormalizedExpr::OpNode(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            },

            NormalizedExpr::TransformNode(_, arr, transform) => {
                write!(
                    f,
                    "transpose(fill(pad({}, {:?}), {:?}), {:?})",
                    arr,
                    transform.pad_sizes,
                    transform.fill_sizes,
                    transform.transpose
                )
            },

            NormalizedExpr::LiteralNode(val) => write!(f, "{}", val),
        }
    }
}

struct LinearIndexingData { scale: i64, offset: i64 }

#[derive(Clone)]
enum PathInfo {
    Index { index: IndexName, extent: Extent },
    Reduce { op: ExprOperator }
}

#[derive(Copy,Clone,Debug,Eq,Hash,PartialEq)]
struct ConstraintVar(usize);

struct ExtentConstraint { var1: ConstraintVar, var2: ConstraintVar }

pub struct Normalizer {
    cur_expr_id: usize,
    cur_constraint_id: usize,
    constraints: Vec<ExtentConstraint>,
    constraint_vars: Vec<ConstraintVar>,
    solution: HashMap<ConstraintVar, Extent>,
    node_vars: HashMap<ExprId, Vec<ConstraintVar>>,
}

impl Normalizer {
    pub fn new() -> Self {
        Normalizer {
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

    fn index_expr_to_interval(&self, index_expr: &IndexExpr, index_store: &IndexEnvironment) -> Extent {
        match index_expr {
            IndexVar(var) => {
                index_store[var]
            },

            IndexLiteral(val) => {
                Interval::new(*val, *val)
            }

            IndexOp(op, expr1, expr2) => {
                let interval1 = self.index_expr_to_interval(expr1, index_store);
                let interval2 = self.index_expr_to_interval(expr2, index_store);
                match op {
                    OpAdd => interval1 + interval2,
                    OpSub => interval1 - interval2,
                    OpMul => interval1 * interval2,
                }
            }
        }
    }

    fn get_linear_indexing_data(&self, index_expr: &IndexExpr, index_var: &IndexName) -> Option<LinearIndexingData> {
        match index_expr {
            IndexVar(v) => {
                if v == index_var {
                    Some(LinearIndexingData { scale: 1, offset: 0 })
                } else {
                    None
                }
            },

            IndexLiteral(val) => {
                Some(LinearIndexingData { scale: 0, offset: *val })
            },

            IndexOp(op, expr1, expr2) => {
                let data1 = self.get_linear_indexing_data(expr1, index_var)?;
                let data2 = self.get_linear_indexing_data(expr2, index_var)?;
                match op {
                    OpAdd => {
                        Some(LinearIndexingData {
                            scale: data1.scale + data2.scale,
                            offset: data1.offset + data2.offset
                        })
                    },
                    OpSub => {
                        Some(LinearIndexingData {
                            scale: data1.scale - data2.scale,
                            offset: data1.offset - data2.offset
                        })
                    },
                    OpMul => {
                        if data1.scale == 0 {
                            Some(LinearIndexingData {
                                scale: data2.scale * data1.offset,
                                offset: data2.offset * data1.offset
                            })
                        } else if data2.scale == 0 {
                            Some(LinearIndexingData {
                                scale: data1.scale * data2.offset,
                                offset: data1.offset * data2.offset
                            })
                        } else {
                            None
                        }
                    }
                }
            },
        }
    }

    /// transformation that removes indices from source expressions.
    fn lower(&mut self, expr: &SourceExpr, store: &ArrayEnvironment, path: &im::Vector<PathInfo> ) -> NormalizedExpr {
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
                let mut index_store: IndexEnvironment = im::HashMap::new();

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
                    pad_sizes.push((pad_min as u64, pad_max as u64));
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
            },

            SourceExpr::LiteralNode(val) => {
                NormalizedExpr::LiteralNode(*val)
            }
        }
    }

    fn collect_extent_constraints(&mut self, expr: &NormalizedExpr) -> Option<(usize, Vec<ConstraintVar>)> {
        match expr {
            NormalizedExpr::ReduceNode(_, body) => {
                match self.collect_extent_constraints(body) {
                    Some((i, extent_list)) => {
                        Some((i+1, extent_list))
                    },

                    None => {
                        panic!("trying to reduce dimension of a scalar value")
                    }
                }
            },

            NormalizedExpr::OpNode(_, expr1, expr2) => {
                let shape1 = self.collect_extent_constraints(expr1);
                let shape2 = self.collect_extent_constraints(expr2);

                match (shape1, shape2) {
                    (Some((i1, extent_list1)),
                     Some((i2, extent_list2))) => {
                        assert!(extent_list1[i1..].len() == extent_list2[i2..].len());

                        let zipped_extents = extent_list1[i1..].iter().zip(extent_list2[i2..].iter());
                        for (&extent1, &extent2) in zipped_extents {
                            self.constraints.push(
                                ExtentConstraint { var1: extent1, var2: extent2 }
                            )
                        }

                        // arbitrarily pick one extent list from operands to return
                        Some((i1, extent_list1))
                     },

                    (Some((i1, extent_list1)), None) => {
                        Some((i1, extent_list1))
                    },
                    
                    (None, Some((i2, extent_list2))) => {
                        Some((i2, extent_list2))
                    },

                     _ => None
                }
            },

            NormalizedExpr::TransformNode(id, _, transform) => {
                let mut extent_vars: Vec<ConstraintVar> = Vec::new();
                for extent in transform.extent_list.iter() {
                    let extent_var = self.fresh_constraint_var();
                    extent_vars.push(extent_var);
                    self.solution.insert(extent_var, *extent);
                }

                self.node_vars.insert(*id, extent_vars.clone());
                Some((0, extent_vars))
            },

            NormalizedExpr::LiteralNode(_) => {
                None
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
                            let new_pad_min = (cur_extent.lower() - sol_extent.lower()) as u64 + pad.0;
                            let new_pad_max = (sol_extent.upper() - cur_extent.upper()) as u64 + pad.1;
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
            },

            NormalizedExpr::LiteralNode(_) => expr.clone()
        }
    }

    pub fn run(&mut self, program: &SourceProgram) -> Result<NormalizedProgram, String> {
        let mut store: im::HashMap<ArrayName, Shape> = im::HashMap::new();
        program.inputs.iter().try_for_each(|input|
            match store.insert(input.0.clone(), input.1.clone()) {
                Some(_) => Err(format!("duplicate input bindings for {}", input.0)),
                None => Ok(())
            }
        )?;
        let norm_expr = self.lower(&program.expr, &store, &im::Vector::new());
        self.collect_extent_constraints(&norm_expr);
        let node_solution = self.solve_extent_constraints();
        let final_expr = self.apply_extent_solution(&norm_expr, &node_solution);
        Ok(NormalizedProgram { store, expr: final_expr })
    }
}