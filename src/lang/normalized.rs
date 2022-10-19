use std::{cmp::max, ops::Index};

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

type PadSize = (usize, usize);

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

#[derive(Clone,Debug)]
pub enum TransformedExpr {
    ReduceNode(usize, ExprOperator, Box<TransformedExpr>),
    OpNode(ExprOperator, Box<TransformedExpr>, Box<TransformedExpr>),
    TransformNode(ArrayName, ArrayTransformInfo),
    LiteralNode(i64),
}

impl Display for TransformedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformedExpr::ReduceNode(dim, op, body) => {
                write!(f, "reduce({}, {}, {})", dim, op, body)
            },

            TransformedExpr::OpNode(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            },

            TransformedExpr::TransformNode(arr, transform) => {
                write!(f, "{}[{}]", arr, transform)
            },

            TransformedExpr::LiteralNode(lit) => {
                write!(f, "{}", lit)
            },
        }
    }
}

pub trait Transform: Display {}

#[derive(Clone, Debug)]
pub enum TransformedDim {
    Input(usize),
    Fill(Extent),
}

impl Transform for TransformedDim {}

impl Display for TransformedDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformedDim::Input(i) => write!(f, "{}", i),
            TransformedDim::Fill(extent) => write!(f, "fill({})", extent),
        }
    }
}

#[derive(Clone, Debug)]
pub enum TransformedIndexDim {
    Index(IndexName),
    Fill(Extent),
}

impl Transform for TransformedIndexDim {}

impl Display for TransformedIndexDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformedIndexDim::Index(index) => write!(f, "{}", index),
            TransformedIndexDim::Fill(extent) => write!(f, "fill({})", extent),
        }
    }
}

pub struct TransformShape<T: Transform>(Vec<T>);

impl<T: Transform> Display for TransformShape<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strs =
            self.0.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>().join(", ");

        write!(f, "[{}]", strs)
    }
}

impl<T: Transform> Default for TransformShape<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

#[derive(Clone,Debug)]
pub struct TransformedDimInfo {
    dim: TransformedDim,
    pad: PadSize,
    extent: Extent,
}

impl Display for TransformedDimInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(pad {} ({},{}))", self.dim, self.pad.0, self.pad.1)
    }
}

#[derive(Clone,Debug)]
pub struct ArrayTransformInfo(Vec<TransformedDimInfo>);

impl Display for ArrayTransformInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strs =
            self.0.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>().join(", ");

        write!(f, "[{}]", strs)
    }
}

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
                    ExprOperator::OpAdd => interval1 + interval2,
                    ExprOperator::OpSub => interval1 - interval2,
                    ExprOperator::OpMul => interval1 * interval2,
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
                    ExprOperator::OpAdd => {
                        Some(LinearIndexingData {
                            scale: data1.scale + data2.scale,
                            offset: data1.offset + data2.offset
                        })
                    },
                    ExprOperator::OpSub => {
                        Some(LinearIndexingData {
                            scale: data1.scale - data2.scale,
                            offset: data1.offset - data2.offset
                        })
                    },
                    ExprOperator::OpMul => {
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
                let mut index_store: IndexEnvironment = HashMap::new();

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
            },

            SourceExpr::LiteralNode(val) => {
                NormalizedExpr::LiteralNode(*val)
            }
        }
    }

    fn compute_expr_extent(&self, expr: &SourceExpr, store: &ArrayEnvironment) -> Shape  {
        match expr {
            SourceExpr::ForNode(index, extent, body) => {
                let body_extent = self.compute_expr_extent(body, store);
                im::vector![extent.clone()] + body_extent
            },

            SourceExpr::ReduceNode(_, body) => {
                let mut body_extent = self.compute_expr_extent(body, store);
                body_extent.split_off(1)
            },

            SourceExpr::OpNode(_, expr1, expr2) => {
                let extent1 = self.compute_expr_extent(expr1, store);
                let extent2 = self.compute_expr_extent(expr2, store);
                assert!(extent1 == extent2);
                extent1
            },

            SourceExpr::IndexingNode(arr, index_list) => {
                let arr_extent = store.get(arr).unwrap();
                arr_extent.clone().split_off(index_list.len())
            },

            SourceExpr::LiteralNode(_) => im::Vector::new()
        }
    }

    fn compute_prog_extent(&mut self, program: &SourceProgram) -> ArrayEnvironment {
        let mut store: ArrayEnvironment = HashMap::new();

        for input in program.inputs.iter() {
            if let Some(_) = store.insert(input.0.clone(), input.1.clone()) {
                panic!("duplicate binding for {}", input.0)
            }
        }

        for binding in program.letBindings.iter() {
            let extent = self.compute_expr_extent(&binding.1, &store);
            if let Some(_) = store.insert(binding.0.clone(), extent) {
                panic!("duplicate binding for {}", binding.0)
            }
        }

        store
    }

    fn lower2(
        &mut self,
        expr: &SourceExpr,
        output_shape: &TransformShape<TransformedDim>,
        store: &ArrayEnvironment,
        path: im::Vector<PathInfo>
    ) -> Result<(TransformedExpr, Option<Vec<usize>>), String> {
        match expr {
            SourceExpr::ForNode(index, extent, body) => {
                let new_path = 
                    path +
                    im::Vector::unit(PathInfo::Index {
                        index: index.clone(), extent: *extent
                    });

                self.lower2(body, output_shape, store, new_path)
            },

            SourceExpr::ReduceNode(op, body) => {
                let new_path = 
                    im::Vector::unit(PathInfo::Reduce { op: *op }) + path;
                let (new_body, reduced_dim_positions_opt) = self.lower2(body, output_shape, store, new_path)?;
                let mut reduced_dim_position = reduced_dim_positions_opt.unwrap();
                let rest = reduced_dim_position.split_off(1);
                let dim = *reduced_dim_position.first().unwrap();
                Ok((TransformedExpr::ReduceNode(dim, *op, Box::new(new_body)), Some(rest)))
            },

            SourceExpr::OpNode(op, expr1, expr2) => {
                let (new_expr1, reduced_dim_position1_opt) = self.lower2(expr1, output_shape, store, path.clone())?;
                let (new_expr2, reduced_dim_position2_opt) = self.lower2(expr2, output_shape, store, path)?;
                let reduced_dim_position_opt =
                    match (&reduced_dim_position1_opt, &reduced_dim_position2_opt) {
                        (None, None) => Ok(None),
                        (None, Some(_)) => Ok(reduced_dim_position2_opt),
                        (Some(_), None) => Ok(reduced_dim_position1_opt),
                        (Some(reduced_dim_position1), Some(reduced_dim_position2)) => {
                            if reduced_dim_position1 == reduced_dim_position2 {
                                Ok(reduced_dim_position1_opt)
                            } else {
                                Err(
                                    format!("op node operands do not have the same reduced dim positions: {:?} {:?}",
                                        reduced_dim_position1,
                                        reduced_dim_position2
                                    )
                                )
                            }
                        }
                    }?;
                Ok((TransformedExpr::OpNode(*op, Box::new(new_expr1), Box::new(new_expr2)), reduced_dim_position_opt))
            },

            SourceExpr::LiteralNode(lit) => {
                Ok((TransformedExpr::LiteralNode(*lit), None))
            },

            SourceExpr::IndexingNode(arr, index_list) => {
                // first, determine the computed shape of the array
                // based on path info and the output shape

                // dimensions that are reduced
                let mut reduced_dims: Vec<(&IndexName, &Extent)> = Vec::new();
                let mut reduced_dim_position: Vec<usize> = Vec::new();

                // dimensions that are part of the output
                let mut output_dims: Vec<(&IndexName, &Extent)> = Vec::new();

                // in-scope indices and their extents
                let mut index_store: IndexEnvironment = HashMap::new();

                let mut num_reductions = 0;
                for info in path.iter() {
                    match info {
                        PathInfo::Index { index, extent } => {
                            if num_reductions > 0 {
                                reduced_dims.push((index, extent));
                                num_reductions -= 1;
                            } else {
                                output_dims.push((index, extent))
                            }
                            index_store.insert(index.clone(), *extent);
                        },
                        PathInfo::Reduce { op: _ } => {
                            num_reductions += 1;
                        },
                    }
                }

                // TODO: support reducing dims not named by an index
                if num_reductions > 0 {
                    return Err(String::from("All reduced dimensions must be indexed"))
                }

                // output index shape is like output shape,
                // but the dimensions are now named by index
                // invariant: every output and reduced dim appears
                // exactly once in output index shape
                let mut out_index_shape: TransformShape<TransformedIndexDim> = TransformShape::default();
                let mut used_rdim = 0;
                for (i, out_dim) in output_shape.0.iter().enumerate() {
                    match out_dim {
                        TransformedDim::Input(dim_index) => {
                            out_index_shape.0.push(
                                TransformedIndexDim::Index(output_dims[*dim_index].0.to_string())
                            );
                        },

                        TransformedDim::Fill(extent) => {
                            if reduced_dims.len() > 0 {
                                // TODO: have a better heuristic for picking which reduced dim to use
                                let index = reduced_dims[used_rdim].0;
                                used_rdim += 1;
                                out_index_shape.0.push(TransformedIndexDim::Index(index.to_string()));

                            } else {
                                out_index_shape.0.push(TransformedIndexDim::Fill(*extent))
                            }
                        },

                        _ => {
                            return Err(String::from("output shape should not have index dims"))
                        }
                    }
                }

                // if some reduced dims are not used, they are added to the front
                if used_rdim < reduced_dims.len() {
                    for rdim in reduced_dims[used_rdim..].iter() {
                        out_index_shape.0.insert(
                            0,
                             TransformedIndexDim::Index(rdim.0.to_string())
                        );
                    }
                }

                // compute the positions of reduced dimensions
                for dim in reduced_dims.iter() {
                    let index_opt =
                        out_index_shape.0.iter().position(|x|
                            match x {
                                TransformedIndexDim::Index(index) => index == dim.0,
                                TransformedIndexDim::Fill(_) => false,
                            }
                        );

                    if let Some(index) = index_opt { 
                        reduced_dim_position.push(index);

                    } else {
                        return Err(format!("reduced indexed dimension {} not in output shape", dim.0))
                    }
                }

                // compute the original shape
                let mut index_position: HashMap<IndexName, usize> = HashMap::new();
                let mut orig_shape: Vec<IndexName> = Vec::new();
                for (i, index_expr) in index_list.iter().enumerate() {
                    match index_expr.get_single_var() {
                        Some(var) => {
                            orig_shape.push(var.clone());
                            index_position.insert(var, i);
                        },

                        None => {
                            return Err(String::from("only one index var required per indexed dimension"))
                        }
                    }
                }

                let mut computed_shape: TransformShape<TransformedDim> = TransformShape::default();
                for dim in out_index_shape.0 {
                    match dim {
                        TransformedIndexDim::Index(index) => {
                            match index_position.get(&index) {
                                // indexed dim is in the original shape; add its position
                                Some(i) => {
                                    computed_shape.0.push(TransformedDim::Input(*i));
                                },

                                // index is missing from original shape; add as a fill dimension
                                None => {
                                    let index_extent_opt =
                                        path.iter().find(|info| 
                                            match info {
                                                PathInfo::Index { index: pindex, extent }
                                                if pindex == &index => true,

                                                _ => false,
                                            }
                                        );

                                    if let Some(PathInfo::Index { index: _, extent }) = index_extent_opt {
                                        computed_shape.0.push(TransformedDim::Fill(*extent));
                                    } else {
                                        return Err(format!("cannot index indexed dimension {} in output shape", index));
                                    }
                                }
                            }
                        },

                        // fill dimension from output shape
                        TransformedIndexDim::Fill(extent) => {
                            computed_shape.0.push(TransformedDim::Fill(extent));
                        },

                        _ => {
                            return Err(String::from("out index shape should not have input or fill index dims"))
                        }
                    }
                }

                let dim_info: Vec<TransformedDimInfo> =
                    computed_shape.0.into_iter().map(|dim| {
                        match dim {
                            // for indexed dims, perform interval analysis to determine padding
                            TransformedDim::Input(i) => {
                                let index_expr = &index_list[i];
                                let index_interval = self.index_expr_to_interval(index_expr, &index_store);
                                let dim_interval = store[arr][i];

                                let pad_min = max(0, dim_interval.lower() - index_interval.lower());
                                let pad_max = max(0, index_interval.upper() - dim_interval.upper());

                                let extent =
                                    Interval::new(dim_interval.lower() - pad_min, dim_interval.upper() + pad_max);

                                Ok(TransformedDimInfo {
                                    dim,
                                    pad: (pad_min as usize, pad_max as usize),
                                    extent,
                                })
                            },

                            // fill dims never have padding
                            TransformedDim::Fill(extent) => {
                                Ok(TransformedDimInfo {
                                    dim,
                                    pad: (0,0),
                                    extent,
                                })
                            },
                        }
                    }).collect::<Result<Vec<TransformedDimInfo>,String>>()?;

                let transform =  TransformedExpr::TransformNode(arr.to_string(), ArrayTransformInfo(dim_info));
                Ok((transform, Some(reduced_dim_position)))
            },
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
                            let new_pad_min = (cur_extent.lower() - sol_extent.lower()) as usize + pad.0;
                            let new_pad_max = (sol_extent.upper() - cur_extent.upper()) as usize + pad.1;
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
        let mut store: ArrayEnvironment = HashMap::new();
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

impl Default for Normalizer {
    fn default() -> Self {
        Normalizer::new()
    }
}

#[cfg(test)]
mod tests{
    use crate::lang::parser::ProgramParser;
    use super::*;

    fn test_lowering(src: &str, out_shape: TransformShape<TransformedDim>) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let mut normalizer = Normalizer::new();
        let store = normalizer.compute_prog_extent(&program);
        let res =
            normalizer.lower2(&program.expr, &out_shape, &store, im::Vector::new());
        
        assert!(res.is_ok());
        println!("{}", res.unwrap().0);

    }

    #[test]
    fn imgblur() {
        let parser = ProgramParser::new();
        let program: SourceProgram =
            parser.parse("
                input img: [(0,16),(0,16)]
                for x: (0, 16) {
                    for y: (0, 16) {
                        img[x-1][y-1] + img[x+1][y+1]
                    }
                }
            ").unwrap();

        let res = Normalizer::new().run(&program);
        assert!(res.is_ok());
        println!("{}", res.unwrap().expr)
    }

    #[test]
    fn imgblur2() {
        test_lowering(
        "input img: [(0,16),(0,16)]
            for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }",
            TransformShape(vec![
                TransformedDim::Input(0),
                TransformedDim::Input(1),
            ]),
        );
    }

    #[test]
    fn matmatmul() {
        let parser = ProgramParser::new();
        let prog: SourceProgram =
            parser.parse("
            input A: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            let x = A + B
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A[i][k] * B[k][j] })
                }
            }
            ").unwrap();

        let res = Normalizer::new().run(&prog);
        assert!(res.is_ok());
        println!("{}", res.unwrap().expr);
    }

    #[test]
    fn matmatmul2() {
        test_lowering(
            "input A: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            let x = A + B
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A[i][k] * B[k][j] })
                }
            }",
            TransformShape(vec![
                TransformedDim::Input(0),
                // TransformedDim::Fill(Interval::new(0, 4)),
                TransformedDim::Input(1),
            ])
        );
    }

    #[test]
    fn matmatmul3() {
        test_lowering(
            "input A: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            let x = A + B
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A[i][k] * B[k][j] })
                }
            }",
            TransformShape(vec![
                TransformedDim::Input(0),
                TransformedDim::Fill(Interval::new(0, 4)),
                TransformedDim::Input(1),
            ])
        );
    }
}