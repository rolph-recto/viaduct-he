use std::hash::{*, Hasher};

use interval::ops::Range;
use gcollections::ops::bounded::Bounded;

use crate::{
    lang::{
        *,
        IndexExpr::*,
        extent_analysis::{ExtentAnalysis, ShapeId},
    },
    circ::{ Ciphertext, Dimensions },
    util::NameGenerator
};

type PadSize = (usize, usize);

struct SimpleIndexingData { scale: isize, offset: isize }

#[derive(Clone,Debug)]
enum PathInfo {
    Index { index: IndexName, extent: Extent },
    Reduce { op: Operator }
}

#[derive(Eq,PartialEq,Clone,Copy,Debug)]
pub enum ReducedDimType { Hidden, Reused }

/// expression with associated data about lowering to an index-free representation.
#[derive(Clone,Debug)]
pub enum TransformedExpr {
    ReduceNode(ReducedDimType, usize, Operator, Box<TransformedExpr>),
    Op(Operator, Box<TransformedExpr>, Box<TransformedExpr>),
    Literal(isize),
    ExprRef(ExprId),
}

impl Display for TransformedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformedExpr::ReduceNode(_, dim, op, body) => {
                write!(f, "reduce({}, {}, {})", dim, op, body)
            },

            TransformedExpr::Op(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            },

            TransformedExpr::ExprRef(id) => {
                write!(f, "expr{}", id)
            },

            TransformedExpr::Literal(lit) => {
                write!(f, "{}", lit)
            },
        }
    }
}

pub trait Transform: Display {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransformedDim {
    Input(usize),
    Fill(Extent),
}

impl Hash for TransformedDim {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            TransformedDim::Input(i) => {
                i.hash(state);
            },

            TransformedDim::Fill(interval) => {
                interval.lower().hash(state);
                interval.upper().hash(state);
            }
        }
    }
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
    ReducedIndex(IndexName),
    Fill(Extent),
}

impl Transform for TransformedIndexDim {}

impl Display for TransformedIndexDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformedIndexDim::Index(index) => write!(f, "{}", index),
            TransformedIndexDim::ReducedIndex(index) => write!(f, "{}", index),
            TransformedIndexDim::Fill(extent) => write!(f, "fill({})", extent),
        }
    }
}

#[derive(Debug)]
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

#[derive(Clone,Debug,Eq)]
pub struct TransformedDimInfo {
    dim: TransformedDim,
    pad: PadSize,
    extent: Extent,
    offset: isize,
}

impl TransformedDimInfo {
    pub fn is_input(&self) -> bool {
        match self.dim {
            TransformedDim::Input(_) => true,
            TransformedDim::Fill(_) => false,
        }
    }

    pub fn is_fill(&self) -> bool {
        match self.dim {
            TransformedDim::Input(_) => false,
            TransformedDim::Fill(_) => true,
        }
    }
}

impl Display for TransformedDimInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.pad != (0,0) {
            write!(f, "(pad {} ({},{}))", self.dim, self.pad.0, self.pad.1)
        } else {
            write!(f, "{}", self.dim)
        }
    }
}

impl PartialEq for TransformedDimInfo {
    fn eq(&self, other: &Self) -> bool {
        self.dim == other.dim && self.pad == other.pad && self.extent == other.extent
    }
}

#[derive(Clone,Eq,PartialEq,Debug)]
pub struct ArrayTransformInfo(ArrayName, Vec<TransformedDimInfo>);

impl ArrayTransformInfo {
    fn to_transformed_shape(&self) -> TransformShape<TransformedDim> {
        TransformShape(
            self.1.iter()
            .map(|dim_info| dim_info.dim.clone())
            .collect()
        )
    }

    fn to_shape(&self) -> Shape {
        self.1.iter().map(|dim_info| {
            dim_info.extent.clone()
        }).collect()
    }
    
    fn to_dimensions(&self) -> Dimensions {
        Dimensions::from(
            self.1.iter().map(|dim_info| {
                (dim_info.extent.upper() - dim_info.extent.lower()) as usize
            }).collect::<im::Vector<usize>>()
        )
    }
}

impl Hash for ArrayTransformInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        for dim_info in self.1.iter() {
            dim_info.dim.hash(state);
            dim_info.pad.hash(state);
        }
    }
}

impl Display for ArrayTransformInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strs =
            self.1.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>().join(", ");

        write!(f, "{}[{}]", self.0, strs)
    }
}

#[derive(Debug)]
struct TransformResult {
    expr: TransformedExpr,
    reduced_dim_position: Option<Vec<(ReducedDimType, usize)>>,
    transformed_inputs: im::HashSet<ExprId>,
}

pub struct IndexElimination {
    // ID of the output expression
    output_id: ExprId,

    // counter for creating fresh expressions IDs
    cur_expr_id: usize,

    // map source-level bindings to their shapes 
    store: ArrayEnvironment,

    // map from input nodes to their shape
    input_map: HashMap<ArrayName, Shape>,

    // map from variables to their bound expressions
    expr_binding_map: HashMap<ArrayName, SourceExpr>,

    // map from expressions to their computed shapes
    transform_info_map: HashMap<ExprId, ArrayTransformInfo>,

    // map from expressions to transformed representation
    transform_map: HashMap<ExprId, TransformedExpr>,

    // topological sort of expressions
    transform_list: Vec<ExprId>,

    // map from expressions to shape IDs from extent analysis
    shape_map: HashMap<ExprId, (usize, ShapeId)>,

    // module to compute required padding
    extent_analysis: ExtentAnalysis,

    // module to generate fresh names for ciphertexts
    name_generator: NameGenerator,
}

static OUTPUT_EXPR_NAME: &'static str = "$root";

impl IndexElimination {
    pub fn new() -> Self {
        IndexElimination {
            input_map: HashMap::new(),
            expr_binding_map: HashMap::new(),
            output_id: 0,
            cur_expr_id: 1,
            transform_info_map: HashMap::new(),
            transform_map: HashMap::new(),
            transform_list: Vec::new(),
            shape_map: HashMap::new(),
            store: HashMap::new(),
            extent_analysis: ExtentAnalysis::new(),
            name_generator: NameGenerator::new(),
        }
    }

    fn fresh_expr_id(&mut self) -> usize {
        let id = self.cur_expr_id;
        self.cur_expr_id += 1;
        id
    }

    fn index_expr_to_interval(&self, index_expr: &IndexExpr, index_store: &IndexEnvironment) -> Extent {
        match index_expr {
            IndexVar(var) => {
                index_store[var]
            },

            IndexLiteral(val) => {
                Interval::new(*val as i64, *val as i64)
            }

            IndexOp(op, expr1, expr2) => {
                let interval1 = self.index_expr_to_interval(expr1, index_store);
                let interval2 = self.index_expr_to_interval(expr2, index_store);
                match op {
                    Operator::Add => interval1 + interval2,
                    Operator::Sub => interval1 - interval2,
                    Operator::Mul => interval1 * interval2,
                }
            }
        }
    }

    fn get_simple_indexing_data(&self, index_expr: &IndexExpr, index_var: &IndexName) -> Option<SimpleIndexingData> {
        match index_expr {
            IndexVar(v) => {
                if v == index_var {
                    Some(SimpleIndexingData { scale: 1, offset: 0 })
                } else {
                    None
                }
            },

            IndexLiteral(val) => {
                Some(SimpleIndexingData { scale: 0, offset: *val })
            },

            IndexOp(op, expr1, expr2) => {
                let data1 = self.get_simple_indexing_data(expr1, index_var)?;
                let data2 = self.get_simple_indexing_data(expr2, index_var)?;
                match op {
                    Operator::Add => {
                        Some(SimpleIndexingData {
                            scale: data1.scale + data2.scale,
                            offset: data1.offset + data2.offset
                        })
                    },
                    Operator::Sub => {
                        Some(SimpleIndexingData {
                            scale: data1.scale - data2.scale,
                            offset: data1.offset - data2.offset
                        })
                    },
                    Operator::Mul => {
                        if data1.scale == 0 {
                            Some(SimpleIndexingData {
                                scale: data2.scale * data1.offset,
                                offset: data2.offset * data1.offset
                            })
                        } else if data2.scale == 0 {
                            Some(SimpleIndexingData {
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

    fn compute_expr_extent(&self, expr: &SourceExpr) -> Shape  {
        match expr {
            SourceExpr::For(_, extent, body) => {
                let body_extent = self.compute_expr_extent(body);
                im::vector![extent.clone()] + body_extent
            },

            SourceExpr::Reduce(_, body) => {
                let mut body_extent = self.compute_expr_extent(body);
                body_extent.split_off(1)
            },

            SourceExpr::ExprOp(_, expr1, expr2) => {
                let extent1 = self.compute_expr_extent(expr1);
                let extent2 = self.compute_expr_extent(expr2);
                assert!(extent1 == extent2);
                extent1
            },

            SourceExpr::Indexing(arr, index_list) => {
                if let Some(arr_extent) = self.store.get(arr) {
                    arr_extent.clone().split_off(index_list.len())
                } else {
                    panic!("no binding for {}", arr)
                }
            },

            SourceExpr::Literal(_) => im::Vector::new()
        }
    }

    fn compute_extent_prog(&mut self, program: &SourceProgram) {
        for input in program.inputs.iter() {
            if let Some(_) = self.store.insert(input.0.clone(), input.1.clone()) {
                panic!("duplicate binding for {}", input.0)
            }
        }

        for binding in program.let_bindings.iter() {
            let extent = self.compute_expr_extent(&*binding.1);
            if let Some(_) = self.store.insert(binding.0.clone(), extent) {
                panic!("duplicate binding for {}", binding.0)
            }
        }

        let output_extent = self.compute_expr_extent(&program.expr);
        if let Some(_) = self.store.insert(String::from(OUTPUT_EXPR_NAME), output_extent) {
            panic!("duplicate binding for {}", OUTPUT_EXPR_NAME)
        }
    }

    fn register_transformed_expr(&mut self, transform: ArrayTransformInfo) -> ExprId {
        let id = self.fresh_expr_id();
        self.transform_info_map.insert(id, transform);
        id
    }

    // transform expression to compute data about its necessary layout and computation
    fn transform_expr(
        &mut self,
        expr: &SourceExpr,
        output_shape: &TransformShape<TransformedDim>,
        path: im::Vector<PathInfo>
    ) -> Result<TransformResult, String> {
        match expr {
            SourceExpr::For(index, extent, body) => {
                let new_path = 
                    path +
                    im::Vector::unit(PathInfo::Index {
                        index: index.clone(), extent: *extent
                    });

                self.transform_expr(body, output_shape, new_path)
            },

            SourceExpr::Reduce(op, body) => {
                let new_path = 
                    path + im::Vector::unit(PathInfo::Reduce { op: *op });
                let body_res = self.transform_expr(body, output_shape, new_path)?;
                let mut reduced_dim_position = body_res.reduced_dim_position.unwrap();
                let rest = reduced_dim_position.split_off(1);
                let (dim_type, dim ) = *reduced_dim_position.first().unwrap();
                let res =
                    TransformResult {
                        expr: TransformedExpr::ReduceNode(dim_type, dim, *op, Box::new(body_res.expr)),
                        reduced_dim_position: Some(rest),
                        transformed_inputs: body_res.transformed_inputs
                    };
                Ok(res)
            },

            SourceExpr::ExprOp(op, expr1, expr2) => {
                let res1 = self.transform_expr(expr1, output_shape, path.clone())?;
                let res2 = self.transform_expr(expr2, output_shape, path)?;
                let reduced_dim_position_opt =
                    match (&res1.reduced_dim_position, &res2.reduced_dim_position) {
                        (None, None) => Ok(None),
                        (None, Some(reduced_dim_position2)) => Ok(Some(reduced_dim_position2.clone())),
                        (Some(reduced_dim_position1), None) => Ok(Some(reduced_dim_position1.clone())),
                        (Some(reduced_dim_position1), Some(reduced_dim_position2)) => {
                            if reduced_dim_position1 == reduced_dim_position2 {
                                Ok(Some(reduced_dim_position1.clone()))
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
                let res =
                    TransformResult {
                        expr: TransformedExpr::Op(*op, Box::new(res1.expr), Box::new(res2.expr)),
                        reduced_dim_position: reduced_dim_position_opt,
                        transformed_inputs: res1.transformed_inputs.union(res2.transformed_inputs),
                    };
                Ok(res)
            },

            SourceExpr::Literal(lit) => {
                Ok(TransformResult {
                    expr: TransformedExpr::Literal(*lit as isize),
                    reduced_dim_position: None,
                    transformed_inputs: im::HashSet::new(),
                })
            },

            SourceExpr::Indexing(array, index_list) => {
                // first, determine the computed shape of the array
                // based on path info and the output shape

                // dimensions that are reduced
                let mut reduced_dims: Vec<(&IndexName, &Extent)> = Vec::new();
                let mut reduced_dim_position: Vec<(ReducedDimType, usize)> = Vec::new();

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

                // TODO: what to do when output shape has less dimensions than computed shape?
                // (e.g. a reduced dim is not used in the output shape)

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
                            return Err(String::from("output shape should not have hidden dims"))
                        }
                    }
                }

                // if some reduced dims are not used, they are added to the front
                if used_rdim < reduced_dims.len() {
                    for rdim in reduced_dims[used_rdim..].iter() {
                        let dim = TransformedIndexDim::ReducedIndex(rdim.0.to_string());
                        out_index_shape.0.insert(0, dim);
                    }
                }

                // compute the positions of reduced dimensions
                for dim in reduced_dims.iter() {
                    let index_opt =
                        out_index_shape.0.iter().position(|x|
                            match x {
                                TransformedIndexDim::Index(index) => index == dim.0,
                                TransformedIndexDim::ReducedIndex(index) => index == dim.0,
                                TransformedIndexDim::Fill(_) => false,
                            }
                        );

                    if let Some(index) = index_opt { 
                        let reduced_dim_type =
                            match out_index_shape.0[index] {
                                TransformedIndexDim::Index(_) => ReducedDimType::Reused,
                                TransformedIndexDim::ReducedIndex(_) => ReducedDimType::Hidden,
                                TransformedIndexDim::Fill(_) => panic!("reduced dim cannot be a fill")
                            };
                        reduced_dim_position.push((reduced_dim_type, index));

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

                // computed_shape is the final shape computed from the output index shape
                let mut computed_shape: TransformShape<TransformedDim> = TransformShape::default();
                for dim in out_index_shape.0 {
                    match dim {
                        TransformedIndexDim::Index(index) |
                        TransformedIndexDim::ReducedIndex(index) => {
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
                                                PathInfo::Index { index: path_index, extent: _ }
                                                if *path_index == index => true,

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
                    }
                }

                let dim_info: Vec<TransformedDimInfo> =
                    computed_shape.0.into_iter().map(|dim| {
                        let extent = 
                            match dim {
                                // for indexed dims, perform interval analysis to determine padding
                                TransformedDim::Input(i) => {
                                    let index_expr = &index_list[i];
                                    let index_interval = self.index_expr_to_interval(index_expr, &index_store);
                                    Interval::new(index_interval.lower(), index_interval.upper())
                                },

                                // fill dims never have padding
                                TransformedDim::Fill(extent) => extent
                            };

                        let offset: isize =
                            match dim {
                                // for indexed dims, perform interval analysis to determine padding
                                TransformedDim::Input(i) => {
                                    let index_expr = &index_list[i];
                                    let index_var = index_expr.get_single_var().unwrap();
                                    if let Some(data) = self.get_simple_indexing_data(index_expr, &index_var) {
                                        data.offset

                                    } else {
                                        panic!("index expr is not simple")
                                    }
                                },

                                // fill dims never have padding
                                TransformedDim::Fill(_) => 0
                            };

                        Ok(TransformedDimInfo { dim, pad: (0,0), extent, offset })
                    }).collect::<Result<Vec<TransformedDimInfo>,String>>()?;

                let expr_id =
                    self.register_transformed_expr(
                        ArrayTransformInfo(String::from(array), dim_info)
                    );

                let res =
                    TransformResult {
                        expr: TransformedExpr::ExprRef(expr_id),
                        reduced_dim_position: Some(reduced_dim_position),
                        transformed_inputs: im::HashSet::unit(expr_id)
                    };
                Ok(res)
            },
        }
    }

    fn gen_extent_constraints_expr(&mut self, expr: &TransformedExpr) -> Option<(usize, ShapeId)> {
        match expr {
            TransformedExpr::ReduceNode(dim_type, _, _, body) => {
                if let Some((head, shape)) = self.gen_extent_constraints_expr(body) {
                    match dim_type {
                        ReducedDimType::Hidden => Some((head+1, shape)),
                        ReducedDimType::Reused => Some((head, shape)),
                    }
                } else {
                    panic!("attempting to reduce dimension of scalar value")
                }
            },

            TransformedExpr::Op(_, expr1, expr2) => {
                let res_opt1 = self.gen_extent_constraints_expr(expr1);
                let res_opt2  = self.gen_extent_constraints_expr(expr2);
                match (res_opt1, res_opt2) {
                    (None, None) => None,
                    (None, Some(res2)) => Some(res2),
                    (Some(res1), None) => Some(res1),
                    (Some((head1, shape1)), Some((head2, shape2))) => {
                        self.extent_analysis.add_equals_constraint(shape1, head1, shape2, head2);

                        // arbitrarily return the first shape, since it should
                        // be the same as the second shape anyway
                        Some((head1, shape1))
                    }
                }
            },

            TransformedExpr::Literal(_) => None,

            TransformedExpr::ExprRef(id) =>
                self.shape_map.get(id).map(|x| *x)
        }
    }

    /// collect constraints for extent analysis
    fn gen_extent_constraints_prog(&mut self) {
        // temporarily move transform_list to a tmp so we can iterate through it
        // we need to do this because the loop mutates the self data structure,
        // so the compiler prevents us from immutably borrowing transform_list
        // to iterate through it.
        // by moving transform_list to a temporary owned ref, we ensure that
        // the mutations to self in the loop don't change it
        let tmp_transform_list = std::mem::replace(&mut self.transform_list, Vec::new());

        for id in tmp_transform_list.iter() {
            // don't process inputs; they don't have mappings in transform_map
            if let Some(expr) = self.transform_map.get(id).map(|x| x.clone()) {
                if let Some((head, shape_id)) = self.gen_extent_constraints_expr(&expr) {
                    self.shape_map.insert(*id, (head, shape_id));

                    let transform_info = &self.transform_info_map[&id];
                    let required_shape: Shape =
                        transform_info.1.iter()
                        .map(|info| info.extent)
                        .collect();

                    self.extent_analysis.add_atleast_constraint(shape_id, head, required_shape);
                }
            }
        }

        // restore transform_list
        self.transform_list = tmp_transform_list;
    }

    /// apply extent solutions to determine padding
    fn apply_extent_solution(&mut self) {
        let extent_solution = self.extent_analysis.solve();
        let expr_extent_map: HashMap<ExprId, Shape> =
            self.shape_map.iter().map(|(id, (head, shape_id))| {
                let mut shape = extent_solution[shape_id].clone();
                (*id, shape.split_off(*head))
            }).collect();

        for (id, transform) in self.transform_info_map.iter_mut() {
            let computed_shape = &expr_extent_map[id];
            let input_shape =  self.input_map.get(&transform.0);

            // update extents
            for (i, dim_info) in transform.1.iter_mut().enumerate() {
                let computed_extent = computed_shape[i];
                dim_info.extent = computed_extent.clone();

                // if input, compute padding as well
                if let Some(orig_shape) = input_shape {
                    match dim_info.dim {
                        // for input dimensions, add padding
                        TransformedDim::Input(orig_i) => {
                            let orig_extent = &orig_shape[orig_i];
                            let pad_min = (orig_extent.lower() - computed_extent.lower()) as usize;
                            let pad_max = (computed_extent.upper() - orig_extent.upper()) as usize;
                            dim_info.pad = (pad_min, pad_max);
                        },

                        // for fill dimensions, just overwrite the fill extent to the computed one
                        TransformedDim::Fill(_) => {
                            dim_info.dim = TransformedDim::Fill(computed_extent);
                        }
                    }
                }
            }
        }
    }

    fn fill_and_rotate(&self, info: &ArrayTransformInfo, expr: IndexFreeExpr) -> IndexFreeExpr {
        // fill
        let fill_dims: im::Vector<usize> =
            info.1.iter().enumerate()
            .filter(|(_, dim_info)| dim_info.is_fill())
            .map(|(i,_)| i)
            .collect();

        let filled_expr =
            fill_dims.iter().fold(expr, |acc, dim| {
                IndexFreeExpr::Fill(Box::new(acc), *dim)
            });

        let need_offset = 
            info.1.iter()
            .any(|dim_info| dim_info.offset != 0);

        let offset_expr =
            if need_offset {
                IndexFreeExpr::Offset(
                    Box::new(filled_expr),
                    info.1.iter().map(|dim_info| {
                        dim_info.offset
                    }).collect()
                )

            } else {
                filled_expr
            };

        offset_expr
    }

    /// generate client transformation from transform info
    fn lower_to_client_transform(&self, info: &ArrayTransformInfo) -> ClientTransform {
        let input_array = ClientTransform::InputArray(info.0.clone());

        let cmp_input_dims =
            |dim1: &&TransformedDimInfo, dim2: &&TransformedDimInfo| -> std::cmp::Ordering {
                match (&dim1.dim, &dim2.dim){
                    (TransformedDim::Input(i1), TransformedDim::Input(i2)) => {
                        i1.cmp(i2)
                    },

                    _ => {
                        panic!("fill dim cannot be one of the original dims")
                    }
                }
            };

        let transposed_dims: Vec<&TransformedDimInfo> =
            info.1.iter()
            .filter(|dim_info| dim_info.is_input())
            .collect();

        let need_transpose =
            transposed_dims.iter().enumerate().any(|(i, dim_info)| {
                match dim_info.dim {
                    TransformedDim::Input(idim) => i != idim,
                    TransformedDim::Fill(_) => true,
                }
            });

        let mut orig_dims = transposed_dims.clone();

        if need_transpose {
            orig_dims.sort_by(cmp_input_dims);
        }

        // pad
        let need_padding =
            orig_dims.iter().any(|dim_info| {
                dim_info.pad.0 != 0 || dim_info.pad.1 != 0
            });

        let padded_transform =
            if need_padding {
                ClientTransform::Pad(
                    Box::new(input_array),
                    transposed_dims.iter()
                    .map(|dim_info| dim_info.pad)
                    .collect::<im::Vector<PadSize>>()
                )

            } else {
                input_array
            };

        // transpose
        let transpose_transform =
            if need_transpose {
                ClientTransform::Transpose(
                    Box::new(padded_transform),
                    transposed_dims.iter().map(|dim_info| {
                        match dim_info.dim {
                            TransformedDim::Input(i) => i,

                            TransformedDim::Fill(_) => {
                                panic!("fill dim should not be an original dim")
                            }
                        }
                    }).collect()
                )

            } else {
                padded_transform
            };

        // extend with new dimensions
        let fill_dims: Vec<(usize, &TransformedDimInfo)> = 
            info.1.iter().enumerate()
            .filter(|(_, dim_info)| dim_info.is_fill())
            .collect();

        let expand_transform =
            if fill_dims.is_empty() {
                transpose_transform

            } else {
                ClientTransform::Expand(
                    Box::new(transpose_transform),
                    fill_dims.iter().map(|(i,_)| *i).collect()
                )
            };

        expand_transform
    }

    fn lower_to_index_free_expr(
        &self,
        expr: &TransformedExpr,
        client_object_map: &HashMap<ArrayTransformInfo, HEObjectName>,
        indfree_expr_map: &mut HashMap<ExprId, IndexFreeExpr>,
    ) -> IndexFreeExpr {
        match expr {
            TransformedExpr::ReduceNode(dim_type, dim, op, body) => {
                let new_body = self.lower_to_index_free_expr(body, client_object_map, indfree_expr_map);
                match dim_type {
                    ReducedDimType::Hidden => {
                        IndexFreeExpr::Reduce(*dim, *op, Box::new(new_body))
                    },

                    // if a dim is reused, zero it out
                    ReducedDimType::Reused => {
                        IndexFreeExpr::Zero(
                            Box::new(
                                IndexFreeExpr::Reduce(*dim, *op, Box::new(new_body))
                            ),
                            *dim
                        )
                    }
                }
            },

            TransformedExpr::Op(op, expr1, expr2) => {
                let new_expr1 = self.lower_to_index_free_expr(expr1, client_object_map, indfree_expr_map);
                let new_expr2 = self.lower_to_index_free_expr(expr2, client_object_map, indfree_expr_map);
                IndexFreeExpr::Op(*op, Box::new(new_expr1), Box::new(new_expr2))
            },

            TransformedExpr::Literal(lit) => {
                IndexFreeExpr::Literal(*lit)
            },

            TransformedExpr::ExprRef(id) => {
                if self.transform_map.contains_key(id) { // let-binding
                    indfree_expr_map.remove(id).unwrap()

                } else { // input
                    let info = &self.transform_info_map[id];
                    let name = &client_object_map[info];
                    let input = IndexFreeExpr::InputArray(String::from(name));
                    self.fill_and_rotate(info, input)
                }
            }
        }
    }

    fn lower_to_index_free_prog(&mut self) -> Result<IndexFreeProgram, String> {
        // generate map of client ciphertexts
        let mut transform_object_map: HashMap<ArrayTransformInfo, HEObjectName> = HashMap::new();
        let mut indfree_expr_map: HashMap<ExprId, IndexFreeExpr> = HashMap::new();
        let mut ciphertexts: HashMap<HEObjectName, Ciphertext> = HashMap::new();

        for id in self.transform_list.iter() {
            if let Some(cur_expr) = self.transform_map.get(id) { // let-binding
                let indfree_expr = self.lower_to_index_free_expr(cur_expr, &transform_object_map, &mut indfree_expr_map);

                // add fills
                let info = &self.transform_info_map[id];
                let final_expr = self.fill_and_rotate(info, indfree_expr);
                indfree_expr_map.insert(*id, final_expr);

            } else { // input
                let info = &self.transform_info_map[id];
                if !transform_object_map.contains_key(&info) {
                    let name = self.name_generator.get_fresh_name(&format!("c_{}", info.0));
                    transform_object_map.insert(info.clone(), name.clone());
                    ciphertexts.insert(name, Ciphertext { dimensions: info.to_dimensions() });
                }
            }
        }

        // build client store
        let client_store: HEClientStore = 
            transform_object_map.iter().map(|(info, name)| {
                (String::from(name), self.lower_to_client_transform(info))
            }).collect();

        let expr = indfree_expr_map.remove(&self.output_id).unwrap();

        Ok(IndexFreeProgram { client_store, expr, ciphertexts })
    }

    fn transform_program(&mut self) -> Result<IndexFreeProgram, String> {
        let output_extent =
            self.store.get(OUTPUT_EXPR_NAME)
            .ok_or(format!("No binding for output expression {}", OUTPUT_EXPR_NAME))?;

        let output_transform: ArrayTransformInfo =
            ArrayTransformInfo(
                String::from(OUTPUT_EXPR_NAME),
                output_extent.iter().enumerate().map(|(i, extent)| {
                    TransformedDimInfo {
                        dim: TransformedDim::Input(i),
                        pad: (0, 0),
                        extent: extent.clone(),
                        offset: 0,
                    }
                }).collect()
            );
        self.transform_info_map.insert(self.output_id, output_transform);

        // backwards analysis to determine the transformations
        // needed for indexed arrays
        self.transform_list.push(self.output_id);
        let mut worklist: Vec<ExprId> = vec![self.output_id];
        while !worklist.is_empty() {
            let cur_id = worklist.pop().unwrap();
            let transform = &self.transform_info_map[&cur_id];
            let cur_expr_opt =
                self.expr_binding_map.get(&transform.0).map(|x| x.clone());

            // add transformed arrays to the worklist, if they are let bound
            if let Some(cur_expr) = cur_expr_opt {
                let cur_res =
                    self.transform_expr(
                        &cur_expr, 
                        &transform.to_transformed_shape(),
                        im::Vector::new()
                    )?;

                worklist.extend(cur_res.transformed_inputs.iter());
                self.transform_list.extend(cur_res.transformed_inputs.iter());
                self.transform_map.insert(cur_id, cur_res.expr);

            // for inputs, add directly to the shape map
            } else if let Some(_) = self.input_map.get(&transform.0) {
                let dims = transform.1.len();
                let required_shape: Shape = 
                    transform.1.iter()
                    .map(|info| info.extent)
                    .collect();

                let shape_id = self.extent_analysis.register_shape(dims);
                self.shape_map.insert(cur_id, (0, shape_id));
                self.extent_analysis.add_atleast_constraint(shape_id, 0, required_shape);

            } else {
                panic!("variable {} is not bound", &transform.0)
            }
        }

        // since transform list stores a toposort of expressions
        // and it was populated above starting from the output expression,
        // we need to reverse it
        self.transform_list.reverse();

        // generate constraints for the extent analysis
        self.gen_extent_constraints_prog();

        // apply extent solutions to determine padding
        self.apply_extent_solution();

        // lower to index-free expression and return client-side transforms
        self.lower_to_index_free_prog()
    }

    pub fn run(mut self, program: &SourceProgram) -> Result<IndexFreeProgram, String> {
        program.inputs.iter().for_each(|input| {
            if let Some(_) = self.input_map.insert(input.0.clone(), input.1.clone()) {
                panic!("duplicate bindings for {}", &input.0)
            }
        });

        program.let_bindings.iter().for_each(|let_binding| {
            if let Some(_) = self.expr_binding_map.insert(let_binding.0.clone(), *let_binding.1.clone()) {
                panic!("duplicate bindings for {}", &let_binding.0)
            }
        });
        self.expr_binding_map.insert(String::from(OUTPUT_EXPR_NAME), program.expr.clone());

        self.compute_extent_prog(program);
        self.transform_program()
    }
}

impl Default for IndexElimination {
    fn default() -> Self {
        IndexElimination::new()
    }
}

#[cfg(test)]
mod tests{
    use crate::lang::parser::ProgramParser;
    use super::*;

    fn test_lowering(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let mut index_elim = IndexElimination::new();
        let res = index_elim.run(&program);
        
        assert!(res.is_ok());

        let prog = res.unwrap();
        for (name, transform) in prog.client_store.iter() {
            println!("{} => {}", name, transform)
        }
        println!("{}", prog.expr);
    }

    #[test]
    fn test_imgblur() {
        test_lowering(
        "input img: [(0,16),(0,16)]
            for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }"
        );
    }

    #[test]
    fn test_imgblur2() {
        test_lowering(
        "input img: [(0,16),(0,16)]
            let res = 
                for x: (0, 16) {
                    for y: (0, 16) {
                        img[x-1][y-1] + img[x+1][y+1]
                    }
                }
            for x: (0, 16) {
                for y: (0, 16) {
                    res[x-2][y-2] + res[x+2][y+2]
                }
            }
            "
        );
    }

    #[test]
    fn test_convolve() {
        test_lowering(
        "input img: [(0,16),(0,16)]
            let conv1 = 
                for x: (0, 15) {
                    for y: (0, 15) {
                        img[x][y] + img[x+1][y+1]
                    }
                }

            for x: (0, 14) {
                for y: (0, 14) {
                    conv1[x][y] + conv1[x+1][y+1]
                }
            }
            "
        );
    }

    #[test]
    fn test_matmatmul() {
        test_lowering(
            "input A: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            let x = A + B
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A[i][k] * B[k][j] })
                }
            }"
        );
    }

    #[test]
    fn test_matmatmul2() {
        test_lowering(
            "input A1: [(0,4),(0,4)]
            input A2: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            let res =
                for i: (0,4) {
                    for j: (0,4) {
                        sum(for k: (0,4) { A1[i][k] * B[k][j] })
                    }
                }
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A2[i][k] * res[k][j] })
                }
            }
            "
        );
    }
}