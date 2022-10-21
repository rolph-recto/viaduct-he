use std::cmp::max;

use interval::ops::Range;
use gcollections::ops::{bounded::Bounded, Subset};

use crate::{lang::{*, source::{*, IndexExpr::*}}, circ::circ_gen::IndexFreeExpr};

use super::extent_analysis::{ExtentAnalysis, ShapeId};

type PadSize = (usize, usize);

struct SimpleIndexingData { scale: i64, offset: i64 }

#[derive(Clone,Debug)]
enum PathInfo {
    Index { index: IndexName, extent: Extent },
    Reduce { op: ExprOperator }
}

#[derive(Eq,PartialEq,Clone,Copy,Debug)]
pub enum ReducedDimType { Hidden, Reused }

#[derive(Clone,Debug)]
pub enum TransformedExpr {
    ReduceNode(ReducedDimType, usize, ExprOperator, Box<TransformedExpr>),
    Op(ExprOperator, Box<TransformedExpr>, Box<TransformedExpr>),
    Literal(i64),
    ExprRef(ExprId),
}

impl Display for TransformedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformedExpr::ReduceNode(dim_type, dim, op, body) => {
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
pub struct ArrayTransformInfo(ArrayName, Vec<TransformedDimInfo>);

impl ArrayTransformInfo {
    fn to_transformed_shape(&self) -> TransformShape<TransformedDim> {
        TransformShape(
            self.1.iter()
            .map(|dim_info| dim_info.dim.clone())
            .collect()
        )
    }
}

impl Display for ArrayTransformInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let strs =
            self.1.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>().join(", ");

        write!(f, "[{}]", strs)
    }
}

#[derive(Debug)]
struct TransformResult {
    expr: TransformedExpr,
    reduced_dim_position: Option<Vec<(ReducedDimType, usize)>>,
    transformed_inputs: im::HashSet<ExprId>,
}

pub struct Normalizer {
    output_id: ExprId,
    cur_expr_id: usize,
    transform_info_map: HashMap<ExprId, ArrayTransformInfo>,
    transform_map: HashMap<ExprId, TransformedExpr>,
    shape_map: HashMap<ExprId, (usize, ShapeId)>,
    extent_analysis: ExtentAnalysis,
}

static OUTPUT_EXPR_NAME: &'static str = "$root";

impl Normalizer {
    pub fn new() -> Self {
        Normalizer {
            output_id: 0,
            cur_expr_id: 1,
            transform_info_map: HashMap::new(),
            transform_map: HashMap::new(),
            shape_map: HashMap::new(),
            extent_analysis: ExtentAnalysis::new(),
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

    fn get_linear_indexing_data(&self, index_expr: &IndexExpr, index_var: &IndexName) -> Option<SimpleIndexingData> {
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
                let data1 = self.get_linear_indexing_data(expr1, index_var)?;
                let data2 = self.get_linear_indexing_data(expr2, index_var)?;
                match op {
                    ExprOperator::OpAdd => {
                        Some(SimpleIndexingData {
                            scale: data1.scale + data2.scale,
                            offset: data1.offset + data2.offset
                        })
                    },
                    ExprOperator::OpSub => {
                        Some(SimpleIndexingData {
                            scale: data1.scale - data2.scale,
                            offset: data1.offset - data2.offset
                        })
                    },
                    ExprOperator::OpMul => {
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

    fn compute_expr_extent(&self, expr: &SourceExpr, store: &ArrayEnvironment) -> Shape  {
        match expr {
            SourceExpr::ForNode(_, extent, body) => {
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

        let output_extent = self.compute_expr_extent(&program.expr, &store);
        if let Some(_) = store.insert(String::from(OUTPUT_EXPR_NAME), output_extent) {
            panic!("duplicate binding for {}", OUTPUT_EXPR_NAME)
        }

        store
    }

    fn register_transformed_expr(&mut self, transform: ArrayTransformInfo) -> ExprId{
        let id = self.fresh_expr_id();
        self.transform_info_map.insert(id, transform);
        id
    }

    fn transform_expr(
        &mut self,
        expr: &SourceExpr,
        output_shape: &TransformShape<TransformedDim>,
        store: &ArrayEnvironment,
        path: im::Vector<PathInfo>
    ) -> Result<TransformResult, String> {
        match expr {
            SourceExpr::ForNode(index, extent, body) => {
                let new_path = 
                    path +
                    im::Vector::unit(PathInfo::Index {
                        index: index.clone(), extent: *extent
                    });

                self.transform_expr(body, output_shape, store, new_path)
            },

            SourceExpr::ReduceNode(op, body) => {
                let new_path = 
                    path + im::Vector::unit(PathInfo::Reduce { op: *op });
                let body_res = self.transform_expr(body, output_shape, store, new_path)?;
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

            SourceExpr::OpNode(op, expr1, expr2) => {
                let res1 = self.transform_expr(expr1, output_shape, store, path.clone())?;
                let res2 = self.transform_expr(expr2, output_shape, store, path)?;
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

            SourceExpr::LiteralNode(lit) => {
                Ok(TransformResult {
                    expr: TransformedExpr::Literal(*lit),
                    reduced_dim_position: None,
                    transformed_inputs: im::HashSet::new(),
                })
            },

            SourceExpr::IndexingNode(array, index_list) => {
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
                        Ok(TransformedDimInfo { dim, pad: (0,0), extent })
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

    fn transform_program(
        &mut self, prog: &SourceProgram, store: &ArrayEnvironment
    ) -> Result<IndexFreeExpr, String> {
        let input_map: HashMap<ArrayName, Shape> =
            prog.inputs.iter().map(|input| {
                (input.0.clone(), input.1.clone())
            }).collect();

        let let_binding_map: HashMap<ArrayName, SourceExpr> =
            prog.letBindings.iter().map(|let_binding| {
                (let_binding.0.clone(), *let_binding.1.clone())
            }).collect();

        let output_extent =
            store.get(OUTPUT_EXPR_NAME).ok_or(format!("No binding for output expression {}", OUTPUT_EXPR_NAME))?;
        let output_transform: ArrayTransformInfo =
            ArrayTransformInfo(
                String::from(OUTPUT_EXPR_NAME),
                output_extent.iter().enumerate().map(|(i, extent)| {
                    TransformedDimInfo {
                        dim: TransformedDim::Input(i),
                        pad: (0, 0),
                        extent: extent.clone()
                    }
                }).collect()
            );
        self.transform_info_map.insert(self.output_id, output_transform);

        // backwards analysis to determine the transformations
        // needed for indexed arrays
        let mut transform_list: Vec<ExprId> = vec![self.output_id];
        let mut worklist: Vec<ExprId> = vec![self.output_id];
        while !worklist.is_empty() {
            let cur_id = worklist.pop().unwrap();
            let transform = &self.transform_info_map[&cur_id];

            // add transformed arrays to the worklist, if they are let bound
            if let Some(cur_expr) = let_binding_map.get(&transform.0) {
                let cur_res =
                    self.transform_expr(
                        cur_expr, 
                        &transform.to_transformed_shape(),
                        store,
                        im::Vector::new()
                    )?;

                worklist.extend(cur_res.transformed_inputs.iter());
                transform_list.extend(cur_res.transformed_inputs.iter());
                self.transform_map.insert(cur_id, cur_res.expr);

            // for inputs, add directly to the shape map
            } else if let Some(_) = input_map.get(&transform.0) {
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

        // extent analysis to determine necessary padding
        transform_list.reverse();
        for id in transform_list {
            let expr = self.transform_map[&id].clone();
            if let Some((head, shape_id)) = self.gen_extent_constraints(&expr) {
                self.shape_map.insert(id, (head, shape_id));

                let transform_info = &self.transform_info_map[&id];
                let required_shape: Shape =
                    transform_info.1.iter()
                    .map(|info| info.extent)
                    .collect();

                self.extent_analysis.add_atleast_constraint(shape_id, head, required_shape);
            }
        }

        let extent_solution = self.extent_analysis.solve();
        let expr_extent_map: HashMap<ExprId, Shape> =
            self.shape_map.iter().map(|(id, (head, shape_id))| {
                let mut shape = extent_solution[shape_id].clone();
                (*id, shape.split_off(*head))
            }).collect();

        for (id, transform) in self.transform_info_map.iter_mut() {
            let computed_shape = &expr_extent_map[id];
            let input_shape =  input_map.get(&transform.0);

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

        todo!()
    }

    fn gen_extent_constraints(&mut self, expr: &TransformedExpr) -> Option<(usize, ShapeId)> {
        match expr {
            TransformedExpr::ReduceNode(dim_type, dim, op, body) => {
                if let Some((head, shape)) = self.gen_extent_constraints(body) {
                    match dim_type {
                        ReducedDimType::Hidden => Some((head+1, shape)),
                        ReducedDimType::Reused => Some((head, shape)),
                    }
                } else {
                    panic!("attempting to reduce dimension of scalar value")
                }
            },

            TransformedExpr::Op(_, expr1, expr2) => {
                let res_opt1 = self.gen_extent_constraints(expr1);
                let res_opt2  = self.gen_extent_constraints(expr2);
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
            normalizer.transform_expr(&program.expr, &out_shape, &store, im::Vector::new());
        
        assert!(res.is_ok());
        println!("{:?}", res.unwrap().expr);
    }

    #[test]
    fn test_imgblur() {
        test_lowering(
        "input img: [(0,16),(0,16)]
            for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }",
            TransformShape(vec![
                TransformedDim::Input(1),
                TransformedDim::Input(0),
            ]),
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
            }",
            TransformShape(vec![
                TransformedDim::Input(0),
                // TransformedDim::Fill(Interval::new(0, 4)),
                TransformedDim::Input(1),
            ])
        );
    }

    #[test]
    fn test_matmatmul2() {
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