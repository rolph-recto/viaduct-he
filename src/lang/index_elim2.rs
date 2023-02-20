use std::{hash::{*, Hasher}, collections::{HashSet, hash_set}};

use disjoint_sets::UnionFind;
use gcollections::ops::Bounded;

use crate::{
    lang::{*, IndexExpr::*},
};


/// expression with associated data about lowering to an index-free representation
#[derive(Clone,Debug)]
pub enum TransformedExpr {
    ReduceNode(usize, Operator, Box<TransformedExpr>),
    Op(Operator, Box<TransformedExpr>, Box<TransformedExpr>),
    Literal(isize),
    ExprRef(ExprRefId),
}

impl TransformedExpr {
    pub fn get_expr_refs(&self) -> im::HashSet<ExprRefId> {
        match self {
            TransformedExpr::ReduceNode(_, _, body) => {
                body.get_expr_refs()
            },

            TransformedExpr::Op(_, expr1, expr2) => {
                let set1 = expr1.get_expr_refs();
                let set2 = expr2.get_expr_refs();
                set1.union(set2)
            },

            TransformedExpr::Literal(_) => im::HashSet::new(),

            TransformedExpr::ExprRef(id) => im::hashset![*id]
        }
    }
}

impl Display for TransformedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformedExpr::ReduceNode(dim, op, body) => {
                write!(f, "reduce({}, {}, {})", dim, op, body)
            },

            TransformedExpr::Op(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            },

            TransformedExpr::ExprRef(id) => {
                write!(f, "expr({})", id)
            },

            TransformedExpr::Literal(lit) => {
                write!(f, "{}", lit)
            },
        }
    }
}

pub type InputArrayDim = (ExprRefId, DimIndex);

pub struct TransformedProgram {
    pub expr: TransformedExpr,
    pub inputs: HashMap<ExprRefId, ArrayTransform>,
    pub array_shapes: HashMap<ArrayName, Shape>,
}

impl TransformedProgram {
    pub fn compute_dim_equiv_classes(&self) -> HashMap<InputArrayDim, usize> {
        let (dim_eqs, _) = self.compute_dim_equalities(&self.expr);
        let id_map: HashMap<InputArrayDim, usize> =
            self.get_dim_set().into_iter().enumerate()
            .map(|(i, dim)| (dim, i))
            .collect();

        let mut uf: UnionFind<usize> = UnionFind::new(id_map.keys().len());
        for (dim1, dim2) in dim_eqs {
            uf.union(id_map[&dim1], id_map[&dim2]);
        }

        let mut class_map: HashMap<InputArrayDim, usize> = HashMap::new();
        for dim in id_map.keys() {
            class_map.insert(*dim, uf.find(id_map[&dim]));
        }

        class_map
    }

    fn find_parent(&self, dim: &InputArrayDim, parent_map: &HashMap<InputArrayDim,InputArrayDim>) -> InputArrayDim {
        let parent = &parent_map[dim];
        if parent == dim {
            *dim

        } else {
            self.find_parent(parent, parent_map)
        }
    }

    fn get_dim_set(&self) -> HashSet<InputArrayDim> {
        let mut dim_set: HashSet<InputArrayDim> = HashSet::new();
        for (ref_id, transform) in self.inputs.iter() {
            for dim in 0..transform.dims.len() {
                dim_set.insert((*ref_id, dim));
            }
        }

        dim_set
    }

    fn compute_dim_equalities(&self, expr: &TransformedExpr) -> (im::Vector<(InputArrayDim, InputArrayDim)>, im::Vector<InputArrayDim>) {
        match expr {
            TransformedExpr::ReduceNode(reduced_index, _, body) => {
                let (eq_body, mut body_dims) = self.compute_dim_equalities(body);
                assert!(body_dims.len() > 0);
                body_dims.remove(*reduced_index);
                (eq_body, body_dims)
            },

            TransformedExpr::Op(_, expr1, expr2) => {
                let (eqs1, dims1) = self.compute_dim_equalities(expr1);
                let (eqs2, dims2) = self.compute_dim_equalities(expr2);
                let (len1, len2) = (dims1.len(), dims2.len());

                if len1 == 0 && len2 != 0 {
                    (eqs1 + eqs2, dims2)

                } else if len1 != 0 && len2 == 0 {
                    (eqs1 + eqs2, dims1)

                } else {
                    assert!(len1 == len2);
                    let result_dims = dims1.clone();
                    let new_eqs: im::Vector<(InputArrayDim, InputArrayDim)> =
                        dims1.into_iter().zip(dims2.into_iter()).collect();
                    (eqs1 + eqs2 + new_eqs, result_dims)
                }
            },

            TransformedExpr::Literal(_) => (im::vector![], im::vector![]),

            TransformedExpr::ExprRef(ref_id) => {
                let input_dims: im::Vector<InputArrayDim> =
                    self.inputs[ref_id].dims.iter().enumerate()
                    .map(|(i, _)| (*ref_id, i))
                    .collect();

                (im::vector![], input_dims)
            }
        }
    }
}

#[derive(Clone,Debug)]
enum PathInfo {
    Index { index: IndexName, extent: usize },
    Reduce { op: Operator }
}

struct PathContext {
    path: im::Vector<PathInfo>
}

impl PathContext {
    fn new(path: im::Vector<PathInfo>) -> Self {
        PathContext { path }
    }

    fn get_index_extent(&self, index_var: &IndexName) -> Option<usize> {
        self.path.iter().find_map(|path_info| {
            match path_info {
                PathInfo::Index { index, extent } if index == index_var => {
                    Some(*extent)
                },

                _ => None
            }
        })
    }
}

#[derive(Clone, Debug)]
struct IndexingVarData {
    index_var: IndexName,
    stride: usize, 
}

#[derive(Clone, Debug)]
struct IndexingData {
    var_data: im::Vector<IndexingVarData>,
    offset: isize
}


// index elimination pass
pub struct IndexElimination2 {
    // map from source-level bindings to their shapes 
    store: ArrayEnvironment,

    // map from expr ref ids to array shapes
    transform_shape_map: HashMap<ExprRefId, ArrayTransform>,

    // map from expr ref ids to transformed exprs
    transform_map: HashMap<ExprRefId, TransformedExpr>,

    // counter for generating fresh expr ids
    cur_expr_id: usize,
}

impl IndexElimination2 {
    pub fn new() -> Self {
        IndexElimination2 {
            store: HashMap::new(),
            transform_shape_map: HashMap::new(),
            transform_map: HashMap::new(),
            cur_expr_id: 1,
        }
    }

    fn fresh_expr_id(&mut self) -> usize {
        let id = self.cur_expr_id;
        self.cur_expr_id += 1;
        id
    }

    fn register_transformed_expr(&mut self, array_shape: ArrayTransform) -> ExprRefId {
        let id = self.fresh_expr_id();
        self.transform_shape_map.insert(id, array_shape);
        id
    }

    fn compute_extent_expr(&self, expr: &SourceExpr) -> Shape  {
        match expr {
            SourceExpr::For(_, extent, body) => {
                let body_extent = self.compute_extent_expr(body);
                im::vector![extent.clone()] + body_extent
            },

            SourceExpr::Reduce(_, body) => {
                let mut body_extent = self.compute_extent_expr(body);
                body_extent.split_off(1)
            },

            SourceExpr::ExprOp(_, expr1, expr2) => {
                let extent1 = self.compute_extent_expr(expr1);
                let extent2 = self.compute_extent_expr(expr2);
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
            let extent = self.compute_extent_expr(&*binding.1);
            if let Some(_) = self.store.insert(binding.0.clone(), extent) {
                panic!("duplicate binding for {}", binding.0)
            }
        }

        let output_extent = self.compute_extent_expr(&program.expr);
        if let Some(_) = self.store.insert(String::from(OUTPUT_EXPR_NAME), output_extent) {
            panic!("duplicate binding for {}", OUTPUT_EXPR_NAME)
        }
    }

    fn process_index_expr(&self, index_expr: &IndexExpr) -> Result<IndexingData, String> {
        match index_expr {
            IndexVar(var) => {
                Ok(IndexingData {
                    var_data: im::vector![
                        IndexingVarData { index_var: var.clone(), stride: 1 }
                    ],
                    offset: 0
                })
            },

            IndexLiteral(lit) => {
                Ok(IndexingData {
                    var_data: im::Vector::new(),
                    offset: *lit
                })
            }

            IndexOp(op, expr1, expr2) => {
                let data1 = self.process_index_expr(expr1)?;
                let data2 = self.process_index_expr(expr2)?;

                match op {
                    Operator::Add => {
                        Ok(IndexingData {
                            var_data: data1.var_data + data2.var_data,
                            offset: data1.offset + data2.offset
                        })
                    },

                    Operator::Sub => {
                        // TODO cannot handle negative strides for now
                        assert!(data2.var_data.is_empty());

                        Ok(IndexingData {
                            var_data: data1.var_data,
                            offset: data1.offset - data2.offset
                        })
                    },

                    Operator::Mul => {
                        if data1.var_data.len() > 0 && data2.offset > 0 {
                            let mul_var_data1: im::Vector<IndexingVarData> =
                                data1.var_data.into_iter().map(|var| {
                                    IndexingVarData {
                                        index_var: var.index_var,
                                        stride: var.stride * (data2.offset as usize),
                                    }
                                }).collect();

                            Ok(IndexingData {
                                var_data: mul_var_data1,
                                offset: data1.offset * data2.offset
                            })

                        } else if data2.var_data.len() > 0 && data1.offset > 0 {
                            let mul_var_data2: im::Vector<IndexingVarData> =
                                data2.var_data.into_iter().map(|var| {
                                    IndexingVarData {
                                        index_var: var.index_var,
                                        stride: var.stride * (data1.offset as usize),
                                    }
                                }).collect();

                            Ok(IndexingData {
                                var_data: mul_var_data2,
                                offset: data1.offset * data2.offset
                            })

                        } else {
                            Err(format!("attempting to multiply index variables: {}", index_expr))
                        }
                    }
                }
            }
        }
    }

    fn transform_expr(
        &mut self,
        expr: &SourceExpr,
        output_transform: &ArrayTransform,
        path_ctx: &PathContext,
    ) -> Result<TransformedExpr,String> {
        match expr {
            SourceExpr::For(index, extent, body) => {
                let new_path_ctx = 
                    PathContext::new(
                        &path_ctx.path +
                        &im::Vector::unit(PathInfo::Index {
                            index: index.clone(),
                            extent: extent.upper() as usize
                        })
                    );

                self.transform_expr(body, output_transform, &new_path_ctx)
            },

            SourceExpr::ExprOp(op, expr1, expr2) => {
                let res1 = self.transform_expr(expr1, output_transform, path_ctx)?;
                let res2 = self.transform_expr(expr2, output_transform, path_ctx)?;
                Ok(TransformedExpr::Op(*op, Box::new(res1), Box::new(res2)))
            },

            SourceExpr::Literal(num) => {
                Ok(TransformedExpr::Literal(*num))
            }

            SourceExpr::Reduce(op, body) => {
                let new_path_ctx = 
                    PathContext::new(
                        &path_ctx.path + &im::Vector::unit(PathInfo::Reduce { op: *op })
                    );
                let body_res = self.transform_expr(body, output_transform, &new_path_ctx)?;

                // index of reduced dim should always 0
                // this invariant is enforced by the Indexing case
                Ok(TransformedExpr::ReduceNode(0, *op, Box::new(body_res)))
            },

            // this should compute a new output shape for the indexed array
            SourceExpr::Indexing(array, index_list) => {
                let array_extent = self.store[array].clone();
                let array_dims = array_extent.len();

                // process indexing sites
                let mut index_to_output_dim_map: HashMap<IndexName, (DimIndex, usize)> = HashMap::new();
                let mut array_offset_map = BaseOffsetMap::new(array_extent.len());
                for (index_dim, index_expr) in index_list.iter().enumerate() {
                    let indexing_data = self.process_index_expr(index_expr)?;
                    array_offset_map.set(index_dim, indexing_data.offset);
                    for var_data in indexing_data.var_data.into_iter() {
                        if !index_to_output_dim_map.contains_key(&var_data.index_var) {
                            let stride = var_data.stride;
                            index_to_output_dim_map.insert(
                                var_data.index_var,
                                (index_dim, stride)
                            );

                        } else {
                            return Err(format!("Index variable {} used in multiple indexing sites", &var_data.index_var))
                        }
                    }
                }

                // build output dims in the order of their original shape and stride
                // (i.e. without recourse to output_shape)
                // the created output dimensions come first,
                // and then the "retained" output dimensions that were not indexed
                let mut num_reductions = 0;
                let mut output_dims: Vec<DimContent> = Vec::new();
                let mut reduced_dims: Vec<DimContent> = Vec::new();
                for info in path_ctx.path.iter() {
                    match info {
                        PathInfo::Index { index: index_var, extent } => {
                            let dim = 
                                match index_to_output_dim_map.get(index_var) {
                                    Some((dim_index, stride)) => {
                                        DimContent::FilledDim {
                                            dim: *dim_index,
                                            extent: *extent,
                                            stride: *stride
                                        }
                                    },

                                    // index var wasn't used;
                                    // then this output dim is empty
                                    None => {
                                        DimContent::EmptyDim {
                                            extent: *extent
                                        }
                                    }
                                };

                            if num_reductions > 0 { // dim is reduced
                                reduced_dims.push(dim);
                                num_reductions -= 1;

                            } else { // dim is in output
                                output_dims.push(dim);
                            }
                        },

                        PathInfo::Reduce { op: _ } => {
                            num_reductions += 1;
                        },
                    }
                }

                // process reduced dims that are not introduced by a for node
                // but rather is in array tail that is not indexed
                let mut cur_retained_dim = index_list.len();
                while num_reductions > 0 {
                    reduced_dims.push(
                        DimContent::FilledDim {
                            dim: cur_retained_dim,
                            extent: array_extent[cur_retained_dim].upper() as usize,
                            stride: 1,
                        }
                    );
                    num_reductions -= 1;
                    cur_retained_dim += 1;
                }

                // dims in array tail that are not reduced are added to the output
                while cur_retained_dim < array_dims {
                    output_dims.push(
                        DimContent::FilledDim {
                            dim: cur_retained_dim,
                            extent: array_extent[cur_retained_dim].upper() as usize,
                            stride: 1,
                        }
                    );
                    cur_retained_dim += 1;
                }

                // build a shape for the indexed array given the required output shape
                let mut indexed_output_shape =
                    ArrayTransform {
                        array: array.to_string(),
                        offset_map: array_offset_map,
                        dims: im::Vector::new()
                    };

                // first, add reduced dims in LIFO order
                while let Some(dim) = reduced_dims.pop() {
                    indexed_output_shape.dims.push_back(dim);
                }

                // next, add the output dimensions
                for dim in output_transform.dims.iter() {
                    let indexed_output_dim =
                        match *dim {
                            DimContent::FilledDim { dim, extent: output_extent, stride: output_stride } => {
                                match output_dims[dim] {
                                    DimContent::FilledDim {
                                        dim: indexed_dim,
                                        extent: indexed_extent,
                                        stride: indexed_stride
                                    } => {
                                        let new_offset =
                                            indexed_output_shape.offset_map.get(dim) + 
                                            (output_transform.offset_map.get(dim) * (indexed_stride as isize));

                                        indexed_output_shape.offset_map.set(indexed_dim, new_offset);

                                        DimContent::FilledDim {
                                            dim: indexed_dim,
                                            extent: output_extent,
                                            stride: output_stride*indexed_stride
                                        }
                                    },

                                    // TODO: does this make sense??
                                    DimContent::EmptyDim { extent: _ } => {
                                        DimContent::EmptyDim { extent: output_extent }
                                    },
                                }
                            },

                            DimContent::EmptyDim { extent: _ } => {
                                dim.clone()
                            },
                        };

                    indexed_output_shape.dims.push_back(indexed_output_dim);
                }

                let expr_id = self.register_transformed_expr(indexed_output_shape);

                Ok(TransformedExpr::ExprRef(expr_id))
            }
        }
    }

    // resolve expr refs of intermediate arrays
    fn resolve_expr_refs(&self, expr: TransformedExpr) -> TransformedExpr {
        match expr {
            TransformedExpr::ReduceNode(i, op, body) => {
                let new_body = self.resolve_expr_refs(*body);
                TransformedExpr::ReduceNode(i, op, Box::new(new_body))
            },

            TransformedExpr::Op(op, expr1, expr2) => {
                let new_expr1 = self.resolve_expr_refs(*expr1);
                let new_expr2 = self.resolve_expr_refs(*expr2);
                TransformedExpr::Op(op, Box::new(new_expr1), Box::new(new_expr2))
            },

            TransformedExpr::Literal(_) => expr,

            TransformedExpr::ExprRef(id) => {
                if let Some(ref_expr) = self.transform_map.get(&id) {
                    ref_expr.clone()

                } else {
                    expr
                }
            }
        }
    }

    pub fn run(&mut self, program: &SourceProgram) -> Result<TransformedProgram, String> {
        self.compute_extent_prog(program);

        let output_extent = &self.store[OUTPUT_EXPR_NAME];
        let output_shape = 
            ArrayTransform {
                array: String::from(OUTPUT_EXPR_NAME),
                offset_map: BaseOffsetMap::new(output_extent.len()),
                dims: output_extent.iter().enumerate().map(|(i, dim_extent)| {
                    DimContent::FilledDim {
                        dim: i,
                        extent: dim_extent.upper() as usize,
                        stride: 1
                    }
                }).collect()
            };

        let output_id = self.fresh_expr_id();
        self.transform_shape_map.insert(output_id, output_shape);

        // backwards analysis goes from output expression and computes
        // required shapes for indexed arrays
        let mut transformed_inputs: HashMap<ExprRefId, ArrayTransform> = HashMap::new();
        let mut worklist = vec![output_id];
        while worklist.len() > 0 {
            let cur_id = worklist.pop().unwrap();
            let array_transform = self.transform_shape_map.remove(&cur_id).unwrap();
            
            if let Some(source_expr) = program.get_expr_binding(&array_transform.array) {
                let transformed_expr = 
                    self.transform_expr(
                        source_expr, 
                        &array_transform, 
                        &PathContext::new(im::Vector::new())
                    )?;

                let child_ids = transformed_expr.get_expr_refs();
                self.transform_map.insert(cur_id, transformed_expr);
                worklist.extend(child_ids.iter());

            } else { // array is an input
                transformed_inputs.insert(cur_id, array_transform);
            }
        }

        let unresolved_output_expr = self.transform_map.remove(&output_id).unwrap();
        let output_expr = self.resolve_expr_refs(unresolved_output_expr);

        let array_shapes: HashMap<ArrayName, Shape>  =
            transformed_inputs.iter().map(|(_, transform)| { 
                (transform.array.clone(), program.get_input_shape(&transform.array).unwrap().clone())
            }).collect();

        Ok(TransformedProgram { expr: output_expr, inputs: transformed_inputs, array_shapes })
    }
}

#[cfg(test)]
mod tests{
    use crate::lang::parser::ProgramParser;
    use super::*;

    fn test_index_elim(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let mut index_elim = IndexElimination2::new();
        let res = index_elim.run(&program);
        
        assert!(res.is_ok());

        let prog = res.unwrap();
        for (ref_id, transform) in prog.inputs {
            println!("{} => {}", ref_id, transform);
        }
        println!("{}", prog.expr);
    }

    #[test]
    fn test_imgblur() {
        test_index_elim(
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
        test_index_elim(
        "input img: [(0,16),(0,16)]
            let res = 
                for x: (0, 16) {
                    for y: (0, 16) {
                        img[x-1][y-1] + img[x+1][y+1]
                    }
                }
            in
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
        test_index_elim(
        "input img: [(0,16),(0,16)]
            let conv1 = 
                for x: (0, 15) {
                    for y: (0, 15) {
                        img[x][y] + img[x+1][y+1]
                    }
                }
            in
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
        test_index_elim(
            "input A: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A[i][k] * B[k][j] })
                }
            }"
        );
    }

    #[test]
    fn test_matmatmul2() {
        test_index_elim(
            "input A1: [(0,4),(0,4)]
            input A2: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            let res =
                for i: (0,4) {
                    for j: (0,4) {
                        sum(for k: (0,4) { A1[i][k] * B[k][j] })
                    }
                }
            in
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A2[i][k] * res[k][j] })
                }
            }
            "
        );
    }

    #[test]
    fn test_dotprod_pointless() {
        test_index_elim(
        "
            input A: [(0,3)]
            input B: [(0,3)]
            sum(A * B)
            "
        );
    }

    #[test]
    fn test_matvecmul() {
        test_index_elim(
        "
            input M: [(0,1),(0,1)]
            input v: [(0,1)]
            for i: (0,1) {
                sum(M[i] * v)
            }
            "
        );
    }
}