use std::{
    collections::{hash_set, HashSet},
    hash::{Hasher, *}, array,
};

use disjoint_sets::UnionFind;
use gcollections::ops::Bounded;
use indexmap::IndexMap;
use log::debug;

use crate::lang::{elaborated::ElaboratedProgram, *};

use super::source::{IndexExpr, SourceExpr};

/// expression with associated data about lowering to an index-free representation
#[derive(Clone, Debug)]
pub enum InlinedExpr {
    ReduceNode(usize, Operator, Box<InlinedExpr>),
    Op(Operator, Box<InlinedExpr>, Box<InlinedExpr>),
    Literal(isize),
    ExprRef(IndexingId, ArrayTransform),
}

impl InlinedExpr {
    pub fn get_indexed_arrays(&self) -> HashSet<ArrayName> {
        match self {
            InlinedExpr::ReduceNode(_, _, body) => body.get_indexed_arrays(),

            InlinedExpr::Op(_, expr1, expr2) => {
                let mut sites1 = expr1.get_indexed_arrays();
                let sites2 = expr2.get_indexed_arrays();
                sites1.extend(sites2);
                sites1
            }

            InlinedExpr::Literal(_) => HashSet::new(),

            InlinedExpr::ExprRef(_, transform) => HashSet::from([transform.array.clone()]),
        }
    }

    pub fn get_indexing_sites(&self) -> HashMap<IndexingId, ArrayTransform> {
        match self {
            InlinedExpr::ReduceNode(_, _, body) => body.get_indexing_sites(),

            InlinedExpr::Op(_, expr1, expr2) => {
                let mut sites1 = expr1.get_indexing_sites();
                let sites2 = expr2.get_indexing_sites();
                sites1.extend(sites2);
                sites1
            }

            InlinedExpr::Literal(_) => HashMap::new(),

            InlinedExpr::ExprRef(indexing_id, transform) => {
                HashMap::from([(indexing_id.clone(), transform.clone())])
            }
        }
    }
}

impl Display for InlinedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InlinedExpr::ReduceNode(dim, op, body) => {
                write!(f, "reduce({}, {}, {})", dim, op, body)
            }

            InlinedExpr::Op(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            }

            InlinedExpr::ExprRef(indexing_id, transform) => {
                write!(f, "expr({}, {})", indexing_id, transform)
            }

            InlinedExpr::Literal(lit) => {
                write!(f, "{}", lit)
            }
        }
    }
}

pub type ArrayDim = (IndexingId, DimIndex);

pub struct InlinedProgram {
    pub input_map: IndexMap<ArrayName, (Shape, ArrayType)>,
    pub expr_map: IndexMap<ArrayName, InlinedExpr>,
}

impl InlinedProgram {
    pub fn is_expr(&self, array: &ArrayName) -> bool {
        !self.is_input(array)
    }

    pub fn is_input(&self, array: &ArrayName) -> bool {
        self.input_map.contains_key(array)
    }

    pub fn get_output_expr(&self) -> &InlinedExpr {
        &self.expr_map.get(OUTPUT_EXPR_NAME).unwrap()
    }

    pub fn get_dim_equiv_classes(&self) -> HashMap<ArrayDim, usize> {
        let mut dim_eqs: Vec<(ArrayDim, ArrayDim)> = Vec::new();
        for (_, expr) in self.expr_map.iter() {
            let (eqs, _) = self.compute_dim_equalities(expr);
            dim_eqs.extend(eqs);
        }

        let id_map: HashMap<ArrayDim, usize> = self
            .get_dim_set()
            .into_iter()
            .enumerate()
            .map(|(i, dim)| (dim, i))
            .collect();

        let mut uf: UnionFind<usize> = UnionFind::new(id_map.keys().len());
        for (dim1, dim2) in dim_eqs {
            uf.union(id_map[&dim1], id_map[&dim2]);
        }

        let mut class_map: HashMap<ArrayDim, usize> = HashMap::new();
        for dim in id_map.keys() {
            class_map.insert(dim.clone(), uf.find(id_map[&dim]));
        }

        class_map
    }

    fn get_dim_set(&self) -> HashSet<ArrayDim> {
        let mut dim_set: HashSet<ArrayDim> = HashSet::new();
        for (_, expr) in self.expr_map.iter() {
            for (indexing_id, transform) in expr.get_indexing_sites() {
                for i in 0..transform.dims.len() {
                    dim_set.insert((indexing_id.clone(), i));
                }
            }
        }

        dim_set
    }

    /// generate equality constraints about which dimensions should be considered the same
    fn compute_dim_equalities(
        &self,
        expr: &InlinedExpr,
    ) -> (Vec<(ArrayDim, ArrayDim)>, Vec<ArrayDim>) {
        match expr {
            InlinedExpr::ReduceNode(reduced_index, _, body) => {
                let (eq_body, mut body_dims) = self.compute_dim_equalities(body);
                assert!(body_dims.len() > 0);
                body_dims.remove(*reduced_index);
                (eq_body, body_dims)
            }

            InlinedExpr::Op(_, expr1, expr2) => {
                let (mut eqs1, dims1) = self.compute_dim_equalities(expr1);
                let (eqs2, dims2) = self.compute_dim_equalities(expr2);
                let (len1, len2) = (dims1.len(), dims2.len());

                eqs1.extend(eqs2);
                if len1 == 0 && len2 != 0 {
                    (eqs1, dims2)
                } else if len1 != 0 && len2 == 0 {
                    (eqs1, dims1)
                } else {
                    assert!(len1 == len2);
                    let result_dims = dims1.clone();
                    let new_eqs: Vec<(ArrayDim, ArrayDim)> =
                        dims1.into_iter().zip(dims2.into_iter()).collect();
                    eqs1.extend(new_eqs);
                    (eqs1, result_dims)
                }
            }

            InlinedExpr::Literal(_) => (vec![], vec![]),

            InlinedExpr::ExprRef(indexing_id, transform) => {
                let input_dims: Vec<ArrayDim> = (0..transform.dims.len())
                    .map(|i| (indexing_id.clone(), i))
                    .collect();

                (vec![], input_dims)
            }
        }
    }
}

impl Display for InlinedProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.input_map
            .iter()
            .try_for_each(|(array, shape)| write!(f, "input {}: {:?}\n", array, shape))?;

        self.expr_map
            .iter()
            .try_for_each(|(id, expr)| write!(f, "let {} = {}\n", id, expr))?;

        Ok(())
    }
}

#[derive(Clone, Debug)]
enum PathInfo {
    Index { index: IndexVar, extent: usize },
    Reduce { op: Operator },
}

struct PathContext {
    path: im::Vector<PathInfo>,
}

impl PathContext {
    fn new(path: im::Vector<PathInfo>) -> Self {
        PathContext { path }
    }

    fn get_index_extent(&self, index_var: &IndexVar) -> Option<usize> {
        self.path.iter().find_map(|path_info| match path_info {
            PathInfo::Index { index, extent } if index == index_var => Some(*extent),

            _ => None,
        })
    }
}

#[derive(Clone, Debug)]
struct IndexingVarData {
    index_var: IndexVar,
    stride: usize,
}

#[derive(Clone, Debug)]
struct IndexingData {
    var_data: im::Vector<IndexingVarData>,
    offset: isize,
}

// index elimination pass
pub struct IndexElimination {
    // map from source-level bindings to their shapes
    store: ArrayEnvironment,
}

impl IndexElimination {
    pub fn new() -> Self {
        IndexElimination {
            store: HashMap::new(),
        }
    }

    fn compute_shape_expr(&self, program: &ElaboratedProgram, expr: &SourceExpr) -> Shape {
        match expr {
            SourceExpr::For(_, extent, body) => {
                let body_extent = self.compute_shape_expr(program, body);
                im::vector![extent.clone()] + body_extent
            }

            SourceExpr::Reduce(_, body) => {
                let mut body_extent = self.compute_shape_expr(program, body);
                body_extent.split_off(1)
            }

            SourceExpr::ExprOp(_, expr1, expr2) => {
                let extent1 = self.compute_shape_expr(program, expr1);
                let extent2 = self.compute_shape_expr(program, expr2);
                assert!(extent1 == extent2);
                extent1
            }

            SourceExpr::Indexing(indexing_id, index_list) => {
                if program.is_input(indexing_id) {
                    let array = program.rename_map.get(indexing_id).unwrap();
                    if let Some(arr_extent) = self.store.get(array) {
                        arr_extent.clone().split_off(index_list.len())
                    } else {
                        panic!("no binding for {}", array)
                    }
                } else {
                    if let Some(arr_extent) = self.store.get(indexing_id) {
                        arr_extent.clone().split_off(index_list.len())
                    } else {
                        panic!("no binding for {}", indexing_id)
                    }
                }
            }

            SourceExpr::Literal(_) => im::Vector::new(),
        }
    }

    fn compute_shape_program(&mut self, program: &ElaboratedProgram) {
        for (array, (shape, _)) in program.input_map.iter() {
            if let Some(_) = self.store.insert(array.clone(), shape.clone()) {
                panic!("duplicate binding for {}", array)
            }
        }

        for (array, expr) in program.expr_map.iter() {
            let extent = self.compute_shape_expr(program, expr);
            if let Some(_) = self.store.insert(array.clone(), extent) {
                panic!("duplicate binding for {}", array)
            }
        }
    }

    fn process_index_expr(&self, index_expr: &IndexExpr) -> Result<IndexingData, String> {
        match index_expr {
            IndexExpr::Var(var) => Ok(IndexingData {
                var_data: im::vector![IndexingVarData {
                    index_var: var.clone(),
                    stride: 1
                }],
                offset: 0,
            }),

            IndexExpr::Literal(lit) => Ok(IndexingData {
                var_data: im::Vector::new(),
                offset: *lit,
            }),

            IndexExpr::Op(op, expr1, expr2) => {
                let data1 = self.process_index_expr(expr1)?;
                let data2 = self.process_index_expr(expr2)?;

                match op {
                    Operator::Add => Ok(IndexingData {
                        var_data: data1.var_data + data2.var_data,
                        offset: data1.offset + data2.offset,
                    }),

                    Operator::Sub => {
                        // TODO cannot handle negative strides for now
                        assert!(data2.var_data.is_empty());

                        Ok(IndexingData {
                            var_data: data1.var_data,
                            offset: data1.offset - data2.offset,
                        })
                    }

                    Operator::Mul => {
                        if data1.var_data.len() > 0 && data2.offset > 0 {
                            let mul_var_data1: im::Vector<IndexingVarData> = data1
                                .var_data
                                .into_iter()
                                .map(|var| IndexingVarData {
                                    index_var: var.index_var,
                                    stride: var.stride * (data2.offset as usize),
                                })
                                .collect();

                            Ok(IndexingData {
                                var_data: mul_var_data1,
                                offset: data1.offset * data2.offset,
                            })
                        } else if data2.var_data.len() > 0 && data1.offset > 0 {
                            let mul_var_data2: im::Vector<IndexingVarData> = data2
                                .var_data
                                .into_iter()
                                .map(|var| IndexingVarData {
                                    index_var: var.index_var,
                                    stride: var.stride * (data1.offset as usize),
                                })
                                .collect();

                            Ok(IndexingData {
                                var_data: mul_var_data2,
                                offset: data1.offset * data2.offset,
                            })
                        } else {
                            Err(format!(
                                "attempting to multiply index variables: {}",
                                index_expr
                            ))
                        }
                    }
                }
            }
        }
    }

    fn transform_expr(
        &mut self,

        // indexing sites to inline
        inline_set: &HashSet<IndexingId>,

        // determines how to group top-level indexing sites
        array_group_map: &HashMap<IndexingId, ArrayName>,

        program: &ElaboratedProgram,

        expr: &SourceExpr,
        output_transform: &ArrayTransform,
        path_ctx: &PathContext,
    ) -> Result<InlinedExpr, String> {
        match expr {
            SourceExpr::For(index, extent, body) => {
                let new_path_ctx = PathContext::new(
                    &path_ctx.path
                        + &im::Vector::unit(PathInfo::Index {
                            index: index.clone(),
                            extent: extent.upper() as usize,
                        }),
                );

                self.transform_expr(
                    inline_set,
                    array_group_map,
                    program,
                    body,
                    output_transform,
                    &new_path_ctx,
                )
            }

            SourceExpr::ExprOp(op, expr1, expr2) => {
                let res1 = self.transform_expr(
                    inline_set,
                    array_group_map,
                    program,
                    expr1,
                    output_transform,
                    path_ctx,
                )?;
                let res2 = self.transform_expr(
                    inline_set,
                    array_group_map,
                    program,
                    expr2,
                    output_transform,
                    path_ctx,
                )?;
                Ok(InlinedExpr::Op(*op, Box::new(res1), Box::new(res2)))
            }

            SourceExpr::Literal(num) => Ok(InlinedExpr::Literal(*num)),

            SourceExpr::Reduce(op, body) => {
                let new_path_ctx = PathContext::new(
                    &path_ctx.path + &im::Vector::unit(PathInfo::Reduce { op: *op }),
                );
                let body_res = self.transform_expr(
                    inline_set,
                    array_group_map,
                    program,
                    body,
                    output_transform,
                    &new_path_ctx,
                )?;

                // index of reduced dim should always 0
                // this invariant is enforced by the Indexing case
                Ok(InlinedExpr::ReduceNode(0, *op, Box::new(body_res)))
            }

            // this should compute a new output shape for the indexed array
            SourceExpr::Indexing(indexing_id, index_list) => {
                let array_shape = if program.is_input(indexing_id) {
                    program.input_map[&program.rename_map[indexing_id]]
                        .0
                        .clone()
                } else {
                    self.store[indexing_id].clone()
                };

                let array_dims = array_shape.len();

                // process indexing sites
                // TODO: check if this supports constant indexing like A[1]?
                let mut index_to_output_dim_map: HashMap<IndexVar, (DimIndex, usize)> =
                    HashMap::new();
                let mut array_offset_map = BaseOffsetMap::new(array_shape.len());
                for (index_dim, index_expr) in index_list.iter().enumerate() {
                    let indexing_data = self.process_index_expr(index_expr)?;
                    array_offset_map.set(index_dim, indexing_data.offset);
                    for var_data in indexing_data.var_data.into_iter() {
                        if !index_to_output_dim_map.contains_key(&var_data.index_var) {
                            let stride = var_data.stride;
                            if stride > 0 {
                                index_to_output_dim_map
                                    .insert(var_data.index_var, (index_dim, stride));
                            }
                        } else {
                            return Err(format!(
                                "Index variable {} used in multiple indexing sites",
                                &var_data.index_var
                            ));
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
                        PathInfo::Index {
                            index: index_var,
                            extent,
                        } => {
                            let dim = match index_to_output_dim_map.get(index_var) {
                                Some((dim_index, stride)) => DimContent::FilledDim {
                                    dim: *dim_index,
                                    extent: *extent,
                                    stride: *stride,
                                },

                                // index var wasn't used;
                                // then this output dim is empty
                                None => DimContent::EmptyDim { extent: *extent },
                            };

                            if num_reductions > 0 {
                                // dim is reduced
                                reduced_dims.push(dim);
                                num_reductions -= 1;
                            } else {
                                // dim is in output
                                output_dims.push(dim);
                            }
                        }

                        PathInfo::Reduce { op: _ } => {
                            num_reductions += 1;
                        }
                    }
                }

                // process reduced dims that are not introduced by a for node
                // but rather is in array tail that is not indexed
                let mut cur_retained_dim = index_list.len();
                while num_reductions > 0 {
                    reduced_dims.push(DimContent::FilledDim {
                        dim: cur_retained_dim,
                        extent: array_shape[cur_retained_dim].upper() as usize,
                        stride: 1,
                    });
                    num_reductions -= 1;
                    cur_retained_dim += 1;
                }

                // dims in array tail that are not reduced are added to the output
                while cur_retained_dim < array_dims {
                    output_dims.push(DimContent::FilledDim {
                        dim: cur_retained_dim,
                        extent: array_shape[cur_retained_dim].upper() as usize,
                        stride: 1,
                    });
                    cur_retained_dim += 1;
                }

                // build a shape for the indexed array given the required output shape
                let mut indexed_transform = ArrayTransform {
                    array: array_group_map.get(indexing_id).unwrap().clone(),
                    offset_map: array_offset_map,
                    dims: im::Vector::new(),
                };

                // first, add reduced dims in LIFO order
                while let Some(dim) = reduced_dims.pop() {
                    indexed_transform.dims.push_back(dim);
                }

                // next, add the output dimensions
                for dim in output_transform.dims.iter() {
                    let indexed_output_dim = match *dim {
                        DimContent::FilledDim {
                            dim,
                            extent: output_extent,
                            stride: output_stride,
                        } => {
                            match output_dims[dim] {
                                // TODO: does this make sense??
                                // what if the inner extent is smaller than the outer extent?
                                DimContent::FilledDim {
                                    dim: indexed_dim,
                                    extent: indexed_extent,
                                    stride: indexed_stride,
                                } => {
                                    let new_offset = indexed_transform.offset_map.get(dim)
                                        + (output_transform.offset_map.get(dim)
                                            * (indexed_stride as isize));

                                    indexed_transform.offset_map.set(indexed_dim, new_offset);

                                    DimContent::FilledDim {
                                        dim: indexed_dim,
                                        extent: output_extent,
                                        stride: output_stride * indexed_stride,
                                    }
                                }

                                // TODO: does this make sense??
                                // what if the inner extent is smaller than the outer extent?
                                DimContent::EmptyDim { extent: _ } => DimContent::EmptyDim {
                                    extent: output_extent,
                                },
                            }
                        }

                        DimContent::EmptyDim { extent: _ } => dim.clone(),
                    };

                    indexed_transform.dims.push_back(indexed_output_dim);
                }

                if inline_set.contains(indexing_id) {
                    // inline
                    let indexed_expr = &program.expr_map[indexing_id];

                    let transformed_indexed_expr = self.transform_expr(
                        inline_set,
                        array_group_map,
                        program,
                        indexed_expr,
                        &indexed_transform,
                        &PathContext {
                            path: im::Vector::new(),
                        },
                    )?;

                    Ok(transformed_indexed_expr)
                } else {
                    // don't inline
                    Ok(InlinedExpr::ExprRef(
                        indexing_id.clone(),
                        indexed_transform,
                    ))
                }
            }
        }
    }

    pub fn run(
        &mut self,
        inline_set: &HashSet<IndexingId>,
        array_group_map: &HashMap<IndexingId, ArrayName>,
        program: &ElaboratedProgram,
    ) -> Result<InlinedProgram, String> {
        self.compute_shape_program(&program);

        let mut expr_map: IndexMap<ArrayName, InlinedExpr> = IndexMap::new();
        let mut worklist: Vec<String> = vec![String::from(OUTPUT_EXPR_NAME)];

        while worklist.len() > 0 {
            let indexing_id = worklist.pop().unwrap();
            let expr = program.expr_map.get(&indexing_id).unwrap();
            let shape = self.store.get(&indexing_id).unwrap();
            let transform = ArrayTransform::from_shape(String::from(&indexing_id), shape);

            let transformed_expr = self.transform_expr(
                inline_set,
                array_group_map,
                &program,
                expr,
                &transform,
                &PathContext::new(im::Vector::new()),
            )?;

            let indexed_arrays: HashSet<ArrayName> =
                transformed_expr
                .get_indexing_sites()
                .into_iter()
                .filter_map(|(indexing_id, _)| {
                    if !program.is_input(&indexing_id) {
                        Some(indexing_id)

                    } else {
                        None
                    }
                })
                .collect();

            let array_name = array_group_map[&indexing_id].clone();
            worklist.extend(indexed_arrays);
            expr_map.insert(array_name, transformed_expr);
        }

        // the expressions were processed backwards from the output expr,
        // so reverse the insertion order to get the program order
        expr_map.reverse();

        let transformed_program = InlinedProgram {
            input_map: program.input_map.clone(),
            expr_map,
        };

        Ok(transformed_program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{elaborated::Elaborator, parser::ProgramParser, source::SourceProgram};

    fn test_index_elim(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let elaborated: ElaboratedProgram = Elaborator::new().run(program);
        println!("{}", elaborated);
        for inline_set in elaborated.inline_sets() {
            println!("inline set: {:?}", inline_set)
        }

        let inline_set = elaborated.default_inline_set();
        let array_group_map = elaborated.default_array_group_map();

        let res =
            IndexElimination::new().run(&inline_set, &array_group_map, &elaborated);

        assert!(res.is_ok());

        let prog = res.unwrap();
        println!("{}", prog);
    }

    #[test]
    fn test_imgblur() {
        test_index_elim(
            "input img: [16,16] from client
            for x: 16 {
                for y: 16 {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }",
        );
    }

    #[test]
    fn test_imgblur2() {
        test_index_elim(
            "input img: [16,16] from client
            let res = 
                for x: 16 {
                    for y: 16 {
                        img[x-1][y-1] + img[x+1][y+1]
                    }
                }
            in
            for x: 16 {
                for y: 16 {
                    res[x-2][y-2] + res[x+2][y+2]
                }
            }
            ",
        );
    }

    #[test]
    fn test_convolve() {
        test_index_elim(
            "input img: [16,16] from client
            let conv1 = 
                for x: 15 {
                    for y: 15 {
                        img[x][y] + img[x+1][y+1]
                    }
                }
            in
            for x: 14 {
                for y: 14 {
                    conv1[x][y] + conv1[x+1][y+1]
                }
            }
            ",
        );
    }

    #[test]
    fn test_matmatmul() {
        test_index_elim(
            "input A: [4,4] from client
            input B: [4,4] from client
            for i: 4 {
                for j: 4 {
                    sum(for k: 4 { A[i][k] * B[k][j] })
                }
            }",
        );
    }

    #[test]
    fn test_matmatmul2() {
        test_index_elim(
            "input A1: [4,4] from client
            input A2: [4,4] from client
            input B: [4,4] from client
            let res =
                for i: 4 {
                    for j: 4 {
                        sum(for k: 4 { A1[i][k] * B[k][j] })
                    }
                }
            in
            for i: 4 {
                for j: 4 {
                    sum(for k: 4 { A2[i][k] * res[k][j] })
                }
            }
            ",
        );
    }

    #[test]
    fn test_dotprod_pointless() {
        test_index_elim(
            "
            input A: [3] from client
            input B: [3] from client
            sum(A * B)
            ",
        );
    }

    #[test]
    fn test_matvecmul() {
        test_index_elim(
            "
            input M: [2,2] from client
            input v: [2] from client
            for i: 2 {
                sum(M[i] * v)
            }
            ",
        );
    }
}
