use std::{collections::{HashMap, HashSet}, fmt::Display, ops::Range, cmp::{min, max}};
use bimap::BiHashMap;
use gcollections::ops::Bounded;
use itertools::{Itertools, MultiProduct};

use crate::{
    circ2::{
        IndexCoordinateMap, CiphertextObject, ParamCircuitExpr,
        VectorRegistry, ParamCircuitProgram
    },
    lang::{
        Operator, BaseArrayTransform,
        index_elim2::{TransformedProgram, TransformedExpr}, ExprRefId, Shape, DimIndex, DimContent, ArrayTransform, ArrayName, OffsetMap
    },
    scheduling::{
        ArraySchedule, ExprSchedule, DimName, OffsetExpr, ParamArrayTransform,
        Schedule, ScheduleDim
    }
};

use super::IndexCoord;

// like DimContent, but with padding information
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum VectorDimContent {
    FilledDim {
        dim: DimIndex, extent: usize, stride: isize,
        pad_left: usize, pad_right: usize
    },

    EmptyDim { extent: usize }
}

impl Display for VectorDimContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorDimContent::FilledDim {
                dim, extent, stride, pad_left, pad_right
            } => {
                if *pad_left == 0 && *pad_right == 0 {
                    write!(f, "{{{}:{}::{}}}", dim, extent, stride)

                } else {
                    write!(f, "{{{}:{}::{}[pad=({},{})]}}", dim, extent, stride, pad_left, pad_right)
                }
            },

            VectorDimContent::EmptyDim { extent } => {
                write!(f, "{{{}}}", extent)
            },
        }
    }
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct VectorInfo {
    array: ArrayName,
    offset_map: OffsetMap<usize>,
    dims: im::Vector<VectorDimContent>,
}

impl Display for VectorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}]<{}>",
            self.array,
            self.offset_map,
            self.dims.iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl VectorInfo {
    pub fn clip(transform: &BaseArrayTransform, array_shape: &Shape) -> Self {
        let mut materialized_dims: im::Vector<VectorDimContent> = im::Vector::new();
        for dim in transform.dims.iter() {
            let materialized_dim = 
                match *dim {
                    // clip dimension to (0, array dimension's extent)
                    DimContent::FilledDim { dim, extent, stride } => {
                        let dim_offset = *transform.offset_map.get(dim);
                        let array_extent = array_shape[dim].upper() as isize;
                        let iextent = extent as isize;

                        // indexing less than 0; clip
                        let pad_left =
                            if dim_offset < 0 {
                                -dim_offset
                            } else {
                                0
                            };
                        
                        // indexing beyond array_extent; clip
                        let mut pad_right = 0;
                        while dim_offset + (stride*(iextent-1-pad_right)) >= array_extent {
                            pad_right += 1;
                        }
                        
                        let new_extent = (iextent - pad_left - pad_right) as usize;

                        VectorDimContent::FilledDim {
                            dim,
                            extent: new_extent,
                            stride,
                            pad_left: pad_left as usize,
                            pad_right: pad_right as usize
                        }
                    },

                    DimContent::EmptyDim { extent } => {
                        VectorDimContent::EmptyDim { extent }
                    }
                };

            materialized_dims.push_back(materialized_dim);
        }

        let clipped_offset_map =
            transform.offset_map.map(|o| max(*o, 0) as usize);

        VectorInfo {
            array: transform.array.clone(),
            offset_map: clipped_offset_map,
            dims: materialized_dims,
        }
    }

    // derive other from self
    pub fn derive(&self, other: &VectorInfo) -> Option<isize> {
        if self.dims.len() != other.dims.len() {
            return None
        }

        if self == other {
            Some(0)

        } else if self.dims.len() != 0 {
            let mut seen_dims: HashSet<DimIndex> = HashSet::new();

            // check derivability conditions
            let dims_derivable = 
                self.dims.iter()
                .zip(other.dims.iter())
                .all(|(dim1, dim2)| {
                    match (*dim1, *dim2) {
                        (VectorDimContent::FilledDim {
                            dim: dim1, extent: extent1, stride: stride1,
                            pad_left: pad_left1, pad_right: pad_right1
                        },
                        VectorDimContent::FilledDim {
                            dim: dim2, extent: extent2, stride: stride2,
                            pad_left: pad_left2, pad_right: pad_right2
                        }) => {
                            // dimensions point to the same indexed dimension (duh)
                            let same_dim = dim1 == dim2;

                            // multiple dims cannot stride the same indexed dim
                            let dim_unseen = !seen_dims.contains(&dim1);

                            // dimensions have the same stride
                            let same_stride = stride1 == stride2;

                            // the offsets of self and other ensure that they
                            // have the same elements
                            let offset1 = self.offset_map[dim1];
                            let offset2 = other.offset_map[dim2];
                            let offset_equiv =
                                offset1 % (stride1 as usize) == offset2 % (stride2 as usize);

                            // the dimensions have the same size
                            let same_size =
                                pad_left1 + extent1 + pad_right1 == pad_left2 + extent2 + pad_right2;

                            let iextent1: isize = extent1 as isize;
                            let iextent2: isize = extent2 as isize;

                            // all of the elements of other's dim is in self's dim
                            let in_extent =
                                offset2 >= offset1 &&
                                (offset1 as isize) + (stride1*iextent1) >=
                                (offset2 as isize) + (stride2*iextent2);

                            seen_dims.insert(dim1);
                            same_dim && dim_unseen && same_stride && offset_equiv && same_size && in_extent
                        },
                        
                        (VectorDimContent::EmptyDim { extent: extent1 },
                        VectorDimContent::EmptyDim { extent: extent2 }) => {
                            extent1 == extent2
                        }

                        (VectorDimContent::FilledDim { dim, extent, stride, pad_left, pad_right },
                            VectorDimContent::EmptyDim { extent: _ }) |
                        (VectorDimContent::EmptyDim { extent },
                            VectorDimContent::FilledDim { dim, extent: _, stride, pad_left, pad_right })
                        => false,
                    }
                });

            if dims_derivable && self.array == other.array {
                let mut block_size: usize = 1;
                let mut rotate_steps = 0;

                self.dims.iter()
                .zip(other.dims.iter()).rev()
                .for_each(|(dim1, dim2)| {
                    match (*dim1, *dim2) {
                        (VectorDimContent::FilledDim {
                            dim: dim1, extent: extent1, stride: stride1,
                            pad_left: pad_left1, pad_right: pad_right1
                        },
                        VectorDimContent::FilledDim {
                            dim: dim2, extent: extent2, stride: stride2,
                            pad_left: pad_left2, pad_right: pad_right2
                        }) => {
                            let offset1 = self.offset_map[dim1];
                            let offset2 = other.offset_map[dim2];

                            // two derivation cases:
                            let steps = 
                                // (1) other offset is greater than self offset,
                                //     rotate self to the left to compensate
                                if offset1 < offset2 {
                                    -(((offset2 - offset1) / (stride1 as usize)) as isize)
                                    
                                // (2) other pad_left is greater than self pad_left,
                                //     rotate to the right to compensate
                                } else if pad_left1 < pad_left2 {
                                    (pad_left2 - pad_left1) as isize

                                } else if offset1 == offset2 && pad_left1 == pad_left2 {
                                    0

                                } else {
                                    panic!("this path should be unreachable")
                                };

                            rotate_steps += steps * (block_size as isize);
                            block_size *= pad_left1 + extent1 + pad_right1;
                        },
                        
                        (VectorDimContent::EmptyDim { extent: extent1 },
                        VectorDimContent::EmptyDim { extent: extent2 }) => {
                            block_size += extent1;
                        }

                        (VectorDimContent::FilledDim { dim, extent, stride, pad_left, pad_right },
                            VectorDimContent::EmptyDim { extent: _ }) |
                        (VectorDimContent::EmptyDim { extent },
                            VectorDimContent::FilledDim { dim, extent: _, stride, pad_left, pad_right })
                        => panic!("this should be unreachable"),
                    }
                });

                // TODO add mask
                Some(rotate_steps)

            } else {
                None
            }

        } else {
            if self.array == other.array && self.offset_map == other.offset_map {
                Some(0)

            } else {
                None
            }
        }
    }
}

pub trait ArrayMaterializer {
    fn can_materialize(&self, param_transform: &ParamArrayTransform, array_shape: &Shape) -> bool;
    fn materialize(&mut self, param_transform: &ParamArrayTransform, array_shape: &Shape, registry: &mut VectorRegistry) -> ParamCircuitExpr;
}

/// materializes a schedule for an index-free program.
pub struct Materializer {
    array_materializers: Vec<Box<dyn ArrayMaterializer>>,
    registry: VectorRegistry,
}

impl Materializer {
    pub fn new(array_materializers: Vec<Box<dyn ArrayMaterializer>>) -> Self {
        Materializer { array_materializers, registry: VectorRegistry::new() }
    }

    /// packages the materialized expr with the vector registry
    pub fn materialize(
        mut self,
        program: &TransformedProgram,
        schedule: &Schedule
    ) -> Result<ParamCircuitProgram, String> {
        let (schedule, expr) = self.materialize_expr(&program, &program.expr, schedule)?;
        Ok(
            ParamCircuitProgram {
                schedule, expr, registry: self.registry
            }
        )
    }

    // TODO: refactor logic of computing output schedules into a separate struct
    // since this share the same logic as Schedule::compute_output_schedule
    fn materialize_expr(
        &mut self,
        program: &TransformedProgram,
        expr: &TransformedExpr,
        schedule: &Schedule
    ) -> Result<(ExprSchedule, ParamCircuitExpr), String> {
        match expr {
            TransformedExpr::Literal(lit) => {
                Ok((ExprSchedule::Any, ParamCircuitExpr::Literal(*lit)))
            },

            TransformedExpr::Op(op, expr1, expr2) => {
                let (sched1, mat1) = self.materialize_expr(program, expr1, schedule)?;
                let (sched2, mat2) = self.materialize_expr(program, expr2, schedule)?;

                let expr = 
                    ParamCircuitExpr::Op(op.clone(), Box::new(mat1), Box::new(mat2));

                let schedule = 
                    match (sched1, sched2) {
                        (ExprSchedule::Any, ExprSchedule::Any) =>
                            ExprSchedule::Any,

                        (ExprSchedule::Any, ExprSchedule::Specific(sched2)) => 
                            ExprSchedule::Specific(sched2),

                        (ExprSchedule::Specific(sched1), ExprSchedule::Any) =>
                            ExprSchedule::Specific(sched1),

                        (ExprSchedule::Specific(sched1), ExprSchedule::Specific(sched2)) => {
                            assert!(sched1 == sched2);
                            ExprSchedule::Specific(sched1)
                        }
                    };

                Ok((schedule, expr))
            },

            // TODO support reduction in vectorized dims
            TransformedExpr::ReduceNode(reduced_index, op, body) => {
                let (body_sched, mat_body) =
                    self.materialize_expr(program, body, schedule)?;

                match body_sched {
                    ExprSchedule::Any => Err("Cannot reduce a literal expression".to_string()),

                    ExprSchedule::Specific(body_sched_spec) => {
                        let mut new_exploded_dims: im::Vector<ScheduleDim> = im::Vector::new();
                        let mut reduced_index_vars: HashSet<(DimName,usize)> = HashSet::new();
                        for mut dim in body_sched_spec.exploded_dims {
                            if dim.index == *reduced_index { // dim is reduced, remove it
                                reduced_index_vars.insert((dim.name, dim.extent));

                            } else if dim.index > *reduced_index { // decrease dim index
                                dim.index -= 1;
                                new_exploded_dims.push_back(dim);

                            } else {
                                new_exploded_dims.push_back(dim);
                            }
                        }

                        let schedule = 
                            ExprSchedule::Specific(
                                ArraySchedule {
                                    exploded_dims: new_exploded_dims,
                                    vectorized_dims: body_sched_spec.vectorized_dims,
                                }
                            );

                        let expr = 
                            ParamCircuitExpr::ReduceVectors(
                                reduced_index_vars,
                                op.clone(),
                                Box::new(mat_body)
                            );

                        Ok((schedule, expr))
                    }
                }
            },

            // this is assumed to be a transformation of an input array
            TransformedExpr::ExprRef(ref_id) => {
                let transform = &program.inputs[ref_id];
                let transform_schedule = &schedule.schedule_map[ref_id];
                let param_transform = transform_schedule.apply_schedule(transform);
                let array_shape = &program.array_shapes[&transform.array];

                for amat in self.array_materializers.iter_mut() {
                    if amat.can_materialize(&param_transform, array_shape) {
                        let expr = amat.materialize(&param_transform, array_shape, &mut self.registry);
                        let schedule = ExprSchedule::Specific(transform_schedule.clone());
                        return Ok((schedule, expr))
                    }
                }

                Err(format!("No array materializer can process {}", param_transform))
            },
        }
    }
}

// array materializer that doesn't attempt to derive vectors
pub struct DummyArrayMaterializer {}

impl ArrayMaterializer for DummyArrayMaterializer {
    // the dummy materializer can materialize any transform
    fn can_materialize(&self, _param_transform: &ParamArrayTransform, _array_shape: &Shape) -> bool {
        true
    }

    fn materialize(&mut self, param_transform: &ParamArrayTransform, array_shape: &Shape, registry: &mut VectorRegistry) -> ParamCircuitExpr {
        let ct_var = registry.fresh_ciphertext_var();
        let coord_map: IndexCoordinateMap<CiphertextObject> =
            IndexCoordinateMap::new(param_transform.exploded_dims.iter());
        registry.set_ciphertext_coord_map(ct_var.clone(), coord_map);
        
        ParamCircuitExpr::CiphertextVar(ct_var)
    }
}

// array materialize that will derive vectors through rotation and masking
type VectorId = usize;

pub struct DefaultArrayMaterializer {
    cur_vector_id: VectorId,
    vector_map: BiHashMap<VectorId, VectorInfo>,
    parent_map: HashMap<VectorId, VectorId>,
}

impl DefaultArrayMaterializer {
    pub fn new() -> Self {
        DefaultArrayMaterializer {
            cur_vector_id: 1,
            vector_map: BiHashMap::new(),
            parent_map: HashMap::new(),
        }
    }

    fn register_vector(&mut self, vector: VectorInfo) -> VectorId {
        if let Some(id) = self.vector_map.get_by_right(&vector) {
            *id

        } else {
            let id = self.cur_vector_id;
            self.cur_vector_id += 1;
            self.vector_map.insert(id, vector);
            id
        }
    }

    fn find_immediate_parent(&self, id: VectorId) -> VectorId {
        let vector = self.vector_map.get_by_left(&id).unwrap();
        for (id2, vector2) in self.vector_map.iter() {
            if id != *id2 {
                if vector2.derive(vector).is_some() {
                    return *id2
                }
            }
        }

        id
    }

    fn find_transitive_parent(&self, id: VectorId) -> VectorId {
        let parent_id = self.parent_map[&id];
        if parent_id != id {
            self.find_transitive_parent(parent_id)

        } else {
            parent_id
        }
    }

    // assume that the rotation steps have a linear relationship to the index vars,
    // then probe certain coordinates to compute an offset expr
    fn compute_linear_offset(
        &self,
        step_map: &HashMap<IndexCoord, isize>,
        index_vars: &Vec<DimName>
    ) -> Option<OffsetExpr> {
        // probe at (0,...,0) to get the base offset
        let base_coord: im::Vector<usize> = im::Vector::from(vec![0; index_vars.len()]);
        let base_offset: isize = step_map[&base_coord];

        // probe at (0,..,1,..,0) to get the coefficient for the ith index var
        let mut coefficients: Vec<isize> = Vec::new();
        for i in 0..index_vars.len() {
            let mut index_coord = base_coord.clone();
            index_coord[i] = 1;
            coefficients.push(step_map[&index_coord] - base_offset);
        }

        // build offset expr from base offset and coefficients
        let offset_expr =
            coefficients.iter()
            .zip(index_vars)
            .fold(OffsetExpr::Literal(base_offset), |acc, (coeff, index_var)| {
                if *coeff != 0 {
                    OffsetExpr::Add(
                        Box::new(acc),
                        Box::new(
                            OffsetExpr::Mul(
                                Box::new(OffsetExpr::Literal(*coeff)),
                                Box::new(OffsetExpr::ExplodedIndexVar(index_var.clone()))
                            )
                        )
                    )
                } else {
                    acc
                }
            });

        // validate computed offset expr
        for (coord, value) in step_map.iter() {
            let index_map: HashMap<DimName, usize> =
                index_vars.clone().into_iter().zip(coord.clone()).collect();

            let predicted_value = offset_expr.eval(&index_map);
            if *value != predicted_value {
                return None
            }
        }

        Some(offset_expr)
    }
}

impl ArrayMaterializer for DefaultArrayMaterializer {
    // the default materializer always applies
    fn can_materialize(&self, _param_transform: &ParamArrayTransform, _array_shape: &Shape) -> bool {
        true
    }

    fn materialize(&mut self, param_transform: &ParamArrayTransform, array_shape: &Shape, registry: &mut VectorRegistry) -> ParamCircuitExpr {
        let ct_var = registry.fresh_ciphertext_var();
        let mut coord_map: IndexCoordinateMap<CiphertextObject> =
            IndexCoordinateMap::new(param_transform.exploded_dims.iter());

        let mut vector_id_map: HashMap<IndexCoord, VectorId> = HashMap::new();
        let index_vars = coord_map.index_vars();

        println!("offset expr: {:?}", param_transform.transform.offset_map);

        // register vectors
        for coord in coord_map.coord_iter() {
            let index_map: HashMap<DimName, usize> =
                index_vars.clone().into_iter().zip(coord.clone()).collect();

            let base_offset_map =
                param_transform.transform.offset_map
                .map(|offset| offset.eval(&index_map));

            let base_transform =
                BaseArrayTransform {
                    array: param_transform.transform.array.clone(),
                    offset_map: base_offset_map,
                    dims: param_transform.transform.dims.clone(),
                };

            let vector = VectorInfo::clip(&base_transform, array_shape);
            println!("coord: {:?} base transform: {} clipped vector: {}", coord, base_transform, vector);
            let vector_id = self.register_vector(vector);
            vector_id_map.insert(coord, vector_id);
        }

        // find immediate parent for each vector
        for (vector_id, _) in self.vector_map.iter() {
            let parent_id = self.find_immediate_parent(*vector_id);
            self.parent_map.insert(*vector_id, parent_id);
        }

        // find transitive parents
        let mut step_map: HashMap<IndexCoord, isize> = HashMap::new();
        for coord in coord_map.coord_iter() {
            let vector_id = vector_id_map.get(&coord).unwrap();
            let parent_id = self.find_transitive_parent(*vector_id);

            if *vector_id != parent_id {
                let vector = self.vector_map.get_by_left(vector_id).unwrap();
                let parent = self.vector_map.get_by_left(&parent_id).unwrap();

                let steps = parent.derive(vector).unwrap();
                println!("coord: {:?} vector: {} parent: {} steps: {}", coord, vector, parent, steps);
                step_map.insert(coord.clone(), steps);
                coord_map.set(coord, CiphertextObject::Vector(parent.clone()));

            } else {
                let vector = self.vector_map.get_by_left(vector_id).unwrap();
                println!("coord: {:?} vector: {} self parent", coord, vector);
                step_map.insert(coord.clone(), 0);
                coord_map.set(coord, CiphertextObject::Vector(vector.clone()));
            }
        }

        if let Some(offset_expr) = self.compute_linear_offset(&step_map, &index_vars) {
            for coord in coord_map.coord_iter() {
                println!("coord {:?} {}", coord, coord_map.get(&coord));
            }

            registry.set_ciphertext_coord_map(ct_var.clone(), coord_map);
            ParamCircuitExpr::Rotate(
                Box::new(offset_expr),
                Box::new(ParamCircuitExpr::CiphertextVar(ct_var))
            )

        } else {
            panic!("cannot process derivation with nonlinear offsets")
        }
    }
}


#[cfg(test)]
mod tests {
    use interval::{Interval, ops::Range};

    use crate::lang::{parser::ProgramParser, index_elim2::IndexElimination2, source::SourceProgram};
    use super::*;

    // generate an initial schedule for a program
    fn test_materializer(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let mut index_elim = IndexElimination2::new();
        let res = index_elim.run(&program);
        
        assert!(res.is_ok());

        let program = res.unwrap();
        let init_schedule = Schedule::gen_initial_schedule(&program);

        let materializer =
            Materializer::new(vec![
                Box::new(DefaultArrayMaterializer::new())
            ]);

        let res_mat = materializer.materialize(&program, &init_schedule);
        assert!(res_mat.is_ok());

        let param_circ = res_mat.unwrap();
        println!("{}", param_circ.schedule);
        println!("{}", param_circ.expr);
    }

    fn test_array_materializer(transform: ParamArrayTransform, shape: Shape) {
        let mut amat = DefaultArrayMaterializer::new();
        let mut registry = VectorRegistry::new();
        let circ = amat.materialize(&transform, &shape, &mut registry);
        println!("{}", circ);
    }

    #[test]
    fn test_imgblur() {
        test_materializer(
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
        test_materializer(
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
        test_materializer(
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
        test_materializer(
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
        test_materializer(
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
        test_materializer(
        "
            input A: [(0,3)]
            input B: [(0,3)]
            sum(A * B)
            "
        );
    }

    #[test]
    fn test_matvecmul() {
        test_materializer(
        "
            input M: [(0,1),(0,1)]
            input v: [(0,1)]
            for i: (0,1) {
                sum(M[i] * v)
            }
            "
        );
    }

    #[test]
    fn test_materialize_img_array() {
        let mut offset_map = OffsetMap::new(2);
        offset_map.set(0, OffsetExpr::ExplodedIndexVar("i".to_string()));
        offset_map.set(1, OffsetExpr::ExplodedIndexVar("j".to_string()));

        let transform =
            ParamArrayTransform {
                exploded_dims: im::vector![
                    ScheduleDim { index: 0, stride: 1, extent: 3, name: "i".to_string() },
                    ScheduleDim { index: 1, stride: 1, extent: 3, name: "j".to_string() },
                ],

                transform: ArrayTransform {
                    array: "img".to_string(),
                    offset_map,
                    dims: im::vector![
                        DimContent::FilledDim { dim: 0, extent: 16, stride: 1 },
                        DimContent::FilledDim { dim: 1, extent: 16, stride: 1 },
                    ]
                }
            };

        let shape: Shape = im::vector![Interval::new(0, 16), Interval::new(0, 16)];
        test_array_materializer(transform, shape);
    }
}