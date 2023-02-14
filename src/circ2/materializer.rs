use std::{collections::{HashMap, HashSet}, fmt::Display, ops::{Range, Index}, cmp::{min, max}};
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
        ArraySchedule, ExprSchedule, DimName, OffsetExpr, ScheduledArrayTransform,
        Schedule, ScheduleDim, ClientPreprocessing
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
    preprocessing: Option<ClientPreprocessing>,
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

        // TODO only clip offsets if they have dims in the vector;
        // otherwise, keep the offset out-of-bounds
        let clipped_offset_map =
            transform.offset_map.map(|o| max(*o, 0) as usize);

        VectorInfo {
            array: transform.array.clone(),
            preprocessing: None,
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

                        => unreachable!()
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
    fn can_materialize(
        &self,
        array_shape: &Shape,
        schedule: &ArraySchedule,
        base: &BaseArrayTransform,
        scheduled: &ScheduledArrayTransform,
    ) -> bool;

    fn materialize(
        &mut self,
        array_shape: &Shape,
        schedule: &ArraySchedule,
        base: &BaseArrayTransform,
        scheduled: &ScheduledArrayTransform,
        registry: &mut VectorRegistry
    ) -> ParamCircuitExpr;
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
                                    preprocessing: None,
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
                let schedule = &schedule.schedule_map[ref_id];
                let base = &program.inputs[ref_id];
                let scheduled = schedule.apply(base);
                let array_shape = &program.array_shapes[&base.array];

                for amat in self.array_materializers.iter_mut() {
                    if amat.can_materialize(array_shape, schedule, base, &scheduled) {
                        let expr = amat.materialize(array_shape, schedule, base, &scheduled, &mut self.registry);
                        let schedule = ExprSchedule::Specific(schedule.clone());
                        return Ok((schedule, expr))
                    }
                }

                Err(format!("No array materializer can process {}", scheduled))
            },
        }
    }
}

// array materializer that doesn't attempt to derive vectors
pub struct DummyArrayMaterializer {}

impl ArrayMaterializer for DummyArrayMaterializer {
    // the dummy materializer can materialize any transform
    fn can_materialize(
        &self,
        _array_shape: &Shape,
        _schedule: &ArraySchedule,
        _base: &BaseArrayTransform,
        _scheduled: &ScheduledArrayTransform,
    ) -> bool {
        true
    }

    fn materialize(
        &mut self,
        array_shape: &Shape,
        _schedule: &ArraySchedule,
        _base: &BaseArrayTransform,
        scheduled: &ScheduledArrayTransform,
        registry: &mut VectorRegistry
    ) -> ParamCircuitExpr {
        let ct_var = registry.fresh_ciphertext_var();
        let mut coord_map: IndexCoordinateMap<CiphertextObject> =
            IndexCoordinateMap::new(scheduled.exploded_dims.iter());

        let index_vars = coord_map.index_vars();

        // register vectors
        for coord in coord_map.coord_iter() {
            let index_map: HashMap<DimName, usize> =
                index_vars.clone().into_iter().zip(coord.clone()).collect();

            let base_offset_map =
                scheduled.transform.offset_map
                .map(|offset| offset.eval(&index_map));

            let base_transform =
                BaseArrayTransform {
                    array: scheduled.transform.array.clone(),
                    offset_map: base_offset_map,
                    dims: scheduled.transform.dims.clone(),
                };

            let vector = VectorInfo::clip(&base_transform, array_shape);
            coord_map.set(coord, CiphertextObject::Vector(vector));
        }
        
        registry.set_ciphertext_coord_map(ct_var.clone(), coord_map);
        ParamCircuitExpr::CiphertextVar(ct_var)
    }
}

// array materialize that will derive vectors through rotation and masking
type VectorId = usize;

pub struct VectorDeriver {
    cur_vector_id: VectorId,
    vector_map: BiHashMap<VectorId, VectorInfo>,
    parent_map: HashMap<VectorId, VectorId>,
}

impl VectorDeriver {
    pub fn new() -> Self {
        VectorDeriver {
            cur_vector_id: 1,
            vector_map: BiHashMap::new(),
            parent_map: HashMap::new(),
        }
    }

    pub fn register_vector(&mut self, vector: VectorInfo) -> VectorId {
        if let Some(id) = self.vector_map.get_by_right(&vector) {
            *id

        } else {
            let id = self.cur_vector_id;
            self.cur_vector_id += 1;
            self.vector_map.insert(id, vector);
            id
        }
    }

    pub fn find_immediate_parent(&self, id: VectorId) -> VectorId {
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

    // find immediate parent for each vector
    pub fn compute_immediate_parents(&mut self) {
        for (vector_id, _) in self.vector_map.iter() {
            let parent_id = self.find_immediate_parent(*vector_id);
            self.parent_map.insert(*vector_id, parent_id);
        }
    }

    pub fn find_transitive_parent(&self, id: VectorId) -> VectorId {
        let parent_id = self.parent_map[&id];
        if parent_id != id {
            self.find_transitive_parent(parent_id)

        } else {
            parent_id
        }
    }

    pub fn get_vector(&self, id: VectorId) -> &VectorInfo {
        self.vector_map.get_by_left(&id).unwrap()
    }

    pub fn register_and_derive_vectors(
        &mut self,
        array_shape: &Shape,
        coords: impl Iterator<Item=IndexCoord> + Clone,
        scheduled: &ScheduledArrayTransform,
        obj_map: &mut IndexCoordinateMap<CiphertextObject>,
        step_map: &mut IndexCoordinateMap<isize>,
    ) {
        let mut vector_id_map: HashMap<IndexCoord, VectorId> = HashMap::new();
        let index_vars = obj_map.index_vars();
        for coord in coords.clone() {
            let index_map: HashMap<DimName, usize> =
                index_vars.clone().into_iter().zip(coord.clone()).collect();

            let base_offset_map =
                scheduled.transform.offset_map
                .map(|offset| offset.eval(&index_map));

            let base_transform =
                BaseArrayTransform {
                    array: scheduled.transform.array.clone(),
                    offset_map: base_offset_map,
                    dims: scheduled.transform.dims.clone(),
                };

            let vector = VectorInfo::clip(&base_transform, array_shape);
            let vector_id = self.register_vector(vector);
            vector_id_map.insert(coord, vector_id);
        }

        self.compute_immediate_parents();

        // find transitive parents
        for coord in coords {
            let vector_id = *vector_id_map.get(&coord).unwrap();
            let parent_id = self.find_transitive_parent(vector_id);

            if vector_id != parent_id { // the vector is derived from some parent 
                let vector = self.get_vector(vector_id);
                let parent = self.get_vector(parent_id);
                let steps = parent.derive(vector).unwrap();

                step_map.set(coord.clone(), steps);
                obj_map.set(coord, CiphertextObject::Vector(parent.clone()));

            } else { // the vector is not derived
                let vector = self.get_vector(vector_id);
                step_map.set(coord.clone(), 0);
                obj_map.set(coord, CiphertextObject::Vector(vector.clone()));
            }
        }
    }

    // assume that the rotation steps have a linear relationship to the index vars,
    // then probe certain coordinates to compute an offset expr
    // this can compute linear offsets for a *subset* of defined coords;
    // hence this function takes in extra arguments
    // valid_coords and processed_index_vars
    pub fn compute_linear_offset(
        &self,
        step_map: &IndexCoordinateMap<isize>,
        valid_coords: impl Iterator<Item=IndexCoord> + Clone,
        processed_index_vars: Vec<DimName>,
    ) -> Option<OffsetExpr> {
        let index_vars = step_map.index_vars();

        // probe at (0,...,0) to get the base offset
        let base_coord: im::Vector<usize> = im::Vector::from(vec![0; index_vars.len()]);
        let base_offset: isize = *step_map.get(&base_coord);

        // probe at (0,..,1,..,0) to get the coefficient for the ith index var
        // only do this for processed_index_vars, not *all* index vars
        let mut coefficients: Vec<isize> = Vec::new();
        for i in 0..processed_index_vars.len() {
            let mut index_coord = base_coord.clone();
            index_coord[i] = 1;

            let step_offset = *step_map.get(&index_coord);
            coefficients.push(step_offset - base_offset);
        }

        // build offset expr from base offset and coefficients
        let offset_expr =
            coefficients.iter()
            .zip(index_vars.clone())
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
        for coord in valid_coords {
            let value = *step_map.get(&coord);
            let index_map: HashMap<DimName, usize> =
                index_vars.clone().into_iter().zip(coord.clone()).collect();

            let predicted_value = offset_expr.eval(&index_map);
            if value != predicted_value {
                return None
            }
        }

        // this expression is correct for all valid_coords; return it
        Some(offset_expr)
    }
}

pub struct DefaultArrayMaterializer {
    deriver: VectorDeriver,
}

impl DefaultArrayMaterializer {
    pub fn new() -> Self {
        DefaultArrayMaterializer {
            deriver: VectorDeriver::new(),
        }
    }
}

impl ArrayMaterializer for DefaultArrayMaterializer {
    // the default materializer always applies
    fn can_materialize(
        &self,
        _array_shape: &Shape,
        _schedule: &ArraySchedule,
        _base: &BaseArrayTransform,
        _scheduled: &ScheduledArrayTransform,
    ) -> bool {
        true
    }

    fn materialize(
        &mut self,
        array_shape: &Shape,
        _schedule: &ArraySchedule,
        _base: &BaseArrayTransform,
        scheduled: &ScheduledArrayTransform,
        registry: &mut VectorRegistry
    ) -> ParamCircuitExpr {
        let mut obj_map: IndexCoordinateMap<CiphertextObject> =
            IndexCoordinateMap::new(scheduled.exploded_dims.iter());
        let mut step_map: IndexCoordinateMap<isize> =
            IndexCoordinateMap::new(scheduled.exploded_dims.iter());
        let coords = obj_map.coord_iter();

        self.deriver.register_and_derive_vectors(
            array_shape,
            coords.clone(),
            scheduled,
            &mut obj_map,
            &mut step_map
        );

        let ct_var = registry.fresh_ciphertext_var();
        let offset_expr_opt =
            self.deriver.compute_linear_offset(
                &step_map,
                coords,
                obj_map.index_vars()
            );

        if let Some(offset_expr) = offset_expr_opt {
            registry.set_ciphertext_coord_map(ct_var.clone(), obj_map);
            ParamCircuitExpr::Rotate(
                Box::new(offset_expr),
                Box::new(ParamCircuitExpr::CiphertextVar(ct_var))
            )

        } else {
            // TODO fix this
            panic!("cannot process derivation with nonlinear offsets")
        }
    }
}

pub struct DiagonalArrayMaterializer { deriver: VectorDeriver }

impl DiagonalArrayMaterializer {
    pub fn new() -> Self {
        DiagonalArrayMaterializer { deriver: VectorDeriver::new() }
    }
}

impl ArrayMaterializer for DiagonalArrayMaterializer {
    fn can_materialize(
        &self,
        _array_shape: &Shape,
        schedule: &ArraySchedule,
        _base: &BaseArrayTransform,
        scheduled: &ScheduledArrayTransform,
    ) -> bool {
        if let Some(ClientPreprocessing::Permute(dim_i, dim_j)) = scheduled.preprocessing {
            // dim i must be exploded and dim j must be the outermost vectorized dim
            let i_exploded = 
                schedule.exploded_dims.iter().any(|edim| edim.index == dim_i);

            let j_outermost_vectorized =
                schedule.vectorized_dims.len() > 0 &&
                schedule.vectorized_dims.head().unwrap().index == dim_j;

            // dim i and j must have both have the same tiling that corresponds
            // to the permutation transform
            // TODO: for now, assume i and j are NOT tiled
            let tiling_i = schedule.get_tiling(dim_i);
            let tiling_j = schedule.get_tiling(dim_j);
            tiling_i == tiling_j && tiling_i.len() == 1 && i_exploded && j_outermost_vectorized

        } else {
            false
        }
    }

    fn materialize(
        &mut self,
        array_shape: &Shape,
        schedule: &ArraySchedule,
        base: &BaseArrayTransform,
        scheduled: &ScheduledArrayTransform,
        registry: &mut VectorRegistry
    ) -> ParamCircuitExpr {
        if let Some(ClientPreprocessing::Permute(dim_i, dim_j)) = scheduled.preprocessing {
            let mut obj_map: IndexCoordinateMap<CiphertextObject> =
                IndexCoordinateMap::new(scheduled.exploded_dims.iter());
            let mut step_map: IndexCoordinateMap<isize> =
                IndexCoordinateMap::new(scheduled.exploded_dims.iter());

            match (&base.dims[dim_i], &base.dims[dim_j]) {
                // if dim j is a filled dim, then the permutation must actually
                // be done by the client; record this fact and then materialize
                // the schedule normally
                (DimContent::FilledDim { dim: idim_i, extent: extent_i, stride: stride_i },
                DimContent::FilledDim { dim: idim_j, extent: extent_j, stride: stride_j }) => {
                    unimplemented!()
                },

                // if dim j is an empty dim, then we can apply the "diagonal"
                // trick from Halevi and Schoup for matrix-vector multiplication
                // to do this, follow these steps:
                // 1. switch innermost tiles of dim i and dim j
                //    (assuming all tiles of i is exploded and only innermost tile of j is vectorized)
                // 2. derive vectors assuming j = 0
                // 3. to fill in the rest of the vectors along dim j by rotating
                //    the vectors at dim j = 0
                (DimContent::FilledDim { dim: dim_i_dim, extent: extent_i, stride: stride_i },
                DimContent::EmptyDim { extent: extent_j }) => {
                    // switch innermost tiles of i and j in the schedule
                    let mut new_schedule = schedule.clone();

                    let inner_i_dim = 
                        new_schedule.exploded_dims.iter_mut()
                        .find(|dim| dim.index == dim_i && dim.stride == 1)
                        .unwrap();
                    inner_i_dim.index = dim_j;
                    let inner_i_dim_name = inner_i_dim.name.clone();
                    let inner_i_dim_extent = inner_i_dim.extent;

                    let inner_j_dim = &mut new_schedule.exploded_dims[0];
                    inner_j_dim.index = dim_i;
                    let inner_j_dim_name = inner_j_dim.name.clone();
                    let inner_j_dim_extent = inner_j_dim.extent;

                    // let inner_i_dim_name = inner_i_dim.name.clone();
                    // let inner_j_dim_name = inner_j_dim.name.clone();

                    // TODO should we replace names as well as indices?
                    // inner_i_dim.name = inner_j_dim_name;
                    // inner_j_dim.name = inner_i_dim_name;

                    let new_scheduled = new_schedule.apply(base);

                    let mut obj_map: IndexCoordinateMap<CiphertextObject> =
                        IndexCoordinateMap::new(new_scheduled.exploded_dims.iter());
                    let mut step_map: IndexCoordinateMap<isize> =
                        IndexCoordinateMap::new(new_scheduled.exploded_dims.iter());
                    let zero_inner_j_coords =
                        obj_map.coord_iter_subset(&inner_i_dim_name, 0..1);

                    self.deriver.register_and_derive_vectors(
                        array_shape,
                        zero_inner_j_coords.clone(),
                        &new_scheduled,
                        &mut obj_map,
                        &mut step_map);

                    let mut processed_index_vars = obj_map.index_vars();

                    // remember, inner i and inner j are swapped,
                    // so inner j now has the name of inner i!
                    let inner_j_name_index =
                        processed_index_vars.iter()
                        .position(|name| *name == inner_i_dim_name)
                        .unwrap();

                    processed_index_vars.remove(inner_j_name_index);

                    let ct_var = registry.fresh_ciphertext_var();
                    let offset_expr_opt =
                        self.deriver.compute_linear_offset(
                            &step_map,
                            zero_inner_j_coords,
                            processed_index_vars
                        );

                    if let Some(offset_expr) = offset_expr_opt {
                        let rest_inner_j_coords =
                            obj_map.coord_iter_subset(&inner_i_dim_name, 1..inner_i_dim_extent);

                        // given expr e is at coord where inner_j=0,
                        // expr rot(inner_j, e) is at coord where inner_j != 0
                        for coord in rest_inner_j_coords {
                            let mut ref_coord = coord.clone();
                            ref_coord[inner_j_name_index] = 0;

                            let ref_obj = obj_map.get(&ref_coord).clone();
                            obj_map.set(coord.clone(), ref_obj);
                        }

                        let new_offset_expr =
                            OffsetExpr::Add(
                                Box::new(offset_expr),
                                Box::new(OffsetExpr::ExplodedIndexVar(inner_i_dim_name.clone()))
                            );

                        registry.set_ciphertext_coord_map(ct_var.clone(), obj_map);
                        ParamCircuitExpr::Rotate(
                            Box::new(new_offset_expr),
                            Box::new(ParamCircuitExpr::CiphertextVar(ct_var))
                        )

                    } else {
                        // TODO fix this
                        panic!("cannot process derivation with nonlinear offsets")
                    }
                },

                // if dim i is empty, then the permutation is a no-op
                // materialize the schedule normally
                (DimContent::EmptyDim { extent }, _) => {
                    unimplemented!()
                }
            }

        } else {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use interval::{Interval, ops::Range};

    use crate::lang::{parser::ProgramParser, index_elim2::IndexElimination2, source::SourceProgram, BaseOffsetMap};
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

    fn test_array_materializer(
        shape: Shape,
        schedule: ArraySchedule,
        base: BaseArrayTransform, 
        scheduled: ScheduledArrayTransform
    ) {
        let mut amat = DefaultArrayMaterializer::new();
        let mut registry = VectorRegistry::new();
        let circ = amat.materialize(&shape, &schedule, &base, &scheduled, &mut registry);
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
            input M: [(0,2),(0,2)]
            input v: [(0,2)]
            for i: (0,2) {
                sum(M[i] * v)
            }
            "
        );
    }

    #[test]
    fn test_materialize_img_array() {
        let shape: Shape = im::vector![Interval::new(0, 16), Interval::new(0, 16)];

        let base =
            BaseArrayTransform {
                array: String::from("img"),
                offset_map: BaseOffsetMap::new(4),
                dims: im::vector![
                    DimContent::FilledDim { dim: 0, extent: 3, stride: 1 },
                    DimContent::FilledDim { dim: 1, extent: 3, stride: 1 },
                    DimContent::FilledDim { dim: 0, extent: 16, stride: 1 },
                    DimContent::FilledDim { dim: 1, extent: 16, stride: 1 },
                ]
            };

        let schedule = 
            ArraySchedule {
                preprocessing: None,
                exploded_dims: im::vector![
                    ScheduleDim { index: 0, stride: 1, extent: 3, name: String::from("i") },
                    ScheduleDim { index: 1, stride: 1, extent: 3, name: String::from("j") }
                ],
                vectorized_dims: im::vector![
                    ScheduleDim { index: 2, stride: 1, extent: 16, name: String::from("x") },
                    ScheduleDim { index: 3, stride: 1, extent: 16, name: String::from("y") }
                ]
            };

        let scheduled =
            schedule.apply(&base);

        test_array_materializer(shape, schedule, base, scheduled);
    }
}