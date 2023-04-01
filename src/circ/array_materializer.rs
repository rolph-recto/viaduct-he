use std::{ops::Range, collections::{HashMap, HashSet}};

use crate::{
    circ::{ *, cost::*, vector_deriver::* },
    scheduling::{IndexingSiteSchedule, ArrayPreprocessing},
    lang::{Shape, ArrayType, ArrayTransform, DimName, Extent, OffsetExpr, DimContent}
};

pub trait InputArrayMaterializer<'a> {
    fn create(&self) -> Box<dyn InputArrayMaterializer + 'a>;

    fn name(&self) -> &str;

    fn can_materialize(
        &self,
        array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    ) -> bool;

    fn register(
        &mut self,
        array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    );

    fn materialize(
        &mut self,
        array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId;

    fn estimate_cost(
        &mut self,
        array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    ) -> CostFeatures;
}

// array materializer that doesn't attempt to derive vectors
pub struct DummyArrayMaterializer;

impl<'a> InputArrayMaterializer<'a> for DummyArrayMaterializer {
    fn create(&self) -> Box<dyn InputArrayMaterializer + 'a> {
        Box::new(Self {})
    }

    fn name(&self) -> &str { "dummy array materializer" }

    // the dummy materializer can only materialize arrays w/o client preprocessing
    fn can_materialize(
        &self,
        _array_type: ArrayType,
        _array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        _transform: &ArrayTransform,
    ) -> bool {
        schedule.preprocessing.is_none()
    }

    fn register(
        &mut self,
        _array_type: ArrayType,
        _array_shape: &Shape,
        _schedule: &IndexingSiteSchedule,
        _transform: &ArrayTransform,
    ) {}

    fn materialize(
        &mut self,
        _array_type: ArrayType,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId {
        let ct_var = registry.fresh_ct_var();

        // register vectors
        let circuit_val = VectorInfo::get_input_vector_value(
            IndexCoordinateSystem::new(schedule.exploded_dims.iter()),
            shape,
            schedule,
            transform,
        )
        .map(|_, vector| CiphertextObject::InputVector(vector.clone()));

        registry.set_ct_var_value(ct_var.clone(), circuit_val);
        registry.register_circuit(ParamCircuitExpr::CiphertextVar(ct_var))
    }

    fn estimate_cost(
        &mut self,
        array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    ) -> CostFeatures {
        let coord_system = IndexCoordinateSystem::new(schedule.exploded_dims.iter());
        let dim_range: HashMap<DimName, Range<usize>> =
            coord_system.index_vars().into_iter()
            .map(|var| (var, 0..1)).collect();

        let mut vectors: HashSet<VectorInfo> = HashSet::new();
        for coord in coord_system.coord_iter_subset(dim_range) {
            let vector = 
                VectorInfo::get_input_vector_at_coord(
                    coord_system.coord_as_index_map(coord.clone()),
                    array_shape,
                    schedule,
                    transform,
                );

            vectors.insert(vector);
        }

        let mut cost = CostFeatures::default();
        match array_type {
            ArrayType::Ciphertext => {
                cost.input_ciphertexts += vectors.len();
            },

            ArrayType::Plaintext => {
                cost.input_plaintexts += vectors.len();
            }
        }

        cost
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

    fn estimate_ciphertext_cost(
        &mut self,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    ) -> CostFeatures {
        let mut obj_map: IndexCoordinateMap<CiphertextObject> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());

        if !obj_map.is_empty() {
            // there is an array of vectors
            let mut mask_map: IndexCoordinateMap<PlaintextObject> =
                IndexCoordinateMap::new(schedule.exploded_dims.iter());
            let mut step_map: IndexCoordinateMap<isize> =
                IndexCoordinateMap::new(schedule.exploded_dims.iter());

            let indices = obj_map.coord_system.index_vars();
            let zeros: IndexCoord = indices.iter().map(|_| 0 ).collect();

            let dim_probes: Vec<IndexCoord> =
                indices.iter().enumerate().map(|(i, _)| {
                    let mut probe = zeros.clone();
                    probe[i] = 1;
                    probe
                }).collect();

            let probes =
                vec![zeros.clone()].into_iter().chain(dim_probes.clone().into_iter());

            self.deriver.locally_register_and_derive_vectors(
                array_shape,
                schedule,
                transform,
                schedule.preprocessing,
                probes.clone(),
                &mut obj_map,
                &mut mask_map,
                &mut step_map,
            );

            let linear_offset_opt = 
                VectorDeriver::compute_linear_offset_coefficient(
                    &step_map,
                    probes.clone(),
                    indices.clone()
                );

            let num_rotates = 
                if let Some((_, coefficients)) = linear_offset_opt {
                    let nonzero_coeff_extents: Vec<(usize, isize)> =
                        obj_map.coord_system.extents().into_iter()
                        .zip(coefficients)
                        .filter(|(_, coeff)| *coeff != 0)
                        .collect();

                    if nonzero_coeff_extents.len() > 0 {
                        nonzero_coeff_extents.into_iter()
                        .fold(1, |acc, (extent, _)| acc * extent)

                    } else {
                        0
                    }

                } else {
                    // rotations too complicated; assume every coordinate
                    // needs a rotation
                    obj_map.coord_system.extents().into_iter().product()
                };

            let distinct_vectors: usize =
                obj_map.coord_system.extents().into_iter()
                .zip(dim_probes.clone())
                .filter(|(_, probe)| {
                    let probe_vec = obj_map.get(probe).unwrap();
                    let base_vec = obj_map.get(&zeros).unwrap();
                    probe_vec != base_vec
                }).fold(1, |acc, (extent, _)| acc * extent);

            let num_masks: usize =
                mask_map.coord_system.extents().into_iter()
                .zip(dim_probes)
                .filter(|(_, probe)| {
                    if let PlaintextObject::Mask(_) = mask_map.get(probe).unwrap() {
                        true
                    } else {
                        false
                    }
                }).fold(1, |acc, (extent, _)| acc * extent);

            let mut cost = CostFeatures::default();
            cost.input_ciphertexts += distinct_vectors;
            cost.ct_rotations += num_rotates;
            cost.ct_pt_mul += num_masks;
            cost.ct_pt_muldepth += if num_masks > 0 { 1 } else { 0 };
            cost
        
        } else {
            let mut cost = CostFeatures::default();
            cost.input_ciphertexts += 1;
            cost
        }
    }
}

impl<'a> InputArrayMaterializer<'a> for DefaultArrayMaterializer {
    fn create(&self) -> Box<dyn InputArrayMaterializer + 'a> {
        Box::new(DefaultArrayMaterializer::new())
    }

    fn name(&self) -> &str { "default array materializer" }

    /// the default materializer can only apply when there is no client preprocessing
    fn can_materialize(
        &self,
        _array_type: ArrayType,
        _array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        _transform: &ArrayTransform,
    ) -> bool {
        schedule.preprocessing.is_none()
    }
    
    fn register(
        &mut self,
        _array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    ) {
        self.deriver.register_vectors(
            array_shape,
            schedule,
            transform,
            &IndexCoordinateSystem::new(schedule.exploded_dims.iter()),
        );
    }

    fn materialize(
        &mut self,
        array_type: ArrayType,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId {
        match array_type {
            ArrayType::Ciphertext => self
                .deriver
                .derive_vectors_and_gen_circuit::<CiphertextObject>(
                    shape,
                    schedule,
                    transform,
                    registry,
                ),

            ArrayType::Plaintext => self
                .deriver
                .derive_vectors_and_gen_circuit::<PlaintextObject>(
                    shape,
                    schedule,
                    transform,
                    registry,
                ),
        }
    }

    fn estimate_cost(
        &mut self,
        array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    ) -> CostFeatures {
        match array_type {
            ArrayType::Ciphertext => {
                self.estimate_ciphertext_cost(array_shape, schedule, transform)
            },

            ArrayType::Plaintext => {
                let mut cost =
                    self.estimate_ciphertext_cost(array_shape, schedule, transform);
                cost.input_plaintexts = cost.input_ciphertexts;
                cost.input_ciphertexts = 0;
                cost.pt_pt_mul = cost.ct_pt_mul;
                cost.pt_rotations = cost.ct_rotations;
                cost
            }
        }
    }
}

// how to materialize a roll preprocessing
enum RollMaterializationType {
    // lay out the vectors as generalized diagonals
    Diagonal(usize, usize),

    // rotate the vector
    Rotate(usize, usize),

    // the roll preprocessing is a no-op
    None
}

pub struct RollArrayMaterializer {
    deriver: VectorDeriver,
}

impl RollArrayMaterializer {
    pub fn new() -> Self {
        RollArrayMaterializer {
            deriver: VectorDeriver::new(),
        }
    }

    // switch the innermost tiles of dim_i (exploded) and dim_j (vectorized)
    fn rotate_schedule(
        &self,
        schedule: &IndexingSiteSchedule,
        dim_i: usize,
        dim_j: usize
    ) -> (IndexingSiteSchedule, String, Extent) {
        let mut new_schedule = schedule.clone();

        // switch innermost tiles of i and j in the schedule
        let inner_i_dim = new_schedule
            .exploded_dims
            .iter_mut()
            .find(|dim| dim.index == dim_i && dim.stride == 1)
            .unwrap();
        let inner_i_dim_name = inner_i_dim.name.clone();
        let inner_i_dim_extent = inner_i_dim.extent;
        inner_i_dim.index = dim_j;

        let inner_j_dim = new_schedule.vectorized_dims.get_mut(0).unwrap();
        inner_j_dim.index = dim_i;

        (new_schedule, inner_i_dim_name, inner_i_dim_extent)
    }

    // if dim j is an empty dim, then we can apply the "diagonal"
    // trick from Halevi and Schoup for matrix-vector multiplication
    // to do this, follow these steps:
    // 1. switch innermost tiles of dim i and dim j
    //    (assuming all tiles of i is exploded and only innermost tile of j is vectorized)
    // 2. derive vectors assuming j = 0
    // 3. to fill in the rest of the vectors along dim j by rotating
    //    the vectors at dim j = 0
    fn diagonal_materialize<'a, T: CircuitObject + Clone>(
        &mut self,
        dim_i: usize,
        dim_j: usize,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId
    where
        CircuitObjectRegistry: CanRegisterObject<'a, T>,
        ParamCircuitExpr: CanCreateObjectVar<T>,
    {
        let (new_schedule, inner_i_dim_name, inner_i_dim_extent) =
            self.rotate_schedule(schedule, dim_i, dim_j);

        let mut obj_map: IndexCoordinateMap<T> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());
        let mut mask_map: IndexCoordinateMap<PlaintextObject> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());
        let mut step_map: IndexCoordinateMap<isize> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());

        let zero_inner_j_coords =
            obj_map.coord_iter_subset(
                HashMap::from([(inner_i_dim_name.clone(), 0..1)])
            );

        self.deriver.derive_vectors::<T>(
            shape,
            &new_schedule.with_preprocessing(None),
            transform,
            zero_inner_j_coords.clone(),
            &mut obj_map,
            &mut mask_map,
            &mut step_map,
        );

        let mut processed_index_vars = obj_map.index_vars();

        // remember, inner i and inner j are swapped,
        // so inner j now has the name of inner i!
        let inner_j_name_index = processed_index_vars
            .iter()
            .position(|name| *name == inner_i_dim_name)
            .unwrap();

        processed_index_vars.remove(inner_j_name_index);

        let obj_var = registry.fresh_obj_var();

        // given expr e is at coord where inner_j = 0,
        // expr rot(inner_j, e) is at coord where inner_j != 0
        let rest_inner_j_coords =
            obj_map.coord_iter_subset(
                HashMap::from([(inner_i_dim_name.clone(), 1..inner_i_dim_extent)])
            );

        for coord in rest_inner_j_coords {
            let mut ref_coord = coord.clone();
            ref_coord[inner_j_name_index] = 0;

            let ref_obj: T = obj_map.get(&ref_coord).unwrap().clone();
            let ref_step = *step_map.get(&ref_coord).unwrap();
            let inner_j_value = coord[inner_j_name_index];

            obj_map.set(coord.clone(), ref_obj);
            step_map.set(coord, ref_step + (inner_j_value as isize));
        }

        // attempt to compute offset expr
        let offset_expr_opt = if processed_index_vars.len() > 0 {
            VectorDeriver::compute_linear_offset(
                &step_map,
                Box::new(zero_inner_j_coords),
                processed_index_vars,
            )
        } else {
            let zero_j_coord = im::vector![0];
            let step = *step_map.get(&zero_j_coord).unwrap();
            Some(OffsetExpr::Literal(step))
        };

        registry.set_obj_var_value(obj_var.clone(), CircuitValue::CoordMap(obj_map));

        let obj_var_id = registry.register_circuit(ParamCircuitExpr::obj_var(obj_var));

        let output_expr = if let Some(offset_expr) = offset_expr_opt {
            let new_offset_expr = OffsetExpr::Add(
                Box::new(offset_expr),
                Box::new(OffsetExpr::Var(inner_i_dim_name.clone())),
            );

            ParamCircuitExpr::Rotate(new_offset_expr, obj_var_id)
        } else {
            let offset_var = registry.fresh_offset_fvar();
            registry.set_offset_var_value(offset_var.clone(), CircuitValue::CoordMap(step_map));

            ParamCircuitExpr::Rotate(OffsetExpr::Var(offset_var), obj_var_id)
        };

        registry.register_circuit(output_expr)
    }

    fn roll_materialization_type(
        &self,
        preprocessing: ArrayPreprocessing,
        transform: &ArrayTransform
    ) -> RollMaterializationType {
        if let ArrayPreprocessing::Roll(dim_i, dim_j) = preprocessing {
            match (&transform.dims[dim_i], &transform.dims[dim_j]) {
                // if dim i is empty, then the permutation is a no-op
                // materialize the schedule normally
                (DimContent::EmptyDim { extent: _ }, _) =>
                    RollMaterializationType::None,

                // if dim j is a filled dim, then the permutation must actually
                // be done by the client; record this fact and then materialize
                // the schedule normally
                (
                    DimContent::FilledDim {
                        dim: cdim_i,
                        extent: _,
                        stride: _,
                    },
                    DimContent::FilledDim {
                        dim: cdim_j,
                        extent: _,
                        stride: _,
                    },
                ) => RollMaterializationType::Diagonal(*cdim_i, *cdim_j),

                // the preprocessing can be eliminated by rotations
                (
                    DimContent::FilledDim {
                        dim: _,
                        extent: _,
                        stride: _,
                    },
                    DimContent::EmptyDim { extent: _ },
                ) => RollMaterializationType::Rotate(dim_i, dim_j)
            }

        } else {
            unreachable!()
        }
    }
}

impl<'a> InputArrayMaterializer<'a> for RollArrayMaterializer {
    fn create(&self) -> Box<dyn InputArrayMaterializer + 'a> {
        Box::new(RollArrayMaterializer::new())
    }

    fn name(&self) -> &str { "roll array materializer" }

    fn can_materialize(
        &self,
        _array_type: ArrayType,
        _array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        _base: &ArrayTransform,
    ) -> bool {
        if let Some(ArrayPreprocessing::Roll(dim_i, dim_j)) = schedule.preprocessing {
            // dim i must be exploded and dim j must be the outermost vectorized dim
            let i_exploded = schedule
                .exploded_dims
                .iter()
                .any(|edim| edim.index == dim_i);

            let j_outermost_vectorized = schedule.vectorized_dims.len() > 0
                && schedule.vectorized_dims.head().unwrap().index == dim_j;

            // dim i and j must have both have the same tiling that corresponds
            // to the permutation transform
            // TODO: for now, assume i and j are NOT tiled
            let tiling_i = schedule.get_dim_tiling(dim_i);
            let tiling_j = schedule.get_dim_tiling(dim_j);

            // dim i and j cannot have any padding
            let no_padding = schedule.vectorized_dims.iter().all(|dim| {
                (dim.index == dim_i && dim.pad_left == 0 && dim.pad_right == 0)
                    || (dim.index == dim_j && dim.pad_left == 0 && dim.pad_right == 0)
                    || (dim.index != dim_i || dim.index != dim_j)
            });

            // TODO: for now, assume i and j are NOT tiled
            tiling_i == tiling_j
                && tiling_i.len() == 1
                && i_exploded
                && j_outermost_vectorized
                && no_padding
        } else {
            false
        }
    }

    fn register(
        &mut self,
        _array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    ) {
        let mat_type =
            self.roll_materialization_type(
                schedule.preprocessing.unwrap(),
                transform
            );

        let new_schedule = match mat_type {
            RollMaterializationType::Diagonal(cdim_i, cdim_j)=>
                schedule.with_preprocessing(
                    Some(ArrayPreprocessing::Roll(cdim_i, cdim_j))
                ),

            RollMaterializationType::Rotate(dim_i, dim_j) => {
                let (rotated_schedule, _, _) =
                    self.rotate_schedule(schedule, dim_i, dim_j);

                rotated_schedule.with_preprocessing(None)
            }

            RollMaterializationType::None =>
                schedule.with_preprocessing(None)
        };

        self.deriver.register_vectors(
            array_shape,
            &new_schedule,
            transform,
            &IndexCoordinateSystem::new(schedule.exploded_dims.iter()),
        );
    }

    fn materialize(
        &mut self,
        array_type: ArrayType,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId {
        let preprocessing = schedule.preprocessing.unwrap();
        match self.roll_materialization_type(preprocessing, transform) {
            // if dim i is empty, then the permutation is a no-op
            // materialize the schedule normally
            RollMaterializationType::None => match array_type {
                ArrayType::Ciphertext => self
                    .deriver
                    .derive_vectors_and_gen_circuit::<CiphertextObject>(
                        shape,
                        &schedule.with_preprocessing(None),
                        transform,
                        registry,
                    ),

                ArrayType::Plaintext => self
                    .deriver
                    .derive_vectors_and_gen_circuit::<PlaintextObject>(
                        shape,
                        &schedule.with_preprocessing(None),
                        transform,
                        registry,
                    ),
            },

            // if dim j is a filled dim, then the permutation must actually
            // be done by the client; record this fact and then materialize
            // the schedule normally
            RollMaterializationType::Diagonal(cdim_i, cdim_j) => {
                let new_schedule = 
                    schedule.with_preprocessing(
                        Some(ArrayPreprocessing::Roll(cdim_i, cdim_j))
                    );

                match array_type {
                    ArrayType::Ciphertext => self
                        .deriver
                        .derive_vectors_and_gen_circuit::<CiphertextObject>(
                            shape,
                            &new_schedule,
                            transform,
                            registry,
                        ),

                    ArrayType::Plaintext => self
                        .deriver
                        .derive_vectors_and_gen_circuit::<PlaintextObject>(
                            shape,
                            &new_schedule,
                            transform,
                            registry,
                        ),
                }
            },

            RollMaterializationType::Rotate(dim_i, dim_j) => match array_type {
                ArrayType::Ciphertext => self.diagonal_materialize::<CiphertextObject>(
                    dim_i, dim_j, shape, schedule, transform, registry,
                ),

                ArrayType::Plaintext => self.diagonal_materialize::<PlaintextObject>(
                    dim_i, dim_j, shape, schedule, transform, registry,
                ),
            },
        }
    }

    fn estimate_cost(
        &mut self,
        _array_type: ArrayType,
        _array_shape: &Shape,
        _schedule: &IndexingSiteSchedule,
        _transform: &ArrayTransform,
    ) -> CostFeatures {
        todo!()
    }
}