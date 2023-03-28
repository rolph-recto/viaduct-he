use std::array;

use bimap::BiHashMap;
use log::info;

use crate::{
    circ::{vector_info::VectorInfo, *},
    scheduling::{ArrayPreprocessing, IndexingSiteSchedule, ExprSchedule}, util,
};

use super::cost::CostFeatures;

/// general methods for deriving vectors through rotation and masking
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
                    return *id2;
                }
            }
        }

        id
    }

    // find immediate parent for each vector
    pub fn compute_immediate_parents(&mut self, vector_ids: Vec<usize>) {
        for vector_id in vector_ids {
            let parent_id = self.find_immediate_parent(vector_id);
            self.parent_map.insert(vector_id, parent_id);
        }
    }

    pub fn find_transitive_parent(&self, id: VectorId) -> VectorId {
        let mut cur_id = id;
        let mut parent_id = self.parent_map[&cur_id];

        while parent_id != cur_id {
            cur_id = parent_id;
            parent_id = self.parent_map[&cur_id];
        }

        parent_id
    }

    pub fn get_vector(&self, id: VectorId) -> &VectorInfo {
        self.vector_map.get_by_left(&id).unwrap()
    }

    pub fn register_and_derive_vectors<T: CircuitObject>(
        &mut self,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        preprocessing: Option<ArrayPreprocessing>,
        coords: impl Iterator<Item = IndexCoord> + Clone,
        obj_map: &mut IndexCoordinateMap<T>,
        mask_map: &mut IndexCoordinateMap<PlaintextObject>,
        step_map: &mut IndexCoordinateMap<isize>,
    ) {
        let mut vector_id_map: HashMap<IndexCoord, VectorId> = HashMap::new();
        for coord in coords.clone() {
            let index_map = obj_map.coord_as_index_map(coord.clone());

            let vector = VectorInfo::get_input_vector_at_coord(
                index_map,
                array_shape,
                schedule,
                transform,
                preprocessing,
            );

            let vector_id = self.register_vector(vector);
            vector_id_map.insert(coord, vector_id);
        }

        let vector_ids =
            vector_id_map.iter()
            .map(|(_, id)| *id)
            .collect();
        self.compute_immediate_parents(vector_ids);

        // find transitive parents
        for coord in coords {
            let vector_id = *vector_id_map.get(&coord).unwrap();
            let parent_id = self.find_transitive_parent(vector_id);

            if vector_id != parent_id {
                // the vector is derived from some parent
                let vector = self.get_vector(vector_id);
                let parent = self.get_vector(parent_id);
                let (steps, mask) = parent.derive(vector).unwrap();

                step_map.set(coord.clone(), steps);
                mask_map.set(coord.clone(), mask);
                obj_map.set(coord, T::input_vector(parent.clone()));
            } else {
                // the vector is not derived
                let vector = self.get_vector(vector_id);
                step_map.set(coord.clone(), 0);
                mask_map.set(coord.clone(), PlaintextObject::Const(1));
                obj_map.set(coord, T::input_vector(vector.clone()));
            }
        }
    }

    fn derive_from_list<'a>(
        src_list: impl Iterator<Item = (IndexCoord, Option<&'a VectorInfo>)>,
        dst: &VectorInfo,
    ) -> Option<(VectorInfo, IndexCoord, isize, PlaintextObject)> {
        for (reg_coord, reg_vector_opt) in src_list {
            if let Some(reg_vector) = reg_vector_opt {
                if let Some((steps, mask)) = reg_vector.derive(dst) {
                    return Some((reg_vector.clone(), reg_coord.clone(), steps, mask));
                }
            }
        }

        None
    }

    pub fn probe_from_source<T: CircuitObject>(
        src: &CircuitValue<VectorInfo>,
        dst: &CircuitValue<VectorInfo>,
        ref_expr_sched: &ExprSchedule,
        array_type: ArrayType,
    ) -> Option<CostFeatures>
    {
        match dst {
            CircuitValue::CoordMap(dst_map) => {
                let indices = dst_map.index_vars();
                let extents = dst_map.extents();
                let zeros: IndexCoord = indices.iter().map(|_| 0).collect();
                let mut probes: Vec<IndexCoord> = Vec::new();
                for i in 0..indices.len() {
                    let mut probe = zeros.clone();
                    probe[i] = 1;
                    probes.push(probe);
                }

                let deriver = |dst_vector| {
                    match src {
                        CircuitValue::CoordMap(src_map) => {
                            VectorDeriver::derive_from_list(src_map.object_iter(), dst_vector)
                        },

                        CircuitValue::Single(src_vector) => {
                            src_vector.derive(dst_vector)
                            .map(|(step, mask)|
                                (src_vector.clone(), im::Vector::new(), step, mask)
                            )
                        }
                    }
                };

                let base_vector_opt =
                    deriver(dst_map.get(&zeros).unwrap());

                let base_vector_coord: IndexCoord =
                    match base_vector_opt {
                        Some(res) => {
                            res.1
                        },

                        None => {
                            return None
                        }
                    };

                let mut rotate_extents: Vec<Extent> = Vec::new();
                let mut mult_extents: Vec<Extent> = Vec::new();
                let mut src_vecs_to_derive_from: usize = 1;
                for (i, coord) in probes.iter().enumerate() {
                    let dst_vector = dst_map.get(coord).unwrap();
                    let derive_opt = deriver(dst_vector);

                    if let Some((_, reg_coord, steps, mask)) = derive_opt {
                        if reg_coord != base_vector_coord {
                            src_vecs_to_derive_from *= extents[i];
                        }

                        if steps != 0 {
                            rotate_extents.push(extents[i]);
                        }

                        if let PlaintextObject::Mask(_) = mask {
                            mult_extents.push(extents[i]);
                        }

                    } else {
                        return None;
                    }
                }

                let dims_to_fill = ref_expr_sched.dims_to_fill().0;
                let reduction_list  = util::get_reduction_list(dims_to_fill);

                let reduce_and_sum_ops = src_vecs_to_derive_from * reduction_list.len();

                let mut cost = CostFeatures::default();

                match array_type {
                    ArrayType::Ciphertext => {
                        cost.ct_rotations =
                            if rotate_extents.len() > 0 {
                                rotate_extents.iter().product()

                            } else {
                                0
                            };

                        cost.ct_pt_mul =
                            if mult_extents.len() > 0 {
                                mult_extents.iter().product()

                            } else {
                                0
                            };
                        cost.ct_ct_add += reduce_and_sum_ops;
                        cost.ct_rotations += reduce_and_sum_ops;
                    },

                    ArrayType::Plaintext => {
                        cost.pt_rotations =
                            if rotate_extents.len() > 0 {
                                rotate_extents.iter().product()

                            } else {
                                0
                            };

                        cost.pt_pt_mul =
                            if mult_extents.len() > 0 {
                                mult_extents.iter().product()

                            } else {
                                0
                            };
                        cost.pt_pt_add += reduce_and_sum_ops;
                        cost.pt_rotations += reduce_and_sum_ops;
                    }
                }

                Some(cost)
            }

            CircuitValue::Single(dst_vector) => {
                let derive_opt =
                    match src {
                        CircuitValue::CoordMap(src_map) => {
                            VectorDeriver::derive_from_list(src_map.object_iter(), dst_vector)
                        }

                        CircuitValue::Single(src_vector) => {
                            src_vector.derive(dst_vector)
                            .map(|(steps, mask)| {
                                (src_vector.clone(), im::Vector::new(), steps, mask)
                            })
                        }
                    };
                
                if let Some((_, _, steps, mask)) = derive_opt {
                    let dims_to_fill = ref_expr_sched.dims_to_fill().0;
                    let reduction_list  = util::get_reduction_list(dims_to_fill);
                    let mut cost = CostFeatures::default();

                    match array_type {
                        ArrayType::Ciphertext => {
                            if steps != 0 {
                                cost.ct_rotations += 1;
                            }

                            if let PlaintextObject::Mask(_) = mask {
                                cost.ct_pt_mul += 1;
                            }

                            cost.ct_ct_add += reduction_list.len();
                            cost.ct_rotations += reduction_list.len();
                        },
                        
                        ArrayType::Plaintext => {
                            if steps != 0 {
                                cost.pt_rotations += 1;
                            }

                            if let PlaintextObject::Mask(_) = mask {
                                cost.pt_pt_mul += 1;
                            }

                            cost.pt_pt_add += reduction_list.len();
                            cost.pt_rotations += reduction_list.len();
                        }
                    };

                    return Some(cost)

                } else {
                    return None;
                }
            },
        }
    }

    // derive a circuit value from a source circuit value
    pub fn derive_from_source<T: CircuitObject>(
        src: &CircuitValue<VectorInfo>,
        dst: &CircuitValue<VectorInfo>,
    ) -> Option<(
        CircuitValue<T>,
        CircuitValue<isize>,
        CircuitValue<PlaintextObject>,
    )> {
        match dst {
            CircuitValue::CoordMap(dst_map) => {
                // attempt to derive the dst map from the src map
                match src {
                    CircuitValue::CoordMap(src_map) => {
                        // coordinate in src_map that can be used to derive dst_map vector
                        let mut obj_map: IndexCoordinateMap<T> =
                            IndexCoordinateMap::from_coord_system(dst_map.coord_system.clone());
                        let mut step_map: IndexCoordinateMap<isize> =
                            IndexCoordinateMap::from_coord_system(dst_map.coord_system.clone());
                        let mut mask_map: IndexCoordinateMap<PlaintextObject> =
                            IndexCoordinateMap::from_coord_system(dst_map.coord_system.clone());

                        for (coord, dst_vector_opt) in dst_map.object_iter() {
                            let dst_vector = dst_vector_opt.unwrap();

                            let derive_opt =
                                VectorDeriver::derive_from_list(src_map.object_iter(), dst_vector);

                            if let Some((vector, reg_coord, steps, mask)) = derive_opt {
                                let object = T::expr_vector(vector.array, reg_coord);

                                obj_map.set(coord.clone(), object);
                                step_map.set(coord.clone(), steps);
                                mask_map.set(coord.clone(), mask);
                            } else {
                                return None;
                            }
                        }

                        Some((
                            CircuitValue::CoordMap(obj_map),
                            CircuitValue::CoordMap(step_map),
                            CircuitValue::CoordMap(mask_map),
                        ))
                    }

                    CircuitValue::Single(src_vector) => {
                        let mut step_map: IndexCoordinateMap<isize> =
                            IndexCoordinateMap::from_coord_system(dst_map.coord_system.clone());
                        let mut mask_map: IndexCoordinateMap<PlaintextObject> =
                            IndexCoordinateMap::from_coord_system(dst_map.coord_system.clone());

                        for (coord, dst_vector_opt) in dst_map.object_iter() {
                            let dst_vector = dst_vector_opt.unwrap();
                            let derive_opt = src_vector.derive(dst_vector);

                            if let Some((steps, mask)) = derive_opt {
                                step_map.set(coord.clone(), steps);
                                mask_map.set(coord.clone(), mask);

                            } else {
                                return None;
                            }
                        }

                        Some((
                            CircuitValue::Single(T::expr_vector(
                                src_vector.array.clone(),
                                im::Vector::new(),
                            )),
                            CircuitValue::CoordMap(step_map),
                            CircuitValue::CoordMap(mask_map),
                        ))
                    }
                }
            }

            CircuitValue::Single(dst_vector) => match src {
                CircuitValue::CoordMap(src_map) => {
                    let derive_opt =
                        VectorDeriver::derive_from_list(src_map.object_iter(), dst_vector);

                    if let Some((vector, reg_coord, steps, mask)) = derive_opt {
                        let object = T::expr_vector(vector.array, reg_coord);

                        Some((
                            CircuitValue::Single(object),
                            CircuitValue::Single(steps),
                            CircuitValue::Single(mask),
                        ))
                    } else {
                        return None;
                    }
                }

                CircuitValue::Single(src_vector) => {
                    if let Some((steps, mask)) = src_vector.derive(dst_vector) {
                        let object = T::expr_vector(src_vector.array.clone(), im::Vector::new());

                        Some((
                            CircuitValue::Single(object),
                            CircuitValue::Single(steps),
                            CircuitValue::Single(mask),
                        ))
                    } else {
                        None
                    }
                }
            },
        }
    }

    // assume that the rotation steps have a linear relationship to the index vars,
    // then probe certain coordinates to compute an offset expr
    // this can compute linear offsets for a *subset* of defined coords;
    // hence this function takes in extra arguments
    // valid_coords and processed_index_vars
    pub fn compute_linear_offset_coefficient(
        step_map: &IndexCoordinateMap<isize>,
        valid_coords: impl Iterator<Item = IndexCoord>,
        index_vars_to_process: Vec<DimName>,
    ) -> Option<(isize, Vec<isize>)> {
        let index_vars = step_map.index_vars();

        // probe at (0,...,0) to get the base offset
        let base_coord: im::Vector<usize> = im::Vector::from(vec![0; index_vars.len()]);
        let base_offset: isize = *step_map.get(&base_coord).unwrap();

        // probe at (0,..,1,..,0) to get the coefficient for the ith index var
        // only do this for processed_index_vars, not *all* index vars
        let mut coefficients: Vec<isize> = Vec::new();
        for i in 0..index_vars_to_process.len() {
            let mut index_coord = base_coord.clone();
            index_coord[i] = 1;

            let step_offset = *step_map.get(&index_coord).unwrap();
            coefficients.push(step_offset - base_offset);
        }

        // validate computed offset expr
        for coord in valid_coords {
            let value = *step_map.get(&coord).unwrap();

            let predicted_value: isize =
                base_offset + 
                coord.into_iter()
                .zip(coefficients.iter())
                .map(|(x, c)| (x as isize) * (*c))
                .sum::<isize>();

            if value != predicted_value {
                return None;
            }
        }

        Some((base_offset, coefficients))
    }

    pub fn compute_linear_offset(
        step_map: &IndexCoordinateMap<isize>,
        valid_coords: impl Iterator<Item = IndexCoord>,
        index_vars_to_process: Vec<DimName>,
    ) -> Option<OffsetExpr> {
        let (base_offset, coefficients) =
            Self::compute_linear_offset_coefficient(step_map, valid_coords, index_vars_to_process)?;

        let index_vars = step_map.index_vars();

        // build offset expr from base offset and coefficients
        let offset_expr = coefficients.iter().zip(index_vars.clone()).fold(
            OffsetExpr::Literal(base_offset),
            |acc, (coeff, index_var)| {
                if *coeff != 0 {
                    OffsetExpr::Add(
                        Box::new(acc),
                        Box::new(OffsetExpr::Mul(
                            Box::new(OffsetExpr::Literal(*coeff)),
                            Box::new(OffsetExpr::Var(index_var.clone())),
                        )),
                    )
                } else {
                    acc
                }
            },
        );

        // this expression is correct for all valid_coords; return it
        Some(offset_expr)
    }

    // function that uses derivation data to generate a circuit
    // the basic pattern it uses is 
    pub fn gen_circuit_expr<'a, T: CircuitObject>(
        obj_val: CircuitValue<T>,
        step_val: CircuitValue<isize>,
        mask_val: CircuitValue<PlaintextObject>,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId
    where
        CircuitObjectRegistry: CanRegisterObject<'a, T>,
        ParamCircuitExpr: CanCreateObjectVar<T>,
    {
        let obj_var = registry.fresh_obj_var();

        let mask_is_nonconst = match &mask_val {
            CircuitValue::CoordMap(mask_map) => mask_map.object_iter().any(|(_, mask)| {
                if let Some(PlaintextObject::Const(_)) = mask {
                    false
                } else {
                    true
                }
            }),

            CircuitValue::Single(obj) => {
                if let PlaintextObject::Const(_) = obj {
                    false
                } else {
                    true
                }
            }
        };

        let masked_expr = if mask_is_nonconst {
            let pt_var = registry.fresh_pt_var();
            registry.set_obj_var_value(obj_var.clone(), obj_val);
            registry.set_pt_var_value(pt_var.clone(), mask_val);

            let var_id = registry.register_circuit(ParamCircuitExpr::obj_var(obj_var));
            let mask_id = registry.register_circuit(ParamCircuitExpr::PlaintextVar(pt_var));

            ParamCircuitExpr::Op(Operator::Mul, var_id, mask_id)
        } else {
            registry.set_obj_var_value(obj_var.clone(), obj_val);
            ParamCircuitExpr::obj_var(obj_var)
        };

        let masked_expr_id = registry.register_circuit(masked_expr.clone());

        let offset_expr_opt = match &step_val {
            // a vector map derived from a vector map
            CircuitValue::CoordMap(step_map) => {
                // attempt to compute offset expr
                VectorDeriver::compute_linear_offset(
                    &step_map,
                    step_map.coord_iter(),
                    step_map.coord_system.index_vars(),
                )
            }

            // a single vector derived from a either a single vector or a vector map
            CircuitValue::Single(step) => Some(OffsetExpr::Literal(*step)),
        };

        let output_expr = if let Some(linear_offset_expr) = offset_expr_opt {
            if let Some(0) = linear_offset_expr.const_value() {
                masked_expr
            } else {
                ParamCircuitExpr::Rotate(linear_offset_expr, masked_expr_id)
            }
        } else {
            // introduce new offset variable, since we can't create an offset expr
            let offset_var = registry.fresh_offset_fvar();
            registry.set_offset_var_value(offset_var.clone(), step_val);

            ParamCircuitExpr::Rotate(OffsetExpr::Var(offset_var), masked_expr_id)
        };

        registry.register_circuit(output_expr)
    }

    // default method for generating circuit expression for an array materializer
    pub fn derive_vectors_and_gen_circuit_expr<'a, T: CircuitObject>(
        &mut self,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        preprocessing: Option<ArrayPreprocessing>,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId
    where
        CircuitObjectRegistry: CanRegisterObject<'a, T>,
        ParamCircuitExpr: CanCreateObjectVar<T>,
    {
        let mut obj_map: IndexCoordinateMap<T> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());

        if !obj_map.is_empty() {
            // there is an array of vectors
            let mut mask_map: IndexCoordinateMap<PlaintextObject> =
                IndexCoordinateMap::new(schedule.exploded_dims.iter());
            let mut step_map: IndexCoordinateMap<isize> =
                IndexCoordinateMap::new(schedule.exploded_dims.iter());

            let coords = obj_map.coord_iter();

            self.register_and_derive_vectors(
                array_shape,
                schedule,
                transform,
                preprocessing,
                coords.clone(),
                &mut obj_map,
                &mut mask_map,
                &mut step_map,
            );

            VectorDeriver::gen_circuit_expr(
                CircuitValue::CoordMap(obj_map),
                CircuitValue::CoordMap(step_map),
                CircuitValue::CoordMap(mask_map),
                registry,
            )
        } else {
            // there is only a single vector
            let vector =
                VectorInfo::get_input_vector_at_coord(
                    HashMap::new(),
                    array_shape,
                    schedule,
                    transform,
                    preprocessing,
                );

            let vector_id = self.register_vector(vector.clone());
            self.compute_immediate_parents(vec![vector_id]);

            let parent_id = self.find_transitive_parent(vector_id);
            if vector_id != parent_id {
                let parent = self.get_vector(parent_id);
                let (steps, mask) = parent.derive(&vector).unwrap();
                VectorDeriver::gen_circuit_expr(
                    CircuitValue::Single(T::input_vector(parent.clone())),
                    CircuitValue::Single(steps),
                    CircuitValue::Single(mask),
                    registry,
                )

            } else {
                VectorDeriver::gen_circuit_expr(
                    CircuitValue::Single(T::input_vector(vector)),
                    CircuitValue::Single(0),
                    CircuitValue::Single(PlaintextObject::Const(1)),
                    registry,
                )
            }
        }
    }
}
