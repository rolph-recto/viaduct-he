use std::{
    collections::{HashMap, HashSet},
    fmt::Display, ops::Index, mem::MaybeUninit, array,
};

use indexmap::IndexMap;
use log::info;

use crate::{
    circ::{vector_info::VectorInfo, CircuitValue, IndexCoordinateMap},
    lang::{
        index_elim::{ArrayDim, InlinedExpr, InlinedProgram},
        *,
    },
};

pub mod scheduler;
pub mod transformer;

// a schedule for a dimension
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScheduleDim {
    pub index: DimIndex,
    pub stride: usize,
    pub extent: usize,
    pub name: String,

    // pad_left and pad_right should only be nonzero for vectorized dims!
    pub pad_left: usize,
    pub pad_right: usize,
}

impl ScheduleDim {
    pub fn size(&self) -> usize {
        self.extent + self.pad_left + self.pad_right
    }
}

impl Display for ScheduleDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}.{}:{}::{}",
            self.name, self.index, self.extent, self.stride
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArrayPreprocessing {
    // TODO add more complicated permutations
    // Permute(i, j) means to permute dim i along dim j
    Roll(DimIndex, DimIndex),
}

impl Display for ArrayPreprocessing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayPreprocessing::Roll(dim_i, dim_j) =>
                write!(f, "roll({},{})", dim_i, dim_j),
        }
    }
}

impl ArrayPreprocessing {
    pub fn transformed_dims(&self) -> HashSet<DimIndex> {
        match self {
            ArrayPreprocessing::Roll(dim_i, _) => {
                let mut set: HashSet<DimIndex> = HashSet::new();
                set.insert(*dim_i);
                set
            }
        }
    }
}

pub trait HasExplodedDims {
    fn get_exploded_dims(&self) -> Vec<&ScheduleDim>;

    // get a parameterized offset map for the *array* indexed by transform
    fn get_indexed_offset_map(&self, transform: &ArrayTransform) -> OffsetMap<OffsetExpr> {
        let num_dims = transform.offset_map.num_dims();
        let mut param_offset_map: OffsetMap<OffsetExpr> = OffsetMap::new(num_dims);
        for i in 0..num_dims {
            let cur_offset = *transform.offset_map.get(i);
            param_offset_map.set(i, OffsetExpr::Literal(cur_offset));
        }

        for sched_dim in self.get_exploded_dims() {
            // exploded dims should not have padding!
            assert!(sched_dim.pad_left == 0 && sched_dim.pad_right == 0);

            let dim_content = transform.dims.get(sched_dim.index).unwrap();
            match dim_content {
                DimContent::FilledDim {
                    dim,
                    extent: _,
                    stride,
                } => {
                    let cur_offset = param_offset_map.get(*dim).clone();
                    let new_offset = OffsetExpr::Add(
                        Box::new(cur_offset),
                        Box::new(OffsetExpr::Mul(
                            Box::new(OffsetExpr::Literal(*stride as isize)),
                            Box::new(OffsetExpr::Var(sched_dim.name.clone())),
                        )),
                    );

                    param_offset_map.set(*dim, new_offset);
                }

                // if the dim is empty, no offset needs to be updated
                DimContent::EmptyDim { extent: _ } => {}
            }
        }

        param_offset_map
    }

    // get a parameterized offset map for the transform itself
    fn get_transform_offset_map(&self, transform: &ArrayTransform) -> OffsetMap<OffsetExpr> {
        let num_dims = transform.dims.len();
        let mut param_offset_map: OffsetMap<OffsetExpr> = OffsetMap::new(num_dims);
        for i in 0..num_dims {
            param_offset_map.set(i, OffsetExpr::Literal(0));
        }

        for sched_dim in self.get_exploded_dims() {
            let cur_offset = param_offset_map.get(sched_dim.index).clone();
            let new_offset = OffsetExpr::Add(
                Box::new(cur_offset),
                Box::new(OffsetExpr::Mul(
                    Box::new(OffsetExpr::Literal(sched_dim.stride as isize)),
                    Box::new(OffsetExpr::Var(sched_dim.name.clone())),
                )),
            );

            param_offset_map.set(sched_dim.index, new_offset);
        }

        param_offset_map
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndexingSiteSchedule {
    pub preprocessing: Option<ArrayPreprocessing>,
    pub exploded_dims: im::Vector<ScheduleDim>,
    pub vectorized_dims: im::Vector<ScheduleDim>,
}

impl IndexingSiteSchedule {
    // compute the scheduled tiling for a given dimension
    pub fn get_dim_tiling(&self, dim: DimIndex) -> Vec<usize> {
        let mut sdims: Vec<(usize, usize)> = Vec::new();

        sdims.extend(
            self.exploded_dims
                .iter()
                .filter(|edim| edim.index == dim)
                .map(|edim| (edim.stride, edim.extent)),
        );

        sdims.extend(self.vectorized_dims.iter().filter_map(|vdim| {
            if vdim.index == dim {
                Some((vdim.stride, vdim.extent))
            } else {
                None
            }
        }));

        sdims.sort_by(|(s1, _), (s2, _)| s1.cmp(s2));
        sdims.into_iter().map(|(_, extent)| extent).collect()
    }

    pub fn get_tiling(&self) -> Vec<Vec<usize>> {
        let mut sdims_map: IndexMap<DimIndex, Vec<(usize, usize)>> = IndexMap::new();

        let mut dims_index: Vec<DimIndex> =
            self.exploded_dims.iter()
            .map(|dim| dim.index)
            .chain(
                self.vectorized_dims.iter()
                .map(|dim| dim.index)
            )
            .collect();
        dims_index.sort();

        for index in dims_index.iter() {
            sdims_map.insert(*index, vec![]);
        }

        for edim in self.exploded_dims.iter() {
            let dim_list = sdims_map.get_mut(&edim.index).unwrap();
            dim_list.push((edim.stride, edim.extent));
        }

        for vdim in self.vectorized_dims.iter() {
            let dim_list = sdims_map.get_mut(&vdim.index).unwrap();
            dim_list.push((vdim.stride, vdim.extent));
        }

        sdims_map.into_iter().map(|(_, mut dims_list)| {
            dims_list.sort_by(|(s1, _), (s2, _)| s1.cmp(s2));
            dims_list.into_iter().map(|(_, extent)| extent).collect()
        }).collect()
    }

    pub fn vector_size(&self) -> usize {
        self.vectorized_dims.iter().fold(1, |acc, dim| {
            acc * dim.size()
        })
    }

    pub fn to_expr_schedule(&self, shape: Shape) -> ExprSchedule {
        ExprSchedule {
            shape,
            preprocessing: self.preprocessing.clone(),
            exploded_dims: self.exploded_dims.clone(),
            vectorized_dims: self
                .vectorized_dims
                .clone()
                .into_iter()
                .map(|dim| VectorScheduleDim::Filled(dim))
                .collect(),
        }
    }
}

impl HasExplodedDims for IndexingSiteSchedule {
    fn get_exploded_dims(&self) -> Vec<&ScheduleDim> {
        self.exploded_dims.iter().collect()
    }
}

impl Display for IndexingSiteSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let preprocessing_str = 
            if let Some(preprocessing) = self.preprocessing {
                preprocessing.to_string()

            } else {
                String::new()
            };

        let exploded_str = self
            .exploded_dims
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        let vectorized_str = self
            .vectorized_dims
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        write!(f, "{}{{{}}}[{}]", preprocessing_str, exploded_str, vectorized_str)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprScheduleType {
    Any, // the schedule is arbitrary (i.e. like for literals)
    Specific(ExprSchedule),
}

impl ExprScheduleType {
    /// counts how many exprs are represented by the schedule
    /// the multiplicity is the
    pub fn multiplicity(&self) -> usize {
        match self {
            ExprScheduleType::Any => 1,
            ExprScheduleType::Specific(spec_sched) => spec_sched
                .exploded_dims
                .iter()
                .map(|dim| dim.extent)
                .fold(1, |acc, x| acc * x),
        }
    }
}

impl Display for ExprScheduleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprScheduleType::Any => write!(f, "*"),
            ExprScheduleType::Specific(sched) => write!(f, "{}", sched),
        }
    }
}

// an output schedule for a vectorized dimension
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum VectorScheduleDim {
    // a regular dimension that contains elements of the scheduled array
    Filled(ScheduleDim),

    // reduced dim with the reduced value in the first position,
    // and the rest are "junk" values
    // e.g. 1 x x x 2 x x x
    // (extent, pad_left, pad_right)
    Reduced(usize, usize, usize),

    // reduced dim that is repeated with elements from other dimensions
    // e.g. 1 1 1 1 2 2 2 2
    ReducedRepeated(usize),
}

impl Display for VectorScheduleDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorScheduleDim::Filled(sched_dim) => write!(f, "{}", sched_dim),

            VectorScheduleDim::Reduced(extent, pad_left, pad_right) =>
                write!(f, "R:{}", extent),

            VectorScheduleDim::ReducedRepeated(extent) => write!(f, "RR:{}", extent),
        }
    }
}

impl VectorScheduleDim {
    pub fn is_reduced(&self) -> bool {
        match self {
            VectorScheduleDim::Filled(_) => false,

            VectorScheduleDim::Reduced(_, _, _) |
            VectorScheduleDim::ReducedRepeated(_) => true,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            VectorScheduleDim::Filled(dim) => dim.size(),

            VectorScheduleDim::Reduced(extent, pad_left, pad_right) =>
                extent + pad_left + pad_right,

            VectorScheduleDim::ReducedRepeated(extent) => {
                *extent
            }
        }
    }
}

// like ArraySchedule, except vectorized dims can have special reduced dimensions
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExprSchedule {
    pub shape: Shape,
    pub preprocessing: Option<ArrayPreprocessing>,
    pub exploded_dims: im::Vector<ScheduleDim>,
    pub vectorized_dims: im::Vector<VectorScheduleDim>,
}

impl Display for ExprSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape_str = self
            .shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        let exploded_str = self
            .exploded_dims
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        let vectorized_str = self
            .vectorized_dims
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        write!(
            f,
            "[{}], {{{}}}[{}]",
            shape_str, exploded_str, vectorized_str
        )
    }
}

#[derive(Copy,Clone,Debug)]
pub enum ScheduleDerivationFailure {
    // the derivation is a dead end; do not try to transform it further
    DimLengthMismatch,

    DimMismatch,

    OpFailure,

    ReduceFailure,

    LiteralFailure,

    IndexingSiteNotFound,


    // further transformations can maybe make schedule valid
    MaybeTransformableToValid
}

impl ExprSchedule {
    /// this schedule can be derived from the expr schedule of indexed array
    /// this can used to prune schedules out of the search space
    pub fn can_derive(
        &self,
        transform: &ArrayTransform,
        array_sched: &ExprSchedule
    ) -> Option<ScheduleDerivationFailure> {
        // first, check that the vectorized dimensions match
        let transform_vec_dims  = self.vectorized_dims.clone();
        let mut array_vec_dims = array_sched.vectorized_dims.clone();

        while transform_vec_dims.len() < array_vec_dims.len() {
            if let VectorScheduleDim::ReducedRepeated(_) = array_vec_dims.front().unwrap() {
                array_vec_dims.pop_front();
            } else {
                break
            }
        }

        if transform_vec_dims.len() < array_vec_dims.len() {
            return Some(ScheduleDerivationFailure::MaybeTransformableToValid)

        } else if transform_vec_dims.len() > array_vec_dims.len() {
            return Some(ScheduleDerivationFailure::DimLengthMismatch)
        }

        let dims_valid =
            transform_vec_dims.iter()
            .zip(array_vec_dims.iter())
            .all(|(dim1, dim2)| -> bool {
                match (dim1, dim2) {
                    (VectorScheduleDim::Filled(fdim1), VectorScheduleDim::Filled(fdim2)) => {
                        let same_extent =
                            fdim1.pad_left + fdim1.extent + fdim1.pad_right ==
                            fdim2.pad_left + fdim2.extent + fdim2.pad_right;
                        let dims_match =
                            match transform.dims[fdim1.index] {
                                DimContent::FilledDim { dim, extent, stride } =>
                                    dim == fdim2.index,

                                DimContent::EmptyDim { extent } =>
                                    false,
                            };

                        same_extent && dims_match
                    },

                    (VectorScheduleDim::Filled(fdim1), VectorScheduleDim::ReducedRepeated(extent)) => {
                        let same_extent =
                            fdim1.pad_left + fdim1.extent + fdim1.pad_right == *extent;
                        let is_empty = 
                            if let DimContent::EmptyDim { extent: _ } = transform.dims[fdim1.index] {
                                true
                            } else {
                                false
                            };

                        same_extent && is_empty
                    },

                    (VectorScheduleDim::Filled(fdim1), VectorScheduleDim::Reduced(extent, pad_left, pad_right)) => {
                        let same_extent =
                            fdim1.pad_left + fdim1.extent + fdim1.pad_right ==
                            *pad_left + *extent + *pad_right;
                        let is_empty = 
                            if let DimContent::EmptyDim { extent: _ } = transform.dims[fdim1.index] {
                                true
                            } else {
                                false
                            };

                        same_extent && is_empty
                    },

                    // dim1 has to be filled, so these are impossible cases
                    _ => false
                }
            });

        if dims_valid {
            None

        } else {
            Some(ScheduleDerivationFailure::DimMismatch)
        }
    }

    /// size of a vector (the product of vectorized dims' extents)
    pub fn vector_size(&self) -> usize {
        self.vectorized_dims
            .iter()
            .fold(1, |acc, dim| acc * dim.size())
    }

    // materialize schedule into a coordinate map of vectors
    // similar to DummyArrayMaterializer
    pub fn materialize(&self, array: &ArrayName) -> CircuitValue<VectorInfo> {
        if self.exploded_dims.len() > 0 {
            let mut coord_map = IndexCoordinateMap::new(self.exploded_dims.iter());
            for index_map in coord_map.index_map_iter() {
                let vector = VectorInfo::get_expr_vector_at_coord(
                    array.clone(),
                    index_map.clone(),
                    self,
                    None,
                );

                let coord = coord_map.index_map_as_coord(index_map);
                coord_map.set(coord, vector);
            }

            CircuitValue::CoordMap(coord_map)
        } else {
            let index_map: HashMap<DimName, usize> = HashMap::new();
            CircuitValue::Single(VectorInfo::get_expr_vector_at_coord(
                array.clone(),
                index_map,
                self,
                None,
            ))
        }
    }

    // returns a list of (extent,blocksize) to clean and fill (reduced dims) and a mask
    pub fn dims_to_fill(&self) -> (Vec<(usize,usize)>, im::Vector<(usize,usize,usize)>) {
        let mut block_size = 1;
        let mut dims_to_fill: Vec<(usize, usize)> = Vec::new();
        let mut mask_vector: im::Vector<(usize,usize,usize)> = im::Vector::new();
        for dim in self.vectorized_dims.iter().rev() {
            match dim {
                VectorScheduleDim::Filled(sdim) => {
                    let size = sdim.size();
                    block_size *= size;
                    mask_vector.push_front((size, 0, size-1));
                }

                VectorScheduleDim::ReducedRepeated(extent) => {
                    block_size *= *extent;
                    mask_vector.push_front((*extent, 0, extent-1));
                }

                VectorScheduleDim::Reduced(extent, pad_left, pad_right) => {
                    // only fill if the extent is more than 1
                    if *extent > 1 {
                        dims_to_fill.push((*extent, block_size));
                        block_size *= extent + pad_left + pad_right;
                        mask_vector.push_front((*extent, 0, 0));
                    } else {
                        mask_vector.push_front((*extent, 0, extent-1));
                    }
                }
            }
        }

        (dims_to_fill, mask_vector)
    }
}

impl HasExplodedDims for ExprSchedule {
    fn get_exploded_dims(&self) -> Vec<&ScheduleDim> {
        self.exploded_dims.iter().collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Schedule {
    pub schedule_map: im::HashMap<IndexingId, IndexingSiteSchedule>,
}

impl Display for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.schedule_map
            .iter()
            .try_for_each(|(ref_id, schedule)| writeln!(f, "{} => {}", ref_id, schedule))
    }
}

impl Index<&str> for Schedule {
    type Output = IndexingSiteSchedule;

    fn index(&self, index: &str) -> &Self::Output {
        &self.schedule_map[index]
    }
}

impl Schedule {
    // generate an initial schedule
    // the initial schedule explodes *all* dims
    pub fn gen_initial_schedule(program: &InlinedProgram) -> Self {
        let mut schedule_map: im::HashMap<IndexingId, IndexingSiteSchedule> = im::HashMap::new();
        let dim_class_map: HashMap<ArrayDim, usize> = program.get_dim_classes();

        for (_, expr) in program.expr_map.iter() {
            for (indexing_id, transform) in expr.get_indexing_sites() {
                let mut schedule_dims: im::Vector<ScheduleDim> = im::Vector::new();
                for (i, dim) in transform.dims.iter().enumerate() {
                    let class_id = dim_class_map[&(indexing_id.clone(), i)];
                    schedule_dims.push_back(ScheduleDim {
                        index: i,
                        stride: 1,
                        extent: dim.extent(),
                        name: format!("i{}", class_id),
                        pad_left: 0,
                        pad_right: 0,
                    })
                }

                let schedule = IndexingSiteSchedule {
                    preprocessing: None,
                    exploded_dims: schedule_dims,
                    vectorized_dims: im::Vector::new(),
                };

                schedule_map.insert(indexing_id, schedule);
            }
        }

        Schedule { schedule_map }
    }

    // apply the schedule to an index-free expression and compute the output schedule
    pub fn compute_output_schedule(
        &self,
        program: &InlinedProgram,
        output_schedules: &HashMap<ArrayName, ExprSchedule>,
        expr: &InlinedExpr,
    ) -> Result<ExprScheduleType, ScheduleDerivationFailure> {
        match expr {
            InlinedExpr::ReduceNode(reduced_index, _, body) => {
                let body_sched =
                    self.compute_output_schedule(
                        program,
                        output_schedules,
                        body
                    )?;
                Schedule::schedule_reduce(*reduced_index, &body_sched)
                .map_err(|_| ScheduleDerivationFailure::ReduceFailure)
            }

            InlinedExpr::Op(_, expr1, expr2) => {
                let sched1 =
                    self.compute_output_schedule(
                        program,
                        output_schedules,
                        expr1
                    )?;
                let sched2 =
                    self.compute_output_schedule(
                        program,
                        output_schedules,
                        expr2
                    )?;

                Schedule::schedule_op(&sched1, &sched2)
                .map_err(|_| ScheduleDerivationFailure::OpFailure)
            }

            InlinedExpr::Literal(_) =>
                Schedule::schedule_literal()
                .map_err(|_| ScheduleDerivationFailure::LiteralFailure),

            InlinedExpr::ExprRef(indexing_id, transform) => {
                if let Some(indexing_sched) = self.schedule_map.get(indexing_id) {
                    let expr_schedule = indexing_sched.to_expr_schedule(transform.as_shape());
                    if let Some(array_sched) = output_schedules.get(&transform.array) {
                        match expr_schedule.can_derive(transform, array_sched) {
                            Some(failure) => Err(failure),

                            None => {
                                Ok(ExprScheduleType::Specific(expr_schedule))
                            }
                        }

                    } else { // indexed array is input
                        Ok(ExprScheduleType::Specific(expr_schedule))
                    }

                } else {
                    Err(ScheduleDerivationFailure::IndexingSiteNotFound)
                }
            }
        }
    }

    pub fn schedule_literal() -> Result<ExprScheduleType, String> {
        Ok(ExprScheduleType::Any)
    }

    // this performs a join on the "schedule status lattice",
    // where valid schedules are incomparable,
    // any is bottom and invalid is top
    pub fn schedule_op(
        sched1: &ExprScheduleType,
        sched2: &ExprScheduleType,
    ) -> Result<ExprScheduleType, String> {
        match (sched1, sched2) {
            (ExprScheduleType::Any, ExprScheduleType::Any) => Ok(ExprScheduleType::Any),

            (ExprScheduleType::Any, ExprScheduleType::Specific(sched))
            | (ExprScheduleType::Specific(sched), ExprScheduleType::Any) => {
                Ok(ExprScheduleType::Specific(sched.clone()))
            }

            (ExprScheduleType::Specific(sched1), ExprScheduleType::Specific(sched2)) => {
                // it's okay if the number of dimensions of sched1 and sched2 don't match;
                // some of the outermost dimensions (empty dims and reduced repeated)
                // can be safely ignored as they only repeat elements anyway
                let mut sched1_new = sched1.clone();
                let mut sched2_new = sched1.clone();

                while sched1_new.vectorized_dims.len() < sched2_new.vectorized_dims.len() {
                    if let VectorScheduleDim::ReducedRepeated(_) = sched2_new.vectorized_dims.front().unwrap() {
                        sched2_new.vectorized_dims.pop_front();
                    } else {
                        break
                    }
                }

                while sched2_new.vectorized_dims.len() > sched2_new.vectorized_dims.len() {
                    if let VectorScheduleDim::ReducedRepeated(_) = sched1_new.vectorized_dims.front().unwrap() {
                        sched1_new.vectorized_dims.pop_front();
                    } else {
                        break
                    }
                }

                if sched1_new == sched2_new {
                    Ok(ExprScheduleType::Specific(sched1.clone()))

                } else {
                    Err(String::from("Operand schedules don't match"))
                }
            }
        }
    }

    // TODO support preprocessing
    pub fn schedule_reduce(
        reduced_index: usize,
        body_sched: &ExprScheduleType,
    ) -> Result<ExprScheduleType, String> {
        match body_sched {
            ExprScheduleType::Any => Err(String::from("Cannot reduce a literal expression")),

            ExprScheduleType::Specific(body_sched_spec) => {
                let mut new_exploded_dims: im::Vector<ScheduleDim> = im::Vector::new();

                for dim in body_sched_spec.exploded_dims.iter() {
                    if dim.index == reduced_index {
                        // don't add dimension to the output schedule
                    } else if dim.index > reduced_index {
                        // decrease dim index
                        let mut new_dim = dim.clone();
                        new_dim.index -= 1;
                        new_exploded_dims.push_back(new_dim);
                    } else {
                        new_exploded_dims.push_back(dim.clone());
                    }
                }

                let mut new_vectorized_dims: im::Vector<VectorScheduleDim> = im::Vector::new();
                for (i, dim) in body_sched_spec.vectorized_dims.iter().enumerate() {
                    let new_dim = match dim {
                        VectorScheduleDim::Filled(sched_dim) => {
                            if sched_dim.index == reduced_index {
                                // if outermost dim is reduced and there's no padding,
                                // the dim contents are repeated
                                if i == 0 && sched_dim.pad_left == 0 && sched_dim.pad_right == 0 {
                                    VectorScheduleDim::ReducedRepeated(sched_dim.extent)

                                } else {
                                    VectorScheduleDim::Reduced(sched_dim.extent, sched_dim.pad_left, sched_dim.pad_right)
                                }

                            } else if sched_dim.index > reduced_index {
                                let mut new_sched_dim = sched_dim.clone();
                                new_sched_dim.index -= 1;
                                VectorScheduleDim::Filled(new_sched_dim)
                            } else {
                                dim.clone()
                            }
                        }

                        VectorScheduleDim::Reduced(_, _, _) |
                        VectorScheduleDim::ReducedRepeated(_) => {
                            dim.clone()
                        }
                    };

                    new_vectorized_dims.push_back(new_dim);
                }

                let mut new_shape = body_sched_spec.shape.clone();
                new_shape.remove(reduced_index);

                Ok(ExprScheduleType::Specific(
                    // TODO support preprocessing here
                    ExprSchedule {
                        shape: new_shape,
                        preprocessing: None,
                        exploded_dims: new_exploded_dims,
                        vectorized_dims: new_vectorized_dims,
                    },
                ))
            }
        }
    }

    pub fn is_schedule_valid(&self, program: &InlinedProgram) -> Result<(), ScheduleDerivationFailure> {
        let mut output_schedules: HashMap<ArrayName, ExprSchedule> = HashMap::new();
        for (array_name, expr) in program.expr_map.iter() {
            match self.compute_output_schedule(program, &output_schedules, expr) {
                Ok(out_sched_type) => {
                    match out_sched_type {
                        ExprScheduleType::Any =>
                            panic!("array literals not supported yet"),

                        ExprScheduleType::Specific(out_sched) => {
                            // info!("array: {}; expr_schedule: {}", array_name, out_sched);
                            output_schedules.insert(array_name.clone(), out_sched);
                        }
                    };
                },
                
                Err(failure) => {
                    return Err(failure)
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::{
        elaborated::Elaborator, index_elim::IndexElimination, parser::ProgramParser,
        source::SourceProgram,
    };

    // generate an initial schedule for a program
    fn test_schedule(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let elaborated = Elaborator::new().run(program);
        let inline_set = elaborated.all_inlined_set();
        let array_group_map = elaborated.array_group_from_inline_set(&inline_set);

        let res =
            IndexElimination::new()
            .run(&inline_set, &array_group_map, &elaborated);

        assert!(res.is_ok());

        let tprogram = res.unwrap();
        println!("{}", tprogram);

        let init_schedule = Schedule::gen_initial_schedule(&tprogram);
        println!("{}", &init_schedule);

        // the initial schedule should always be valid!
        assert!(init_schedule.is_schedule_valid(&tprogram).is_ok());
    }

    #[test]
    fn test_imgblur() {
        test_schedule(
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
        test_schedule(
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
        test_schedule(
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
        test_schedule(
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
        test_schedule(
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
        test_schedule(
            "
            input A: [3] from client
            input B: [3] from client
            sum(A * B)
            ",
        );
    }

    #[test]
    fn test_matvecmul() {
        test_schedule(
            "
            input M: [2,2] from client
            input v: [2] from client
            for i: 2 {
                sum(M[i] * v)
            }
            ",
        );
    }

    #[test]
    fn test_pop_reduced_repeated() {
        let sched1: ExprSchedule = ExprSchedule {
            shape: im::vector![16],
            preprocessing: None,
            exploded_dims: im::Vector::new(),
            vectorized_dims: im::vector![
                VectorScheduleDim::ReducedRepeated(16),
                VectorScheduleDim::Filled(
                    ScheduleDim {
                        index: 0,
                        stride: 1,
                        extent: 16,
                        name: String::from("i"),
                        pad_left: 0,
                        pad_right: 0,
                    }
                ),
            ]
        };

        let sched2: ExprSchedule = ExprSchedule {
            shape: im::vector![16],
            preprocessing: None,
            exploded_dims: im::Vector::new(),
            vectorized_dims: im::vector![
                VectorScheduleDim::Filled(
                    ScheduleDim {
                        index: 0,
                        stride: 1,
                        extent: 16,
                        name: String::from("i"),
                        pad_left: 0,
                        pad_right: 0,
                    }
                ),
            ]
        };

        let sched_type1 = ExprScheduleType::Specific(sched1);
        let sched_type2 = ExprScheduleType::Specific(sched2);

        let out_sched1 =
            Schedule::schedule_op(
                &sched_type1,
                &sched_type2,
            );

        let out_sched2 =
            Schedule::schedule_op(
                &sched_type2,
                &sched_type1,
            );

        assert!(out_sched1.is_ok());
        assert!(out_sched2.is_ok());
    }
}
