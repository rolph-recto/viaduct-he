use std::{
    cmp::min,
    collections::{HashMap, HashSet},
    fmt::Display,
};

use gcollections::ops::Bounded;

use crate::{circ::PlaintextObject, lang::*, scheduling::*};

use super::{CircuitValue, IndexCoordinateMap, IndexCoordinateSystem};

// like DimContent, but with padding information
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum VectorDimContent {
    // we assume that dimensions have the following structure
    // | pad_left | oob_left | extent | oob_right | pad_right |
    FilledDim {
        dim: DimIndex,
        extent: usize,
        stride: usize,

        // out of bounds intervals
        oob_left: usize,
        oob_right: usize,

        // padding
        pad_left: usize,
        pad_right: usize,
    },

    // dim that has repeated elements from other dims
    EmptyDim {
        extent: usize,
        pad_left: usize,
        pad_right: usize,
        oob_right: usize,
    },

    // dim with a reduced element in its first coordinate
    ReducedDim {
        extent: usize,
        pad_left: usize,
        pad_right: usize,
    },
}

impl Display for VectorDimContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorDimContent::FilledDim {
                dim,
                extent,
                stride,
                oob_left,
                oob_right,
                pad_left,
                pad_right,
            } => {
                let mut extra: String = String::new();

                if *oob_left != 0 || *oob_right != 0 {
                    extra.push_str(&format!("[oob=({},{})]", oob_left, oob_right));
                }

                if *pad_left != 0 || *pad_right != 0 {
                    extra.push_str(&format!("[pad=({},{})]", pad_left, pad_right));
                }

                write!(f, "{{{}:{}::{}{}}}", dim, extent, stride, extra)
            }

            VectorDimContent::EmptyDim {
                extent,
                pad_left,
                pad_right,
                oob_right,
            } => {
                let mut extra: String = String::new();
                if *pad_left != 0 || *pad_right != 0 {
                    extra.push_str(&format!("[pad=({},{})]", pad_left, pad_right));
                }

                if *oob_right != 0 {
                    extra.push_str(&format!("[oob=(0,{})]", oob_right));
                }

                write!(f, "{{{}{}}}", extent, extra)
            }

            VectorDimContent::ReducedDim {
                extent,
                pad_left,
                pad_right,
            } => {
                if *pad_left != 0 || *pad_right != 0 {
                    write!(f, "{{R{}[pad=({},{})]}}", extent, pad_left, pad_right)
                } else {
                    write!(f, "{{R{}}}", extent)
                }
            }
        }
    }
}

impl VectorDimContent {
    pub fn size(&self) -> usize {
        match self {
            VectorDimContent::FilledDim {
                dim: _,
                extent,
                stride: _,
                oob_left,
                oob_right,
                pad_left,
                pad_right,
            } => pad_left + oob_left + extent + oob_right + pad_right,

            VectorDimContent::EmptyDim {
                extent,
                pad_left,
                pad_right,
                oob_right,
            } => pad_left + (*extent) + pad_right + oob_right,

            VectorDimContent::ReducedDim {
                extent,
                pad_left,
                pad_right,
            } => pad_left + (*extent) + pad_right,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VectorInfo {
    pub array: ArrayName,
    pub preprocessing: Option<ArrayPreprocessing>,
    pub offset_map: OffsetMap<isize>,
    pub dims: im::Vector<VectorDimContent>,
}

impl Display for VectorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(preprocess) = self.preprocessing {
            write!(
                f,
                "{}({})[{}]<{}>",
                self.array,
                preprocess,
                self.offset_map,
                self.dims
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        } else {
            write!(
                f,
                "{}[{}]<{}>",
                self.array,
                self.offset_map,
                self.dims
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        }
    }
}

impl VectorInfo {
    fn process_schedule_dim(
        shape: &Shape,
        transform: &ArrayTransform,
        clipped_offset_map: &mut OffsetMap<isize>,
        transform_offset_map: &OffsetMap<usize>,
        dim: &ScheduleDim,
    ) -> VectorDimContent {
        let dim_content = transform.dims.get(dim.index).unwrap();
        let transform_dim_offset = transform_offset_map.get(dim.index);
        match dim_content {
            // clip dimension to (0, array dimension's extent)
            DimContent::FilledDim {
                dim: idim,
                extent: transform_extent,
                stride: istride,
            } => {
                let dim_offset = *clipped_offset_map.get(*idim);
                let array_extent = shape[*idim].upper() as usize;
                let new_stride = *istride * dim.stride;

                // indexing less than 0; clip
                let mut oob_left: usize = 0;
                while dim_offset + ((oob_left * new_stride) as isize) < 0 {
                    oob_left += 1;
                }

                // indexing beyond array_extent; clip
                let mut oob_right: usize = 0;
                while dim_offset + ((new_stride * (dim.extent - 1 - oob_right)) as isize)
                    >= array_extent as isize
                {
                    oob_right += 1;
                }

                // clip data beyond transform bounds
                while transform_dim_offset + (dim.stride * (dim.extent - 1 - oob_right))
                    >= *transform_extent
                {
                    oob_right += 1;
                }

                // increment offset by oob_left
                let cur_offset = *clipped_offset_map.get(*idim);
                clipped_offset_map.set(*idim, cur_offset + ((oob_left * new_stride) as isize));

                let new_extent_signed =
                    (dim.extent as isize) - (oob_left as isize) - (oob_right as isize);

                if new_extent_signed >= 0 {
                    VectorDimContent::FilledDim {
                        dim: *idim,
                        extent: new_extent_signed as usize,
                        stride: new_stride,
                        oob_left,
                        oob_right,
                        pad_left: dim.pad_left,
                        pad_right: dim.pad_right,
                    }
                } else {
                    // make entire dim OOB
                    VectorDimContent::FilledDim {
                        dim: *idim,
                        extent: 0,
                        stride: new_stride,
                        oob_left: dim.extent,
                        oob_right,
                        pad_left: dim.pad_left,
                        pad_right: dim.pad_right,
                    }
                }
            }

            DimContent::EmptyDim {
                extent: transform_extent,
            } => {
                let mut oob_right: usize = 0;
                while transform_dim_offset + (dim.stride * (dim.extent - 1 - oob_right))
                    >= *transform_extent
                {
                    oob_right += 1;
                }

                VectorDimContent::EmptyDim {
                    extent: dim.extent - oob_right,
                    pad_left: dim.pad_left,
                    pad_right: dim.pad_right,
                    oob_right,
                }
            }
        }
    }

    // retrieve vector at specific coordinate of a scheduled array
    pub fn get_input_vector_at_coord(
        index_map: HashMap<DimName, usize>,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        preprocessing: Option<ArrayPreprocessing>,
    ) -> Self {
        let offset_env = OffsetEnvironment::new(index_map);

        let mut clipped_offset_map = schedule
            .get_indexed_offset_map(transform)
            .map(|offset| offset.eval(&offset_env));

        let transform_offset_map = schedule
            .get_transform_offset_map(transform)
            .map(|offset| usize::try_from(offset.eval(&offset_env)).unwrap());

        let mut materialized_dims: im::Vector<VectorDimContent> = im::Vector::new();
        for dim in schedule.vectorized_dims.iter() {
            let materialized_dim = VectorInfo::process_schedule_dim(
                shape,
                transform,
                &mut clipped_offset_map,
                &transform_offset_map,
                dim,
            );
            materialized_dims.push_back(materialized_dim);
        }

        VectorInfo {
            array: transform.array.clone(),
            preprocessing,
            offset_map: clipped_offset_map,
            dims: materialized_dims,
        }
    }

    pub fn get_input_vector_value(
        coord_system: IndexCoordinateSystem,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        preprocessing: Option<ArrayPreprocessing>,
    ) -> CircuitValue<VectorInfo> {
        if !coord_system.is_empty() {
            let mut coord_map = IndexCoordinateMap::from_coord_system(coord_system);
            for index_map in coord_map.index_map_iter() {
                let vector = VectorInfo::get_input_vector_at_coord(
                    index_map.clone(),
                    shape,
                    schedule,
                    transform,
                    preprocessing,
                );

                coord_map.set(coord_map.index_map_as_coord(index_map), vector);
            }

            CircuitValue::CoordMap(coord_map)
        } else {
            let vector = VectorInfo::get_input_vector_at_coord(
                HashMap::new(),
                shape,
                schedule,
                transform,
                preprocessing,
            );

            CircuitValue::Single(vector)
        }
    }

    pub fn get_expr_vector_at_coord(
        array: ArrayName,
        index_map: HashMap<DimName, usize>,
        expr_schedule: &ExprSchedule,
        preprocessing: Option<ArrayPreprocessing>,
    ) -> Self {
        let transform = ArrayTransform::from_shape(array, &expr_schedule.shape);

        let offset_env = OffsetEnvironment::new(index_map);

        let mut clipped_offset_map = expr_schedule
            .get_indexed_offset_map(&transform)
            .map(|offset| offset.eval(&offset_env));

        let transform_offset_map = expr_schedule
            .get_transform_offset_map(&transform)
            .map(|offset| offset.eval(&offset_env) as usize);

        let mut materialized_dims: im::Vector<VectorDimContent> = im::Vector::new();
        for dim in expr_schedule.vectorized_dims.iter() {
            let materialized_dim = match dim {
                VectorScheduleDim::Filled(sched_dim) =>
                    VectorInfo::process_schedule_dim(
                        &expr_schedule.shape,
                        &transform,
                        &mut clipped_offset_map,
                        &transform_offset_map,
                        sched_dim,
                    ),

                // treat like an empty dim
                // this is only possible if there is no padding,
                // so set padding to 0
                VectorScheduleDim::ReducedRepeated(extent) =>
                    VectorDimContent::EmptyDim {
                        extent: *extent,
                        pad_left: 0,
                        pad_right: 0,
                        oob_right: 0,
                    },

                VectorScheduleDim::Reduced(extent, pad_left, pad_right) =>
                    VectorDimContent::ReducedDim {
                        extent: *extent,
                        pad_left: *pad_left,
                        pad_right: *pad_right,
                    }
            };

            materialized_dims.push_back(materialized_dim);
        }

        VectorInfo {
            array: transform.array.clone(),
            preprocessing,
            offset_map: clipped_offset_map,
            dims: materialized_dims,
        }
    }

    // derive other from self
    pub fn derive(&self, other: &VectorInfo) -> Option<(isize, PlaintextObject)> {
        use VectorDimContent::*;

        if self.dims.len() != other.dims.len() {
            None
        } else if self == other {
            Some((0, PlaintextObject::Const(1)))
        } else {
            let mut seen_dims: HashSet<DimIndex> = HashSet::new();
            let mut dims_to_fill: Vec<usize> = Vec::new();

            // check derivability conditions
            let dims_derivable =
                self.dims.iter()
                .zip(other.dims.iter())
                .all(|(dim1, dim2)| {
                    match (*dim1, *dim2) {
                        (FilledDim {
                            dim: idim1, extent: extent1, stride: stride1,
                            oob_left: oob_left1, oob_right: oob_right1,
                            pad_left: pad_left1, pad_right: pad_right1,
                        },
                        FilledDim {
                            dim: idim2, extent: extent2, stride: stride2,
                            oob_left: oob_left2, oob_right: oob_right2,
                            pad_left: pad_left2, pad_right: pad_right2,
                        }) => {
                            // dimensions point to the same indexed dimension (duh)
                            let same_dim = idim1 == idim2;

                            // multiple dims cannot stride the same indexed dim
                            // TODO is this necessary
                            let dim_unseen = !seen_dims.contains(&idim1);

                            // dimensions have the same stride
                            let same_stride = stride1 == stride2;

                            // the offsets of self and other ensure that they
                            // have the same elements
                            let offset1 = self.offset_map[idim1];
                            let offset2 = other.offset_map[idim2];
                            let offset_equiv =
                                offset1 % (stride1 as isize) == offset2 % (stride2 as isize);

                            // the dimensions have the same size
                            let same_size = dim1.size() == dim2.size();

                            // all of the elements of other's dim is in self's dim
                            let in_extent =
                                offset2 >= offset1 &&
                                offset1 + ((stride1 * extent1) as isize) >= offset2 + ((stride2 * extent2) as isize);

                            // self cannot have out of bounds values
                            // TODO this is not needed!
                            let self_no_oob = oob_left1 == 0 && oob_right1 == 0;

                            seen_dims.insert(idim1);
                            same_dim && dim_unseen && same_stride && offset_equiv &&
                            same_size && in_extent && self_no_oob
                        },

                        // empty dims will not be rotated for derivation, but
                        // but they might need to be masked
                        (EmptyDim { extent: extent1, pad_left: pad_left1, pad_right: pad_right1, oob_right: oob_right1 },
                        EmptyDim { extent: extent2, pad_left: pad_left2, pad_right: pad_right2, oob_right: oob_right2  }) => {
                            pad_left1 + extent1 + pad_right1 + oob_right1 == pad_left2 + extent2 + pad_right2 + oob_right2 &&

                            // the padding for derived vector must be more
                            // thant the padding for the parent, since we can
                            // only mask parent contents (not add more)
                            pad_left1 <= pad_left2 && pad_right1 + oob_right1 <= pad_right2 + oob_right2 &&

                            // can always truncate empty dims with more padding,
                            // but don't support extending dims (yet)
                            extent1 >= extent2
                        },

                        // we can derive an empty dim
                        // TODO add reduced dim to a list of dims to fill back in
                        (ReducedDim { extent: extent1, pad_left: pad_left1, pad_right: pad_right1 },
                        EmptyDim { extent: extent2, pad_left: pad_left2, pad_right: pad_right2, oob_right: oob_right2 }) =>  {
                            // this has the same restrictions as deriving from
                            // an empty dim, except we need to fill the
                            // reduced parent dim first
                            pad_left1 + extent1 + pad_right1 == pad_left2 + extent2 + pad_right2 + oob_right2 &&
                            pad_left1 <= pad_left2 && pad_right1 <= pad_right2 + oob_right2 &&
                            extent1 >= extent2
                        },

                        (FilledDim { dim: _, extent: _, stride: _, oob_left: _, oob_right: _, pad_left: _, pad_right: _ },
                        EmptyDim { extent: _, pad_left: _, pad_right: _, oob_right: _ }) |

                        (EmptyDim { extent: _, pad_left: _, pad_right: _, oob_right: _ },
                        FilledDim { dim: _, extent: _, stride: _, oob_left: _, oob_right: _, pad_left: _, pad_right: _ }) |

                        (_, ReducedDim { extent: _, pad_left: _, pad_right: _ }) |

                        (ReducedDim { extent:_, pad_left:_, pad_right:_ },
                        FilledDim { dim:_, extent:_, stride:_, oob_left:_, oob_right:_, pad_left:_, pad_right:_ }) =>
                            false,
                    }
                });

            let mut nonvectorized_dims =
                (0..min(self.offset_map.num_dims(), other.offset_map.num_dims())).filter(|dim| {
                    !self.dims.iter().any(|dim2_content| match dim2_content {
                        FilledDim {
                            dim: dim2,
                            extent: _,
                            stride: _,
                            oob_left: _,
                            oob_right: _,
                            pad_left: _,
                            pad_right: _,
                        } => dim == dim2,

                        EmptyDim {
                            extent: _,
                            pad_left: _,
                            pad_right: _,
                            oob_right: _,
                        }
                        | ReducedDim {
                            extent: _,
                            pad_left: _,
                            pad_right: _,
                        } => false,
                    })
                });

            let nonvectorized_dims_equal_offsets =
                nonvectorized_dims.all(|dim| self.offset_map.get(dim) == other.offset_map.get(dim));

            if self.array != other.array || !dims_derivable || !nonvectorized_dims_equal_offsets {
                None
            } else {
                let mut block_size: usize = 1;
                let mut rotate_steps = 0;

                // tuple of (dim_size, mask_opt) for each dim
                // if mask_opt is None, nothing to mask;
                // otherwise, mask along interval (lo, hi) where mask_opt = Some((lo, hi))
                let mut mask_dim_info: Vec<(usize, Option<(usize, usize)>)> = Vec::new();

                self.dims.iter()
                .zip(other.dims.iter()).rev()
                .enumerate()
                .for_each(|(i, (dim1, dim2))| {
                    match (*dim1, *dim2) {
                        // this assumes that parent OOB and pad regions are zeroed out
                        // the masking here will prevent the contents of parent
                        // from being in the OOB region of the derived vector
                        (FilledDim {
                            dim: idim1, extent: extent1, stride: stride1,
                            oob_left: oob_left1, oob_right: oob_right1,
                            pad_left: pad_left1, pad_right: pad_right1,
                        },
                        FilledDim {
                            dim: idim2, extent: extent2, stride: stride2,
                            oob_left: oob_left2, oob_right: oob_right2,
                            pad_left: pad_left2, pad_right: pad_right2,
                        }) => {
                            let offset1 = self.offset_map[idim1];
                            let offset2 = other.offset_map[idim2];

                            let dim_steps =
                                (pad_left2 as isize) + (oob_left2 as isize) - (offset2 as isize) +
                                (offset1 as isize) - (oob_left1 as isize) - (pad_left1 as isize);

                            let dim_size = dim1.size();
                            rotate_steps += dim_steps * (block_size as isize);
                            block_size *= dim_size;

                            // assume that the dimensions are laid out in the parent like:
                            // wrapl_content1 | wrapl pad_right | pad_left1 | oob_left1 | content1 | oob_right1 | pad_right1 | wrapr pad_left1 | wrapr_content1

                            let wrapl_content1_hi =
                                dim_steps - (pad_right1 as isize) - 1;

                            let content1_lo =
                                dim_steps + ((pad_left1 + oob_left1) as isize);

                            let content1_hi =
                                dim_steps + ((pad_left1 + oob_left1 + extent1) as isize) - 1;

                            let wrapr_content1_lo =
                                dim_steps + ((dim_size + pad_left1) as isize);

                            let oob_left2_lo = pad_left2 as isize;
                            let oob_left2_hi = oob_left2_lo + (oob_left2 as isize);

                            let oob_right2_lo = (pad_left2 + oob_left2 + extent2) as isize;
                            let oob_right2_hi = oob_right2_lo + (oob_right2 as isize);

                            // check if wrapl_content1_hi or content1_lo
                            // intersects with the oob_left interval
                            let oob_left2_intersect =
                                wrapl_content1_hi >= oob_left2_lo ||
                                content1_lo <= oob_left2_hi;

                            // if the OOB left interval in derived vector is nonempty
                            // and parent's contents intersect with it, mask OOB left in derived
                            let mask_lo_opt =
                                if oob_left2_lo != oob_left2_hi && oob_left2_intersect {
                                    Some(pad_left2 + oob_left2)

                                } else {
                                    None
                                };

                            // check if content1_hi or wrapr_content1_lo
                            // intersects with the oob_left interval
                            let oob_right2_intersect =
                                content1_hi >= oob_right2_lo ||
                                wrapr_content1_lo <= oob_right2_hi;

                            // if the OOB right interval in derived vector is nonempty
                            // and parent's contents intersect with it, mask OOB right in derived
                            let mask_hi_opt =
                                if oob_right2_hi != oob_right2_lo && oob_right2_intersect {
                                    Some(dim_size - pad_right2 - oob_right2 - 1)

                                } else {
                                    None
                                };

                            let dim_mask =
                                match (mask_lo_opt, mask_hi_opt) {
                                    (None, None) => None,

                                    (None, Some(hi)) => Some((pad_left2, hi)),

                                    (Some(lo), None) => Some((lo, dim_size - pad_right2 - 1)),

                                    (Some(lo), Some(hi)) => Some((lo, hi))
                                };

                            mask_dim_info.push((dim_size, dim_mask));
                        },

                        (EmptyDim { extent: extent1, pad_left: pad_left1, pad_right: pad_right1, oob_right: oob_right1 },
                        EmptyDim { extent: extent2, pad_left: pad_left2, pad_right: pad_right2, oob_right: oob_right2 }) => {
                            let dim_size = pad_left1 + extent1 + pad_right1 + oob_right1;
                            block_size *= dim_size;

                            let dim_mask =
                                // don't mask anything
                                if pad_left1 == pad_left2 && pad_right1 + oob_right1 == pad_right2 + oob_right2 {
                                    None

                                } else {
                                    Some((pad_left2, dim_size - pad_right2 - oob_right2 -1))
                                };
                            mask_dim_info.push((dim_size, dim_mask));
                        },

                        (ReducedDim { extent: extent1, pad_left: pad_left1, pad_right: pad_right1 },
                        EmptyDim { extent: extent2, pad_left: pad_left2, pad_right: pad_right2, oob_right: oob_right2  }) => {
                            let dim_size = pad_left1 + extent1 + pad_right1;
                            block_size *= dim_size;

                            let dim_mask =
                                // don't mask anything
                                if pad_left1 == pad_left2 && pad_right1 == pad_right2 + oob_right2 {
                                    None

                                } else {
                                    Some((pad_left2, dim_size - pad_right2 - oob_right2 -1))
                                };

                            dims_to_fill.push(self.dims.len() - 1 -i);
                            mask_dim_info.push((dim_size, dim_mask));
                        },

                        (ReducedDim { extent:_, pad_left:_, pad_right:_ },
                        FilledDim { dim:_, extent:_, stride:_, oob_left:_, oob_right:_, pad_left:_, pad_right:_ }) |
                        (FilledDim { dim: _, extent: _, stride: _, oob_left: _, oob_right: _, pad_left: _, pad_right: _ },
                        EmptyDim { extent: _, pad_left: _, pad_right: _, oob_right: _ }) |
                        (EmptyDim { extent: _, pad_left: _, pad_right: _, oob_right: _ },
                        FilledDim { dim: _, extent: _, stride: _, oob_left: _, oob_right: _, pad_left: _, pad_right: _ }) |
                        (_, ReducedDim { extent: _, pad_left: _, pad_right: _ }) =>
                            unreachable!(),
                    }
                });

                let is_mask_const = mask_dim_info.iter().all(|(_, dim_mask)| dim_mask.is_none());

                let mask = if is_mask_const {
                    PlaintextObject::Const(1)
                } else {
                    let mask = mask_dim_info
                        .into_iter()
                        .map(|(dim_size, dim_mask)| match dim_mask {
                            Some((lo, hi)) => (dim_size, lo, hi),
                            None => (dim_size, 0, dim_size - 1),
                        })
                        .collect();

                    PlaintextObject::Mask(mask)
                };

                Some((rotate_steps, mask))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_derive() {
        // vec1 equals vec2
        let vec1 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: OffsetMap::new(2),
            dims: im::vector![],
        };

        assert!(vec1.derive(&vec1).is_some())
    }

    #[test]
    fn test_vector_derive1b() {
        // vec1: 0 1 2 3
        let vec1 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: OffsetMap::new(2),
            dims: im::vector![VectorDimContent::FilledDim {
                dim: 0,
                extent: 4,
                stride: 1,
                oob_left: 0,
                oob_right: 0,
                pad_left: 0,
                pad_right: 0,
            }],
        };

        // vec2: 0 1 2 3
        let offset2: OffsetMap<isize> = OffsetMap::new(2);
        let vec2 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: offset2,
            dims: im::vector![VectorDimContent::FilledDim {
                dim: 0,
                extent: 4,
                stride: 1,
                oob_left: 0,
                oob_right: 0,
                pad_left: 0,
                pad_right: 0,
            }],
        };

        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);

        assert_eq!(res.0, 0);
        assert_eq!(res.1, PlaintextObject::Const(1));
    }

    #[test]
    fn test_vector_derive2() {
        let vec1 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: OffsetMap::new(2),
            dims: im::vector![],
        };

        let mut offset2: OffsetMap<isize> = OffsetMap::new(2);
        offset2.set(0, 1);

        let vec2 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: offset2,
            dims: im::vector![],
        };

        // offsets of nonvectorized dims don't match, no derivation
        assert!(vec1.derive(&vec2).is_none())
    }

    #[test]
    fn test_vector_derive3() {
        // vec1: 0 1 2 3
        let vec1 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: OffsetMap::new(2),
            dims: im::vector![VectorDimContent::FilledDim {
                dim: 0,
                extent: 4,
                stride: 1,
                oob_left: 0,
                oob_right: 0,
                pad_left: 0,
                pad_right: 0,
            }],
        };

        let mut offset2: OffsetMap<isize> = OffsetMap::new(2);
        offset2.set(0, 1);

        // vec2: x 1 2 3
        let vec2 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: offset2,
            dims: im::vector![VectorDimContent::FilledDim {
                dim: 0,
                extent: 3,
                stride: 1,
                oob_left: 1,
                oob_right: 0,
                pad_left: 0,
                pad_right: 0,
            }],
        };

        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);

        // vec1 just needs to be masked to derive vec2
        assert_eq!(res.0, 0);
        assert_ne!(res.1, PlaintextObject::Const(1));
    }

    #[test]
    fn test_vector_derive4() {
        // vec1 = 0 1 2 3
        let vec1 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: OffsetMap::new(2),
            dims: im::vector![VectorDimContent::FilledDim {
                dim: 0,
                extent: 4,
                stride: 1,
                oob_left: 0,
                oob_right: 0,
                pad_left: 0,
                pad_right: 0,
            }],
        };

        // vec2 = 0 1 2 x
        let vec2 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: OffsetMap::new(2),
            dims: im::vector![VectorDimContent::FilledDim {
                dim: 0,
                extent: 3,
                stride: 1,
                oob_left: 0,
                oob_right: 1,
                pad_left: 0,
                pad_right: 0,
            }],
        };

        // vec can be masked to get vec2
        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);
        assert_eq!(res.0, 0);
        assert_ne!(res.1, PlaintextObject::Const(1));
    }

    #[test]
    fn test_vector_derive6() {
        // vec1: 0 1 2 3
        let vec1 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: OffsetMap::new(2),
            dims: im::vector![VectorDimContent::FilledDim {
                dim: 0,
                extent: 4,
                stride: 1,
                oob_left: 0,
                oob_right: 0,
                pad_left: 0,
                pad_right: 0,
            }],
        };

        // vec2: x 0 1 2
        let offset2: OffsetMap<isize> = OffsetMap::new(2);
        let vec2 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: offset2,
            dims: im::vector![VectorDimContent::FilledDim {
                dim: 0,
                extent: 3,
                stride: 1,
                oob_left: 1,
                oob_right: 0,
                pad_left: 0,
                pad_right: 0,
            }],
        };

        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);

        // rot(1, vec1) == rot2
        assert_eq!(res.0, 1);
        assert_ne!(res.1, PlaintextObject::Const(1));
    }

    #[test]
    fn test_vector_derive7() {
        // vec1: R X X X
        // (R is reduced value)
        let vec1 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: OffsetMap::new(2),
            dims: im::vector![
                VectorDimContent::FilledDim {
                    dim: 0,
                    stride: 1,
                    extent: 4,
                    pad_left: 0,
                    pad_right: 0,
                    oob_left: 0,
                    oob_right: 0,
                },
                VectorDimContent::ReducedDim {
                    extent: 4,
                    pad_left: 0,
                    pad_right: 0,
                },
            ],
        };

        let offset2: OffsetMap<isize> = OffsetMap::new(2);
        let vec2 = VectorInfo {
            array: String::from("a"),
            preprocessing: None,
            offset_map: offset2,
            dims: im::vector![
                VectorDimContent::FilledDim {
                    dim: 0,
                    stride: 1,
                    extent: 4,
                    pad_left: 0,
                    pad_right: 0,
                    oob_left: 0,
                    oob_right: 0,
                },
                VectorDimContent::EmptyDim {
                    extent: 4,
                    pad_left: 0,
                    pad_right: 0,
                    oob_right: 0,
                }
            ],
        };

        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);

        // rot(1, vec1) == rot2
        assert_eq!(res.0, 0);
        assert_eq!(res.1, PlaintextObject::Const(1));
    }
}
