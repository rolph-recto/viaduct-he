use std::{fmt::Display, collections::{HashMap, HashSet}, cmp::min};

use gcollections::ops::Bounded;

use crate::{
    circ2::PlaintextObject,
    lang::{DimIndex, ArrayName, OffsetMap, Shape, ArrayTransform, DimContent},
    scheduling::{ClientPreprocessing, DimName, ArraySchedule}
};

// like DimContent, but with padding information
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum VectorDimContent {
    // we assume that dimensions have the following structure
    // | pad_left | oob_left | extent | oob_right | pad_right |
    FilledDim {
        dim: DimIndex, extent: usize, stride: usize,

        // out of bounds intervals
        oob_left: usize, oob_right: usize,
        
        // padding
        pad_left: usize, pad_right: usize,
    },

    EmptyDim { extent: usize, pad_left: usize, pad_right: usize, },
}

impl Display for VectorDimContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorDimContent::FilledDim {
                dim, extent, stride, oob_left, oob_right, pad_left, pad_right
            } => {
                let mut extra: String = String::new();

                if *oob_left != 0 || *oob_right != 0 {
                    extra.push_str(&format!("[oob=({},{})]", oob_left, oob_right));
                }

                if *pad_left != 0 || *pad_right != 0 {
                    extra.push_str(&format!("[pad=({},{})]", pad_left, pad_right));
                }

                write!(f, "{{{}:{}::{}{}}}", dim, extent, stride, extra)
            },

            VectorDimContent::EmptyDim { extent, pad_left, pad_right } => {
                if *pad_left != 0 || *pad_right != 0 {
                    write!(f, "{{{}[pad=({},{})]}}", extent, pad_left, pad_right)

                } else {
                    write!(f, "{{{}}}", extent)
                }
            },
        }
    }
}

impl VectorDimContent {
    pub fn size(&self) -> usize {
        match self {
            VectorDimContent::FilledDim {
                dim, extent, stride,
                oob_left, oob_right,
                pad_left, pad_right
            } => {
                pad_left + oob_left + extent + oob_right + pad_right
            }

            VectorDimContent::EmptyDim { extent, pad_left, pad_right } => {
                pad_left + (*extent) + pad_right
            }
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
        if let Some(preprocess) = self.preprocessing {
            write!(f, "{}({})[{}]<{}>",
                self.array,
                preprocess,
                self.offset_map,
                self.dims.iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )
        } else {
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
}

impl VectorInfo {
    // retrieve vector at specific coordinate of a scheduled array
    pub fn get_vector_at_coord(
        array_shape: &Shape,
        index_map: &HashMap<DimName, usize>,
        schedule: &ArraySchedule,
        transform: &ArrayTransform,
        preprocessing: Option<ClientPreprocessing>,
    ) -> Self {
        let mut clipped_offset_map =
            schedule.get_offset_map(transform)
            .map(|offset| offset.eval(index_map));

        let mut materialized_dims: im::Vector<VectorDimContent> = im::Vector::new();
        for dim in schedule.vectorized_dims.iter() {
            let dim_content = transform.dims.get(dim.index).unwrap();
            let materialized_dim = 
                match dim_content {
                    // clip dimension to (0, array dimension's extent)
                    DimContent::FilledDim { dim: idim, extent: _, stride: istride } => {
                        let dim_offset = *clipped_offset_map.get(*idim);
                        let array_extent = array_shape[*idim].upper() as usize;
                        let new_stride = *istride * dim.stride;

                        // indexing less than 0; clip
                        let mut oob_left: usize = 0;
                        while dim_offset + ((oob_left * new_stride) as isize) < 0 {
                            oob_left += 1;
                        }
                        
                        // indexing beyond array_extent; clip
                        let mut oob_right: usize = 0;
                        while dim_offset + ((new_stride * (dim.extent - 1 - oob_right)) as isize) >= array_extent as isize {
                            oob_right += 1;
                        }

                        // increment offset by oob_left
                        let cur_offset = *clipped_offset_map.get(*idim);
                        clipped_offset_map.set(*idim, cur_offset + ((oob_left * new_stride) as isize));
                        
                        let new_extent = dim.extent - oob_left - oob_right;

                        VectorDimContent::FilledDim {
                            dim: *idim,
                            extent: new_extent,
                            stride: new_stride,
                            oob_left,
                            oob_right,
                            pad_left: dim.pad_left,
                            pad_right: dim.pad_right,
                        }
                    },

                    DimContent::EmptyDim { extent: _ } => {
                        VectorDimContent::EmptyDim {
                            extent: dim.extent,
                            pad_left: dim.pad_left,
                            pad_right: dim.pad_right,
                        }
                    }
                };

            materialized_dims.push_back(materialized_dim);
        }

        VectorInfo {
            array: transform.array.clone(),
            preprocessing,
            offset_map: clipped_offset_map.map(|offset| *offset as usize),
            dims: materialized_dims,
        }
    }

    // derive other from self
    pub fn derive(&self, other: &VectorInfo) -> Option<(isize, PlaintextObject)> {
        if self.dims.len() != other.dims.len() {
            None

        } else if self == other {
            Some((0, PlaintextObject::Const(1)))

        } else {
            let mut seen_dims: HashSet<DimIndex> = HashSet::new();

            // check derivability conditions
            let dims_derivable = 
                self.dims.iter()
                .zip(other.dims.iter())
                .all(|(dim1, dim2)| {
                    match (*dim1, *dim2) {
                        (VectorDimContent::FilledDim {
                            dim: idim1, extent: extent1, stride: stride1,
                            oob_left: oob_left1, oob_right: oob_right1,
                            pad_left: pad_left1, pad_right: pad_right1,
                        },
                        VectorDimContent::FilledDim {
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
                                offset1 % (stride1 as usize) == offset2 % (stride2 as usize);

                            // the dimensions have the same size
                            let same_size = dim1.size() == dim2.size();

                            // all of the elements of other's dim is in self's dim
                            let in_extent =
                                offset2 >= offset1 &&
                                offset1 + (stride1 * extent1) >= offset2 + (stride2 * extent2);

                            // self cannot have out of bounds values
                            let self_no_oob = oob_left1 == 0 && oob_right1 == 0;

                            seen_dims.insert(idim1);
                            same_dim && dim_unseen && same_stride && offset_equiv &&
                            same_size && in_extent && self_no_oob
                        },
                        
                        (VectorDimContent::EmptyDim { extent: extent1, pad_left: pad_left1, pad_right: pad_right1 },
                        VectorDimContent::EmptyDim { extent: extent2, pad_left: pad_left2, pad_right: pad_right2  }) => {
                            pad_left1 + extent1 + pad_right1 == pad_left2 + extent2 + pad_right2 &&

                            // can always truncate empty dims with more padding,
                            // but don't support extending dims (yet)
                            extent1 >= extent2
                        },

                        (VectorDimContent::FilledDim { dim: _, extent: _, stride: _, oob_left: _, oob_right: _, pad_left: _, pad_right: _ },
                            VectorDimContent::EmptyDim { extent: _, pad_left: _, pad_right: _ }) |
                        (VectorDimContent::EmptyDim { extent: _, pad_left: _, pad_right: _ },
                            VectorDimContent::FilledDim { dim: _, extent: _, stride: _, oob_left: _, oob_right: _, pad_left: _, pad_right: _ })
                        => false,
                    }
                });

            let mut nonvectorized_dims =
                (0..min(self.offset_map.num_dims(), other.offset_map.num_dims()))
                .filter(|dim| {
                    !self.dims.iter().any(|dim2_content| {
                        match dim2_content {
                            VectorDimContent::FilledDim {
                                dim: dim2, extent: _, stride: _,
                                oob_left: _, oob_right: _, pad_left: _, pad_right: _,
                            } => dim == dim2,

                            VectorDimContent::EmptyDim { extent: _, pad_left: _, pad_right : _ } => false,
                        }
                    })
                });

            let nonvectorized_dims_equal_offsets = 
                nonvectorized_dims.all(|dim| {
                    self.offset_map.get(dim) == other.offset_map.get(dim)
                });

            if self.array == other.array && dims_derivable && nonvectorized_dims_equal_offsets {
                let mut block_size: usize = 1;
                let mut rotate_steps = 0;
                let mut mask_dim_info: im::Vector<(usize, usize, usize)> = im::Vector::new();

                self.dims.iter()
                .zip(other.dims.iter()).rev()
                .for_each(|(dim1, dim2)| {
                    match (*dim1, *dim2) {
                        (VectorDimContent::FilledDim {
                            dim: idim1, extent: extent1, stride: stride1,
                            oob_left: oob_left1, oob_right: oob_right1,
                            pad_left: pad_left1, pad_right: pad_right1,
                        },
                        VectorDimContent::FilledDim {
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

                            let oob_left2_lo = pad_left2;
                            let oob_left2_hi = oob_left2_lo + oob_left2;

                            let oob_left2_lo_intersect =
                                dim_steps - ((pad_right1 + pad_left1) as isize) > oob_left2_lo as isize;

                            let oob_left2_hi_intersect =
                                dim_steps + (pad_left1 as isize) < oob_left2_hi as isize;

                            let need_mask_left =
                                oob_left2_lo != oob_left2_hi &&
                                (oob_left2_lo_intersect || oob_left2_hi_intersect);

                            let mask_lo =
                                if need_mask_left {
                                    pad_left2 + oob_left2

                                } else {
                                    pad_left2
                                };

                            let oob_right2_lo = pad_left2 + oob_left2 + extent2;
                            let oob_right2_hi = oob_right2_lo + oob_right2;

                            let oob_right2_lo_intersect =
                                dim_steps + ((pad_left1 + extent1) as isize) > oob_right2_lo as isize;

                            let oob_right2_hi_intersect =
                                dim_steps + ((pad_left1 + extent1 + pad_right1 + pad_left1) as isize) < oob_right2_hi as isize;

                            let need_mask_right =
                                oob_right2_hi != oob_right2_lo &&
                                (oob_right2_lo_intersect || oob_right2_hi_intersect);

                            let mask_hi = 
                                if need_mask_right {
                                    dim_size - pad_right2 - oob_right2 - 1

                                } else {
                                    dim_size - pad_right2 - 1
                                };

                            let mask_dim =
                                // don't mask anything
                                if mask_lo == pad_left2 && mask_hi == dim_size - pad_right2 - 1 {
                                    (dim_size, 0, dim_size-1)

                                } else { // mask the OOB region
                                    (dim_size, mask_lo, mask_hi)
                                };

                            mask_dim_info.push_back(mask_dim);
                        },
                        
                        (VectorDimContent::EmptyDim { extent: extent1, pad_left: pad_left1, pad_right: pad_right1 },
                        VectorDimContent::EmptyDim { extent: extent2, pad_left: pad_left2, pad_right: pad_right2 }) => {
                            let dim_size = pad_left1 + extent1 + pad_right1;
                            block_size *= dim_size;

                            let mask_dim =
                                // don't mask anything
                                if pad_left1 == pad_left2 && pad_right1 == pad_right2 {
                                    (dim_size, 0, dim_size - 1)

                                } else {
                                    (dim_size, pad_left2, pad_right2)
                                };
 
                            mask_dim_info.push_back(mask_dim);
                        },

                        (VectorDimContent::FilledDim { dim: _, extent: _, stride: _, oob_left: _, oob_right: _, pad_left: _, pad_right: _ },
                            VectorDimContent::EmptyDim { extent: _, pad_left: _, pad_right: _ }) |
                        (VectorDimContent::EmptyDim { extent: _, pad_left: _, pad_right: _ },
                            VectorDimContent::FilledDim { dim: _, extent: _, stride: _, oob_left: _, oob_right: _, pad_left: _, pad_right: _ })
                        => unreachable!()
                    }
                });

                let is_mask_const =
                    mask_dim_info.iter().all(|(size, lo, hi)| {
                        *lo == 00 && *hi == size - 1
                    });

                let mask =
                    if is_mask_const {
                        PlaintextObject::Const(1)

                    } else {
                        PlaintextObject::Mask(mask_dim_info)
                    };

                Some((rotate_steps, mask))

            } else {
                None
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
        let vec1 = 
            VectorInfo {
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
        let vec1 = 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 4,
                        stride: 1,
                        oob_left: 0, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        // vec2: 0 1 2 3
        let offset2: OffsetMap<usize> = OffsetMap::new(2);
        let vec2= 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: offset2,
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 4,
                        stride: 1,
                        oob_left: 0, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);

        assert_eq!(res.0, 0);
        assert_eq!(res.1, PlaintextObject::Const(1));
    }


    #[test]
    fn test_vector_derive2() {
        let vec1 = 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![],
            };

        let mut offset2: OffsetMap<usize> = OffsetMap::new(2);
        offset2.set(0, 1);

        let vec2= 
            VectorInfo {
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
        let vec1 = 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 4,
                        stride: 1,
                        oob_left: 0, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        let mut offset2: OffsetMap<usize> = OffsetMap::new(2);
        offset2.set(0, 1);

        // vec2: x 1 2 3
        let vec2= 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: offset2,
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 3,
                        stride: 1,
                        oob_left: 1, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
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
        let vec1 = 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 4,
                        stride: 1,
                        oob_left: 0, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        // vec2 = 0 1 2 x
        let vec2= 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 3,
                        stride: 1,
                        oob_left: 0, oob_right: 1,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        // vec can be masked to get vec2
        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);
        assert_eq!(res.0, 0);
        assert_ne!(res.1, PlaintextObject::Const(1));
    }

    #[test]
    fn test_vector_derive5() {
        // vec1: 0 1 2 3
        let vec1 = 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 4,
                        stride: 1,
                        oob_left: 0, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        // vec2: x 1 2 3
        let mut offset2: OffsetMap<usize> = OffsetMap::new(2);
        offset2.set(0, 1);

        let vec2= 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: offset2,
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 3,
                        stride: 1,
                        oob_left: 1, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);

        // vec1 just needs to be masked to derive vec2, no rotation required
        assert_eq!(res.0, 0);
        assert_ne!(res.1, PlaintextObject::Const(1));
    }

    #[test]
    fn test_vector_derive6() {
        // vec1: 0 1 2 3
        let vec1 = 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 4,
                        stride: 1,
                        oob_left: 0, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        // vec2: x 0 1 2
        let offset2: OffsetMap<usize> = OffsetMap::new(2);
        let vec2= 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: offset2,
                dims: im::vector![
                    VectorDimContent::FilledDim {
                        dim: 0,
                        extent: 3,
                        stride: 1,
                        oob_left: 1, oob_right: 0,
                        pad_left: 0, pad_right: 0,
                    }
                ],
            };

        let res = vec1.derive(&vec2).unwrap();
        println!("{:?}", res);

        // rot(1, vec1) == rot2
        assert_eq!(res.0, 1);
        assert_ne!(res.1, PlaintextObject::Const(1));
    }
}