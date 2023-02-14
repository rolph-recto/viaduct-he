use std::{fmt::Display, collections::{HashMap, HashSet}, cmp::max};

use gcollections::ops::Bounded;

use crate::{lang::{DimIndex, ArrayName, OffsetMap, Shape, BaseArrayTransform, DimContent}, scheduling::{ClientPreprocessing, DimName, ScheduledArrayTransform}};

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
    // retrieve vector at specific coordinate of a scheduled array
    pub fn get_vector_at_coord(
        array_shape: &Shape,
        index_map: &HashMap<DimName, usize>,
        scheduled: &ScheduledArrayTransform,
        preprocessing: Option<ClientPreprocessing>,
    ) -> Self {
        let base_offset_map =
            scheduled.transform.offset_map
            .map(|offset| offset.eval(index_map));

        let transform = 
            BaseArrayTransform {
                array: scheduled.transform.array.clone(),
                offset_map: base_offset_map,
                dims: scheduled.transform.dims.clone(),
            };

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
            preprocessing,
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

            let mut nonvectorized_dims =
                (0..self.offset_map.num_dims())
                .filter(|dim| {
                    !self.dims.iter().any(|dim2_content| {
                        match dim2_content {
                            VectorDimContent::FilledDim {
                                dim: dim2, extent: _, stride: _, pad_left: _, pad_right: _
                            } => dim == dim2,

                            VectorDimContent::EmptyDim { extent: _ } => false,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_derive() {
        let vec1 = 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![],
            };

        let vec2= 
            VectorInfo {
                array: String::from("a"),
                preprocessing: None,
                offset_map: OffsetMap::new(2),
                dims: im::vector![],
            };

        assert!(vec1.derive(&vec2).is_some())
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
                        pad_left: 0,
                        pad_right: 0,
                    }
                ],
            };

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
                        pad_left: 1,
                        pad_right: 0,
                    }
                ],
            };

        // vec can be rotated to the right 1 step to get vec2
        assert!(vec1.derive(&vec2).is_some())
    }

    #[test]
    fn test_vector_derive4() {
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
                        pad_left: 0,
                        pad_right: 0,
                    }
                ],
            };

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
                        pad_left: 0,
                        pad_right: 1,
                    }
                ],
            };

        // vec can be masked to get vec2
        assert!(vec1.derive(&vec2).is_some())
    }
}