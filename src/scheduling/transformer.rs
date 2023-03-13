use crate::util::{self, NameGenerator};

use super::*;

/// object that takes an input schedule and returns a set of "nearby" schedules.
pub trait ScheduleTransformer {
    fn transform(&self, schedule: &Schedule) -> HashSet<Schedule>;
}

pub trait ScheduleTransformerFactory<'a> {
    fn create(&self, program: &InlinedProgram) -> Vec<Box<dyn ScheduleTransformer + 'a>>;
}

pub struct DefaultScheduleTransformerFactory;

impl<'a> ScheduleTransformerFactory<'a> for DefaultScheduleTransformerFactory {
    fn create(&self, program: &InlinedProgram) -> Vec<Box<dyn ScheduleTransformer + 'a>> {
        let dim_classes = program.get_dim_equiv_classes();
        vec![
            Box::new(VectorizeDimTransformer::new(dim_classes.clone())),
            Box::new(SplitDimTransformer::new(Some(2), dim_classes))
        ]
    }
}

/// transformer factory that only has the vectorization transformer, for speed
/// (split dim transformer blows up the search space)
pub struct FastScheduleTransformerFactory;

impl<'a> ScheduleTransformerFactory<'a> for FastScheduleTransformerFactory {
    fn create(&self, program: &InlinedProgram) -> Vec<Box<dyn ScheduleTransformer + 'a>> {
        let dim_classes = program.get_dim_equiv_classes();
        vec![
            Box::new(VectorizeDimTransformer::new(dim_classes.clone())),
        ]
    }
}

/// a transformer that turns exploded dims to vectorized dims
#[derive(Default)]
pub struct VectorizeDimTransformer {
    dim_classes: HashMap<(IndexingId, DimIndex), usize>
}

impl VectorizeDimTransformer {
    pub fn new(dim_classes: HashMap<(IndexingId, DimIndex), usize>) -> Self {
        Self { dim_classes }
    }
}

impl ScheduleTransformer for VectorizeDimTransformer {
    fn transform(&self, schedule: &Schedule) -> HashSet<Schedule> {
        let mut neighbors = HashSet::new();

        // find candidate dims to be vectorized
        let mut candidate_dims: HashSet<(usize, usize, usize)> = HashSet::new();
        for (site, ischedule) in schedule.schedule_map.iter() {
            let last_vdim_index =
                ischedule.vectorized_dims.last()
                .map(|vdim| vdim.index);

            for edim in ischedule.exploded_dims.iter() {
                // don't add vectorization candidate if it will be next
                // to a dim of the same index
                let adjacent_same_index =
                    last_vdim_index
                    .map_or(false, |index| edim.index == index);

                if !adjacent_same_index {
                    let class = self.dim_classes[&(site.clone(), edim.index)];
                    candidate_dims.insert((class, edim.stride, edim.extent));
                }
            }
        }

        for (class, stride, extent) in candidate_dims {
            let mut new_schedule_map: im::HashMap<IndexingId, IndexingSiteSchedule> = im::HashMap::new();
            for (site, ischedule) in schedule.schedule_map.iter() {
                let mut new_site_schedule =
                    IndexingSiteSchedule {
                        preprocessing: ischedule.preprocessing,
                        exploded_dims: im::Vector::new(),
                        vectorized_dims: ischedule.vectorized_dims.clone(),
                    };

                for edim in ischedule.exploded_dims.iter() {
                    let eclass = self.dim_classes[&(site.clone(), edim.index)];

                    // vectorize the exploded dim
                    if class == eclass && edim.stride == stride && edim.extent == extent {
                        let vec_extent = util::get_nearest_pow2(extent);
                        let new_sched_dim = ScheduleDim {
                            index: edim.index,
                            stride,
                            extent: vec_extent,
                            name: edim.name.clone(),
                            pad_left: edim.pad_left,
                            pad_right: edim.pad_right,
                        };
                        new_site_schedule.vectorized_dims.push_back(new_sched_dim);

                    } else { // keep the exploded dim in the same place
                        new_site_schedule.exploded_dims.push_back(edim.clone());
                    }
                }

                new_schedule_map.insert(site.clone(), new_site_schedule);
            }

            neighbors.insert(
                Schedule {
                    schedule_map: new_schedule_map
                }
            );
        }

        neighbors
    }
}

pub struct SplitDimTransformer {
    split_limit: Option<usize>,
    dim_classes: HashMap<(IndexingId, DimIndex), usize>,
}

impl SplitDimTransformer {
    pub fn new(
        split_limit: Option<usize>,
        dim_classes: HashMap<(IndexingId, DimIndex), usize>
    ) -> Self {
        Self { split_limit, dim_classes }
    }
}

impl ScheduleTransformer for SplitDimTransformer {
    fn transform(&self, schedule: &Schedule) -> HashSet<Schedule> {
        let mut neighbors = HashSet::new();

        // find candidate dims to be vectorized
        let mut candidate_dims: HashSet<(usize, usize, usize)> = HashSet::new();
        for (site, ischedule) in schedule.schedule_map.iter() {
            let tiling = ischedule.get_tiling();
            for edim in ischedule.exploded_dims.iter() {
                let dim_tiling = tiling[edim.index].len();

                // the dimension has not been split beyond the limit (if there is one)
                let within_tiling_limit = 
                    self.split_limit.map_or(true, |l| dim_tiling < l);

                // only split innermost dims (stride = 1)
                let innermost_dim = edim.stride == 1;

                if within_tiling_limit && innermost_dim {
                    let class = self.dim_classes[&(site.clone(), edim.index)];
                    candidate_dims.insert((class, edim.stride, edim.extent));
                }
            }
        }

        for (class, stride, extent) in candidate_dims {
            let factor_pairs = util::get_factor_pairs(extent);
            for (f_in, f_out) in factor_pairs {
                let mut new_schedule_map: im::HashMap<IndexingId, IndexingSiteSchedule> = im::HashMap::new();
                for (site, ischedule) in schedule.schedule_map.iter() {
                    let mut new_site_schedule =
                        IndexingSiteSchedule {
                            preprocessing: ischedule.preprocessing,
                            exploded_dims: im::Vector::new(),
                            vectorized_dims: ischedule.vectorized_dims.clone(),
                        };

                    for edim in ischedule.exploded_dims.iter() {
                        let eclass = self.dim_classes[&(site.clone(), edim.index)];

                        // split dimension
                        // given dim with stride s and extent e, create two new dims
                        // where s = 1, e = e_in * e_out, s_in = s, s_out = s * e_in
                        if class == eclass && edim.stride == stride && edim.extent == extent {
                            new_site_schedule.exploded_dims.push_back(
                                ScheduleDim {
                                    index: edim.index,
                                    stride: edim.stride * f_in,
                                    extent: f_out,
                                    name: format!("{}o", edim.name),
                                    pad_left: edim.pad_left,
                                    pad_right: edim.pad_right
                                }
                            );
                            new_site_schedule.exploded_dims.push_back(
                                ScheduleDim {
                                    index: edim.index,
                                    stride: edim.stride,
                                    extent: f_in,
                                    name: format!("{}i", edim.name),
                                    pad_left: edim.pad_left,
                                    pad_right: edim.pad_right
                                }
                            );

                        } else { // keep the exploded dim in the same place
                            new_site_schedule.exploded_dims.push_back(edim.clone());
                        }
                    }

                    new_schedule_map.insert(site.clone(), new_site_schedule);
                }

                neighbors.insert(
                    Schedule {
                        schedule_map: new_schedule_map
                    }
                );
            }
        }

        neighbors
    }
}


// transformers:
// - dimension split transformer
// - permute vectorize dim transformer
// - permute preprocessing transformer

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorize_dim() {
        let dim_classes = HashMap::from([
            ((String::from("a1"), 0), 0),
            ((String::from("a1"), 1), 1),
            ((String::from("b1"), 0), 0),
            ((String::from("b1"), 1), 1),
        ]);

        let mut schedule_map: im::HashMap<IndexingId, IndexingSiteSchedule> = im::HashMap::new();
        schedule_map.insert(
            String::from("a1"),
            IndexingSiteSchedule {
                preprocessing:  None,
                exploded_dims: im::Vector::from(vec![
                    ScheduleDim {
                        index: 0,
                        stride: 1,
                        extent: 4,
                        name: String::from("i0"),
                        pad_left: 0,
                        pad_right: 0,
                    },
                    ScheduleDim {
                        index: 1,
                        stride: 1,
                        extent: 4,
                        name: String::from("i1"),
                        pad_left: 0,
                        pad_right: 0,
                    },
                ]),
                vectorized_dims: im::Vector::new(),
            }
        );
        schedule_map.insert(
            String::from("b1"),
            IndexingSiteSchedule {
                preprocessing:  None,
                exploded_dims: im::Vector::from(vec![
                    ScheduleDim {
                        index: 0,
                        stride: 1,
                        extent: 4,
                        name: String::from("i0"),
                        pad_left: 0,
                        pad_right: 0,
                    },
                    ScheduleDim {
                        index: 1,
                        stride: 1,
                        extent: 4,
                        name: String::from("i1"),
                        pad_left: 0,
                        pad_right: 0,
                    },
                ]),
                vectorized_dims: im::Vector::new(),
            }
        );

        let schedule = Schedule { schedule_map };

        let transformer = VectorizeDimTransformer::new(dim_classes);
        let neighbors = transformer.transform(&schedule);
        assert_eq!(neighbors.len(), 2);

        for neighbor in neighbors {
            println!("neighbor:\n{}", neighbor)
        }
    }
}