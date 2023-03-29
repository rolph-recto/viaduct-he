use std::cmp::max;

use crate::util;

use super::*;

/// object that takes an input schedule and returns a set of "nearby" schedules.
pub trait ScheduleTransformer {
    fn name(&self) -> &str;
    fn transform(&self, schedule: &Schedule) -> HashSet<Schedule>;
    fn next_epoch(&mut self, _epoch: usize) {}
}

pub trait ScheduleTransformerFactory<'a> {
    fn create(&self, program: &InlinedProgram) -> Vec<Box<dyn ScheduleTransformer + 'a>>;
}

pub struct DefaultScheduleTransformerFactory;

impl<'a> ScheduleTransformerFactory<'a> for DefaultScheduleTransformerFactory {
    fn create(&self, program: &InlinedProgram) -> Vec<Box<dyn ScheduleTransformer + 'a>> {
        let dim_classes = program.get_dim_classes();
        let indexing_levels = program.get_indexing_levels();
        vec![
            Box::new(VectorizeDimTransformer::new(dim_classes.clone(), indexing_levels.clone())),
            Box::new(SplitDimTransformer::new(Some(2), dim_classes, indexing_levels.clone()))
        ]
    }
}

/// a transformer that turns exploded dims to vectorized dims
#[derive(Default)]
pub struct VectorizeDimTransformer {
    dim_classes: HashMap<(IndexingId, DimIndex), usize>,
    indexing_levels: HashMap<IndexingId, usize>,
}

impl VectorizeDimTransformer {
    pub fn new(
        dim_classes: HashMap<(IndexingId, DimIndex), usize>,
        indexing_levels: HashMap<IndexingId, usize>,
    ) -> Self {
        Self { dim_classes, indexing_levels }
    }
}

impl ScheduleTransformer for VectorizeDimTransformer {
    fn name(&self) -> &str { "VectorizeDimTransformer "}

    fn transform(&self, schedule: &Schedule) -> HashSet<Schedule> {
        let mut neighbors = HashSet::new();

        let cur_level: usize = 
            schedule.schedule_map.iter()
            .filter(|(_, isched)| {
                isched.get_tiling().iter().any(|dim| dim.len() > 1) ||
                isched.vectorized_dims.len() > 0
            })
            .fold(0, |acc, (site, _)| {
                max(acc, self.indexing_levels[site])
            });

        // find candidate dims to be vectorized
        let mut candidate_dims: HashSet<(usize, usize, usize)> = HashSet::new();
        for (site, ischedule) in schedule.schedule_map.iter() {
            if self.indexing_levels[site] >= cur_level { 
                let vectorized_dims: HashSet<DimIndex> =
                    ischedule.vectorized_dims.iter()
                    .map(|vdim| vdim.index)
                    .collect();

                for edim in ischedule.exploded_dims.iter() {
                    // don't add vectorization candidate if some tile of the dimension
                    // has already been vectorized
                    if !vectorized_dims.contains(&edim.index) {
                        let class = self.dim_classes[&(site.clone(), edim.index)];
                        candidate_dims.insert((class, edim.stride, edim.extent));
                    }
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
    num_dims_to_split: Option<usize>,
    dim_classes: HashMap<(IndexingId, DimIndex), usize>,
   indexing_levels: HashMap<IndexingId, usize>,
}

impl SplitDimTransformer {
    pub fn new(
        split_limit: Option<usize>,
        dim_classes: HashMap<(IndexingId, DimIndex), usize>,
        indexing_levels: HashMap<IndexingId, usize>,
    ) -> Self {
        Self { split_limit, num_dims_to_split: None, dim_classes, indexing_levels, }
    }
}

impl ScheduleTransformer for SplitDimTransformer {
    fn name(&self) -> &str { "SplitDimTransformer "}

    fn transform(&self, schedule: &Schedule) -> HashSet<Schedule> {
        let num_split_dims =
            schedule.schedule_map.iter()
            .fold(0, |acc,(_, isched)| {
                let isched_tiled_dims =
                    isched.get_tiling().iter().filter(|t| t.len() > 1).count();

                acc + isched_tiled_dims
            });

        let num_split_dims_within_limit =
            self.num_dims_to_split
            .map_or(true, |limit| num_split_dims <= limit);

        if !num_split_dims_within_limit {
            return HashSet::new()
        }

        let mut neighbors = HashSet::new();

        let cur_level: usize =
            schedule.schedule_map.iter()
            .filter(|(_, isched)| {
                isched.get_tiling().iter().any(|dim| dim.len() > 1) ||
                isched.vectorized_dims.len() > 0
            })
            .fold(0, |acc, (site, _)| {
                max(acc, self.indexing_levels[site])
            });


        // find candidate dims to be vectorized
        let mut candidate_dims: HashSet<(usize, usize, usize)> = HashSet::new();
        for (site, ischedule) in schedule.schedule_map.iter() {
            let tiling = ischedule.get_tiling();

            if self.indexing_levels[site] >= cur_level && num_split_dims_within_limit {
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

    // increase the number of dims to split by 1 per epoch
    fn next_epoch(&mut self, epoch: usize) {
        self.num_dims_to_split = Some(epoch - 1);
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

        let indexing_levels = HashMap::from([
            (String::from("a1"), 1),
            (String::from("b1"), 1),
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

        let transformer = VectorizeDimTransformer::new(dim_classes, indexing_levels);
        let neighbors = transformer.transform(&schedule);
        assert_eq!(neighbors.len(), 2);

        for neighbor in neighbors {
            println!("neighbor:\n{}", neighbor)
        }
    }
}