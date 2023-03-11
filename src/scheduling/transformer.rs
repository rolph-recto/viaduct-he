use crate::util;

use super::*;

/// object that takes an input schedule and returns a set of "nearby" schedules.
pub trait ScheduleTransformer {
    fn transform(&mut self, schedule: &Schedule) -> HashSet<Schedule>;
}

/// a transformer that turns exploded dims to vectorized dims
pub struct VectorizeDimTransformer<'a> {
    dim_classes: &'a HashMap<(IndexingId, DimIndex), usize>
}

impl<'a> VectorizeDimTransformer<'a> {
    pub fn new(dim_classes: &'a HashMap<(IndexingId, DimIndex), usize>) -> Self {
        Self { dim_classes }
    }
}

impl<'a> ScheduleTransformer for VectorizeDimTransformer<'a> {
    fn transform(&mut self, schedule: &Schedule) -> HashSet<Schedule> {
        let mut neighbors = HashSet::new();

        // find candidate dims to be vectorized
        let mut candidate_dims: HashSet<(usize, usize, usize)> = HashSet::new();
        for (site, ischedule) in schedule.schedule_map.iter() {
            for edim in ischedule.exploded_dims.iter() {
                let class = self.dim_classes[&(site.clone(), edim.index)];
                candidate_dims.insert((class, edim.stride, edim.extent));
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

        let mut transformer = VectorizeDimTransformer::new(&dim_classes);
        let neighbors = transformer.transform(&schedule);
        assert_eq!(neighbors.len(), 2);

        for neighbor in neighbors {
            println!("neighbor:\n{}", neighbor)
        }
    }
}