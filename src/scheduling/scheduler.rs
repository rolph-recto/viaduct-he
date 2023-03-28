use std::{collections::{HashSet, HashMap}, time::Instant};

use log::{info, debug};

use crate::{
    lang::{index_elim::{InlinedProgram}, DimIndex},
    circ::{materializer::MaterializerFactory, cost::{CostEstimator, CostFeatures},
    pseudomaterializer::{PseudoMaterializer, PseudoMaterializerFactory}},
    scheduling::{
        Schedule,
        transformer::{ScheduleTransformer, ScheduleTransformerFactory}
    },
};

use super::ScheduleDerivationFailure;

struct InlineScheduler<'a> {
    pub program: InlinedProgram,
    pub transformers: Vec<Box<dyn ScheduleTransformer + 'a>>,
    pub visited: HashSet<Schedule>,
    pub frontier: HashSet<Schedule>,
    pub valid_schedules_visited: usize,
    pub vector_size: usize,
}

impl<'t> InlineScheduler<'t> {
    pub fn new(
        program: InlinedProgram,
        transformers: Vec<Box<dyn ScheduleTransformer + 't>>,
        init_schedule: Schedule,
        vector_size: usize,
    ) -> Self {
        Self {
            program,
            transformers,
            vector_size,
            visited: HashSet::from([init_schedule.clone()]),
            frontier: HashSet::from([init_schedule]),
            valid_schedules_visited: 1,
        }
    }

    // check if this the cost for this schedule should be estimated
    // if not, it will just be recorded as visited
    // this will allow visits of schedules we need to find other schedules
    // we actually want to remember
    fn should_estimate_cost(&self, schedule: &Schedule) -> bool {
        // if a dimension has multiple exploded tiles, don't estimate its cost! 
        for (_, ischedule) in schedule.schedule_map.iter() {
            let mut tile_count_map: HashMap<DimIndex, usize> = HashMap::new();
            for edim in ischedule.exploded_dims.iter() {
                if tile_count_map.contains_key(&edim.index) {
                    *tile_count_map.get_mut(&edim.index).unwrap() += 1;

                } else {
                    tile_count_map.insert(edim.index, 1);
                }
            }

            if tile_count_map.iter().any(|(_, count)| *count > 1) {
                return false
            }
        }

        true
    }

    pub fn iterate<'m>(
        &mut self,
        materializer_factory: &dyn PseudoMaterializerFactory,
        cost_estimator: &CostEstimator,
    ) -> (usize, Vec<(Schedule, CostFeatures)>) {
        let mut cur: HashSet<Schedule> = HashSet::new();
        std::mem::swap(&mut self.frontier, &mut cur);
        let mut valid_neighbors: Vec<(Schedule, CostFeatures)> = Vec::new();
        for schedule in cur {
            for transformer in self.transformers.iter() {
                let neighbors = transformer.transform(&schedule);
                for neighbor in neighbors {
                    if !self.visited.contains(&neighbor) {
                        // neighbor is a newly visited schedule;
                        // try to materialize it into a circuit

                        // if the schedule can be materialized into a circuit,
                        // give a cost to the schedule and save it if it's
                        // in the Pareto frontier
                        let valid = neighbor.is_schedule_valid(&self.program);
                        let within_size_limit =
                            neighbor.schedule_map.iter().all(|(_, isched)| {
                                isched.vector_size() <= self.vector_size
                            });
                        let should_estimate = self.should_estimate_cost(&neighbor);

                        debug!("neighbor: {}; valid? {:?} within_size? {} should estimate? {}", neighbor, valid, within_size_limit, should_estimate);
                        if should_estimate && valid.is_ok() && within_size_limit
                        {
                            let mat = materializer_factory.create();
                            if let Ok(circuit) = mat.run(&self.program, &neighbor) {
                                // let cost = cost_estimator.estimate_cost(&circuit);
                                let cost = cost_estimator.estimate_pseudo_cost(&circuit);
                                debug!("neighbor cost {:?}", cost);

                                valid_neighbors.push((neighbor.clone(), cost));
                                self.valid_schedules_visited += 1;
                            }
                        }

                        if let Err(ScheduleDerivationFailure::MaybeTransformableToValid) | Ok(()) = valid {
                            self.frontier.insert(neighbor.clone());
                            self.visited.insert(neighbor);
                        }
                    }
                }
            }
        }

        (self.frontier.len(), valid_neighbors)
    }
}

pub struct SchedulingResult {
    pub inlined_program: InlinedProgram,
    pub pareto_frontier: Vec<(Schedule, CostFeatures)>,
    pub visited: HashSet<Schedule>,
    pub valid_schedules_visited: usize,
}

/// scheduler for a particular inlined program
/// (identified by array group map and inline set)
pub struct Scheduler<'m, 't> {
    materializer_factory: Box<dyn PseudoMaterializerFactory + 'm>,
    cost_estimator: CostEstimator,
    inline_schedulers: HashMap<usize, (bool, InlineScheduler<'t>)>,

    // schedules in the Pareto frontier
    // (i.e. no visited schedule is strictly cheaper than these)
    pareto_frontier: HashMap<(usize, Schedule), CostFeatures>
}

impl<'m, 't> Scheduler<'m, 't> {
    pub fn new(
        inlined_programs: Vec<InlinedProgram>,
        transformer_factory: Box<dyn ScheduleTransformerFactory<'t> + 't>,
        materializer_factory: Box<dyn PseudoMaterializerFactory + 'm>,
        vector_size: usize,
    ) -> Self {
        let cost_estimator = CostEstimator::default();
        let mut init_schedules: Vec<(usize, Schedule, CostFeatures)> = Vec::new();
        let mut inline_schedulers: HashMap<usize, (bool, InlineScheduler)> = HashMap::new();
        let mut inline_id = 1;

        for inlined_program in inlined_programs {
            let init_schedule =
                Schedule::gen_initial_schedule(&inlined_program);

            // let mat = materializer_factory.create();
            // let mat_res = mat.run(&inlined_program, &init_schedule);
            let mat = materializer_factory.create();
            let mat_res = mat.run(&inlined_program, &init_schedule);

            if let Ok(circuit) = mat_res {
                // let cost = cost_estimator.estimate_cost(&circuit);
                let cost = cost_estimator.estimate_pseudo_cost(&circuit);
                init_schedules.push((inline_id, init_schedule.clone(), cost));
            }

            let transformers =
                transformer_factory.create(&inlined_program);

            let inline_scheduler =
                InlineScheduler::new(
                    inlined_program,
                    transformers,
                    init_schedule,
                    vector_size
                );

            inline_schedulers.insert(inline_id, (true, inline_scheduler));
            inline_id += 1;
        }

        let mut scheduler = 
            Self {
                materializer_factory,
                cost_estimator,
                inline_schedulers,
                pareto_frontier: HashMap::new(),
            };

        info!("initializing pareto frontier...");
        for (inline_id, init_schedule, cost) in init_schedules {
            scheduler.update_pareto_frontier(inline_id, init_schedule, cost);
        }

        info!("inline schedulers: {:?}", scheduler.inline_schedulers.len());
        info!("init pareto frontier size: {:?}", scheduler.pareto_frontier.len());

        scheduler
    }

    pub fn get_results(&self) -> Vec<SchedulingResult> {
        let mut pareto_map: HashMap<usize, Vec<(Schedule, CostFeatures)>> = HashMap::new();

        for (id, _) in self.inline_schedulers.iter() {
            pareto_map.insert(*id, vec![]);
        }

        for ((id, schedule), cost) in self.pareto_frontier.iter() {
            let schedule_list = pareto_map.get_mut(&id).unwrap();
            schedule_list.push((schedule.clone(), cost.clone()));
        }

        pareto_map.into_iter().map(|(id, schedule_list)| {  
            let (_, inline_sched) = self.inline_schedulers.get(&id).unwrap();

            SchedulingResult {
                inlined_program: inline_sched.program.clone(),
                visited: inline_sched.visited.clone(),
                pareto_frontier: schedule_list,
                valid_schedules_visited: inline_sched.valid_schedules_visited,
            }
        }).collect()
    }

    // TODO finish
    pub fn get_best_schedule(
        &self,
        weights: CostFeatures
    ) -> Option<(InlinedProgram, Schedule, CostFeatures)> {
        let mut best: Option<(InlinedProgram, Schedule, CostFeatures)> = None;
        let results = self.get_results();

        for res in results {
            for (schedule, cost) in res.pareto_frontier {
                info!("pareto schedule: {}", schedule);
                info!("cost: {:?}", cost);
                let mut replace_best = false;
                if let Some((_, _, best_cost)) = best {
                    if cost.weighted_cost(&weights) < best_cost.weighted_cost(&weights) {
                        replace_best = true;
                    }

                } else {
                    replace_best = true
                }

                if replace_best {
                    best = Some((res.inlined_program.clone(), schedule, cost));
                }
            }
        }

        best
    }

    /// add a schedule to the pareto frontier, unless it is strictly dominated
    /// by another schedule in the frontier
    pub fn update_pareto_frontier(&mut self, id: usize, schedule: Schedule, cost: CostFeatures) {
        let mut to_remove: Vec<(usize, Schedule)> = Vec::new();
        let mut add_new = true;
        for ((pid, pschedule), pcost) in self.pareto_frontier.iter() {
            let pcost_dominates = pcost.dominates(&cost);
            let cost_dominates = cost.dominates(pcost);

            // if the new schedule is strictly costlier, don't add it to the frontier
            if pcost_dominates {
                add_new = false;
                break;

            } else if cost_dominates {
                to_remove.push((*pid, pschedule.clone()));
            }
        }

       for pschedule in to_remove {
           self.pareto_frontier.remove(&pschedule);
       }

        if add_new {
            self.pareto_frontier.insert((id, schedule), cost);
        }
    }

    /// apply transformers to the current set of visited
    /// this uses a trick similar to semi-naive evaluation to Datalog
    /// returns true if new schedules were visited; otherwise
    pub fn iterate(&mut self) -> (usize, usize) {
        let mut frontier_size = 0;
        let mut new_schedules: Vec<(usize, Vec<(Schedule, CostFeatures)>)> = Vec::new();
        for (id, (do_run, scheduler)) in self.inline_schedulers.iter_mut() {
            if *do_run {
                let (inline_frontier_size, new_inline_schedules) =
                    scheduler.iterate(
                        self.materializer_factory.as_ref(),
                        &self.cost_estimator
                    );

                frontier_size += inline_frontier_size;
                if inline_frontier_size == 0 {
                    *do_run = false;
                }

                if new_inline_schedules.len() > 0 { 
                    new_schedules.push((*id, new_inline_schedules));
                }
            }
        }

        let new_schedules_count =
            new_schedules.iter()
            .fold(0, |acc, (_, sched_list)| {   
                acc + sched_list.len()
            });
        for (id, new_inline_schedules) in new_schedules {
            for (schedule, cost) in new_inline_schedules {
                self.update_pareto_frontier(id, schedule, cost);
            }
        }

        (frontier_size, new_schedules_count)
    }

    /// run a certain number of iterations, or until reaching quiescence
    pub fn run(&mut self, iter_limit: Option<usize>) {
        info!("running scheduler with iter limit: {:?}", iter_limit);
        let mut iter = 0;
        let mut changed = true;
        let within_limit = |x: usize| {
            match iter_limit {
                Some(limit) => x < limit,
                None => true
            }
        };
    
        while changed && within_limit(iter) {
            let iter_res = self.iterate();
            changed = iter_res.0 > 0;
            info!("iteration {}\nnew schedules visited: {}; new valid schedules found: {}",
                iter, iter_res.0, iter_res.1);
            iter += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use crate::{
        circ::{materializer::DefaultMaterializerFactory, pseudomaterializer::DefaultPseudoMaterializerFactory},
        lang::{
            index_elim::IndexElimination,
            elaborated::Elaborator,
            source::SourceProgram,
            parser::ProgramParser
        },
        scheduling::transformer::{DefaultScheduleTransformerFactory, FastScheduleTransformerFactory}
    };
    use super::*;

    fn test_scheduler_from_src(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let elaborated = Elaborator::new().run(program);

        let inlined_programs: Vec<InlinedProgram> =
            elaborated.inline_sets().iter()
            .filter_map(|inline_set| {
                let array_group_map =
                    elaborated.array_group_from_inline_set(&inline_set);

                let index_elim_res =
                    IndexElimination::new()
                    .run(&inline_set, &array_group_map, &elaborated);

                match index_elim_res {
                    Ok(inlined_program) => {
                        println!("inline set: {:?}", inline_set);
                        println!("inlined:\n{}", inlined_program);
                        Some(inlined_program)
                    }
                    Err(_) => None
                }
            }).collect();
        /*
        let inlined_programs = vec![
            IndexElimination::new().run(
                &elaborated.default_inline_set(),
                &elaborated.default_array_group_map(),
                &elaborated
            ).unwrap()
        ];
        */

        let mut scheduler =
            Scheduler::new(
                inlined_programs,
                Box::new(FastScheduleTransformerFactory), 
                Box::new(DefaultPseudoMaterializerFactory), 
                4096
            );
        
        scheduler.run(None);

        let mat_factory = DefaultMaterializerFactory;
        for (i, result) in scheduler.get_results().iter().enumerate() {
            println!("inline scheduler {}", i);
            println!("inlined program:\n{}", result.inlined_program);
            println!("schedules visited: {}", result.visited.len());
            println!("valid schedules visited: {}", result.valid_schedules_visited);
            println!("pareto frontier size: {}", result.pareto_frontier.len());
            for (schedule, cost) in result.pareto_frontier.iter() {
                // could still fail because we use the pseudomaterializer currently
                let circuit =
                    mat_factory.create()
                    .run(&result.inlined_program, &schedule)
                    .unwrap();

                println!("schedule:\n{}\ncost:\n{:?}", schedule, cost);
            }
        }
    }

    #[test]
    #[ignore]
    fn test_imgblur0() {
        test_scheduler_from_src(
            "input img: [16,16] from client
            for x: 16 {
                for y: 16 {
                    img[x][y] + img[x+1][y+1]
                }
            }",
        );
    }

    #[test]
    #[ignore]
    fn test_imgblur() {
        test_scheduler_from_src(
            "input img: [16,16] from client
            for x: 16 {
                for y: 16 {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }",
        );
    }

    #[test]
    #[ignore]
    fn test_imgblur2() {
        test_scheduler_from_src(
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
    #[ignore]
    fn test_convolve() {
        test_scheduler_from_src(
            "input img: [16,16] from client
            let conv1 = 
                for x: 16 {
                    for y: 16 {
                        img[x][y] + img[x+1][y+1]
                    }
                }
            in
            for x: 15 {
                for y: 15 {
                    conv1[x][y] + conv1[x+1][y+1]
                }
            }
            ",
        );
    }
    
    #[test]
    fn test_matvecmul() {
        test_scheduler_from_src(
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
    fn test_matmatmul() {
        test_scheduler_from_src(
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
    #[ignore]
    fn test_matmatmul2() {
        test_scheduler_from_src(
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
        test_scheduler_from_src(
            "
            input A: [3] from client
            input B: [3] from client
            sum(A * B)
            ",
        );
    }
}