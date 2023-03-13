use std::{collections::{HashSet, HashMap}};

use crate::{
    lang::{index_elim::{InlinedProgram}},
    circ::{materializer::MaterializerFactory, cost::{CostEstimator, CostFeatures}},
    scheduling::{
        Schedule,
        transformer::{ScheduleTransformer, ScheduleTransformerFactory}
    },
};

struct InlineScheduler<'a> {
    pub program: InlinedProgram,
    pub transformers: Vec<Box<dyn ScheduleTransformer + 'a>>,
    pub visited: HashSet<Schedule>,
    pub frontier: HashSet<Schedule>
}

impl<'t> InlineScheduler<'t> {
    pub fn new(
        program: InlinedProgram,
        transformers: Vec<Box<dyn ScheduleTransformer + 't>>,
        init_schedule: Schedule
    ) -> Self {
        Self {
            program,
            transformers,
            visited: HashSet::from([init_schedule.clone()]),
            frontier: HashSet::from([init_schedule]),
        }
    }

    pub fn iterate<'m>(
        &mut self,
        materializer_factory: &dyn MaterializerFactory,
        cost_estimator: &CostEstimator,
    ) -> (bool, Vec<(Schedule, CostFeatures)>) {
        let mut has_new = false;

        let mut cur: HashSet<Schedule> = HashSet::new();
        std::mem::swap(&mut self.frontier, &mut cur);
        let mut neighbor_list: Vec<(Schedule, CostFeatures)> = Vec::new();
        for schedule in cur {
            for transformer in self.transformers.iter() {
                let neighbors = transformer.transform(&schedule);
                for neighbor in neighbors {
                    if !self.visited.contains(&neighbor) && neighbor.is_schedule_valid(&self.program) {
                        // neighbor is a newly visited schedule;
                        // try to materialize it into a circuit
                        let mat = materializer_factory.create();

                        // if the schedule can be materialized into a circuit,
                        // give a cost to the schedule and save it if it's
                        // in the Pareto frontier
                        if let Ok(circuit) = mat.run(&self.program, &neighbor) {
                            let cost = cost_estimator.estimate_cost(&circuit);
                            neighbor_list.push((neighbor.clone(), cost));
                        }

                        self.frontier.insert(neighbor.clone());
                        self.visited.insert(neighbor);
                        has_new = true;
                    }
                }
            }
        }

        (has_new, neighbor_list)
    }
}

pub struct SchedulingResult {
    pub inlined_program: InlinedProgram,
    pub pareto_frontier: Vec<(Schedule, CostFeatures)>,
    pub visited: HashSet<Schedule>,
}

/// scheduler for a particular inlined program
/// (identified by array group map and inline set)
pub struct Scheduler<'m, 't> {
    materializer_factory: Box<dyn MaterializerFactory + 'm>,
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
        materializer_factory: Box<dyn MaterializerFactory + 'm>,
    ) -> Self {
        let cost_estimator = CostEstimator::default();
        let mut init_schedules: Vec<(usize, Schedule, CostFeatures)> = Vec::new();
        let mut inline_schedulers: HashMap<usize, (bool, InlineScheduler)> = HashMap::new();
        let mut inline_id = 1;

        for inlined_program in inlined_programs {
            let init_schedule =
                Schedule::gen_initial_schedule(&inlined_program);

            let mat = materializer_factory.create();
            let mat_res = mat.run(&inlined_program, &init_schedule);
            if let Ok(circuit) = mat_res {
                let cost = cost_estimator.estimate_cost(&circuit);
                init_schedules.push((inline_id, init_schedule.clone(), cost));
            }

            let transformers =
                transformer_factory.create(&inlined_program);

            let inline_scheduler =
                InlineScheduler::new(
                    inlined_program,
                    transformers,
                    init_schedule
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

        for (inline_id, init_schedule, cost) in init_schedules {
            scheduler.update_pareto_frontier(inline_id, init_schedule, cost);
        }

        scheduler
    }

    pub fn get_results(mut self) -> Vec<SchedulingResult> {
        let mut pareto_map: HashMap<usize, Vec<(Schedule, CostFeatures)>> = HashMap::new();
        for ((id, schedule), cost) in self.pareto_frontier.into_iter() {
            match pareto_map.get_mut(&id) {
                Some(schedule_list) => {
                    schedule_list.push((schedule, cost));
                },

                None => {
                    pareto_map.insert(id, vec![(schedule, cost)]);
                }
            }
        }

        pareto_map.into_iter().map(|(id, schedule_list)| {  
            let (_, inline_sched) = self.inline_schedulers.remove(&id).unwrap();

            SchedulingResult {
                inlined_program: inline_sched.program,
                visited: inline_sched.visited,
                pareto_frontier: schedule_list
            }
        }).collect()
    }

    /// add a schedule to the pareto frontier, unless it is strictly dominated
    /// by another schedule in the frontier
    pub fn update_pareto_frontier(&mut self, id: usize, schedule: Schedule, cost: CostFeatures) {
        let mut to_remove: Vec<(usize, Schedule)> = Vec::new();
        let mut add_new = true;
        'l: for ((pid, pschedule), pcost) in self.pareto_frontier.iter() {
            let pcost_dominates = pcost.dominates(&cost);
            let cost_dominates = cost.dominates(pcost);

            // if the new schedule is strictly costlier, don't add it to the frontier
            if pcost_dominates {
                add_new = false;
                break 'l;

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
    pub fn iterate(&mut self) -> bool {
        let mut has_new = false;
        let mut new_schedules: Vec<(usize, Vec<(Schedule, CostFeatures)>)> = Vec::new();
        for (id, (do_run, scheduler)) in self.inline_schedulers.iter_mut() {
            if *do_run {
                let (changed, new_inline_schedules) =
                    scheduler.iterate(
                        self.materializer_factory.as_ref(),
                        &self.cost_estimator
                    );

                if new_inline_schedules.len() > 0 { 
                    new_schedules.push((*id, new_inline_schedules));
                }

                if changed {
                    has_new = true;

                } else {
                    *do_run = false;
                }
            }
        }

        for (id, new_inline_schedules) in new_schedules {
            for (schedule, cost) in new_inline_schedules {
                self.update_pareto_frontier(id, schedule, cost);
            }
        }

        has_new
    }

    /// run a certain number of iterations, or until reaching quiescence
    pub fn run(&mut self, iter_limit: Option<usize>) {
        let mut iter = 0;
        let mut changed = true;
        let within_limit = |x: usize| {
            match iter_limit {
                Some(limit) => x < limit,
                None => true
            }
        };

        while changed && within_limit(iter) {
            changed = self.iterate();
            iter += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circ::materializer::DefaultMaterializerFactory,
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

        /*
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
        */
        let inlined_programs = vec![
            IndexElimination::new().run(
                &elaborated.default_inline_set(),
                &elaborated.default_array_group_map(),
                &elaborated
            ).unwrap()
        ];

        let mut scheduler =
            Scheduler::new(
                inlined_programs,
                Box::new(FastScheduleTransformerFactory), 
                Box::new(DefaultMaterializerFactory), 
            );
        
        scheduler.run(None);

        let mat_factory = DefaultMaterializerFactory;
        for result in scheduler.get_results() {
            println!("inlined program:\n{}", result.inlined_program);
            println!("visited schedules: {}", result.visited.len());
            println!("pareto frontier size: {}", result.pareto_frontier.len());
            for (schedule, cost) in result.pareto_frontier.iter() {
                let circuit =
                    mat_factory.create()
                    .run(&result.inlined_program, &schedule).unwrap();
                println!("schedule:\n{}\ncost:\n{:?}", schedule, cost);
                println!("circuit:\n{}", circuit);
            }
        }
    }

    #[test]
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