use std::{collections::{HashSet, HashMap}, cmp::Ordering};

use crate::{
    lang::index_elim::InlinedProgram,
    circ::{materializer::MaterializerFactory, cost::{CostEstimator, CostFeatures}},
    scheduling::{
        Schedule,
        transformer::ScheduleTransformer,
    },
};

/// scheduler for a particular indexing for a particular inline set / array group map
pub struct Scheduler<'a, 'b, 'c> {
    transformers: Vec<Box<dyn ScheduleTransformer + 'b>>,
    program: &'a InlinedProgram,
    materializer_factory: MaterializerFactory<'c>,
    cost_estimator: CostEstimator,

    // schedules that have been visited for all iterations
    visited: HashSet<Schedule>,

    // the schedules visited during the previous iteration
    previous: HashSet<Schedule>,

    // schedules in the Pareto frontier
    // (i.e. no visited schedule is strictly cheaper than these)
    pub pareto_frontier: HashMap<Schedule, CostFeatures>
}

impl<'a, 'b, 'c> Scheduler<'a, 'b, 'c> {
    pub fn new(
        transformers: Vec<Box<dyn ScheduleTransformer + 'b>>,
        program: &'a InlinedProgram,
        materializer_factory: MaterializerFactory<'c>,
        initial: Schedule
    ) -> Self {
        let cost_estimator = CostEstimator::default();
        let previous = HashSet::from([initial.clone()]);
        let visited = HashSet::from([initial.clone()]);
        let pareto_frontier: HashMap<Schedule, CostFeatures> = HashMap::new();
        let mat = materializer_factory.create();
        
        let cost_opt =
            if let Ok(circuit) = mat.run(program, &initial) {
                Some(cost_estimator.estimate_cost(&circuit))

            } else {
                None
            };

        let mut scheduler = 
            Self {
                transformers,
                program,
                materializer_factory,
                cost_estimator,
                visited,
                previous,
                pareto_frontier,
            };

        if let Some(cost) = cost_opt {
            scheduler.update_pareto_frontier(initial, cost);
        }

        scheduler
    }

    /// add a schedule to the pareto frontier, unless it is strictly dominated
    /// by another schedule in the frontier
    pub fn update_pareto_frontier(&mut self, schedule: Schedule, cost: CostFeatures) {
        let mut to_remove: Vec<Schedule> = Vec::new();
        let mut add_new = true;
        for (pschedule, pcost) in self.pareto_frontier.iter() {
            match cost.partial_cmp(pcost) {
                // if the new schedule is strictly costlier, don't add it to the frontier
                Some(Ordering::Greater) => {
                    add_new = false;
                    break;
                },

                // if the new schedule is strictly cheaper than a schedule s2,
                // currently in the frontier, add the new schedule
                // and remove s2 from the frontier
                Some(Ordering::Less) => {
                    to_remove.push(pschedule.clone());
                }

                Some(Ordering::Equal) | None => {}
            }
        }

        for pschedule in to_remove {
            self.pareto_frontier.remove(&pschedule);
        }

        if add_new {
            self.pareto_frontier.insert(schedule, cost);
        }
    }

    /// apply transformers to the current set of visited
    /// this uses a trick similar to semi-naive evaluation to Datalog
    /// returns true if new schedules were visited; otherwise
    pub fn iterate(&mut self) -> bool {
        let mut has_new = false;

        let mut cur: HashSet<Schedule> = HashSet::new();
        std::mem::swap(&mut self.previous, &mut cur);
        let mut neighbor_list: Vec<(Schedule, CostFeatures)> = Vec::new();
        for schedule in cur {
            for transformer in self.transformers.iter_mut() {
                let neighbors = transformer.transform(&schedule);
                for neighbor in neighbors {
                    if !self.visited.contains(&neighbor) && neighbor.is_schedule_valid(self.program) {
                        // neighbor is a newly visited schedule;
                        // try to materialize it into a circuit
                        let mat = self.materializer_factory.create();

                        // if the schedule can be materialized into a circuit,
                        // give a cost to the schedule and save it if it's
                        // in the Pareto frontier
                        if let Ok(circuit) = mat.run(self.program, &neighbor) {
                            let cost = self.cost_estimator.estimate_cost(&circuit);
                            neighbor_list.push((neighbor.clone(), cost));
                        }

                        self.previous.insert(neighbor.clone());
                        self.visited.insert(neighbor);
                        has_new = true;
                    }
                }
            }
        }

        // try to add new schedules to the pareto frontier
        for (neighbor, cost) in neighbor_list {
            self.update_pareto_frontier(neighbor, cost);
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
    use crate::{circ::{ParamCircuitProgram, materializer::{InputArrayMaterializer, DefaultArrayMaterializer}}, lang::{index_elim::IndexElimination, elaborated::Elaborator, source::SourceProgram, parser::ProgramParser}, scheduling::transformer::VectorizeDimTransformer};
    use super::*;

    fn test_scheduler_from_src(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let elaborated = Elaborator::new().run(program);
        let inline_set = elaborated.get_default_inline_set();
        let array_group_map = elaborated.get_default_array_group_map();

        let res = IndexElimination::new().run(&inline_set, &array_group_map, elaborated);

        assert!(res.is_ok());

        let inlined = res.unwrap();
        let init_schedule = Schedule::gen_initial_schedule(&inlined);
        let dim_classes = inlined.compute_dim_equiv_classes();

        let transformers: Vec<Box<dyn ScheduleTransformer + '_>> = vec![
            Box::new(VectorizeDimTransformer::new(&dim_classes))
        ];

        let amats: Vec<Box<dyn InputArrayMaterializer + '_>> =
            vec![Box::new(DefaultArrayMaterializer::new())];

        let mat_factory = MaterializerFactory::new(amats);

        let mut scheduler =
            Scheduler::new(
                transformers, 
                &inlined, 
                mat_factory, 
                init_schedule
            );
        
        scheduler.run(None);
        println!("pareto frontier:");
        for (schedule, cost) in scheduler.pareto_frontier.iter() {
            println!("schedule:\n{}\ncost:\n{:?}", schedule, cost);
        }
        drop(scheduler);
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
}