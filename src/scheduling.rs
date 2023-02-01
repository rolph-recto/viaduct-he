
use std::{collections::HashMap, fmt::Display};

use crate::lang::{*, index_elim2::{TransformedExpr, TransformedProgram}};

pub mod materializer;

type ExplodedIndexName = String;
type ExplodedIndexStore = HashMap<ExplodedIndexName, isize>;

#[derive(Clone,Debug)]
pub enum OffsetExpr {
    Add(Box<OffsetExpr>, Box<OffsetExpr>),
    Mul(Box<OffsetExpr>, Box<OffsetExpr>),
    Literal(isize),
    ExplodedIndexVar(ExplodedIndexName),
}

impl OffsetExpr {
    pub fn eval(&self, store: &ExplodedIndexStore) -> isize {
        match self {
            OffsetExpr::Add(expr1, expr2) => {
                let val1 = expr1.eval(store);
                let val2 = expr2.eval(store);
                val1 + val2
            },

            OffsetExpr::Mul(expr1, expr2) => {
                let val1 = expr1.eval(store);
                let val2 = expr2.eval(store);
                val1 * val2
            },

            OffsetExpr::Literal(lit) => *lit,

            OffsetExpr::ExplodedIndexVar(var) => store[var]
        }
    }
}

impl Display for OffsetExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OffsetExpr::Add(expr1, expr2) => {
                write!(f, "({} + {})", expr1, expr2)
            },

            OffsetExpr::Mul(expr1, expr2) => {
                write!(f, "({} * {})", expr1, expr2)
            },

            OffsetExpr::Literal(lit) => {
                write!(f, "{}", lit)
            },

            OffsetExpr::ExplodedIndexVar(var) => {
                write!(f, "{}", var)
            }
        }
    }
}

type ParameterizedOffsetMap = OffsetMap<OffsetExpr>;

pub struct ParameterizedArrayTransform {
    exploded_dims: im::Vector<ScheduleDim>,
    transform: ArrayTransform<OffsetExpr>,
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct ScheduleDim {
    pub index: DimIndex,
    pub stride: isize,
    pub extent: usize,
    pub name: String,
}

impl Display for ScheduleDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}:{}::{}", self.name, self.index, self.extent, self.stride)
    }
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct ArraySchedule {
    pub exploded_dims: im::Vector<ScheduleDim>,
    pub vectorized_dims: im::Vector<ScheduleDim>,
}

impl Display for ArraySchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let exploded_str =
            self.exploded_dims.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        let vectorized_str =
            self.vectorized_dims.iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ");

        write!(f, "{{{}}}[{}]", exploded_str, vectorized_str)
    }
}

#[derive(Clone,Debug,PartialEq,Eq)]
pub enum OutputScheduleStatus {
    // input schedule is invalid for the program
    Invalid, 

    // the output schedule is arbitrary (i.e. like for literals)
    Any,     

    // the input schedule is valid, with the following output schedule
    Valid(ArraySchedule)
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct Schedule {
    pub schedule_map: im::HashMap<ExprRefId,ArraySchedule>
}

impl Display for Schedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.schedule_map.iter().try_for_each(|(ref_id, schedule)| {
            writeln!(f, "{} => {}", ref_id, schedule)
        })
    }
}

impl Schedule {
    // generate an initial schedule 
    // the initial schedule explodes *all* dims
    pub fn gen_initial_schedule(program: &TransformedProgram) -> Self {
        let mut schedule_map: im::HashMap<ExprRefId,ArraySchedule> = im::HashMap::new();
        let dim_class_map = program.compute_dim_equiv_classes();

        for (ref_id, transform) in program.inputs.iter() {
            let mut schedule_dims: im::Vector<ScheduleDim> = im::Vector::new();
            for (i, dim) in transform.dims.iter().enumerate() {
                let class_id = dim_class_map[&(*ref_id, i)];
                schedule_dims.push_back(
                    ScheduleDim {
                        index: i,
                        stride: 1,
                        extent: dim.extent(),
                        name: format!("i{}", class_id)
                    }
                )
            }
            let schedule =
                ArraySchedule {
                    exploded_dims: schedule_dims,
                    vectorized_dims: im::Vector::new(),
                };

            schedule_map.insert(*ref_id, schedule);
        }

        Schedule { schedule_map }
    }

    // apply the schedule to an index-free expression and compute the output schedule
    pub fn compute_output_schedule(&self, expr: &TransformedExpr) -> OutputScheduleStatus {
        match expr {
            TransformedExpr::ReduceNode(reduced_index, _, body) => {
                match self.compute_output_schedule(body) {
                    OutputScheduleStatus::Invalid => OutputScheduleStatus::Invalid,

                    OutputScheduleStatus::Any => OutputScheduleStatus::Invalid,

                    OutputScheduleStatus::Valid(body_sched) => {
                        // TODO: implement reduction for vectorized dims
                        let mut new_exploded_dims: im::Vector<ScheduleDim> = im::Vector::new();
                        for mut dim in body_sched.exploded_dims {
                            if dim.index == *reduced_index { // dim is reduced, remove it

                            } else if dim.index > *reduced_index { // decrease dim index
                                dim.index -= 1;
                                new_exploded_dims.push_back(dim);

                            } else {
                                new_exploded_dims.push_back(dim);
                            }
                        }

                        OutputScheduleStatus::Valid(
                            ArraySchedule {
                                exploded_dims: new_exploded_dims,
                                vectorized_dims: body_sched.vectorized_dims,
                            }
                        )
                    }
                }
            }

            // this performs a join on the "schedule status lattice",
            // where valid schedules are incomparable,
            // any is bottom and invalid is top
            TransformedExpr::Op(_, expr1, expr2) => {
                let res1 = self.compute_output_schedule(expr1);
                let res2 = self.compute_output_schedule(expr2);
                match (res1, res2) {
                    (OutputScheduleStatus::Invalid, _) | 
                    (_, OutputScheduleStatus::Invalid) =>
                        OutputScheduleStatus::Invalid,

                    (OutputScheduleStatus::Any, OutputScheduleStatus::Any) => 
                        OutputScheduleStatus::Any,

                    (OutputScheduleStatus::Any, OutputScheduleStatus::Valid(sched)) |
                    (OutputScheduleStatus::Valid(sched), OutputScheduleStatus::Any) =>
                        OutputScheduleStatus::Valid(sched),

                    (OutputScheduleStatus::Valid(sched1), OutputScheduleStatus::Valid(sched2)) => {
                        if sched1 == sched2 {
                            OutputScheduleStatus::Valid(sched1)

                        } else {
                            OutputScheduleStatus::Invalid
                        }
                    }
                }
            },

            TransformedExpr::Literal(_) => OutputScheduleStatus::Any,

            TransformedExpr::ExprRef(ref_id) => {
                match self.schedule_map.get(ref_id) {
                    Some(array_sched) =>
                        OutputScheduleStatus::Valid(array_sched.clone()),

                    None =>
                        OutputScheduleStatus::Invalid,
                }
            },
        }
    }

    pub fn is_schedule_valid(&self, expr: &TransformedExpr) -> bool {
        OutputScheduleStatus::Invalid != self.compute_output_schedule(expr)
    }

    // pub fn apply_schedule(&self, )
}

#[cfg(test)]
mod tests{
    use crate::lang::{parser::ProgramParser, index_elim2::IndexElimination2};
    use super::*;

    // generate an initial schedule for a program
    fn test_gen_init_schedule(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let mut index_elim = IndexElimination2::new();
        let res = index_elim.run(&program);
        
        assert!(res.is_ok());

        let program = res.unwrap();
        let init_schedule = Schedule::gen_initial_schedule(&program);
        println!("{}", &init_schedule);

        // the initial schedule should always be valid!
        assert!(init_schedule.is_schedule_valid(&program.expr));
    }

    #[test]
    fn test_imgblur() {
        test_gen_init_schedule(
        "input img: [(0,16),(0,16)]
            for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }"
        );
    }
}