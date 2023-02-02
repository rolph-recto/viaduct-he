
use std::{collections::HashMap, fmt::Display};

use crate::lang::{*, index_elim2::{TransformedExpr, TransformedProgram}};

pub mod materializer;

type DimName = String;
type ExplodedIndexStore = HashMap<DimName, isize>;

#[derive(Clone,Debug)]
pub enum OffsetExpr {
    Add(Box<OffsetExpr>, Box<OffsetExpr>),
    Mul(Box<OffsetExpr>, Box<OffsetExpr>),
    Literal(isize),
    ExplodedIndexVar(DimName),
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

impl Default for OffsetExpr {
    fn default() -> Self {
        OffsetExpr::Literal(0)
    }
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

impl ArraySchedule {
    // apply schedule to an array transform
    pub fn apply_schedule(&self, transform: &BaseArrayTransform) -> ParamArrayTransform {
        let num_dims = transform.offset_map.num_dims();
        let mut param_offset_map: OffsetMap<OffsetExpr> = OffsetMap::new(num_dims);
        for i in 0..num_dims {
            let cur_offset = *transform.offset_map.get_offset(i);
            param_offset_map.set_offset(i, OffsetExpr::Literal(cur_offset));
        }

        // process exploded dims
        for sched_dim in self.exploded_dims.iter() {
            let dim_content = transform.dims.get(sched_dim.index).unwrap();
            match dim_content {
                DimContent::FilledDim { dim, extent, stride } => {
                    let cur_offset = param_offset_map.get_offset(*dim).clone();
                    let new_offset =
                        OffsetExpr::Add(
                            Box::new(cur_offset),
                            Box::new(
                                OffsetExpr::Mul(
                                    Box::new(OffsetExpr::Literal(*stride)),
                                    Box::new(OffsetExpr::ExplodedIndexVar(sched_dim.name.clone()))
                                )
                            )
                        );

                    param_offset_map.set_offset(*dim, new_offset);
                },

                // if the dim is empty, no offset needs to be updated
                DimContent::EmptyDim { extent: _ } => {}
            }
        }

        // process vectorized dims
        let mut new_vectorized_dims: Vec<DimContent> = Vec::new();
        for sched_dim in self.vectorized_dims.iter() {
            let dim_content = transform.dims.get(sched_dim.index).unwrap();
            let new_dim =
                match dim_content {
                    // increase stride according to schedule dim, trunate extent
                    DimContent::FilledDim { dim: dim_index, extent: _, stride: content_stride } => {
                        DimContent::FilledDim {
                            dim: *dim_index,
                            extent: sched_dim.extent,
                            stride: content_stride * sched_dim.stride
                        }
                    }

                    // truncate to schedule dim's extent
                    DimContent::EmptyDim { extent: _ } => {
                        DimContent::EmptyDim{ extent: sched_dim.extent }
                    }
                };
            new_vectorized_dims.push(new_dim);
        }

        ParamArrayTransform {
            exploded_dims: self.exploded_dims.clone(),
            transform: ArrayTransform {
                array: transform.array.clone(),
                offset_map: param_offset_map,
                dims: new_vectorized_dims,
            }
        }
    }
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

pub struct ParamArrayTransform {
    exploded_dims: im::Vector<ScheduleDim>,
    transform: ArrayTransform<OffsetExpr>,
}

impl Display for ParamArrayTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}", self.exploded_dims, self.transform)
    }
}

#[derive(Clone,Debug,PartialEq,Eq)]
pub enum ExprSchedule {
    Any, // the schedule is arbitrary (i.e. like for literals)
    Specific(ArraySchedule)
}

impl Display for ExprSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprSchedule::Any => write!(f, "*"),
            ExprSchedule::Specific(sched) => write!(f, "{}", sched)
        }
    }
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
    pub fn compute_output_schedule(&self, expr: &TransformedExpr) -> Option<ExprSchedule> {
        match expr {
            TransformedExpr::ReduceNode(reduced_index, _, body) => {
                let sched_type = self.compute_output_schedule(body)?;
                match sched_type {
                    ExprSchedule::Any => None,

                    ExprSchedule::Specific(body_sched) => {
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

                        Some(
                            ExprSchedule::Specific(
                                ArraySchedule {
                                    exploded_dims: new_exploded_dims,
                                    vectorized_dims: body_sched.vectorized_dims,
                                }
                            )
                        )
                    }
                }
            }

            // this performs a join on the "schedule status lattice",
            // where valid schedules are incomparable,
            // any is bottom and invalid is top
            TransformedExpr::Op(_, expr1, expr2) => {
                let res1 = self.compute_output_schedule(expr1)?;
                let res2 = self.compute_output_schedule(expr2)?;
                match (res1, res2) {
                    (ExprSchedule::Any, ExprSchedule::Any) => 
                        Some(ExprSchedule::Any),

                    (ExprSchedule::Any, ExprSchedule::Specific(sched)) |
                    (ExprSchedule::Specific(sched), ExprSchedule::Any) =>
                        Some(ExprSchedule::Specific(sched)),

                    (ExprSchedule::Specific(sched1), ExprSchedule::Specific(sched2)) => {
                        if sched1 == sched2 {
                            Some(ExprSchedule::Specific(sched1))

                        } else {
                            None
                        }
                    }
                }
            },

            TransformedExpr::Literal(_) => Some(ExprSchedule::Any),

            TransformedExpr::ExprRef(ref_id) => {
                let array_sched = self.schedule_map.get(ref_id)?;
                Some(ExprSchedule::Specific(array_sched.clone()))
            },
        }
    }

    pub fn is_schedule_valid(&self, expr: &TransformedExpr) -> bool {
        self.compute_output_schedule(expr).is_some()
    }
}

#[cfg(test)]
mod tests{
    use crate::lang::{parser::ProgramParser, index_elim2::IndexElimination2};
    use super::*;

    // generate an initial schedule for a program
    fn test_schedule(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let mut index_elim = IndexElimination2::new();
        let res = index_elim.run(&program);
        
        assert!(res.is_ok());

        let program = res.unwrap();
        println!("{}", program.expr);

        let init_schedule = Schedule::gen_initial_schedule(&program);
        println!("{}", &init_schedule);

        // the initial schedule should always be valid!
        assert!(init_schedule.is_schedule_valid(&program.expr))
    }

    #[test]
    fn test_imgblur() {
        test_schedule(
        "input img: [(0,16),(0,16)]
            for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }"
        );
    }

    #[test]
    fn test_imgblur2() {
        test_schedule(
        "input img: [(0,16),(0,16)]
            let res = 
                for x: (0, 16) {
                    for y: (0, 16) {
                        img[x-1][y-1] + img[x+1][y+1]
                    }
                }
            in
            for x: (0, 16) {
                for y: (0, 16) {
                    res[x-2][y-2] + res[x+2][y+2]
                }
            }
            "
        );
    }

    #[test]
    fn test_convolve() {
        test_schedule(
        "input img: [(0,16),(0,16)]
            let conv1 = 
                for x: (0, 15) {
                    for y: (0, 15) {
                        img[x][y] + img[x+1][y+1]
                    }
                }
            in
            for x: (0, 14) {
                for y: (0, 14) {
                    conv1[x][y] + conv1[x+1][y+1]
                }
            }
            "
        );
    }

    #[test]
    fn test_matmatmul() {
        test_schedule(
            "input A: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A[i][k] * B[k][j] })
                }
            }"
        );
    }

    #[test]
    fn test_matmatmul2() {
        test_schedule(
            "input A1: [(0,4),(0,4)]
            input A2: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            let res =
                for i: (0,4) {
                    for j: (0,4) {
                        sum(for k: (0,4) { A1[i][k] * B[k][j] })
                    }
                }
            in
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A2[i][k] * res[k][j] })
                }
            }
            "
        );
    }

    #[test]
    fn test_dotprod_pointless() {
        test_schedule(
        "
            input A: [(0,3)]
            input B: [(0,3)]
            sum(A * B)
            "
        );
    }

    #[test]
    fn test_matvecmul() {
        test_schedule(
        "
            input M: [(0,1),(0,1)]
            input v: [(0,1)]
            for i: (0,1) {
                sum(M[i] * v)
            }
            "
        );
    }
}