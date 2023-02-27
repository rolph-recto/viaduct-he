
use std::{collections::{HashMap, HashSet}, fmt::Display};

use gcollections::ops::Bounded;

use crate::{
    lang::{*,
        index_elim::{TransformedExpr, TransformedProgram, ArrayDim}
    },
    circ2::{
        IndexCoordinateMap,
        vector_info::VectorInfo, CircuitValue
    }
};

pub type DimName = String;

pub struct OffsetEnvironment {
    index_map: HashMap<DimName, usize>,
    function_values: HashMap<String, isize>,
}

impl OffsetEnvironment {
    pub fn new(index_map: HashMap<DimName, usize>) -> Self {
        OffsetEnvironment { index_map, function_values: HashMap::new() }
    }

    pub fn set_function_value(&mut self, func: String, value: isize) {
        self.function_values.insert(func, value);
    }
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub enum OffsetExpr {
    Add(Box<OffsetExpr>, Box<OffsetExpr>),
    Mul(Box<OffsetExpr>, Box<OffsetExpr>),
    Literal(isize),
    Var(DimName),
    FunctionVar(String, im::Vector<DimName>),
}

impl OffsetExpr {
    pub fn eval(&self, store: &OffsetEnvironment) -> isize {
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

            OffsetExpr::Var(var) => store.index_map[var] as isize,

            OffsetExpr::FunctionVar(func, _) => store.function_values[func],
        }
    }

    pub fn const_value(&self) -> Option<isize> {
        match self {
            OffsetExpr::Add(expr1, expr2) => {
                let const1 = expr1.const_value()?;
                let const2 = expr2.const_value()?;
                Some(const1 + const2)
            },

            OffsetExpr::Mul(expr1, expr2) => {
                let const1 = expr1.const_value()?;
                let const2 = expr2.const_value()?;
                Some(const1 + const2)
            },
            
            OffsetExpr::Literal(lit) => Some(*lit),

            OffsetExpr::Var(_) => None,

            OffsetExpr::FunctionVar(_, _) => None,
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

            OffsetExpr::Var(var) => {
                write!(f, "{}", var)
            },

            OffsetExpr::FunctionVar(func, vars) => {
                write!(f, "{}{:?}", func, vars)
            }
        }
    }
}

impl Default for OffsetExpr {
    fn default() -> Self {
        OffsetExpr::Literal(0)
    }
}

// a schedule for a dimension
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct ScheduleDim {
    pub index: DimIndex,
    pub stride: usize,
    pub extent: usize,
    pub name: String,

    // pad_left and pad_right should only be nonzero for vectorized dims!
    pub pad_left: usize,
    pub pad_right: usize,
}

impl Display for ScheduleDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}:{}::{}", self.name, self.index, self.extent, self.stride)
    }
}

#[derive(Copy,Clone,Debug,PartialEq,Eq,Hash)]
pub enum ClientPreprocessing {
    // TODO add more complicated permutations
    // Permute(i, j) means to permute dim i along dim j
    Permute(DimIndex, DimIndex)
}

impl Display for ClientPreprocessing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClientPreprocessing::Permute(dim_i, dim_j) =>
                write!(f, "permute({},{})", dim_i, dim_j)
        }
    }
}

impl ClientPreprocessing {
    pub fn transformed_dims(&self) -> HashSet<DimIndex> {
        match self {
            ClientPreprocessing::Permute(dim_i, _) => {
                let mut set: HashSet<DimIndex> = HashSet::new();
                set.insert(*dim_i);
                set
            }
        }
    }
}

pub trait HasExplodedDims {
    fn get_exploded_dims(&self) -> Vec<&ScheduleDim>;

    // get a parameterized offset map for the *array* indexed by transform
    fn get_indexed_offset_map(&self, transform: &ArrayTransform) -> OffsetMap<OffsetExpr> {
        let num_dims = transform.offset_map.num_dims();
        let mut param_offset_map: OffsetMap<OffsetExpr> = OffsetMap::new(num_dims);
        for i in 0..num_dims {
            let cur_offset = *transform.offset_map.get(i);
            param_offset_map.set(i, OffsetExpr::Literal(cur_offset));
        }

        for sched_dim in self.get_exploded_dims() {
            // exploded dims should not have padding!
            assert!(sched_dim.pad_left == 0 && sched_dim.pad_right == 0);

            let dim_content = transform.dims.get(sched_dim.index).unwrap();
            match dim_content {
                DimContent::FilledDim { dim, extent: _, stride } => {
                    let cur_offset = param_offset_map.get(*dim).clone();
                    let new_offset =
                        OffsetExpr::Add(
                            Box::new(cur_offset),
                            Box::new(
                                OffsetExpr::Mul(
                                    Box::new(OffsetExpr::Literal(*stride as isize)),
                                    Box::new(OffsetExpr::Var(sched_dim.name.clone()))
                                )
                            )
                        );

                    param_offset_map.set(*dim, new_offset);
                },

                // if the dim is empty, no offset needs to be updated
                DimContent::EmptyDim { extent: _ } => {}
            }
        }

        param_offset_map
    }

    // get a parameterized offset map for the transform itself
    fn get_transform_offset_map(&self, transform: &ArrayTransform) -> OffsetMap<OffsetExpr> {
        let num_dims = transform.dims.len();
        let mut param_offset_map: OffsetMap<OffsetExpr> = OffsetMap::new(num_dims);
        for i in 0..num_dims {
            param_offset_map.set(i, OffsetExpr::Literal(0));
        }

        for sched_dim in self.get_exploded_dims() {
            let cur_offset = param_offset_map.get(sched_dim.index).clone();
            let new_offset =
                OffsetExpr::Add(
                    Box::new(cur_offset),
                    Box::new(
                        OffsetExpr::Mul(
                            Box::new(OffsetExpr::Literal(sched_dim.stride as isize)),
                            Box::new(OffsetExpr::Var(sched_dim.name.clone()))
                        )
                    )
                );

            param_offset_map.set(sched_dim.index, new_offset);
        }

        param_offset_map
    }
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct IndexingSiteSchedule {
    pub preprocessing: Option<ClientPreprocessing>,
    pub exploded_dims: im::Vector<ScheduleDim>,
    pub vectorized_dims: im::Vector<ScheduleDim>,
}

impl IndexingSiteSchedule {
    // compute the scheduled tiling for a given dimension
    pub fn get_tiling(&self, dim: DimIndex) -> Vec<usize> {
        let mut sdims: Vec<(usize, usize)> = Vec::new();
        
        sdims.extend(
            self.exploded_dims.iter()
            .filter(|edim| edim.index == dim)
            .map(|edim| (edim.stride, edim.extent))
        );

        sdims.extend(
            self.vectorized_dims.iter()
            .flat_map(|vdim| {
                if vdim.index == dim {
                    vec![(vdim.stride, vdim.extent)]
                } else {
                    vec![]
                }
            })
        );

        sdims.sort_by(|(s1,_), (s2,_)| s1.cmp(s2));
        sdims.into_iter().map(|(_,extent)| extent).collect()
    }

    pub fn to_expr_schedule(&self, shape: Shape) -> ExprSchedule {
        ExprSchedule {
            shape,
            preprocessing: self.preprocessing.clone(),
            exploded_dims: self.exploded_dims.clone(),
            vectorized_dims: 
                self.vectorized_dims.clone().into_iter()
                .map(|dim| VectorScheduleDim::Filled(dim))
                .collect()
        }
    }
}

impl HasExplodedDims for IndexingSiteSchedule {
    fn get_exploded_dims(&self) -> Vec<&ScheduleDim> {
        self.exploded_dims.iter().collect()
    }
}

impl Display for IndexingSiteSchedule {
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
pub enum ExprScheduleType {
    Any, // the schedule is arbitrary (i.e. like for literals)
    Specific(ExprSchedule)
}

impl ExprScheduleType {
    /// counts how many exprs are represented by the schedule
    /// the multiplicity is the 
    pub fn multiplicity(&self) -> usize {
        match self {
            ExprScheduleType::Any => 1,
            ExprScheduleType::Specific(spec_sched) => 
                spec_sched.exploded_dims.iter()
                .map(|dim| dim.extent)
                .fold(1, |acc, x| acc*x)
        }
    }
}

impl Display for ExprScheduleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprScheduleType::Any => write!(f, "*"),
            ExprScheduleType::Specific(sched) => write!(f, "{}", sched)
        }
    }
}

// an output schedule for a vectorized dimension
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub enum VectorScheduleDim {
    // a regular dimension that contains elements of the scheduled array
    Filled(ScheduleDim),

    // reduced dim with the reduced value in the first position,
    // and the rest are "junk" values
    // e.g. 1 x x x 2 x x x
    Reduced(usize),

    // reduced dim that is repeated with elements from other dimensions
    // e.g. 1 1 1 1 2 2 2 2
    ReducedRepeated(usize),
}

impl Display for VectorScheduleDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorScheduleDim::Filled(sched_dim) =>
                write!(f, "{}", sched_dim),

            VectorScheduleDim::Reduced(extent) =>
                write!(f, "R:{}", extent),

            VectorScheduleDim::ReducedRepeated(extent) =>
                write!(f, "RR:{}", extent),
        }
    }
}

impl VectorScheduleDim {
    pub fn is_reduced(&self) -> bool {
        match self {
            VectorScheduleDim::Filled(_) => false,

            VectorScheduleDim::Reduced(_) |
            VectorScheduleDim::ReducedRepeated(_) => true
        }
    }

    pub fn extent(&self) -> usize {
        match self {
            VectorScheduleDim::Filled(dim) => dim.extent,

            VectorScheduleDim::Reduced(extent) |
            VectorScheduleDim::ReducedRepeated(extent) => *extent
        }
    }
}

// like ArraySchedule, except vectorized dims can have special reduced dimensions
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct ExprSchedule {
    pub shape: Shape,
    pub preprocessing: Option<ClientPreprocessing>,
    pub exploded_dims: im::Vector<ScheduleDim>,
    pub vectorized_dims: im::Vector<VectorScheduleDim>,
}

impl Display for ExprSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape_str =
            self.shape.iter()
            .map(|dim| dim.upper().to_string())
            .collect::<Vec<String>>()
            .join(", ");

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

        write!(f, "[{}], {{{}}}[{}]", shape_str, exploded_str, vectorized_str)
    }
}

impl ExprSchedule {
    /// size of a vector (the product of vectorized dims' extents)
    pub fn vector_size(&self) -> usize {
        self.vectorized_dims.iter()
        .fold(1, |acc, dim| acc * dim.extent())
    }

    // materialize schedule into a coordinate map of vectors
    // similar to DummyArrayMaterializer
    pub fn materialize(&self) -> CircuitValue<VectorInfo> {
        if self.exploded_dims.len() > 0 {
            let mut coord_map = IndexCoordinateMap::new(self.exploded_dims.iter());
            for index_map in coord_map.index_map_iter() {
                let vector =
                    VectorInfo::get_expr_vector_at_coord(index_map.clone(), self, None);

                let coord = coord_map.index_map_as_coord(index_map);
                coord_map.set(coord, vector);
            }

            CircuitValue::CoordMap(coord_map)

        } else {
            let index_map: HashMap<DimName, usize> = HashMap::new();
            CircuitValue::Object(
                VectorInfo::get_expr_vector_at_coord(index_map, self, None)
            )
        }
    }
}

impl HasExplodedDims for ExprSchedule {
    fn get_exploded_dims(&self) -> Vec<&ScheduleDim> {
        self.exploded_dims.iter().collect()
    }
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct Schedule {
    pub schedule_map: im::HashMap<IndexingId, IndexingSiteSchedule>
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
        let mut schedule_map: im::HashMap<IndexingId,IndexingSiteSchedule> = im::HashMap::new();
        let dim_class_map: HashMap<ArrayDim, usize> = program.compute_dim_equiv_classes();

        for (_, expr) in program.expr_map.iter() {
            for (indexing_id, transform) in expr.get_indexing_sites() {
                let mut schedule_dims: im::Vector<ScheduleDim> = im::Vector::new();
                for (i, dim) in transform.dims.iter().enumerate() {
                    let class_id = dim_class_map[&(indexing_id.clone(), i)];
                    schedule_dims.push_back(
                        ScheduleDim {
                            index: i,
                            stride: 1,
                            extent: dim.extent(),
                            name: format!("i{}", class_id),
                            pad_left: 0,
                            pad_right: 0,
                        }
                    )
                }

                let schedule =
                    IndexingSiteSchedule {
                        preprocessing: None,
                        exploded_dims: schedule_dims,
                        vectorized_dims: im::Vector::new(),
                    };

                schedule_map.insert(indexing_id, schedule);
            }
        }

        Schedule { schedule_map }
    }

    // apply the schedule to an index-free expression and compute the output schedule
    pub fn compute_output_schedule(&self, program: &TransformedProgram, expr: &TransformedExpr) -> Result<ExprScheduleType, String> {
        match expr {
            TransformedExpr::ReduceNode(reduced_index, _, body) => {
                let body_sched = self.compute_output_schedule(program, body)?;
                Schedule::schedule_reduce(*reduced_index, &body_sched)
            }

            TransformedExpr::Op(_, expr1, expr2) => {
                let sched1 = self.compute_output_schedule(program, expr1)?;
                let sched2 = self.compute_output_schedule(program, expr2)?;
                Schedule::schedule_op(&sched1, &sched2)
            },

            TransformedExpr::Literal(_) => Schedule::schedule_literal(),

            TransformedExpr::ExprRef(indexing_id, transform) => {
                if let Some(indexing_sched) = self.schedule_map.get(indexing_id) {
                    let expr_schedule = indexing_sched.to_expr_schedule(transform.as_shape());
                    Ok(ExprScheduleType::Specific(expr_schedule))

                } else {
                    Err(String::from("indexing site has no schedule"))
                }
            },
        }
    }

    pub fn schedule_literal() -> Result<ExprScheduleType, String> {
        Ok(ExprScheduleType::Any)
    }

    // this performs a join on the "schedule status lattice",
    // where valid schedules are incomparable,
    // any is bottom and invalid is top
    pub fn schedule_op(sched1: &ExprScheduleType, sched2: &ExprScheduleType) -> Result<ExprScheduleType, String> {
        match (sched1, sched2) {
            (ExprScheduleType::Any, ExprScheduleType::Any) => 
                Ok(ExprScheduleType::Any),

            (ExprScheduleType::Any, ExprScheduleType::Specific(sched)) |
            (ExprScheduleType::Specific(sched), ExprScheduleType::Any) =>
                Ok(ExprScheduleType::Specific(sched.clone())),

            (ExprScheduleType::Specific(sched1), ExprScheduleType::Specific(sched2)) => {
                if sched1 == sched2 {
                    Ok(ExprScheduleType::Specific(sched1.clone()))

                } else {
                    Err(String::from("Operand schedules don't match"))
                }
            }
        }
    }

    // TODO support preprocessing
    pub fn schedule_reduce(reduced_index: usize, body_sched: &ExprScheduleType) -> Result<ExprScheduleType, String> {
        match body_sched {
            ExprScheduleType::Any =>
                Err(String::from("Cannot reduce a literal expression")),
            
            ExprScheduleType::Specific(body_sched_spec) => {
                let mut new_exploded_dims: im::Vector<ScheduleDim> = im::Vector::new();

                for dim in body_sched_spec.exploded_dims.iter() {
                    if dim.index == reduced_index {
                        // don't add dimension to the output schedule

                    } else if dim.index > reduced_index { // decrease dim index
                        let mut new_dim = dim.clone();
                        new_dim.index -= 1;
                        new_exploded_dims.push_back(new_dim);

                    } else {
                        new_exploded_dims.push_back(dim.clone());
                    }
                }

                let mut new_vectorized_dims: im::Vector<VectorScheduleDim> = im::Vector::new();
                for (i, dim) in body_sched_spec.vectorized_dims.iter().enumerate() {
                    let new_dim = 
                        match dim {
                            VectorScheduleDim::Filled(sched_dim) => {
                                if sched_dim.index == reduced_index {
                                    // if outermost dim is reduced and there's no padding,
                                    // the dim contents are repeated
                                    if i == 0 && sched_dim.pad_left == 0 && sched_dim.pad_right == 0 {
                                        VectorScheduleDim::ReducedRepeated(sched_dim.extent)

                                    } else {
                                        VectorScheduleDim::Reduced(sched_dim.extent)
                                    }

                                } else if sched_dim.index > reduced_index {
                                    let mut new_sched_dim = sched_dim.clone();
                                    new_sched_dim.index -= 1;
                                    VectorScheduleDim::Filled(new_sched_dim)
                                    
                                } else {
                                    dim.clone()
                                }
                            },

                            VectorScheduleDim::Reduced(_) |
                            VectorScheduleDim::ReducedRepeated(_) => {
                                dim.clone()
                            }
                        };

                    new_vectorized_dims.push_back(new_dim);
                }

                let mut new_shape = body_sched_spec.shape.clone();
                new_shape.remove(reduced_index);

                Ok(
                    ExprScheduleType::Specific(
                        // TODO support preprocessing here
                        ExprSchedule {
                            shape: new_shape,
                            preprocessing: None,
                            exploded_dims: new_exploded_dims,
                            vectorized_dims: new_vectorized_dims
                        }
                    )
                )
            }
        }
    }

    pub fn is_schedule_valid(&self, program: &TransformedProgram) -> bool {
        program.expr_map.iter().all(|(_, expr)| {
            self.compute_output_schedule(program, expr).is_ok()
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::lang::{parser::ProgramParser, index_elim::IndexElimination, elaborated::Elaborator};
    use super::*;

    // generate an initial schedule for a program
    fn test_schedule(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let elaborated = Elaborator::new().run(program);
        let inline_set = elaborated.get_default_inline_set();
        let array_group_map = elaborated.get_default_array_group_map();

        let res =
            IndexElimination::new()
            .run(&inline_set, &array_group_map, elaborated);
        
        assert!(res.is_ok());

        let tprogram = res.unwrap();
        println!("{}", tprogram);

        let init_schedule = Schedule::gen_initial_schedule(&tprogram);
        println!("{}", &init_schedule);

        // the initial schedule should always be valid!
        assert!(init_schedule.is_schedule_valid(&tprogram));
    }

    #[test]
    fn test_imgblur() {
        test_schedule(
        "input img: [16,16]
            for x: 16 {
                for y: 16 {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }"
        );
    }

    #[test]
    fn test_imgblur2() {
        test_schedule(
        "input img: [16,16]
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
            "
        );
    }

    #[test]
    fn test_convolve() {
        test_schedule(
        "input img: [16,16]
            let conv1 = 
                for x: 15 {
                    for y: 15 {
                        img[x][y] + img[x+1][y+1]
                    }
                }
            in
            for x: 14 {
                for y: 14 {
                    conv1[x][y] + conv1[x+1][y+1]
                }
            }
            "
        );
    }

    #[test]
    fn test_matmatmul() {
        test_schedule(
            "input A: [4,4]
            input B: [4,4]
            for i: 4 {
                for j: 4 {
                    sum(for k: 4 { A[i][k] * B[k][j] })
                }
            }"
        );
    }

    #[test]
    fn test_matmatmul2() {
        test_schedule(
            "input A1: [4,4]
            input A2: [4,4]
            input B: [4,4]
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
            "
        );
    }

    #[test]
    fn test_dotprod_pointless() {
        test_schedule(
        "
            input A: [3]
            input B: [3]
            sum(A * B)
            "
        );
    }

    #[test]
    fn test_matvecmul() {
        test_schedule(
        "
            input M: [2,2]
            input v: [2]
            for i: 2 {
                sum(M[i] * v)
            }
            "
        );
    }
}