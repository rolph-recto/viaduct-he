use std::collections::{ HashMap, HashSet };
use std::fmt;
use std::ops::RangeInclusive;
use std::time::*;
use egg::*;
use rand::Rng;

define_language! {
    enum HE {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "rot" = Rot([Id; 2]),
        Symbol(Symbol),
    }
}

impl Copy for HE {}

type EGraph = egg::EGraph<HE, HEData>;

#[derive(Default, Clone, Copy)]
struct HEData;

struct OpSizeFunction;
impl LpCostFunction<HE, HEData> for OpSizeFunction {
    fn node_cost(&mut self, _: &EGraph, _: Id, enode: &HE) -> f64 {
        match enode {
            HE::Num(_) => 0.1,
            HE::Add(_) => 6.0,
            HE::Mul(_) => 15.0,
            HE::Rot(_) => 2.0,
            HE::Symbol(_) => 0.1
        }
    }
}

impl Analysis<HE> for HEData {
    type Data = Option<i32>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |to2: &mut i32, from2: i32| {
            if *to2 == from2 {
                DidMerge { 0: false, 1: false }

            } else {
                println!("attmepting to merge numeric exprs with different values");
                println!("to: {}, from: {}", to2, from2);
                DidMerge { 0: true, 1: true }
            }
        })
    }

    fn make(_egraph: &EGraph, enode: &HE) -> Self::Data {
        if let HE::Num(n) = enode { Some(*n) } else { None }
    }
}

const VEC_LENGTH:i32 = 16;

fn make_rules() -> Vec<Rewrite<HE, HEData>> {
    let mut rules: Vec<Rewrite<HE, HEData>> = vec![
        // bidirectional addition rules
        rewrite!("add-identity"; "(+ ?a 0)" <=> "?a"),
        rewrite!("add-assoc"; "(+ ?a (+ ?b ?c))" <=> "(+ (+ ?a ?b) ?c)"),
        rewrite!("add-commute"; "(+ ?a ?b)" <=> "(+ ?b ?a)"),

        // bidirectional multiplication rules
        rewrite!("mul-identity"; "(* ?a 1)" <=> "?a"),
        rewrite!("mul-assoc"; "(* ?a (* ?b ?c))" <=> "(* (* ?a ?b) ?c)"),
        rewrite!("mul-commute"; "(* ?a ?b)" <=> "(* ?b ?a)"),
        rewrite!("mul-distribute"; "(* (+ ?a ?b) ?c)" <=> "(+ (* ?a ?c) (* ?b ?c))"),

        // bidirectional rotation rules
        rewrite!("rot-distribute-mul"; "(rot (* ?a ?b) ?l)" <=> "(* (rot ?a ?l) (rot ?b ?l))"),
        rewrite!("rot-distribute-add"; "(rot (+ ?a ?b) ?l)" <=> "(+ (rot ?a ?l) (rot ?b ?l))"),
    ].concat();

    rules.extend(vec![
        // unidirectional rules
        rewrite!("mul-annihilator"; "(* ?a 0)" => "0"),

        // constant folding
        rewrite!("add-fold"; "(+ ?a ?b)" => {
            ConstantFold {
                is_add: true,
                a: "?a".parse().unwrap(),
                b: "?b".parse().unwrap()
            }
        } if can_fold("?a", "?b")),

        rewrite!("mul-fold"; "(* ?a ?b)" => {
            ConstantFold {
                is_add: false,
                a: "?a".parse().unwrap(),
                b: "?b".parse().unwrap()
            }
        } if can_fold("?a", "?b")),

        rewrite!("factor-split"; "(* ?a ?b)" => {
            FactorSplit {
                a: "?a".parse().unwrap(),
                b: "?b".parse().unwrap()
            }
        } if can_split_factor("?a", "?b")),

        // rotation of 0 doesn't do anything
        rewrite!("rot-none"; "(rot ?x ?l)" => "?x" if is_zero("?l")),

        // wrap rotation according to vector length
        /*
        rewrite!("rot-wrap"; "(rot ?x ?l)" => {
            RotateWrap { x: "?x".parse().unwrap(), l: "?l".parse().unwrap() }
        } if beyond_vec_length("?l")),
        */

        // squash nested rotations into a single rotation
        rewrite!("rot-squash"; "(rot (rot ?x ?l1) ?l2)" => {
            RotateSquash {
                x: "?x".parse().unwrap(),
                l1: "?l1".parse().unwrap(),
                l2: "?l2".parse().unwrap()
            }
        }),

        // given an operation on rotated vectors,
        // split rotation before and after the operation
        rewrite!("rot-add-split"; "(+ (rot ?x1 ?l1) (rot ?x2 ?l2))" => {
            RotateSplit {
                is_add: true,
                x1: "?x1".parse().unwrap(),
                l1: "?l1".parse().unwrap(),
                x2: "?x2".parse().unwrap(),
                l2: "?l2".parse().unwrap(),
            }
        } if can_split_rot("?l1", "?l2")),

        rewrite!("rot-mul-split"; "(* (rot ?x1 ?l1) (rot ?x2 ?l2))" => {
            RotateSplit {
                is_add: false,
                x1: "?x1".parse().unwrap(),
                l1: "?l1".parse().unwrap(),
                x2: "?x2".parse().unwrap(),
                l2: "?l2".parse().unwrap(),
            }
        } if can_split_rot("?l1", "?l2"))
    ]);

    return rules
}

// This returns a function that implements Condition
fn is_zero(var: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    let zero = HE::Num(0);
    move |egraph, _, subst|
        egraph[subst[var]].nodes.contains(&zero)
}

// This returns a function that implements Condition
fn can_fold(astr: &'static str, bstr: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let avar = astr.parse().unwrap();
    let bvar = bstr.parse().unwrap();
    move |egraph, _, subst| {
        egraph[subst[avar]].data.is_some() && egraph[subst[bvar]].data.is_some()
    }
}

fn can_split_factor(astr: &'static str, bstr: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let avar = astr.parse().unwrap();
    let bvar = bstr.parse().unwrap();
    move |egraph, _, subst| {
        
        match egraph[subst[avar]].data {
            Some(aval) => {
                aval != 0 && egraph[subst[bvar]].data.is_none()
            },
            None => false
        }
    }
}

// This returns a function that implements Condition
fn beyond_vec_length(var: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst: &Subst| {
        let val = egraph[subst[var]].data.unwrap();
        -VEC_LENGTH <= val && val <= VEC_LENGTH
    }
}

fn can_split_rot(l1_str: &'static str, l2_str: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let l1: Var = l1_str.parse().unwrap();
    let l2: Var = l2_str.parse().unwrap();
    move |egraph: &mut EGraph, _, subst: &Subst| {
        match (egraph[subst[l1]].data, egraph[subst[l2]].data) {
            (Some(l1_val), Some(l2_val)) => {
                (l1_val < 0 && l2_val < 0) || (l1_val > 0 && l2_val > 0)
            },

            _ => false
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstantFold { is_add: bool, a: Var, b: Var }

impl Applier<HE, HEData> for ConstantFold {
    fn apply_one(
        &self, egraph: &mut EGraph, matched_id: Id, subst: &Subst,
        _: Option<&PatternAst<HE>>, _: Symbol
    ) -> Vec<Id> {
        let aval: i32 = egraph[subst[self.a]].data.unwrap();
        let bval: i32 = egraph[subst[self.b]].data.unwrap();

        let folded_val =
            if self.is_add { aval + bval } else { aval * bval };
        let folded_id = egraph.add(HE::Num(folded_val));

        if egraph.union(matched_id, folded_id) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FactorSplit { a: Var, b: Var }

impl Applier<HE, HEData> for FactorSplit {
    fn apply_one(
        &self, egraph: &mut EGraph, matched_id: Id, subst: &Subst,
        _: Option<&PatternAst<HE>>, _: Symbol
    ) -> Vec<Id> {
        let factor: i32 = egraph[subst[self.a]].data.unwrap();

        let mut acc: Id = egraph.add(HE::Num(0));
        let mut cur_val = factor;

        let dir = if cur_val > 0 { 1 } else { -1 };
        let dir_id = egraph.add(HE::Num(dir));

        let chunk = egraph.add(HE::Mul([dir_id, subst[self.b]]));

        while cur_val != 0 {
            acc = egraph.add(HE::Add([chunk, acc]));
            cur_val -= dir
        }

        if egraph.union(matched_id, acc) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RotateWrap { x: Var, l: Var }

impl Applier<HE, HEData> for RotateWrap {
    fn apply_one(
        &self, egraph: &mut EGraph, matched_id: Id, subst: &Subst,
        _: Option<&PatternAst<HE>>, _: Symbol
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let lval = egraph[subst[self.l]].data.unwrap();

        let wrapped_lval = egraph.add(HE::Num(lval % VEC_LENGTH));
        let wrapped_rot: Id = egraph.add(HE::Rot([xclass, wrapped_lval]));
        if egraph.union(matched_id, wrapped_rot) {
            vec![matched_id]

        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RotateSquash { x: Var, l1: Var, l2: Var }

impl Applier<HE, HEData> for RotateSquash {
    fn apply_one(
        &self, egraph: &mut EGraph, matched_id: Id, subst: &Subst,
        _: Option<&PatternAst<HE>>, _: Symbol
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let l1_val = egraph[subst[self.l1]].data.unwrap();
        let l2_val = egraph[subst[self.l2]].data.unwrap();

        let lval_sum = egraph.add(HE::Num((l1_val + l2_val) % VEC_LENGTH));
        let rot_sum: Id = egraph.add(HE::Rot([xclass, lval_sum]));

        if egraph.union(matched_id, rot_sum) {
            vec![matched_id]

        } else {
            vec![]
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
struct RotateSplit { is_add: bool, x1: Var, l1: Var, x2: Var, l2: Var }

impl Applier<HE, HEData> for RotateSplit {
    fn apply_one(
        &self, egraph: &mut EGraph, matched_id: Id, subst: &Subst,
        _pattern: Option<&PatternAst<HE>>, _rule: Symbol
    ) -> Vec<Id> {
        let x1_class: Id = subst[self.x1];
        let x2_class: Id = subst[self.x2];

        let l1_val = egraph[subst[self.l1]].data.unwrap();
        let l2_val = egraph[subst[self.l2]].data.unwrap();

        let dir: i32 = if l1_val < 0 { 1 } else { -1 };
        let (mut cur_l1, mut cur_l2) = (l1_val + dir, l2_val + dir);
        let mut outer_rot = -dir;
        let mut has_split = false;

        while l1_val*cur_l1 >= 0 || l2_val*cur_l2 >= 0 {
            let cur_l1_class = egraph.add(HE::Num(cur_l1));
            let cur_l2_class = egraph.add(HE::Num(cur_l2));

            let rot_in1: Id = egraph.add(HE::Rot([x1_class, cur_l1_class]));
            let rot_in2: Id = egraph.add(HE::Rot([x2_class, cur_l2_class]));

            let op: HE =
                if self.is_add {
                    HE::Add([rot_in1, rot_in2])
                } else {
                    HE::Mul([rot_in1, rot_in2])
                };

            let op_class = egraph.add(op);

            let outer_rot_class = egraph.add(HE::Num(outer_rot));
            let rot_outer = egraph.add(HE::Rot([op_class, outer_rot_class]));

            has_split = has_split || egraph.union(matched_id, rot_outer);
            outer_rot += -dir;
            cur_l1 += dir;
            cur_l2 += dir;
        }

        if has_split {
            vec![matched_id]

        } else {
            vec![]
        }
    }
}

#[derive(Copy, Clone)]
enum HEOperand {
    NodeRef(usize),
    ConstSym(Symbol),
    ConstNum(i32),
}


impl fmt::Display for HEOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            HEOperand::NodeRef(i) => format!("i{}", i+1),
            HEOperand::ConstSym(sym) => sym.to_string(),
            HEOperand::ConstNum(n) => n.to_string()
        })
    }
}

#[derive(Copy, Clone)]
enum HEInstr {
    Add(HEOperand, HEOperand),
    Mul(HEOperand, HEOperand),
    Rot(HEOperand, HEOperand),
}

impl HEInstr {
    fn get_operands(&self) -> [&HEOperand; 2] {
        match self {
            HEInstr::Add(op1, op2) |
            HEInstr::Mul(op1, op2) |
            HEInstr::Rot(op1, op2) => {
                [op1, op2]
            },
        }
    }
}

impl fmt::Display for HEInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEInstr::Add(op1, op2) => write!(f, "{} + {}", op1, op2),
            HEInstr::Mul(op1, op2) => write!(f, "{} * {}", op1, op2),
            HEInstr::Rot(op1, op2) => write!(f, "rot {} {}", op1, op2)
        }
    }
}

struct HEProgram { instrs: Vec<HEInstr> }

#[derive(PartialEq, Eq, Clone)]
enum HEValue {
    HEScalar(i32),
    HEVector(Vec<i32>)
}

impl fmt::Display for HEValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEValue::HEScalar(s) => write!(f, "{}", s),
            HEValue::HEVector(v) => write!(f, "{:?}", v)
        }
    }
}

type HESymStore = HashMap<String, HEValue>;
type HERefStore = HashMap<usize, HEValue>;

impl HEProgram {
    /// get the symbols used in this program.
    fn get_symbols(&self) -> HashSet<Symbol> {
        let mut symset = HashSet::new();

        for instr in self.instrs.iter() {
            for op in instr.get_operands() {
                if let HEOperand::ConstSym(sym) = op {
                    symset.insert(*sym);
                }
            }
        }

        symset
    }

    /// generate a random sym store for this program
    fn gen_sym_store(&self, vec_size: usize, range: RangeInclusive<i32>) -> HESymStore {
        let symbols = self.get_symbols();
        let mut sym_store: HESymStore = HashMap::new();
        let mut rng = rand::thread_rng();

        for symbol in symbols {
            let new_vec: Vec<i32> = 
                (0..vec_size).into_iter()
                .map(|_| rng.gen_range(range.clone()))
                .collect();
                
            sym_store.insert(symbol.to_string(), HEValue::HEVector(new_vec));
        }

        sym_store
    }

}

impl fmt::Display for HEProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.instrs.iter().enumerate().map(|(i, instr)| {
            write!(f, "let i{} = {}\n", i+1, instr)
        }).collect()
    }
}

fn gen_program(expr: &RecExpr<HE>) -> HEProgram {
    let mut node_map: HashMap<Id, HEOperand> = HashMap::new();
    let mut program: HEProgram = HEProgram { instrs: Vec::new() };
    let mut instr_index: usize = 0;

    let mut op_processor =
        | nmap: &mut HashMap<Id, HEOperand>,
          id: Id, ctor: fn(HEOperand, HEOperand) -> HEInstr,
          id_op1: &Id, id_op2: &Id |
        {
            let op1 = &nmap[id_op1];
            let op2 = &nmap[id_op2];

            program.instrs.push(ctor(*op1, *op2));
            nmap.insert(id, HEOperand::NodeRef(instr_index));

            instr_index += 1;
        };

    for (i, node) in expr.as_ref().iter().enumerate() {
        let id = Id::from(i);
        match node {
            HE::Num(n) => {
                node_map.insert(id, HEOperand::ConstNum(*n));
            },

            HE::Symbol(sym) => {
                node_map.insert(id, HEOperand::ConstSym(*sym));
            }

            HE::Add([id1, id2]) => {
                op_processor(&mut node_map, id, HEInstr::Add, id1, id2);
            }

            HE::Mul([id1, id2]) => {
                op_processor(&mut node_map, id, HEInstr::Mul, id1, id2);
            }

            HE::Rot([id1, id2]) => {
                op_processor(&mut node_map, id, HEInstr::Rot, id1, id2);
            }
        }
    }

    program
}

fn interp_operand(sym_store: &HESymStore, ref_store: &HERefStore, op: &HEOperand) -> HEValue {
    match op {
        HEOperand::NodeRef(ref_i) =>
            ref_store[ref_i].clone(),

        HEOperand::ConstSym(sym) =>
            sym_store[sym.as_str()].clone(),

        HEOperand::ConstNum(n) =>
            HEValue::HEScalar(*n)
    }
}

fn interp_instr(sym_store: &HESymStore, ref_store: &HERefStore, instr: &HEInstr, vec_size: usize) -> HEValue {
    let exec_binop = |op1: &HEOperand, op2: &HEOperand, f: fn(&i32, &i32)->i32| -> HEValue {
        let val1 = interp_operand(sym_store, ref_store, op1);
        let val2 = interp_operand(sym_store, ref_store, op2);
        match (val1, val2) {
            (HEValue::HEScalar(s1), HEValue::HEScalar(s2)) => {
                HEValue::HEScalar(f(&s1, &s2))
            },

            (HEValue::HEScalar(s1), HEValue::HEVector(v2)) => {
                let new_vec = v2.iter().map(|x| f(x, &s1)).collect();
                HEValue::HEVector(new_vec)
            },

            (HEValue::HEVector(v1), HEValue::HEScalar(s2)) => {
                let new_vec = v1.iter().map(|x| f(x, &s2)).collect();
                HEValue::HEVector(new_vec)
            },

            (HEValue::HEVector(v1), HEValue::HEVector(v2)) => {
                let new_vec = v1.iter().zip(v2).map(|(x1,x2)| f(x1, &x2)).collect();
                HEValue::HEVector(new_vec)
            }
        }
    };

    match instr {
        HEInstr::Add(op1, op2) => {
            exec_binop(op1, op2, |x1, x2| x1 + x2)
        },

        HEInstr::Mul(op1, op2) => {
            exec_binop(op1, op2, |x1, x2| x1 * x2)
        }

        HEInstr::Rot(op1, op2) => {
            let val1 = interp_operand(sym_store, ref_store, op1);
            let val2 = interp_operand(sym_store, ref_store, op2);
            match (val1, val2) {
                (HEValue::HEVector(v1), HEValue::HEScalar(s2)) => {
                    let rot_val = s2 % (vec_size as i32);
                    let mut new_vec: Vec<i32> = v1.clone();
                    if rot_val < 0 {
                        new_vec.rotate_left((-rot_val) as usize)

                    } else {
                        new_vec.rotate_right(rot_val as usize)
                    }

                    HEValue::HEVector(new_vec)
                },

                _ => panic!("Rotate must have vector has 1st operand and scalar as 2nd operand")
            }
        }
    }
}

fn interp_program(sym_store: &HESymStore, program: &HEProgram, vec_size: usize) -> Option<HEValue> {
    let mut ref_store: HERefStore = HashMap::new();

    let mut last_instr = None;
    for (i, instr) in program.instrs.iter().enumerate() {
        let val = interp_instr(&sym_store, &ref_store, instr, vec_size);
        ref_store.insert(i, val);
        last_instr = Some(i);
    }
    
    last_instr.and_then(|i| ref_store.remove(&i))
}

/// parse an expression, simplify it using egg, and pretty print it back out
fn simplify(s: &str) {
    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<HE> = s.parse().unwrap();

    let timeout: u64 = 20;
    println!("Running equality saturation for {} seconds...", timeout);

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let mut runner =
        Runner::default()
        .with_explanations_enabled()
        .with_expr(&expr)
        .with_time_limit(Duration::from_secs(timeout))
        .run(&make_rules());

    let egraph = &mut runner.egraph;
    let root = egraph.add_expr(&expr);

    let mut lp_extractor = LpExtractor::new(egraph, OpSizeFunction);
    let opt_expr = lp_extractor.solve(root);

    /*
    let (best1_cost, best1) =
        Extractor::new(&runner.egraph, AstSize)
        .find_best(root);
    */

    println!(
        "LpExtractor: Simplified from {} with cost {} to {} with cost {}",
        expr, expr.as_ref().len(), opt_expr, opt_expr.as_ref().len()
    );

    let init_prog = gen_program(&expr);

    println!("Initial HE program:");
    println!("{}", init_prog);

    let opt_prog = gen_program(&opt_expr);
    println!("Optimized HE program:");
    println!("{}", opt_prog);

    let vec_size = 16;
    let sym_store: HESymStore = init_prog.gen_sym_store(vec_size, -10..=10);

    let init_out = interp_program(&sym_store, &init_prog, vec_size);
    let opt_out = interp_program(&sym_store, &opt_prog, vec_size);

    // values of the last instructions should be equal
    println!("output for init prog: {}", init_out.unwrap());
    println!("output for opt prog: {}", opt_out.unwrap());
}


fn main() {
    simplify("(+ (+ (+ (rot x -6) (* -1 (rot x -4))) (* 2 (rot x -1))) (+ (+ (rot x 6) (rot x 4)) (* -2 (rot x 1))))");
    // simplify("(+ (+ (+ (rot x -6) (rot x -4)) (rot x -1)) (+ (+ (rot x 6) (rot x 4)) (rot x 1)))");
    // simplify("(+ 2 (* 3 2))")
}