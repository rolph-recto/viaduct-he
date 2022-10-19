use egg::*;
use log::*;
use clap::ValueEnum;
use std::{time::*, cmp::{max, min}};

use self::{greedy_extractor::*, lp_extractor::*};

use super::HECircuit;

mod greedy_extractor;
pub mod lp_extractor;

define_language! {
    /// The language used by egg e-graph engine.
    pub enum HEOptCircuit {
        Num(isize),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "rot" = Rot([Id; 2]),
        CiphertextRef(Symbol),
        PlaintextRef(Symbol),
    }
}

impl From<HECircuit> for RecExpr<HEOptCircuit> {
    fn from(circ: HECircuit) -> Self {
        circ.to_opt_circuit()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RewriteOp { Add, Sub, Mul }

#[derive(Clone, ValueEnum)]
pub enum ExtractorType { GREEDY, LP }

pub struct HELatencyModel {
    pub add: f64,
    pub add_plain: f64,
    pub sub: f64,
    pub sub_plain: f64,
    pub mul: f64,
    pub mul_plain: f64,
    pub rot: f64,
    pub num: f64,
    pub sym: f64,
}

impl Default for HELatencyModel {
    fn default() -> Self {
        Self {
            add: 8.0,
            add_plain: 4.0,
            sub: 8.0,
            sub_plain: 4.0,
            mul: 20.0,
            mul_plain: 8.0,
            rot: 1.0,
            num: 0.1,
            sym: 0.1,
        }
    }
}

pub(crate) type HEGraph = egg::EGraph<HEOptCircuit, HEData>;

#[derive(Debug, Default, Clone, Copy)]
pub struct HEData {
    constval: Option<isize>,
    muldepth: usize
}

impl Analysis<HEOptCircuit> for HEData {
    type Data = HEData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        if to.constval == from.constval {
            merge_min(&mut to.muldepth, from.muldepth)

        } else {
            panic!("attmepting to merge numeric exprs with different values: {:?} and {:?} ", to.constval, from.constval);
        }
        /*
        match (to.constval, from.constval) {
            (None, None) | (Some(_), None) => {},
            (None, Some(from_val)) => {
                to.constval = Some(from_val);
            },
            (Some(to_val), Some(from_val)) => {
                if to_val != from_val {
                    panic!("attmepting to merge numeric exprs with different values: {:?} and {:?} ", to_val, from_val);
                }
            }
        }

        to.muldepth = min(to.muldepth, from.muldepth);
        DidMerge(true, true)
        */
    }

    fn make(egraph: &HEGraph, enode: &HEOptCircuit) -> Self::Data {
        let data = |id: &Id| egraph[*id].data;

        match enode {
            HEOptCircuit::Num(n) =>
                HEData { constval: Some(*n), muldepth: 0 },

            HEOptCircuit::Add([id1, id2]) => {
                let constval: Option<isize> = 
                    data(id1).constval.and_then(|d1|
                        data(id2).constval.map(|d2| d1 + d2)
                    );

                let muldepth = max(data(id1).muldepth, data(id2).muldepth);

                HEData { constval, muldepth }
            },

            HEOptCircuit::Sub([id1, id2]) => {
                let constval: Option<isize> = 
                    data(id1).constval.and_then(|d1|
                        data(id2).constval.map(|d2| d1 - d2)
                    );

                let muldepth = max(data(id1).muldepth, data(id2).muldepth);

                HEData { constval, muldepth }
            },

            HEOptCircuit::Mul([id1, id2]) => {
                let v1 = data(id1).constval;
                let v2 = data(id2).constval;

                // special case multiplicative annihilator (0)
                let constval: Option<isize> = 
                    match (v1, v2) {
                        (Some(0), _) => Some(0),
                        (_, Some(0)) => Some(0),
                        (Some(d1), Some(d2)) => Some(d1 * d2),
                        _ => None
                    };

                // don't count scalar multiplication as muldepth
                let muldepth =
                    if constval.is_none() {
                        max(data(id1).muldepth, data(id2).muldepth) + 1
                    } else {
                        0
                    };

                HEData { constval, muldepth }
            },

            HEOptCircuit::Rot([id1, id2]) => {
                let muldepth = max(data(id1).muldepth, data(id2).muldepth);
                HEData { constval: egraph[*id1].data.constval, muldepth }
            }

            HEOptCircuit::CiphertextRef(_) => HEData { constval: None, muldepth: 0 },

            HEOptCircuit::PlaintextRef(_) => HEData { constval: None, muldepth: 0 },
        }
    }
}

// This returns a function that implements Condition
fn is_zero(var: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    let zero = HEOptCircuit::Num(0);
    move |egraph, _, subst| egraph[subst[var]].nodes.contains(&zero)
}

fn is_constant(str: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let var = str.parse().unwrap();
    move |egraph, _, subst|
        egraph[subst[var]].data.constval.is_some()
}

fn has_constant_factor(
    astr: &'static str,
    bstr: &'static str,
) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let avar = astr.parse().unwrap();
    let bvar = bstr.parse().unwrap();
    move |egraph, _, subst| {
        let a_id = egraph.find(subst[avar]);
        let b_id = egraph.find(subst[bvar]);
        match (egraph[a_id].data.constval, egraph[b_id].data.constval) {
            (None, Some(bval)) => true,
            _ => false
        }
    }
}

// This returns a function that implements Condition
fn can_fold(astr: &'static str, bstr: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let avar = astr.parse().unwrap();
    let bvar = bstr.parse().unwrap();
    move |egraph, _, subst|
        egraph[subst[avar]].data.constval.is_some() && egraph[subst[bvar]].data.constval.is_some()
}

fn can_split_rot(
    l1_str: &'static str,
    l2_str: &'static str,
) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let l1: Var = l1_str.parse().unwrap();
    let l2: Var = l2_str.parse().unwrap();
    move |egraph: &mut HEGraph, _, subst: &Subst| match (
        egraph[subst[l1]].data.constval,
        egraph[subst[l2]].data.constval,
    ) {
        (Some(l1_val), Some(l2_val)) => (l1_val < 0 && l2_val < 0) || (l1_val > 0 && l2_val > 0),

        _ => false,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AddToMul { a: Var, b: Var }

impl Applier<HEOptCircuit, HEData> for AddToMul {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let a_id = subst[self.a];
        let b_id = egraph.find(subst[self.b]);
        let bval = egraph[b_id].data.constval.unwrap();

        let mut changed = false;
        if bval != -1 {
            let b_inc_id = egraph.add(HEOptCircuit::Num(bval + 1));
            let mul_id = egraph.add(HEOptCircuit::Mul([a_id, b_inc_id]));
            changed = changed || egraph.union(matched_id, mul_id);
        }

        if changed {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MulToAdd { a: Var, b: Var }

impl Applier<HEOptCircuit, HEData> for MulToAdd {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let a_id = subst[self.a];
        let b_id = subst[self.b];
        let bval = egraph[b_id].data.constval.unwrap();

        let mut changed = false;
        if bval != 0 {
            for i in 1..bval.abs() {
                let cur_b = if bval > 0 { bval - i } else { bval + i };
                let cur_b_id = egraph.add(HEOptCircuit::Num(cur_b));

                let mut acc = egraph.add(HEOptCircuit::Mul([a_id, cur_b_id]));
                for _ in 0..i {
                    if bval > 0 {
                        acc = egraph.add(HEOptCircuit::Add([acc, a_id]));
                    } else {
                        acc = egraph.add(HEOptCircuit::Sub([acc, a_id]));
                    }
                }

                changed = changed || egraph.union(matched_id, acc);
            }
        } 

        if changed {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SubInverse { a: Var, b: Var }

impl Applier<HEOptCircuit, HEData> for SubInverse {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let a_id = subst[self.a];
        let b_id = subst[self.b];

        let neg_one = egraph.add(HEOptCircuit::Num(-1));
        let neg_b = egraph.add(HEOptCircuit::Mul([neg_one, b_id]));
        let a_plus_neg_b = egraph.add(HEOptCircuit::Add([a_id, neg_b]));

        if egraph.union(matched_id, a_plus_neg_b) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstantFold { op: RewriteOp, a: Var, b: Var }

impl Applier<HEOptCircuit, HEData> for ConstantFold {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let aval: isize = egraph[subst[self.a]].data.constval.unwrap();
        let bval: isize = egraph[subst[self.b]].data.constval.unwrap();

        let folded_val =
            match self.op {
                RewriteOp::Add => aval + bval,
                RewriteOp::Sub => aval - bval,
                RewriteOp::Mul => aval * bval,
            };
        
        let folded_id = egraph.add(HEOptCircuit::Num(folded_val));

        if egraph.union(matched_id, folded_id) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FactorSplit { a: Var, b: Var }

impl Applier<HEOptCircuit, HEData> for FactorSplit {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let factor: isize = egraph[subst[self.a]].data.constval.unwrap();

        let mut acc: Id = egraph.add(HEOptCircuit::Num(0));
        let mut cur_val = factor;

        let dir = if cur_val > 0 { 1 } else { -1 };
        let dir_id = egraph.add(HEOptCircuit::Num(dir));

        let chunk = egraph.add(HEOptCircuit::Mul([dir_id, subst[self.b]]));

        while cur_val != 0 {
            acc = egraph.add(HEOptCircuit::Add([chunk, acc]));
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
struct RotateWrap { size: isize, x: Var, l: Var }

impl Applier<HEOptCircuit, HEData> for RotateWrap {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let lval = egraph[subst[self.l]].data.constval.unwrap();

        let wrapped_lval = egraph.add(HEOptCircuit::Num(lval % self.size));
        let wrapped_rot: Id = egraph.add(HEOptCircuit::Rot([xclass, wrapped_lval]));
        if egraph.union(matched_id, wrapped_rot) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RotateSquash { size: usize, x: Var, l1: Var, l2: Var }

impl Applier<HEOptCircuit, HEData> for RotateSquash {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let l1_val = egraph[subst[self.l1]].data.constval.unwrap();
        let l2_val = egraph[subst[self.l2]].data.constval.unwrap();

        let lval_sum = egraph.add(HEOptCircuit::Num((l1_val + l2_val) % (self.size as isize)));
        let rot_sum: Id = egraph.add(HEOptCircuit::Rot([xclass, lval_sum]));

        if egraph.union(matched_id, rot_sum) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RotateSplit {
    op: RewriteOp,
    x1: Var,
    l1: Var,
    x2: Var,
    l2: Var,
}

impl Applier<HEOptCircuit, HEData> for RotateSplit {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _pattern: Option<&PatternAst<HEOptCircuit>>,
        _rule: Symbol,
    ) -> Vec<Id> {
        let x1_class: Id = subst[self.x1];
        let x2_class: Id = subst[self.x2];

        let l1_val = egraph[subst[self.l1]].data.constval.unwrap();
        let l2_val = egraph[subst[self.l2]].data.constval.unwrap();

        assert!(l1_val >= 0, "l1_val is {}", l1_val);
        assert!(l2_val >= 0, "l2_val is {}", l2_val);

        let max_outer = min(l1_val as usize, l2_val as usize);
        let mut has_split = false;
        for i in 1..max_outer {
            let cur_l1 = l1_val - (i as isize);
            let cur_l2 = l2_val - (i as isize);

            // recall that (rot x 0 ) = x
            let rot_in1 =
                if cur_l1 != 0 {
                    let cur_l1_class = egraph.add(HEOptCircuit::Num(cur_l1));
                    egraph.add(HEOptCircuit::Rot([x1_class, cur_l1_class]))

                } else {
                    x1_class
                };

            let rot_in2 = 
                if cur_l2 != 0 {
                    let cur_l2_class = egraph.add(HEOptCircuit::Num(cur_l2));
                    egraph.add(HEOptCircuit::Rot([x2_class, cur_l2_class]))
                } else {
                    x2_class
                };

            let op: HEOptCircuit =
                match self.op {
                    RewriteOp::Add =>
                        HEOptCircuit::Add([rot_in1, rot_in2]),

                    RewriteOp::Sub =>
                        HEOptCircuit::Sub([rot_in1, rot_in2]),

                    RewriteOp::Mul =>
                        HEOptCircuit::Add([rot_in1, rot_in2]),
                };

            let op_class = egraph.add(op);
            let outer_rot_class = egraph.add(HEOptCircuit::Num(i as isize));
            let rot_outer = egraph.add(HEOptCircuit::Rot([op_class, outer_rot_class]));
            has_split = has_split || egraph.union(matched_id, rot_outer);
        }

        if has_split {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

pub struct Optimizer {
    rules: Vec<Rewrite<HEOptCircuit, HEData>>
}

impl Optimizer {
    pub fn new(size: usize) -> Self {
        let mut rules: Vec<Rewrite<HEOptCircuit, HEData>> = vec![
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
        ]
        .concat();

        rules.extend(vec![
            // unidirectional rules
            rewrite!("mul-annihilator"; "(* ?a 0)" => "0"),

            // a - b = a + (-1 * b)
            rewrite!("sub-inverse"; "(- ?a ?b)" => {
                SubInverse {
                    a: "?a".parse().unwrap(),
                    b: "?b".parse().unwrap(),
                }
            }),

            // x + (x * c) = x * (c + 1), where c is a constant
            rewrite!("add-to-mul"; "(+ ?a (* ?a ?b))" => {
                AddToMul {
                    a: "?a".parse().unwrap(),
                    b: "?b".parse().unwrap()
                }
            } if has_constant_factor("?a", "?b")),

            // x * c = x + (x * (c - 1)), where c is a constant
            rewrite!("mul-to-add"; "(* ?a ?b)" => {
                MulToAdd {
                    a: "?a".parse().unwrap(),
                    b: "?b".parse().unwrap()
                }
            } if has_constant_factor("?a", "?b")),

            // constant folding
            rewrite!("add-fold"; "(+ ?a ?b)" => {
                ConstantFold {
                    op: RewriteOp::Add,
                    a: "?a".parse().unwrap(),
                    b: "?b".parse().unwrap()
                }
            } if can_fold("?a", "?b")),

            rewrite!("sub-fold"; "(- ?a ?b)" => {
                ConstantFold {
                    op: RewriteOp::Sub,
                    a: "?a".parse().unwrap(),
                    b: "?b".parse().unwrap()
                }
            } if can_fold("?a", "?b")),

            rewrite!("mul-fold"; "(* ?a ?b)" => {
                ConstantFold {
                    op: RewriteOp::Mul,
                    a: "?a".parse().unwrap(),
                    b: "?b".parse().unwrap()
                }
            } if can_fold("?a", "?b")),

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
                    size,
                    x: "?x".parse().unwrap(),
                    l1: "?l1".parse().unwrap(),
                    l2: "?l2".parse().unwrap()
                }
            }),

            // given an operation on rotated vectors,
            // split rotation before and after the operation
            rewrite!("rot-add-split"; "(+ (rot ?x1 ?l1) (rot ?x2 ?l2))" => {
                RotateSplit {
                    op: RewriteOp::Add,
                    x1: "?x1".parse().unwrap(),
                    l1: "?l1".parse().unwrap(),
                    x2: "?x2".parse().unwrap(),
                    l2: "?l2".parse().unwrap(),
                }
            } if can_split_rot("?l1", "?l2")),

            rewrite!("rot-sub-split"; "(- (rot ?x1 ?l1) (rot ?x2 ?l2))" => {
                RotateSplit {
                    op: RewriteOp::Sub,
                    x1: "?x1".parse().unwrap(),
                    l1: "?l1".parse().unwrap(),
                    x2: "?x2".parse().unwrap(),
                    l2: "?l2".parse().unwrap(),
                }
            } if can_split_rot("?l1", "?l2")),

            rewrite!("rot-mul-split"; "(* (rot ?x1 ?l1) (rot ?x2 ?l2))" => {
                RotateSplit {
                    op: RewriteOp::Mul,
                    x1: "?x1".parse().unwrap(),
                    l1: "?l1".parse().unwrap(),
                    x2: "?x2".parse().unwrap(),
                    l2: "?l2".parse().unwrap(),
                }
            } if can_split_rot("?l1", "?l2")),
        ]);

        Optimizer { rules }
    }

    pub fn rules(&self) -> &[Rewrite<HEOptCircuit,HEData>] {
        &self.rules
    }

    pub fn optimize(&self, expr: &RecExpr<HEOptCircuit>, size: usize, timeout: usize, extractor_type: ExtractorType) -> RecExpr<HEOptCircuit> {
        info!("running equality saturation for {} seconds...", timeout);

        let optimization_time = Instant::now(); 

        // simplify the expression using a Runner, which creates an e-graph with
        // the given expression and runs the given rules over it
        let mut runner = Runner::default()
            .with_explanations_enabled()
            .with_expr(expr)
            .with_time_limit(Duration::from_secs(timeout as u64))
            .run(&self.rules);

        info!("Optimization time: {}ms", optimization_time.elapsed().as_millis());

        let egraph = &mut runner.egraph;
        let root = egraph.add_expr(expr);

        let extraction_time = Instant::now();

        let opt_expr =
            match extractor_type {
                ExtractorType::GREEDY => {
                    info!("using greedy extractor to derive optimized program...");
                    // let extractor = GreedyExtractor::new(egraph, HECostFunction { egraph, count: 0 });
                    // let extractor = Extractor::new(egraph, HECostFunction { egraph, latency: HELatencyModel::default() });
                    let extractor = Extractor::new(egraph, HECostFunction { egraph, latency: HELatencyModel::default() });
                    let (_, opt_expr) = extractor.find_best(root);
                    info!("optimized solution found: {}", opt_expr.pretty(20));
                    opt_expr
                },
                ExtractorType::LP => {
                    info!("using LP extractor to derive optimized program...");
                    // let mut lp_extractor = LpExtractor::new(egraph, OpSizeFunction { latency: HELatencyModel::default() });
                    let mut lp_extractor = LpExtractor::new(egraph, AstSize);
                    let solution = lp_extractor.solve(root);
                    // let mut lp_extractor = HEExtractor::new(egraph, root, HELatencyModel::default());
                    // let solution = lp_extractor.solve();
                    info!("optimized solution found: {}", solution.pretty(20));
                    solution
                }
            };

        info!("Extraction time: {}ms", extraction_time.elapsed().as_millis());

        opt_expr
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    
    fn run_equiv(s1: &str, s2: &str) -> (bool, String) {
        let optimizer = Optimizer::new(16);
        let expr1 = s1.parse().unwrap();
        let expr2 = s2.parse().unwrap();
        let mut runner =
            Runner::default()
            .with_explanations_enabled()
            .with_expr(&expr1)
            .run(optimizer.rules());

        let equiv = runner.egraph.equivs(&expr1, &expr2).len() > 0;
        if equiv {
            (true, runner.explain_equivalence(&expr1, &expr2).get_flat_string())

        } else {
            (false, String::from(""))
        }
    }

    fn run_extractor(s: &str) -> RecExpr<HEOptCircuit> {
        let optimizer = Optimizer::new(16);
        let expr = s.parse().unwrap();
        let runner =
            Runner::default()
            // .with_explanations_enabled()
            .with_expr(&expr)
            .run(optimizer.rules());
        let root = *runner.roots.first().unwrap();

        // let mut extractor = HEExtractor::new(&runner.egraph, root);
        // extractor.solve().unwrap()

        let mut extractor =
            LpExtractor::new(
                &runner.egraph, 
                OpSizeFunction { latency: HELatencyModel::default() }
            );
        extractor.solve(root)
    }

    // #[ignore] ensures that these long-running equality saturation tests
    // don't run when calling `cargo test`.

    #[test]
    #[ignore]
    fn test_mul_to_add() {
        assert!(run_equiv("(* x 2)", "(+ x x)").0);
    }

    #[test]
    #[ignore]
    fn test_add_to_mul() {
        assert!(run_equiv("(+ (+ x x) x)", "(* x 3)").0);
    }

    #[test]
    #[ignore]
    fn test_factor() {
        assert!(run_equiv("(+ (* x x) (* 2 x))", "(* x (+ x 2))").0);
    }

    #[test]
    #[ignore]
    fn test_constant_fold() {
        assert!(run_equiv("(+ x (* 2 3))", "(+ x 6)").0);
    }

    #[test]
    #[ignore]
    fn test_neg_equiv() {
        assert!(!run_equiv("(+ (* x x) (* 2 x))", "(+ (+ x x) x)").0);
    }

    #[test]
    #[ignore]
    fn test_neg_equiv2() {
        let (equiv, explain) = run_equiv("(+ (* x x) (* 2 x))", "(* x (+ 3 x))");
        println!("{}", explain);
        assert!(!equiv);
    }

    #[test]
    #[ignore]
    fn test_extract() {
        let res = run_extractor("(+ (* x x) (* 2 x))");
        println!("{}", res);
    }
}