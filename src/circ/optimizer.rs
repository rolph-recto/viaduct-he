use egg::*;
use log::*;
use clap::ValueEnum;
use std::{time::*, cmp::max};

use self::{greedy_extractor::*, lp_extractor::*};

mod greedy_extractor;
mod lp_extractor;

define_language! {
    /// The language used by egg e-graph engine.
    pub enum HEOptimizerCircuit {
        Num(isize),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "rot" = Rot([Id; 2]),
        CiphertextRef(Symbol),
        PlaintextRef(Symbol),
    }
}

#[derive(Clone, ValueEnum)]
pub enum ExtractorType { GREEDY, LP }

pub const MUL_LATENCY: usize = 20;
pub const MUL_PLAIN_LATENCY: usize = 8;
pub const ADD_LATENCY: usize = 8;
pub const ADD_PLAIN_LATENCY: usize = 4;
pub const ROT_LATENCY: usize = 1;
pub const SYM_LATENCY: usize = 0;
pub const NUM_LATENCY: usize = 0;

pub(crate) type HEGraph = egg::EGraph<HEOptimizerCircuit, HEData>;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct HEData {
    constval: Option<isize>,
    muldepth: usize
}

impl Analysis<HEOptimizerCircuit> for HEData {
    type Data = HEData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        if to.constval == from.constval {
            merge_min(&mut to.muldepth, from.muldepth)

        } else {
            // println!("to: {:?}, from: {:?}", to.constval, from.constval);
            panic!("attmepting to merge numeric exprs with different values");
        }
    }

    fn make(egraph: &HEGraph, enode: &HEOptimizerCircuit) -> Self::Data {
        let data = |id: &Id| egraph[*id].data;

        match enode {
            HEOptimizerCircuit::Num(n) =>
                HEData { constval: Some(*n), muldepth: 0 },

            HEOptimizerCircuit::Add([id1, id2]) => {
                let constval: Option<isize> = 
                    data(id1).constval.and_then(|d1|
                        data(id2).constval.map(|d2| d1 + d2)
                    );

                let muldepth = max(data(id1).muldepth, data(id2).muldepth);

                HEData { constval, muldepth }
            },

            HEOptimizerCircuit::Mul([id1, id2]) => {
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

            HEOptimizerCircuit::Rot([id1, id2]) => {
                let muldepth = max(data(id1).muldepth, data(id2).muldepth);
                HEData { constval: egraph[*id1].data.constval, muldepth }
            }

            HEOptimizerCircuit::CiphertextRef(_) => HEData { constval: None, muldepth: 0 },

            HEOptimizerCircuit::PlaintextRef(_) => HEData { constval: None, muldepth: 0 },
        }
    }
}

fn make_rules(size: usize) -> Vec<Rewrite<HEOptimizerCircuit, HEData>> {
    let mut rules: Vec<Rewrite<HEOptimizerCircuit, HEData>> = vec![
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
        } if can_split_rot("?l1", "?l2")),
    ]);

    rules
}

// This returns a function that implements Condition
fn is_zero(var: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    let zero = HEOptimizerCircuit::Num(0);
    move |egraph, _, subst| egraph[subst[var]].nodes.contains(&zero)
}

// This returns a function that implements Condition
fn can_fold(astr: &'static str, bstr: &'static str) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let avar = astr.parse().unwrap();
    let bvar = bstr.parse().unwrap();
    move |egraph, _, subst|
        egraph[subst[avar]].data.constval.is_some() && egraph[subst[bvar]].data.constval.is_some()
}

fn can_split_factor(
    astr: &'static str,
    bstr: &'static str,
) -> impl Fn(&mut HEGraph, Id, &Subst) -> bool {
    let avar = astr.parse().unwrap();
    let bvar = bstr.parse().unwrap();
    move |egraph, _, subst| match egraph[subst[avar]].data.constval {
        Some(aval) => aval != 0 && egraph[subst[bvar]].data.constval.is_none(),
        None => false,
    }
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
struct ConstantFold {
    is_add: bool,
    a: Var,
    b: Var,
}

impl Applier<HEOptimizerCircuit, HEData> for ConstantFold {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptimizerCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let aval: isize = egraph[subst[self.a]].data.constval.unwrap();
        let bval: isize = egraph[subst[self.b]].data.constval.unwrap();

        let folded_val = if self.is_add {
            aval + bval
        } else {
            aval * bval
        };
        let folded_id = egraph.add(HEOptimizerCircuit::Num(folded_val));

        if egraph.union(matched_id, folded_id) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FactorSplit {
    a: Var,
    b: Var,
}

impl Applier<HEOptimizerCircuit, HEData> for FactorSplit {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptimizerCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let factor: isize = egraph[subst[self.a]].data.constval.unwrap();

        let mut acc: Id = egraph.add(HEOptimizerCircuit::Num(0));
        let mut cur_val = factor;

        let dir = if cur_val > 0 { 1 } else { -1 };
        let dir_id = egraph.add(HEOptimizerCircuit::Num(dir));

        let chunk = egraph.add(HEOptimizerCircuit::Mul([dir_id, subst[self.b]]));

        while cur_val != 0 {
            acc = egraph.add(HEOptimizerCircuit::Add([chunk, acc]));
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
struct RotateWrap {
    size: isize,
    x: Var,
    l: Var,
}

impl Applier<HEOptimizerCircuit, HEData> for RotateWrap {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptimizerCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let lval = egraph[subst[self.l]].data.constval.unwrap();

        let wrapped_lval = egraph.add(HEOptimizerCircuit::Num(lval % self.size));
        let wrapped_rot: Id = egraph.add(HEOptimizerCircuit::Rot([xclass, wrapped_lval]));
        if egraph.union(matched_id, wrapped_rot) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RotateSquash {
    size: usize,
    x: Var,
    l1: Var,
    l2: Var,
}

impl Applier<HEOptimizerCircuit, HEData> for RotateSquash {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HEOptimizerCircuit>>,
        _: Symbol,
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let l1_val = egraph[subst[self.l1]].data.constval.unwrap();
        let l2_val = egraph[subst[self.l2]].data.constval.unwrap();

        let lval_sum = egraph.add(HEOptimizerCircuit::Num((l1_val + l2_val) % (self.size as isize)));
        let rot_sum: Id = egraph.add(HEOptimizerCircuit::Rot([xclass, lval_sum]));

        if egraph.union(matched_id, rot_sum) {
            vec![matched_id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RotateSplit {
    is_add: bool,
    x1: Var,
    l1: Var,
    x2: Var,
    l2: Var,
}

impl Applier<HEOptimizerCircuit, HEData> for RotateSplit {
    fn apply_one(
        &self,
        egraph: &mut HEGraph,
        matched_id: Id,
        subst: &Subst,
        _pattern: Option<&PatternAst<HEOptimizerCircuit>>,
        _rule: Symbol,
    ) -> Vec<Id> {
        let x1_class: Id = subst[self.x1];
        let x2_class: Id = subst[self.x2];

        let l1_val = egraph[subst[self.l1]].data.constval.unwrap();
        let l2_val = egraph[subst[self.l2]].data.constval.unwrap();

        let dir: isize = if l1_val < 0 { 1 } else { -1 };
        let (mut cur_l1, mut cur_l2) = (l1_val + dir, l2_val + dir);
        let mut outer_rot = -dir;
        let mut has_split = false;

        while l1_val * cur_l1 >= 0 || l2_val * cur_l2 >= 0 {
            let cur_l1_class = egraph.add(HEOptimizerCircuit::Num(cur_l1));
            let cur_l2_class = egraph.add(HEOptimizerCircuit::Num(cur_l2));

            let rot_in1: Id = egraph.add(HEOptimizerCircuit::Rot([x1_class, cur_l1_class]));
            let rot_in2: Id = egraph.add(HEOptimizerCircuit::Rot([x2_class, cur_l2_class]));

            let op: HEOptimizerCircuit = if self.is_add {
                HEOptimizerCircuit::Add([rot_in1, rot_in2])
            } else {
                HEOptimizerCircuit::Mul([rot_in1, rot_in2])
            };

            let op_class = egraph.add(op);

            let outer_rot_class = egraph.add(HEOptimizerCircuit::Num(outer_rot));
            let rot_outer = egraph.add(HEOptimizerCircuit::Rot([op_class, outer_rot_class]));

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

pub fn optimize(expr: &RecExpr<HEOptimizerCircuit>, size: usize, timeout: usize, extractor_type: ExtractorType) -> RecExpr<HEOptimizerCircuit> {
    info!("running equality saturation for {} seconds...", timeout);

    let optimization_time = Instant::now(); 

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let mut runner = Runner::default()
        .with_explanations_enabled()
        .with_expr(expr)
        .with_time_limit(Duration::from_secs(timeout as u64))
        .run(&make_rules(size));

    info!("Optimization time: {}ms", optimization_time.elapsed().as_millis());

    let egraph = &mut runner.egraph;
    let root = egraph.add_expr(expr);

    let extraction_time = Instant::now();

    let opt_expr =
        match extractor_type {
            ExtractorType::GREEDY => {
                info!("using greedy extractor to derive optimized program...");
                let extractor = GreedyExtractor::new(egraph, HECostFunction { egraph, count: 0 });
                // let extractor = Extractor::new(egraph, HECostFunction { egraph, count: 0 });
                let (_, opt_expr) = extractor.find_best(root);
                opt_expr
            },
            ExtractorType::LP => {
                info!("using LP extractor to derive optimized program...");
                let mut lp_extractor = LpExtractor::new(egraph, OpSizeFunction);
                lp_extractor.solve(root)
            }
        };

    info!("Extraction time: {}ms", extraction_time.elapsed().as_millis());

    opt_expr
}