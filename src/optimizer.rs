use egg::*;
use std::{time::*, cmp::max};

use crate::lang::*;

type EGraph = egg::EGraph<HE, HEData>;

#[derive(Debug, Default, Clone, Copy)]
struct HEData {
    constval: Option<i32>,
    muldepth: usize
}

struct OpSizeFunction;
impl LpCostFunction<HE, HEData> for OpSizeFunction {
    fn node_cost(&mut self, egraph: &EGraph, id: Id, enode: &HE) -> f64 {
        match enode {
            HE::Num(_) => 0.1,

            HE::Add([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                6.0 * (muldepth as f64)
            },

            HE::Mul([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                15.0 * (muldepth as f64)
            },

            HE::Rot([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                0.1 * (muldepth as f64)
            },

            HE::Symbol(_) => 0.1,
        }
    }
}

impl Analysis<HE> for HEData {
    type Data = HEData;

    /*
    fn pre_union(egraph: &EGraph, id1: Id, id2: Id) {
        let n1 =
            egraph[id1].nodes.first().unwrap().build_recexpr(|cid|
                egraph[cid].nodes.first().unwrap().clone()
            );
        let n2 =
            egraph[id2].nodes.first().unwrap().build_recexpr(|cid|
                egraph[cid].nodes.first().unwrap().clone()
            );
        println!("merging {} {}", n1, n2);
    }
    */

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        if to.constval == from.constval {
            merge_min(&mut to.muldepth, from.muldepth)

        } else {
            // println!("to: {:?}, from: {:?}", to.constval, from.constval);
            panic!("attmepting to merge numeric exprs with different values");
        }
    }

    fn make(egraph: &EGraph, enode: &HE) -> Self::Data {
        let data = |id: &Id| egraph[*id].data;

        match enode {
            HE::Num(n) =>
                HEData { constval: Some(*n), muldepth: 0 },

            HE::Add([id1, id2]) => {
                let constval: Option<i32> = 
                    data(id1).constval.and_then(|d1|
                        data(id2).constval.and_then(|d2|
                            Some(d1 + d2)
                        )
                    );

                let muldepth = max(data(id1).muldepth, data(id2).muldepth);

                HEData { constval, muldepth }
            },

            HE::Mul([id1, id2]) => {
                let v1 = data(id1).constval;
                let v2 = data(id2).constval;

                // special case multiplicative annihilator (0)
                let constval: Option<i32> = 
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

            HE::Rot([id1, id2]) => {
                let muldepth = max(data(id1).muldepth, data(id2).muldepth);
                HEData { constval: egraph[*id1].data.constval, muldepth }
            }

            HE::Symbol(_) => HEData { constval: None, muldepth: 0 }
        }
    }
}

const VEC_LENGTH: i32 = 16;

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

    return rules;
}

// This returns a function that implements Condition
fn is_zero(var: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    let zero = HE::Num(0);
    move |egraph, _, subst| egraph[subst[var]].nodes.contains(&zero)
}

// This returns a function that implements Condition
fn can_fold(astr: &'static str, bstr: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let avar = astr.parse().unwrap();
    let bvar = bstr.parse().unwrap();
    move |egraph, _, subst|
        egraph[subst[avar]].data.constval.is_some() && egraph[subst[bvar]].data.constval.is_some()
}

fn can_split_factor(
    astr: &'static str,
    bstr: &'static str,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let avar = astr.parse().unwrap();
    let bvar = bstr.parse().unwrap();
    move |egraph, _, subst| match egraph[subst[avar]].data.constval {
        Some(aval) => aval != 0 && egraph[subst[bvar]].data.constval.is_none(),
        None => false,
    }
}

// This returns a function that implements Condition
fn beyond_vec_length(var: &'static str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst: &Subst| {
        let val = egraph[subst[var]].data.constval.unwrap();
        -VEC_LENGTH <= val && val <= VEC_LENGTH
    }
}

fn can_split_rot(
    l1_str: &'static str,
    l2_str: &'static str,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let l1: Var = l1_str.parse().unwrap();
    let l2: Var = l2_str.parse().unwrap();
    move |egraph: &mut EGraph, _, subst: &Subst| match (
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

impl Applier<HE, HEData> for ConstantFold {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HE>>,
        _: Symbol,
    ) -> Vec<Id> {
        let aval: i32 = egraph[subst[self.a]].data.constval.unwrap();
        let bval: i32 = egraph[subst[self.b]].data.constval.unwrap();

        let folded_val = if self.is_add {
            aval + bval
        } else {
            aval * bval
        };
        let folded_id = egraph.add(HE::Num(folded_val));

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

impl Applier<HE, HEData> for FactorSplit {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HE>>,
        _: Symbol,
    ) -> Vec<Id> {
        let factor: i32 = egraph[subst[self.a]].data.constval.unwrap();

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
struct RotateWrap {
    x: Var,
    l: Var,
}

impl Applier<HE, HEData> for RotateWrap {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HE>>,
        _: Symbol,
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let lval = egraph[subst[self.l]].data.constval.unwrap();

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
struct RotateSquash {
    x: Var,
    l1: Var,
    l2: Var,
}

impl Applier<HE, HEData> for RotateSquash {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        matched_id: Id,
        subst: &Subst,
        _: Option<&PatternAst<HE>>,
        _: Symbol,
    ) -> Vec<Id> {
        let xclass: Id = subst[self.x];
        let l1_val = egraph[subst[self.l1]].data.constval.unwrap();
        let l2_val = egraph[subst[self.l2]].data.constval.unwrap();

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
struct RotateSplit {
    is_add: bool,
    x1: Var,
    l1: Var,
    x2: Var,
    l2: Var,
}

impl Applier<HE, HEData> for RotateSplit {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        matched_id: Id,
        subst: &Subst,
        _pattern: Option<&PatternAst<HE>>,
        _rule: Symbol,
    ) -> Vec<Id> {
        let x1_class: Id = subst[self.x1];
        let x2_class: Id = subst[self.x2];

        let l1_val = egraph[subst[self.l1]].data.constval.unwrap();
        let l2_val = egraph[subst[self.l2]].data.constval.unwrap();

        let dir: i32 = if l1_val < 0 { 1 } else { -1 };
        let (mut cur_l1, mut cur_l2) = (l1_val + dir, l2_val + dir);
        let mut outer_rot = -dir;
        let mut has_split = false;

        while l1_val * cur_l1 >= 0 || l2_val * cur_l2 >= 0 {
            let cur_l1_class = egraph.add(HE::Num(cur_l1));
            let cur_l2_class = egraph.add(HE::Num(cur_l2));

            let rot_in1: Id = egraph.add(HE::Rot([x1_class, cur_l1_class]));
            let rot_in2: Id = egraph.add(HE::Rot([x2_class, cur_l2_class]));

            let op: HE = if self.is_add {
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

pub(crate) fn optimize(expr: &RecExpr<HE>, timeout: u64) -> (usize, RecExpr<HE>) {
    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let mut runner = Runner::default()
        .with_explanations_enabled()
        .with_expr(&expr)
        .with_time_limit(Duration::from_secs(timeout))
        .run(&make_rules());

    let egraph = &mut runner.egraph;
    let root = egraph.add_expr(&expr);

    // let mut extractor = Extractor::new(egraph, AstSize);
    // let (opt_cost, opt_expr) = extractor.find_best(root);

    let mut lp_extractor = LpExtractor::new(egraph, OpSizeFunction);
    let opt_expr = lp_extractor.solve(root);

    return (0, opt_expr);
}